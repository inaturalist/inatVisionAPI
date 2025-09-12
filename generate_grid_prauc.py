import os
from pathlib import Path

import tifffile
import geopandas as gpd
import pandas as pd
import numpy as np
import h3pandas
from tqdm.auto import tqdm
import tensorflow as tf
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

from lib.model_taxonomy_dataframe import ModelTaxonomyDataframe
from lib.tf_gp_elev_model import TFGeoPriorModelElev
from taxon_range_evaluation import get_prauc

dataset_dir = Path("/data-ssd/alex/datasets/vision-export-20240929050006-aka-2.17")

elevation_file = "wc2.1_5m_elev.tif"

taxon_range_recalls_file = dataset_dir / "taxon_range_recalls.csv"
taxon_range_indicies = dataset_dir / "taxon_range_csvs/"
tax_file = dataset_dir / "taxonomy.csv"
model_file = "/data-ssd/alex/experiments/geo_prior_tf/2_17/2.17_sample_weights_1024_400e_0_001lr_elev_1733549204/saved_model.h5"
h3_resolution = 4
output_file = "grid_prauc_outputs.csv"


def generate_df_from_elevation_tif(elevation_file):
    im = tifffile.imread(elevation_file)
    
    im_df = pd.DataFrame(im)
    im_df.index = np.linspace(90, -90, 2160)
    im_df.columns = np.linspace(-180, 180, 4320)
    im_df = im_df.reset_index()
    im_df = im_df.melt(
        id_vars=["index"],
    )
    im_df.columns = ["lat", "lng", "elevation"]
    
    print("processing h3 and making features...")
    elev_dfh3 = im_df.h3.geo_to_h3(h3_resolution)
    elev_dfh3 = elev_dfh3.drop(
        columns=["lng", "lat"]
    ).groupby("h3_0" + str(h3_resolution)).mean()
    gdfk = elev_dfh3.h3.h3_to_geo()
    gdfk["lng"] = gdfk["geometry"].x
    gdfk["lat"] = gdfk["geometry"].y
    _ = gdfk.pop("geometry")
    gdfk = gdfk.rename_axis("h3index")

    return gdfk
    
def make_sinr_tax(sinr_tax_file):
    sinr_tax = pd.read_json(sinr_tax_file)
    sinr_tax.reset_index(inplace=True)
    sinr_tax.rename({"index": "spatial_class_id"}, axis=1, inplace=True)
    return sinr_tax

def get_prauc(gdfb, tr_h3, plot):
    bp_h3 = gdfb.copy()
    if bp_h3.shape[0] == 0:
        return None
    auc_presences = bp_h3[bp_h3.index.isin(tr_h3.index)]["pred"]
    auc_absences = bp_h3[~bp_h3.index.isin(tr_h3.index)]["pred"]
    test = np.concatenate(([1] * len(auc_presences), [0] * len(auc_absences)))
    predictions = np.concatenate((auc_presences, auc_absences))
    precision, recall, thresholds = precision_recall_curve(test, predictions)
    p1 = (2 * precision * recall)
    p2 = (precision + recall)
    out = np.zeros((len(p1)))
    fscore = np.divide(p1, p2, out=out, where=p2 != 0)
    index = np.argmax(fscore)
    prthres = thresholds[index]
    prf1 = fscore[index]
    prprecision = precision[index]
    prrecall = recall[index]
    prauc = auc(recall, precision)
    if plot is True:
        print("PR AUC: " + str(prauc))
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color="purple")
        ax.plot([recall[index]], [precision[index]], color="green", marker="o")
        ax.set_title("Precision-Recall Curve")
        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")
        plt.show()
    return prauc, prthres, prf1, prprecision, prrecall


def process_prauc(taxon_id, taxon_range_recalls, taxon_range_indicies, tfgpm, sinr_tax, gdfk, feats):
    if taxon_range_recalls[taxon_range_recalls.taxon_id.eq(taxon_id)].shape[0] == 0:
        return None

    if taxon_range_recalls[
        (taxon_range_recalls["taxon_id"] == taxon_id) & (taxon_range_recalls["recall"] > 0.9)
    ].empty:
        return None
        
    tri = taxon_range_indicies / f"{taxon_id}.csv"
    if not os.path.exists(tri):
        return None

    taxon_range_index = pd.read_csv(tri, header=None)
    taxon_range_index.rename(columns={0: "h3index_new"}, inplace=True)
    tr_h3 = gdfk.loc[gdfk.index.isin(taxon_range_index.h3index_new)]
    
    try:
        class_of_interest = sinr_tax[sinr_tax.taxon_id==taxon_id].iloc[0]["spatial_class_id"]
    except Exception:
        print(f"{taxon_id} is not in the taxonomy")
        return None

    try:
        preds = tfgpm.eval_one_class_elevation_from_features(feats, class_of_interest)
    except Exception as e:
        return None
    
    gdfk["pred"] = tf.squeeze(preds).numpy()

    auc_presences = gdfk[gdfk.index.isin(tr_h3.index)]["pred"]
    auc_absences = gdfk[~gdfk.index.isin(tr_h3.index)]["pred"]    
    test = np.concatenate(([1] * len(auc_presences), [0] * len(auc_absences)))
    predictions = np.concatenate((auc_presences, auc_absences))
    precision, recall, thresholds = precision_recall_curve(test, predictions)
    return auc(recall, precision)

def main():
    taxon_range_recalls = pd.read_csv(taxon_range_recalls_file)
    tfgpm = TFGeoPriorModelElev(model_file)
    tax = pd.read_csv(tax_file)
    
    gdfk = generate_df_from_elevation_tif(elevation_file)
    feats = tfgpm.features_for_one_class_elevation(
        latitude=list(gdfk.lat),
        longitude=list(gdfk.lng),
        elevation=list(gdfk.elevation),
    )

    praucs = {}

    taxon_ids = sorted(list(tax[~tax.leaf_class_id.isna()].taxon_id.values))
    taxon_ids = taxon_ids[:100]
    for i, taxon_id in tqdm(enumerate(taxon_ids), total=len(taxon_ids)):
        prauc = process_prauc(
            taxon_id, 
            taxon_range_recalls, 
            taxon_range_indicies, 
            tfgpm, 
            tax, 
            gdfk, 
            feats
        )
        
        if prauc is not None:
            praucs[taxon_id] = prauc

    df = pd.DataFrame({
        "taxon_id": praucs.keys(),
        "pr_auc": praucs.values(),
    })
    df.to_csv(output_file, index=False)

        
    print(np.array(list(praucs.values())).mean())
    
    
if __name__ == "__main__":
    main()

