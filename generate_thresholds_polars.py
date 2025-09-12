"""
Script to generate thresholds from a (tensorflow or pytorch) model, taxonomy, test and train data
"""

import argparse
import warnings

import tifffile
import pandas as pd
import polars as pl
import numpy as np
import h3
from h3ronpy.polars import vector
import h3pandas  # noqa: F401
import tensorflow as tf
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_curve

from lib.model_taxonomy_dataframe import ModelTaxonomyDataframe
from lib.tf_gp_elev_model import TFGeoPriorModelElev


def ignore_shapely_deprecation_warning(message, category, filename, lineno, file=None, line=None):
    if "array interface is deprecated" in str(message):
        return None
    return warnings.defaultaction(message, category, filename, lineno, file, line)

def _load_train_data_parquet(path):
    print("loading in the training data from parquet...")
    train_df = pl.read_parquet(path)
    train_df = train_df[["taxon_id", "latitude", "longitude", "captive"]]
    
    train_df = train_df.rename({
        "latitude": "lat",
        "longitude": "lng",
    })

    train_df = train_df.filter(pl.col("captive")==0) # no-CID ok, wild only
    train_df = train_df.drop("captive")

    return train_df
 


def main(args):
    print("loading in the model...")
    mtd = ModelTaxonomyDataframe(args.taxonomy, None)
    tfgpm = TFGeoPriorModelElev(args.model)

    print("setting up the map...")
    warnings.showwarning = ignore_shapely_deprecation_warning
    im = tifffile.imread(args.elevation)
    im_df = pd.DataFrame(im)
    im_df.index = np.linspace(90, -90, 2160)
    im_df.columns = np.linspace(-180, 180, 4320)
    im_df = im_df.reset_index()
    im_df = im_df.melt(
        id_vars=["index"],
    )
    im_df.columns = ["lat", "lng", "elevation"]
    elev_dfh3 = im_df.h3.geo_to_h3(args.h3_resolution)
    elev_dfh3 = elev_dfh3.drop(
        columns=["lng", "lat"]
    ).groupby("h3_0" + str(args.h3_resolution)).mean()
    gdfk = elev_dfh3.h3.h3_to_geo()
    gdfk["lng"] = gdfk["geometry"].x
    gdfk["lat"] = gdfk["geometry"].y
    _ = gdfk.pop("geometry")
    gdfk = gdfk.rename_axis("h3index")

    print("making features...")
    feats = tfgpm.features_for_one_class_elevation(
        latitude=list(gdfk.lat),
        longitude=list(gdfk.lng),
        elevation=list(gdfk.elevation)
    )

    print("loading in the training data...")
    train_df = _load_train_data_parquet(
        args.train_spatial_data
    )
    
    print("making h3 labels for training data...")
    train_df_h3 = train_df.with_columns(
        pl.struct(["lat", "lng"])
        .map_batches(
            lambda combined: vector.coordinates_to_cells(
                combined.struct.field("lat"), combined.struct.field("lng"), args.h3_resolution
            ),  
            is_elementwise=True
        )
        .alias("h3_04")
    )

    print("converting h3 indicies to strings...")
    train_df_h3 = train_df_h3.with_columns(
        pl.col("taxon_id"),
        pl.col("h3_04").map_elements(
            lambda x: h3.h3_to_string(x), return_dtype=str
        )
    )
    
    all_spatial_grid_counts = train_df_h3.to_pandas()["h3_04"].value_counts()
    
    presence_absence = pd.DataFrame({
        "background": all_spatial_grid_counts,
    })
    presence_absence = presence_absence.fillna(0)

    print("loading taxonomy...")
    output = []
    taxa = pd.read_csv(
        args.taxonomy,
        usecols=[
            "taxon_id",
            "leaf_class_id",
            "iconic_class_id"
        ]
    ).dropna(subset=["leaf_class_id"])
    taxon_ids = taxa.taxon_id
    if args.stop_after is not None:
        taxon_ids = taxon_ids[0:args.stop_after]
    resolution = args.h3_resolution
    area = h3.hex_area(resolution)

    # we want the taxon id to be the index since we'll be selecting on it
    print("grouping h3 cells by taxon_id...")

    train_df_h3_grouped = train_df_h3.group_by(
        pl.col("taxon_id")
    ).agg(
        pl.col("h3_04")
    )
    print(train_df_h3_grouped.sample(2))

    print("...looping through taxa")
    for taxon_id in tqdm(taxon_ids):
        try:
            class_of_interest = mtd.df.loc[taxon_id]["leaf_class_id"]
        except Exception:
            print("not in the model for some reason")
            continue

        # get predictions
        preds = tfgpm.eval_one_class_elevation_from_features(feats, class_of_interest)
        gdfk["pred"] = tf.squeeze(preds).numpy()

        # make presence absence dataset
        grouped = train_df_h3_grouped.filter(
            pl.col("taxon_id") == taxon_id
        )
        if len(grouped) == 0:
            print(f"taxon id {taxon_id} not present.")
            continue
        target_spatial_grid_counts = grouped["h3_04"].item().to_pandas().value_counts()
        #target_spatial_grid_counts = train_df_h3_grouped[taxon_id]["h3_04"].item().to_pandas().value_counts()
        #target_spatial_grid_counts = \
        #    train_df_h3[train_df_h3.index == taxon_id].h3_04.value_counts()
        
        presences = gdfk.loc[target_spatial_grid_counts.index]["pred"]
        if len(presences) == 0:
            print("not present")
            continue

        # calculate threhold
        presence_absence["forground"] = target_spatial_grid_counts
        presence_absence["predictions"] = gdfk["pred"]
        presence_absence.forground = presence_absence.forground.fillna(0)
        yield_cutoff = np.percentile((
            presence_absence["background"] / presence_absence["forground"]
        )[presence_absence["forground"] > 0], 95)
        absences = presence_absence[
            (presence_absence["forground"] == 0) & (presence_absence["background"] > yield_cutoff)
        ]["predictions"]
        presences = presence_absence[(presence_absence["forground"] > 0)]["predictions"]
        df_x = pd.DataFrame({"predictions": presences, "test": 1})
        df_y = pd.DataFrame({"predictions": absences, "test": 0})
        for_thres = pd.concat([df_x, df_y], ignore_index=False)
        precision, recall, thresholds = precision_recall_curve(
            for_thres.test,
            for_thres.predictions
        )
        p1 = (2 * precision * recall)
        p2 = (precision + recall)
        out = np.zeros((len(p1)))
        fscore = np.divide(p1, p2, out=out, where=p2 != 0)
        index = np.argmax(fscore)
        thres = thresholds[index]

        # store daa
        row = {
            "taxon_id": taxon_id,
            "thres": thres,
            "area": len(gdfk[gdfk.pred >= thres]) * area
        }
        row_dict = dict(row)
        output.append(row_dict)

    print("writing output...")
    output_pd = pd.DataFrame(output)
    output_pd.to_csv(args.output_dir + "/thresholds.csv")


if __name__ == "__main__":
    info_str = "\nrun as follows\n" + \
               "   python generate_thresholds.py --elevation wc2.1_5m_elev.tif \n" + \
               "   --model v2_6/tf_geoprior_2_5_r6_elevation.h5 \n" + \
               "   --taxonomy taxonomy_1_4.csv\n" + \
               "   --train_spatial_data v2_6/taxonomy.csv\n" + \
               "   --output_dir v2_6\n" + \
               "   --h3_resolution 4\n" + \
               "   --stop_after 10\n"

    parser = argparse.ArgumentParser(usage=info_str)
    parser.add_argument("--elevation", type=str,
                        help="Path to elev tif.", required=True)
    parser.add_argument("--model", type=str,
                        help="Path to tf model.", required=True)
    parser.add_argument("--taxonomy", type=str,
                        help="Path to taxonomy csv.", required=True)
    parser.add_argument("--train_spatial_data", type=str,
                        help="Path to train csv for occupancy.", required=True)
    parser.add_argument("--output_dir", type=str,
                        help="directory to write thesholds.", required=True)
    parser.add_argument("--h3_resolution", type=int, default=4,
                        help="grid resolution from 0 - 15, lower numbers are coarser/faster. "
                        "Currently using 4")
    parser.add_argument("--stop_after", type=int,
                        help="just run the first x taxa")
    args = parser.parse_args()

    main(args)
