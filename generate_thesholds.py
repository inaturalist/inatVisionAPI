"""
Script to generate thresholds from a tensor flow model, taxonomy, test and train data
"""

import argparse
from tqdm.auto import tqdm
import pandas as pd
import h3pandas
import numpy as np
import math
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

from lib.taxon import Taxon
from lib.model_taxonomy import ModelTaxonomy
from lib.tf_gp_model import TFGeoPriorModel

def ratios_for_taxon_id(taxon_id, all_spatial_grid_counts, train_df_h3):
    
    #counts in each presence cell for target taxon
    target_spatial_grid_counts = train_df_h3[train_df_h3.taxon_id==taxon_id].index.value_counts()

    #background count per target taxon count in target taxon presence cells
    counts = pd.DataFrame({
        "target": target_spatial_grid_counts,
        "all": all_spatial_grid_counts,
    })
    counts = counts.fillna(0)
    counts["target_ratio"] = counts["all"]/counts["target"]
    counts.replace([np.inf, -np.inf], np.nan, inplace=True)
    good_counts = counts["target_ratio"].dropna()
    return good_counts

def pseudo_absences_for_taxon_id(
        taxon_id,
        all_spatial_grid_counts,
        train_df_h3,
        test_df_h3,
        full_spatial_data_lookup_table
    ):
    
    #count cutoff is mean background count per target taxon count in target taxon presence cells
    count_cutoff = ratios_for_taxon_id(taxon_id, all_spatial_grid_counts, train_df_h3).mean()
    
    #absence candidates from test
    sample = test_df_h3.sample(10_000)
    sample_occupancy_pseudoabsences = []
    for i, row in sample.iterrows():
        if taxon_id in full_spatial_data_lookup_table.loc[row.name].taxon_id:
            #candidate in cell containing target taxon so can't be absence
            pass
        else:
            if row.name in all_spatial_grid_counts.index:
                if all_spatial_grid_counts.loc[row.name] > count_cutoff:
                    #candidate in cell not containing target taxon and
                    #count is greater than count cutoff. Use as absence
                    sample_occupancy_pseudoabsences.append(
                        (row.lat, row.lng)
                    )
                
        #limit number of absences
        if len(sample_occupancy_pseudoabsences) >= 500:
            break

    sample_occupancy_pseudoabsences = pd.DataFrame(
        sample_occupancy_pseudoabsences,
        columns=["lat", "lng"]
    )
    sample_occupancy_pseudoabsences["taxon_id"] = taxon_id
    return sample_occupancy_pseudoabsences

def get_threshold(test, predictions):
  
    precision, recall, thresholds = precision_recall_curve(test, predictions)
    p1 = (2 * precision * recall)
    p2 = (precision + recall)
    out = np.zeros( (len(p1)) )
    fscore = np.divide(p1,p2, out=out, where=p2!=0)
    index = np.argmax(fscore)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    prauc = auc(recall, precision)
    return thresholdOpt, prauc

def get_predictions(latitude,longitude,taxon_id, mt, tfgpm):
    
    class_of_interest = mt.node_key_to_leaf_class_id[taxon_id]
    predictions = tfgpm.eval_one_class(
        np.array(latitude), 
        np.array(longitude), 
        class_of_interest
    )
    return predictions

def main(args):
    
    print("loading in test and train data...")
    test_df = pd.read_csv(args.test_spatial_data_path,
        usecols=["taxon_id","latitude","longitude"]).rename({
        "latitude": "lat",
        "longitude": "lng"
    }, axis=1)    
    train_df = pd.read_csv(args.train_spatial_data_path,
        usecols=["taxon_id","latitude","longitude"]).rename({
        "latitude": "lat",
        "longitude": "lng"
    }, axis=1)
    taxon_ids = test_df.taxon_id.unique()
    
    print("calculating absences...")
    h3_resolution = 3
    test_df_h3 = train_df.h3.geo_to_h3(h3_resolution)
    train_df_h3 = train_df.h3.geo_to_h3(h3_resolution)
    full_spatial_data_lookup_table = pd.concat([train_df_h3, test_df_h3]).pivot_table(
        index="h3_0"+str(h3_resolution),
        values="taxon_id",
        aggfunc=set
    )
    pseudoabsence_df = pd.DataFrame()
    all_spatial_grid_counts = train_df_h3.index.value_counts()
    for taxon_id in tqdm(taxon_ids):
        pa = pseudo_absences_for_taxon_id(
            taxon_id,
            all_spatial_grid_counts,
            train_df_h3,
            test_df_h3,
            full_spatial_data_lookup_table
        )
        pseudoabsence_df = pd.concat([pseudoabsence_df, pa])
    
    print("calculating thresholds...")
    mt = ModelTaxonomy(args.taxonomy_path)
    tfgpm = TFGeoPriorModel(args.model_path, mt)
        
    output = []
    for taxon_id in tqdm(taxon_ids):
        absences = pseudoabsence_df[pseudoabsence_df.taxon_id==taxon_id]
        presences = test_df[test_df.taxon_id==taxon_id]
        test = np.concatenate(([1] * len(presences), [0] * len(absences)))
        
        longitude = np.concatenate((
            presences.lng.values.astype('float32'),
            absences.lng.values.astype('float32')
        ))
        
        latitude = np.concatenate((
            presences.lat.values.astype('float32'),
            absences.lat.values.astype('float32')
        ))
        
        predictions = get_predictions(latitude, longitude, taxon_id, mt, tfgpm)
        threshold, prauc = get_threshold(test, predictions)
        
        row = {
            "taxon_id": taxon_id,
            "threshold": threshold,
            "auc": prauc,
            "num_presences": len(presences),
            "num_absences": len(absences)
        }
        row_dict = dict(row)
        output.append(row_dict)
        
    print("writing output...")
    pd.DataFrame(output).to_csv(args.output_path)

if __name__ == "__main__":
    
    info_str = '\nrun as follows\n' + \
               '   python generate_thresholds.py --model_path tf_geoprior_1_4_r6.h5 \n' + \
               '   --taxonomy_path taxonomy_1_4.csv\n' + \
               '   --test_spatial_data_path test_spatial_data.csv\n' + \
               '   --train_spatial_data_path train_spatial_data.csv\n' + \
               '   --output_path thresholds.csv\n'
    
    parser = argparse.ArgumentParser(usage=info_str)
    parser.add_argument('--model_path', type=str,
                        help='Path to tf model.', required=True)
    parser.add_argument('--taxonomy_path', type=str,
                        help='Path to taxonomy csv.', required=True)
    parser.add_argument('--test_spatial_data_path', type=str,
                        help='Path to test csv for presence/absences.', required=True)
    parser.add_argument('--train_spatial_data_path', type=str,
                        help='Path to train csv for occupancy.', required=True)
    parser.add_argument('--output_path', type=str,
                        help='Path to write thesholds.', required=True)
    args = parser.parse_args()

    main(args)
