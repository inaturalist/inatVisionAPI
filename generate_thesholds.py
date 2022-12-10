"""
Script to generate thresholds from a tensor flow model and test data
"""

import argparse
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import math
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

from lib.taxon import Taxon
from lib.model_taxonomy import ModelTaxonomy
from lib.tf_gp_model import TFGeoPriorModel

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
    
    mt = ModelTaxonomy(args.taxonomy_path)
    tfgpm = TFGeoPriorModel(args.model_path, mt)
    test_df = pd.read_csv(args.test_spatial_data_path,
        usecols=["taxon_id","latitude","longitude"])
    pseudoabsence_df = pd.read_csv(args.absences_path,
        usecols=["lat","lng","occupancy_pseudo_absence_taxonid"]).rename({
            "lat": "latitude",
            "lng": "longitude",
            "occupancy_pseudo_absence_taxonid": "taxon_id"
        }, axis=1)
    
    #get the set of taxa
    taxon_ids = pseudoabsence_df.taxon_id.unique()
    
    output = []
    for taxon_id in tqdm(taxon_ids):
        absences = pseudoabsence_df[pseudoabsence_df.taxon_id==taxon_id]
        presences = test_df[test_df.taxon_id==taxon_id]
        test = np.concatenate(([1] * len(presences), [0] * len(absences)))
        
        longitude = np.concatenate((
            presences.longitude.values.astype('float32'),
            absences.longitude.values.astype('float32')
        ))
        
        latitude = np.concatenate((
            presences.latitude.values.astype('float32'),
            absences.latitude.values.astype('float32')
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
        
        #write results to file
        pd.DataFrame(output).to_csv(args.output_path)

if __name__ == "__main__":
    
    info_str = '\nrun as follows\n' + \
               '   python generate_thresholds.py --model_path tf_geoprior_1_4_r6.h5 \n' + \
               '   --taxonomy_path taxonomy_1_4.csv\n' + \
               '   --test_spatial_data_path test_spatial_data.csv\n' + \
               '   --absences_path occupancy_pseudo_absences.csv\n' + \
               '   --output_path thresholds.csv\n'
    
    
    parser = argparse.ArgumentParser(usage=info_str)
    parser.add_argument('--model_path', type=str,
                        help='Path to tf model.', required=True)
    parser.add_argument('--taxonomy_path', type=str,
                        help='Path to taxonomy csv.', required=True)
    parser.add_argument('--test_spatial_data_path', type=str,
                        help='Path to test (presence) csv.', required=True)
    parser.add_argument('--absences_path', type=str,
                        help='Path to absence csv.', required=True)
    parser.add_argument('--output_path', type=str,
                        help='Path to write thesholds.', required=True)
    args = parser.parse_args()

    main(args)