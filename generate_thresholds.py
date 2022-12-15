"""
Script to generate thresholds from a (tensorflow or pytorch) model, taxonomy, test and train data
"""

import argparse
from tqdm.auto import tqdm
import pandas as pd
import h3pandas
import numpy as np
import math
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import sys

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
    cutoff_grid_cell_indices = set(all_spatial_grid_counts[all_spatial_grid_counts>count_cutoff].index)
    
    #absence candidates from test
    ilocs = np.unique(np.random.randint(0, len(test_df_h3), 10_000))
    sample_occupancy_pseudoabsences = []
    for i in ilocs:
        row = test_df_h3.iloc[i]

        if row.name in cutoff_grid_cell_indices:
            if taxon_id in full_spatial_data_lookup_table.loc[row.name].taxon_id:
                #candidate in cell containing target taxon so can't be absence
                pass
            else:
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

def get_predictions_py(latitude,longitude,taxon_id, net_params, model, torch):
    
    class_of_interest = net_params["params"]["class_to_taxa"].index(taxon_id)
    grid_lon = torch.from_numpy(longitude/180)
    grid_lat = torch.from_numpy(latitude/90)
    grid_lon = grid_lon.repeat(1,1).unsqueeze(2)
    grid_lat = grid_lat.repeat(1, 1).unsqueeze(2)
    loc_ip = torch.cat((grid_lon, grid_lat), 2)
    feats = torch.cat((torch.sin(math.pi*loc_ip), torch.cos(math.pi*loc_ip)), 2)
    model.eval()
    with torch.no_grad():
        pred = model(feats, class_of_interest=class_of_interest)
        predictions = pred.cpu().numpy()[0]
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
    if args.stop_after is not None:
        taxon_ids = taxon_ids[0:args.stop_after]
    
    print("calculating absences...")
    test_df_h3 = test_df.h3.geo_to_h3(args.h3_resolution)
    train_df_h3 = train_df.h3.geo_to_h3(args.h3_resolution)
    full_spatial_data_lookup_table = pd.concat([train_df_h3, test_df_h3]).pivot_table(
        index="h3_0"+str(args.h3_resolution),
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
    if args.model_type == "tf":
        from lib.taxon import Taxon
        from lib.model_taxonomy import ModelTaxonomy
        from lib.tf_gp_model import TFGeoPriorModel
        mt = ModelTaxonomy(args.taxonomy_path)
        tfgpm = TFGeoPriorModel(args.model_path, mt)   
    elif args.model_type == "pytorch":
        import torch
        sys.path.append("../geo_prior_inat")
        from geo_prior import models
        net_params = torch.load(args.model_path, map_location="cpu")
        net_params["params"]['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = models.select_model(net_params["params"]["model"])
        model = model_name(num_inputs=net_params["params"]['num_feats'],
                           num_classes=net_params["params"]['num_classes'],
                           num_filts=net_params["params"]['num_filts'],
                           num_users=net_params["params"]['num_users'],
                           num_context=net_params["params"]['num_context']).to(net_params["params"]['device'])
        model.load_state_dict(net_params['state_dict'])        
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
        
        if args.model_type == "tf":
            predictions = get_predictions(latitude, longitude, taxon_id, mt, tfgpm)
        elif args.model_type == "pytorch":
            predictions = get_predictions_py(latitude, longitude, taxon_id, net_params, model, torch)
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
               '   python generate_thresholds.py --model_type tf \n' + \
               '   --model_path tf_geoprior_1_4_r6.h5 \n' + \
               '   --taxonomy_path taxonomy_1_4.csv\n' + \
               '   --test_spatial_data_path test_spatial_data.csv\n' + \
               '   --train_spatial_data_path train_spatial_data.csv\n' + \
               '   --output_path thresholds.csv\n' + \
               '   --h3_resolution 3\n' + \
               '   --stop_after 10\n'
    
    parser = argparse.ArgumentParser(usage=info_str)
    parser.add_argument('--model_type', type=str,
                        help='Can be either "tf" or "pytorch".', required=True)
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
    parser.add_argument('--h3_resolution', type=int, default=3,
        help='grid resolution from 0 - 15, lower numbers are coarser/faster. Recommend 3, 4, or 5')
    parser.add_argument('--stop_after', type=int,
        help='just run the first x taxa')
    args = parser.parse_args()

    main(args)
