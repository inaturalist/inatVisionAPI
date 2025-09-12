"""
Script to generate thresholds from a (tensorflow or pytorch) model, taxonomy, test and train data
"""

import argparse
import tifffile
import os
import pandas as pd
import numpy as np
import h3
import h3pandas
import tensorflow as tf
import csv
import math
import json
from tqdm.auto import tqdm
import tensorflow as tf
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt    
import warnings

class TFResLayer(tf.keras.layers.Layer):
    def __init__(self, name):
        super(TFResLayer, self).__init__(name=name)
        self.w1 = tf.keras.layers.Dense(
            256, activation="relu", name="{}_w1".format(name)
        )
        self.w2 = tf.keras.layers.Dense(
            256, activation="relu", name="{}_w2".format(name)
        )
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        x = self.w1(inputs)
        x = self.w2(x)
        x = self.add([x, inputs])
        return x

    def get_config(self):
        return {}

class Taxon:

    def __init__(self, row):
        for key in row:
            setattr(self, key, row[key])

    def set(self, attr, val):
        setattr(self, attr, val)

    def is_or_descendant_of(self, taxon):
        if self.id == taxon.id:
            return True
        return self.descendant_of(taxon)

    # using the nested set left and right values, a taxon is a descendant of another
    # as long as its left is higher and its right is lower
    def descendant_of(self, taxon):
        return self.left > taxon.left and self.right < taxon.right

class ModelTaxonomy:

    def __init__(self, path):
        self.load_mapping(path)
        self.assign_nested_values()

    def load_mapping(self, path):
        self.node_key_to_leaf_class_id = {}
        self.leaf_class_to_taxon = {}
        # there is no taxon with ID 0, but roots of the taxonomy with have a parent ID of 0,
        # so create a fake taxon of Life to represent the root of the entire tree
        self.taxa = {0: Taxon({"name": "Life", "depth": 0})}
        self.taxon_children = {}
        try:
            with open(path) as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=",")
                for row in csv_reader:
                    taxon_id = int(row["taxon_id"])
                    rank_level = float(row["rank_level"])
                    leaf_class_id = int(row["leaf_class_id"]) if row["leaf_class_id"] else None
                    parent_id = int(row["parent_taxon_id"]) if row["parent_taxon_id"] else 0
                    # some taxa are not leaves and aren't represented in the leaf layer
                    if leaf_class_id is not None:
                        self.node_key_to_leaf_class_id[taxon_id] = leaf_class_id
                        self.leaf_class_to_taxon[leaf_class_id] = taxon_id
                    self.taxa[taxon_id] = Taxon({
                        "id": taxon_id,
                        "name": row["name"],
                        "parent_id": parent_id,
                        "leaf_class_id": leaf_class_id,
                        "rank_level": rank_level
                    })
                    if parent_id not in self.taxon_children:
                        self.taxon_children[parent_id] = []
                    self.taxon_children[parent_id].append(taxon_id)
        except IOError as e:
            print(e)
            print(f"\n\nCannot open mapping file `{path}`\n\n")
            raise e

    # prints to the console a representation of this tree
    def print(self, taxon_id=0, ancestor_prefix=""):
        children = self.taxon_children[taxon_id]
        index = 0
        for child_id in children:
            last_in_branch = (index == len(children) - 1)
            index += 1
            icon = "└──" if last_in_branch else "├──"
            prefixIcon = "   " if last_in_branch else "│   "
            taxon = self.taxa[child_id]
            print(f'{ancestor_prefix}{icon}{taxon.name} :: {taxon.left}:{taxon.right}')
            if child_id in self.taxon_children:
                self.print(child_id, f"{ancestor_prefix}{prefixIcon}")

    # calculated nested set left and right values and depth representing how many nodes
    # down the taxon is from Life. These can be later used for an efficient way to calculate
    # if a taxon is a descendant of another
    def assign_nested_values(self, taxon_id=0, index=0, depth=1, ancestors=[]):
        for child_id in self.taxon_children[taxon_id]:
            self.taxa[child_id].set("left", index)
            self.taxa[child_id].set("depth", depth)
            self.taxa[child_id].set("ancestors", ancestors)
            index += 1
            if child_id in self.taxon_children:
                child_ancestors = ancestors + [child_id]
                index = self.assign_nested_values(child_id, index, depth + 1, child_ancestors)
            self.taxa[child_id].set("right", index)
            index += 1
        return index


class TFGeoPriorModel:

    def __init__(self, model, taxonomy):
        self.taxonomy = taxonomy
        # initialize the geo model for inference
        self.gpmodel = tf.keras.models.Sequential([
            tf.keras.layers.Input(4, name="input"),
            tf.keras.layers.Dense(256, activation="relu", name="encode_loc"),
            TFResLayer("reslayer1"),
            TFResLayer("reslayer2"),
            TFResLayer("reslayer3"),
            TFResLayer("reslayer4"),
            tf.keras.layers.Dense(90290, use_bias=False, name="class_emb"),
            tf.keras.layers.Activation("sigmoid", dtype="float32", name="predictions"),
        ])
        self.gpmodel.load_weights(model)

    
    def features_for_one_class(self, latitude, longitude):
        """Evalutes the model for a single class and multiple locations

        Args:
            latitude (list): A list of latitudes
            longitude (list): A list of longitudes (same length as latitude)
            class_of_interest (int): The single class to eval

        Returns:
            numpy array: scores for class of interest at each location
        """
        def encode_loc(latitude, longitude):
            latitude = np.array(latitude)
            longitude = np.array(longitude)
            grid_lon = longitude.astype('float32') / 180.0
            grid_lat = latitude.astype('float32') / 90.0
            
            if np.isscalar(grid_lon):
                grid_lon = np.array([grid_lon])
            if np.isscalar(grid_lat):
                grid_lat = np.array([grid_lat])
                        
            norm_loc = tf.stack([grid_lon, grid_lat], axis=1)

            encoded_loc = tf.concat([
                tf.sin(norm_loc * math.pi),
                tf.cos(norm_loc * math.pi),
            ], axis=1)            
            
            return encoded_loc

        encoded_loc = encode_loc(latitude, longitude)
        loc_emb = self.gpmodel.layers[0](encoded_loc)
        
        # res layers - feature extraction
        x = self.gpmodel.layers[1](loc_emb)
        x = self.gpmodel.layers[2](x)
        x = self.gpmodel.layers[3](x)
        x = self.gpmodel.layers[4](x)
        
        # process just the one class
        return x

    def eval_one_class_from_features(self, x, class_of_interest):
        return tf.keras.activations.sigmoid(
            tf.matmul(
                x, 
                tf.expand_dims(self.gpmodel.layers[5].weights[0][:,class_of_interest], axis=0),
                transpose_b=True
            )
        ).numpy()

def ignore_shapely_deprecation_warning(message, category, filename, lineno, file=None, line=None):
    if "array interface is deprecated" in str(message):
        return None
    return warnings.defaultaction(message, category, filename, lineno, file, line)

def main(args):
    print("loading in the model...")
    mt = ModelTaxonomy(args.taxonomy)
    tfgpm = TFGeoPriorModel(args.model, mt)
    
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
    im_df = im_df[["lat", "lng"]]
    dfh3 = im_df.h3.geo_to_h3(args.h3_resolution)
    dfh3 = dfh3.drop(
        columns=['lng', 'lat']
    ).groupby("h3_0"+str(args.h3_resolution)).mean()
    gdfk = dfh3.h3.h3_to_geo()
    gdfk["lng"] = gdfk["geometry"].x
    gdfk["lat"] = gdfk["geometry"].y
    _ = gdfk.pop("geometry")
    gdfk = gdfk.rename_axis('h3index')
    print(gdfk.sample(3))

    print("making features...")
    feats = tfgpm.features_for_one_class(
        latitude=list(gdfk.lat),
        longitude=list(gdfk.lng),
    )

    print("loading in the training data...")
    train_df = pd.read_csv(args.train_spatial_data,
        usecols=["taxon_id","latitude","longitude","captive"]).rename({
        "latitude": "lat",
        "longitude": "lng"
    }, axis=1)
    train_df = train_df[train_df.captive==0] #no-CID ok, wild only
    train_df.drop(["captive"],axis=1)
    train_df_h3 = train_df.h3.geo_to_h3(args.h3_resolution)
    all_spatial_grid_counts = train_df_h3.index.value_counts()
    presence_absence = pd.DataFrame({
        "background": all_spatial_grid_counts,
    })
    presence_absence = presence_absence.fillna(0)

    print("...looping through taxa")
    output = []
    taxa = pd.read_csv(args.taxonomy, usecols=["taxon_id","leaf_class_id","iconic_class_id"]).dropna(subset=['leaf_class_id'])
    taxon_ids = taxa.taxon_id
    if args.stop_after is not None:
            taxon_ids = taxon_ids[0:args.stop_after]
    desired_recall = 0.95
    resolution = args.h3_resolution
    area = h3.hex_area(resolution)
    for taxon_id in tqdm(taxon_ids):
        try:
            class_of_interest = mt.node_key_to_leaf_class_id[taxon_id]
        except:
            print('not in the model for some reason')
            continue

        #get predictions
        preds = tfgpm.eval_one_class_from_features(feats, class_of_interest)
        gdfk["pred"] = tf.squeeze(preds).numpy()
    
        #make presence absence dataset
        target_spatial_grid_counts = train_df_h3[train_df_h3.taxon_id==taxon_id].index.value_counts()
        presences = gdfk.loc[target_spatial_grid_counts.index]["pred"]
        if len(presences) == 0:
            print("not present")
            continue
    
        #calculate threhold
        presence_absence["forground"] = target_spatial_grid_counts
        presence_absence["predictions"] = gdfk["pred"]
        presence_absence.forground = presence_absence.forground.fillna(0)
        yield_cutoff = np.percentile((presence_absence["background"]/presence_absence["forground"])[presence_absence["forground"]>0], 95)
        absences = presence_absence[(presence_absence["forground"]==0) & (presence_absence["background"] > yield_cutoff)]["predictions"]
        presences = presence_absence[(presence_absence["forground"]>0)]["predictions"]
        df_x = pd.DataFrame({'predictions': presences, 'test': 1})
        df_y = pd.DataFrame({'predictions': absences, 'test': 0})
        for_thres = pd.concat([df_x, df_y], ignore_index=False)
        precision, recall, thresholds = precision_recall_curve(for_thres.test, for_thres.predictions)
        p1 = (2 * precision * recall)
        p2 = (precision + recall)
        out = np.zeros( (len(p1)) )
        fscore = np.divide(p1,p2, out=out, where=p2!=0)
        index = np.argmax(fscore)
        thres = thresholds[index]
    
        #store daa
        row = {
            "taxon_id": taxon_id,
            "thres": thres,
            "area": len(gdfk[gdfk.pred >= thres])*area
        }
        row_dict = dict(row)
        output.append(row_dict)
    
    print("writing output...")
    output_pd = pd.DataFrame(output)
    output_pd.to_csv(args.output_dir+"/thresholds.csv")

if __name__ == "__main__":
    
    info_str = '\nrun as follows\n' + \
               '   python generate_thresholds.py \n' + \
               '   --model v2_6/tf_geoprior_2_5_r6_elevation.h5 \n' + \
               '   --taxonomy taxonomy_1_4.csv\n' + \
               '   --train_spatial_data v2_6/taxonomy.csv\n' + \
               '   --output_dir v2_6\n' + \
               '   --h3_resolution 4\n' + \
               '   --stop_after 10\n'
    
    parser = argparse.ArgumentParser(usage=info_str)
    parser.add_argument('--elevation', type=str,
                        help='Path to elev tif.', required=True)
    parser.add_argument('--model', type=str,
                        help='Path to tf model.', required=True)
    parser.add_argument('--taxonomy', type=str,
                        help='Path to taxonomy csv.', required=True)
    parser.add_argument('--train_spatial_data', type=str,
                        help='Path to train csv for occupancy.', required=True)
    parser.add_argument('--output_dir', type=str,
                        help='directory to write thesholds.', required=True)
    parser.add_argument('--h3_resolution', type=int, default=4,
        help='grid resolution from 0 - 15, lower numbers are coarser/faster. Currently using 4')
    parser.add_argument('--stop_after', type=int,
            help='just run the first x taxa')
    args = parser.parse_args()

    main(args)
    
