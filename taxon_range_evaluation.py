"""
Script to evaluate model and thresholds against taxon ranges
"""

import argparse
import csv
import tensorflow as tf
import pandas as pd
import gc
from tqdm.auto import tqdm
from os.path import exists
import tifffile
import numpy as np
import h3
import h3pandas
import math
import geopandas as gpd
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

class ResLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ResLayer, self).__init__()
        self.w1 = tf.keras.layers.Dense(
            256, activation="relu", kernel_initializer="he_normal"
        )
        self.w2 = tf.keras.layers.Dense(
            256, activation="relu", kernel_initializer="he_normal"
        )
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        x = self.w1(inputs)
        x = self.dropout(x)
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

class TFGeoPriorModelEnv:

    def __init__(self, model_path, taxonomy):
        self.taxonomy = taxonomy
        # initialize the geo model for inference
        self.gpmodel = tf.keras.models.load_model(
            model_path,
            custom_objects={'ResLayer': ResLayer},
            compile=False
        )

    def eval_one_class_elevation(self, latitude, longitude, elevation, class_of_interest):
        """Evalutes the model for a single class and multiple locations

        Args:
            latitude (list): A list of latitudes
            longitude (list): A list of longitudes (same length as latitude)
            elevation (list): A list of elevations (same length as latitude)
            class_of_interest (int): The single class to eval

        Returns:
            numpy array: scores for class of interest at each location
        """
        def encode_loc(latitude, longitude, elevation):
            latitude = np.array(latitude)
            longitude = np.array(longitude)
            elevation = np.array(elevation)
            elevation = elevation.astype("float32")
            grid_lon = longitude.astype('float32') / 180.0
            grid_lat = latitude.astype('float32') / 90.0
            
            elevation[elevation>0] = elevation[elevation>0]/6574.0
            elevation[elevation<0] = elevation[elevation<0]/32768.0
            norm_elev = elevation
            
            if np.isscalar(grid_lon):
                grid_lon = np.array([grid_lon])
            if np.isscalar(grid_lat):
                grid_lat = np.array([grid_lat])
            if np.isscalar(norm_elev):
                norm_elev = np.array([norm_elev])
                        
            norm_loc = tf.stack([grid_lon, grid_lat], axis=1)

            encoded_loc = tf.concat([
                tf.sin(norm_loc * math.pi),
                tf.cos(norm_loc * math.pi),
                tf.expand_dims(norm_elev, axis=1),

            ], axis=1)            
            
            return encoded_loc

        encoded_loc = encode_loc(latitude, longitude, elevation)
        loc_emb = self.gpmodel.layers[0](encoded_loc)
        
        # res layers - feature extraction
        x = self.gpmodel.layers[1](loc_emb)
        x = self.gpmodel.layers[2](x)
        x = self.gpmodel.layers[3](x)
        x = self.gpmodel.layers[4](x)
        
        # process just the one class
        return tf.keras.activations.sigmoid(
            tf.matmul(
                x, 
                tf.expand_dims(self.gpmodel.layers[5].weights[0][:,class_of_interest], axis=0),
                transpose_b=True
            )
        ).numpy()
    
    def features_for_one_class_elevation(self, latitude, longitude, elevation):
        """Evalutes the model for a single class and multiple locations

        Args:
            latitude (list): A list of latitudes
            longitude (list): A list of longitudes (same length as latitude)
            elevation (list): A list of elevations (same length as latitude)
            class_of_interest (int): The single class to eval

        Returns:
            numpy array: scores for class of interest at each location
        """
        def encode_loc(latitude, longitude, elevation):
            latitude = np.array(latitude)
            longitude = np.array(longitude)
            elevation = np.array(elevation)
            elevation = elevation.astype("float32")
            grid_lon = longitude.astype('float32') / 180.0
            grid_lat = latitude.astype('float32') / 90.0
            
            elevation[elevation>0] = elevation[elevation>0]/6574.0
            elevation[elevation<0] = elevation[elevation<0]/32768.0
            norm_elev = elevation
            
            if np.isscalar(grid_lon):
                grid_lon = np.array([grid_lon])
            if np.isscalar(grid_lat):
                grid_lat = np.array([grid_lat])
            if np.isscalar(norm_elev):
                norm_elev = np.array([norm_elev])
                        
            norm_loc = tf.stack([grid_lon, grid_lat], axis=1)

            encoded_loc = tf.concat([
                tf.sin(norm_loc * math.pi),
                tf.cos(norm_loc * math.pi),
                tf.expand_dims(norm_elev, axis=1),

            ], axis=1)            
            
            return encoded_loc

        encoded_loc = encode_loc(latitude, longitude, elevation)
        loc_emb = self.gpmodel.layers[0](encoded_loc)
        
        # res layers - feature extraction
        x = self.gpmodel.layers[1](loc_emb)
        x = self.gpmodel.layers[2](x)
        x = self.gpmodel.layers[3](x)
        x = self.gpmodel.layers[4](x)
        
        # process just the one class
        return x

    def eval_one_class_elevation_from_features(self, x, class_of_interest):
        return tf.keras.activations.sigmoid(
            tf.matmul(
                x, 
                tf.expand_dims(self.gpmodel.layers[5].weights[0][:,class_of_interest], axis=0),
                transpose_b=True
            )
        ).numpy()

def evaluate_p_r(thres, gdfb, tr_h3, world, plot):
    bp_h3 = gdfb[gdfb["pred"]>=thres].copy()
    area = bp_h3.shape[0]
    if area == 0:
        return None, None, None
    tt = tr_h3.h3.h3_to_geo_boundary()[['geometry']].copy()
    fp_map = bp_h3[~bp_h3.index.isin(tt.index)].h3.h3_to_geo_boundary()[['geometry']].copy()
    fp_map = fp_map.set_geometry(fp_map.geometry.apply(push_right))
    fp_map["score"] = 1
    tp_map = tt[tt.index.isin(bp_h3.index)][['geometry']].copy()
    tp_map["score"] = 2
    fn_map = tt[~tt.index.isin(bp_h3.index)][['geometry']].copy()
    fn_map["score"] = 3
    kappa_map = pd.concat([fp_map, tp_map, fn_map], axis=0)
    
    fp=kappa_map[kappa_map["score"]==1].shape[0] #fp
    tp=kappa_map[kappa_map["score"]==2].shape[0] #tp
    fn=kappa_map[kappa_map["score"]==3].shape[0] #fn
    p = tp/(tp+fp)
    r = tp/(fn+tp)
    
    if plot==True:
        print("Precision: " + str(p))
        print("Recall: " + str(r))
        kappa_map_geometry_total_bounds = kappa_map.geometry.total_bounds
        if np.isnan(kappa_map_geometry_total_bounds).any():
            minx, miny, maxx, maxy = [-180,  -90, 180,  90]
        else:
            minx, miny, maxx, maxy = kappa_map_geometry_total_bounds
        fig, ax = plt.subplots(figsize=(10, 10))
        kappa_map.plot(
            ax=ax,
            column="score",
            legend="true"
        )
        world.boundary.plot(ax=ax, alpha=0.7, color="black")
        ax.set_xlim(minx - .1, maxx + .1)
        ax.set_ylim(miny - .1, maxy + .1)
        plt.show()
    
    return p, r, area

def push_right(geom):
    def shift_pts(pts):
        for x, y in pts:
            if x < -100:
                x += 360
            yield (x, y)
    ring = geom.exterior
    if any(p < -100 for p in ring.coords.xy[0]) and any(p > 100 for p in ring.coords.xy[0]):
        shell = type(ring)(list(shift_pts(ring.coords)))
    else:
        shell = type(ring)(list(ring.coords))
    holes = list()
    return type(geom)(shell, holes)

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
    out = np.zeros( (len(p1)) )
    fscore = np.divide(p1,p2, out=out, where=p2!=0)
    index = np.argmax(fscore)
    prthres = thresholds[index]
    prf1 = fscore[index]
    prprecision = precision[index]
    prrecall = recall[index]
    prauc = auc(recall, precision)
    if plot==True:
        print("PR AUC: " + str(prauc))
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color='purple')
        ax.plot([recall[index]], [precision[index]], color='green', marker='o')
        ax.set_title('Precision-Recall Curve')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        plt.show()
    return prauc, prthres, prf1, prprecision, prrecall

def main(args):
    print("read in the taxonomy...")
    taxa = pd.read_csv(args.taxonomy, usecols=["taxon_id","leaf_class_id","iconic_class_id"]).dropna(subset=['leaf_class_id'])
    taxon_ids = taxa.taxon_id
    if args.stop_after is not None:
                taxon_ids = taxon_ids[0:args.stop_after]
    mt = ModelTaxonomy(args.taxonomy)

    print("read in the model...")
    tfgpm = TFGeoPriorModelEnv(args.model, mt)

    print("read in the taxon range recalls and thresholds...")
    taxon_range_recalls = pd.read_csv(args.taxon_range_recalls)
    thresholds = pd.read_csv(args.thresholds)

    print("reading in the elevation and world map...")
    im = tifffile.imread(args.elevation)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    print("processing elevation and making features...")
    h3_resolution=4
    im_df = pd.DataFrame(im)
    im_df.index = np.linspace(90, -90, 2160)
    im_df.columns = np.linspace(-180, 180, 4320)
    im_df = im_df.reset_index()
    im_df = im_df.melt(
        id_vars=["index"],
    )
    im_df.columns = ["lat", "lng", "elevation"]
    elev_dfh3 = im_df.h3.geo_to_h3(h3_resolution)
    elev_dfh3 = elev_dfh3.drop(
        columns=['lng', 'lat']
    ).groupby("h3_0"+str(h3_resolution)).mean()
    gdfk = elev_dfh3.h3.h3_to_geo()
    gdfk["lng"] = gdfk["geometry"].x
    gdfk["lat"] = gdfk["geometry"].y
    _ = gdfk.pop("geometry")
    gdfk = gdfk.rename_axis('h3index')
    feats = tfgpm.features_for_one_class_elevation(
        latitude=list(gdfk.lat),
        longitude=list(gdfk.lng),
        elevation=list(gdfk.elevation)
    )
    
    print("looping through the taxa...")
    eval_output = []
    for taxon_id in tqdm(taxon_ids):
        #check whether taxon represented in taxon range eval set
        if taxon_range_recalls[taxon_range_recalls.taxon_id.eq(taxon_id)].shape[0] == 0:
            continue
        if taxon_range_recalls[(taxon_range_recalls['taxon_id'] == taxon_id) & (taxon_range_recalls['r'] > 0.9)].empty:
            continue
        taxon_range_indicies = args.taxon_range_indicies+"/"+ str(taxon_id) +".csv"
        if exists(taxon_range_indicies) == False:
            continue
    
        #process taxon range
        try:
            taxon_range_index = pd.read_csv(taxon_range_indicies, header=None)
            taxon_range_index.rename(columns={0: 'h3index_new'}, inplace=True)
            tr_h3 = gdfk.loc[gdfk.index.isin(taxon_range_index.h3index_new)]
        except:
            gc.collect()
            continue

        #get model predictions and threshold
        try:
            class_of_interest = mt.node_key_to_leaf_class_id[taxon_id]
        except:
            continue
        preds = tfgpm.eval_one_class_elevation_from_features(feats, class_of_interest)
        gdfk["pred"] = tf.squeeze(preds).numpy()
        thres = thresholds[thresholds.taxon_id==taxon_id].thres.values[0]
    
        #get precision, recall, prauc, and f1
        p, r, area = evaluate_p_r(thres, gdfk, tr_h3, world, False)
        if p == None or r == None or ((p+r)==0):
            f1 = None
        else:
            f1 = (2 * p * r) / (p + r)
        prauc, prthres, prf1, prprecision, prrecall = get_prauc(gdfk, tr_h3, False)
        area = h3.hex_area(h3_resolution)
    
        #store results
        row = {
            "taxon_id": taxon_id,
            "prauc": prauc,
            "p": p,
            "r": r,
            "f1": f1,
            "taxon_range_area": len(tr_h3)*area,
        }
        row_dict = dict(row)
        eval_output.append(row_dict)
        
    eval_output_pd = pd.DataFrame(eval_output)
    print("evaluation statistics:")
    print("\tPR-AUC: "+str(round(eval_output_pd.prauc.mean(),3)))
    print("\tPrecision: "+str(round(eval_output_pd.p.mean(),3)))
    print("\tRecall: "+str(round(eval_output_pd.r.mean(),3)))
    print("\tF1: "+str(round(eval_output_pd.f1.mean(),3)))

    print("writing output...")
    eval_output_pd.to_csv(args.output_path)

if __name__ == "__main__":
    
    info_str = '\nrun as follows\n' + \
               '   python taxon_range_evaluation.py --elevation wc2.1_5m_elev.tif \n' + \
               '   --model v2_8/no_full_shuffle_50k_buffer.h5 \n' + \
               '   --taxonomy v2_8/taxonomy.csv\n' + \
               '   --thresholds v2_8/tf_env_thresh.csv\n' + \
               '   --taxon_range_recalls v2_8/taxon_range_recalls.csv\n' + \
               '   --taxon_range_indicies v2_8/taxon_range_indicies\n' + \
               '   --output_path v2_8/tf_env_eval_test.csv\n' + \
               '   --stop_after 10\n'
    
    parser = argparse.ArgumentParser(usage=info_str)
    parser.add_argument('--elevation', type=str,
                        help='Path to elev tif.', required=True)
    parser.add_argument('--model', type=str,
                        help='Path to tf model.', required=True)
    parser.add_argument('--taxonomy', type=str,
                        help='Path to taxonomy csv.', required=True)
    parser.add_argument('--thresholds', type=str,
                        help='Path to thresholds csv.', required=True)
    parser.add_argument('--taxon_range_recalls', type=str,
                        help='Path to taxon_range_recalls csv.', required=True)
    parser.add_argument('--taxon_range_indicies', type=str,
                        help='Path to indices dir.', required=True)                        
    parser.add_argument('--output_path', type=str,
                        help='file to write thesholds.', required=True)
    parser.add_argument('--stop_after', type=int,
            help='just run the first x taxa')
    args = parser.parse_args()

    main(args)
    
