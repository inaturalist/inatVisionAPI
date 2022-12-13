"""
Script to generate kappa statistics from a tf model, thresholds, taxonomy, and taxon_ranges
the model, thresholds, and taxonomy must be local. The taxon_ranges are pulled from iNaturalist
and stored in a geojsons folder. The script outputs visualizations in a pngs folder and kappa
statistics in a csv
"""

import argparse
import requests
import os
from tqdm.auto import tqdm
import numpy as np
import math
import pandas as pd
import geopandas as gpd
import h3pandas
import h3
from h3ronpy import vector, util
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from os.path import exists
import time
import gc

from lib.taxon import Taxon
from lib.model_taxonomy import ModelTaxonomy
from lib.tf_gp_model import TFGeoPriorModel

def get_kappa(hexagon_count, bin_pred_h3_h3index, tr_h3_h3index):
    
    hexagon_count
    tp = bin_pred_h3_h3index.isin(tr_h3_h3index).sum()
    fp = bin_pred_h3_h3index.count() - tp
    fn = (tr_h3_h3index.count() - tp)
    tn = hexagon_count - tp - fp - fn
    po = (tp+tn) / hexagon_count
    pe = (1/hexagon_count**2*((tn+fn)*(tn+fp)+(tp+fp)*(tp+fn)))
    K = (po - pe) / (1 - pe)
    return K

def plot_confusion_map(bp_h3, tr_h3, taxon_id, world):
    
    tr_h3 = tr_h3.set_geometry(tr_h3.geometry.apply(push_right))
    fp_map = bp_h3[~bp_h3.h3index.isin(tr_h3.h3index)].h3.h3_to_geo_boundary()[['geometry']].copy()
    fp_map = fp_map.set_geometry(fp_map.geometry.apply(push_right))
    fp_map["score"] = 1
    tp_map = tr_h3[tr_h3.h3index.isin(bp_h3.h3index)][['geometry']].copy()
    tp_map["score"] = 2
    fn_map = tr_h3[~tr_h3.h3index.isin(bp_h3.h3index)][['geometry']].copy()
    fn_map["score"] = 3
    kappa_map = pd.concat([fp_map, tp_map, fn_map], axis=0)
    minx, miny, maxx, maxy = kappa_map.geometry.total_bounds
    fig, ax = plt.subplots(figsize=(10, 10))
    kappa_map.plot(
        ax=ax,
        column="score",
        legend="true"
    )
    world.boundary.plot(ax=ax, alpha=0.7, color="pink")
    ax.set_xlim(minx - .1, maxx + .1)
    ax.set_ylim(miny - .1, maxy + .1)
    plt.savefig("pngs/" + str(taxon_id) + ".png")
    fig.clf()
    plt.close()
    gc.collect()

def push_right(geom):
    def shift_pts(pts):
        for x, y in pts:
            if x < -100:
                x += 360
            yield (x, y)
    ring = geom.exterior
    if any(p <100 for p in ring.coords.xy[0]) and any(p > 100 for p in ring.coords.xy[0]):
        shell = type(ring)(list(shift_pts(ring.coords)))
    else:
        shell = type(ring)(list(ring.coords))
    holes = list()
    return type(geom)(shell, holes)

def main(args):
    
    print("load in the model, taxonomy, and thresholds...")
    taxon_ids = pd.read_csv(args.taxonomy_path)["leaf_class_id"].dropna()
    if args.first_20 == True:
        taxon_ids = taxon_ids.loc[0:19]
    mt = ModelTaxonomy(args.taxonomy_path)
    tfgpm = TFGeoPriorModel(args.model_path, mt)   
    thresholds = pd.read_csv(args.threshold_path)
    if not os.path.exists('geojsons'):
       os.makedirs('geojsons')
    if not os.path.exists('pngs'):
       os.makedirs('pngs')

    print("setting up the map...")
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    geoJson1 = {'type': 'Polygon', 'coordinates': [[[90,-180],[90,0],[-90,0],[-90,-180]]]}
    geoJson2 = {'type': 'Polygon', 'coordinates': [[[90,0],[90,180],[-90,180],[-90,0]]]}
    h3_resolution = 4
    hexagons = list(h3.polyfill(geoJson1, h3_resolution)) + list(h3.polyfill(geoJson2, h3_resolution))
    polygonise = lambda hex_id: Polygon(
            h3.h3_to_geo_boundary(hex_id, geo_json=True)
        )
    all_polys = gpd.GeoSeries(list(map(polygonise, hexagons)),
            index=hexagons,
            crs="EPSG:4326"
        )
    gdfb = gpd.GeoDataFrame(all_polys)
    gdfb = gdfb.rename(columns={0: 'geometry'})
    gdfb = gdfb.to_crs("epsg:4326")
    gdfb = gdfb.set_geometry(gdfb.geometry.apply(push_right))
    longitude = gdfb.centroid.x.values.astype('float32')
    latitude = gdfb.centroid.y.values.astype('float32')
    hexagon_count = len(hexagons)
    
    output = []
    for taxon_id in tqdm(taxon_ids):
        taxon_id = int(taxon_id)
        try:
            class_of_interest = mt.node_key_to_leaf_class_id[taxon_id]
        except:
            continue
        
        #get the taxon_range
        kml_url = 'https://www.inaturalist.org/taxa/'+ str(taxon_id) +'/range.kml'
        kml_path = r'geojsons/'+ str(taxon_id) +'.kml'
        geojson_path = r'geojsons/'+ str(taxon_id) +'.geojson'
        if exists(geojson_path)==False:
            kml_response = requests.get(kml_url)
            if kml_response.status_code != 200:
                if exists(kml_path)==True:
                    os.remove(kml_path)
                time.sleep(2.5) #avoid hammering server
                continue
            open(kml_path, "wb").write(kml_response.content)
            with open(kml_path) as f:
                first_line = f.readline()
            if first_line == "<!DOCTYPE html>\n":
                if exists(kml_path)==True:
                    os.remove(kml_path)
                time.sleep(2.5) #avoid hammering server
                continue
            cmd = "ogr2ogr -f GeoJSON " + geojson_path + " " + kml_path
            os.system(cmd)
            os.remove(kml_path)
        
        tr = gpd.read_file(geojson_path)
        try:
            tr_h3 = util.h3index_column_to_geodataframe(vector.geodataframe_to_h3(tr, h3_resolution))
        except:
            continue
        
        #get tensorflow predictions map
        pred = tfgpm.eval_one_class(
            np.array(latitude), 
            np.array(longitude), 
            class_of_interest
        )
        gdfb["pred"] = pred[:,0]
        
        #theshold map
        thres = thresholds[thresholds["taxon_id"]==taxon_id]["threshold"]
        bin_pred_h3 = gdfb[gdfb["pred"]>np.float32(thres)[0]].copy()
        bin_pred_h3["h3_hex"] =  bin_pred_h3.index.astype(str)
        bin_pred_h3["h3index"] = bin_pred_h3["h3_hex"].apply(int, base=16)
        
        #calculate kappa
        kappa_stat = get_kappa(hexagon_count, bin_pred_h3.h3index, tr_h3.h3index)
        
        #plot map
        plot_confusion_map(bin_pred_h3, tr_h3, taxon_id, world)
        
        row = {
            "taxon_id": taxon_id,
            "kappa_stat": kappa_stat
        }
        row_dict = dict(row)
        output.append(row_dict)
        
    print("writing output...")
    pdoutput = pd.DataFrame(output)
    pdoutput.to_csv(args.output_path)

if __name__ == "__main__":
    
    info_str = '\nrun as follows\n' + \
               '   python calculate_kappa.py --model_path tf_geoprior_1_4_r6.h5 \n' + \
               '   --taxonomy_path taxonomy_1_4.csv\n' + \
               '   --threshold_path thresholds.csv\n' + \
               '   --output_path kappas.csv\n'
    
    parser = argparse.ArgumentParser(usage=info_str)
    parser.add_argument('--model_path', type=str,
                        help='Path to tf model.', required=True)
    parser.add_argument('--taxonomy_path', type=str,
                        help='Path to taxonomy csv.', required=True)
    parser.add_argument('--threshold_path', type=str,
                        help='Path to threshold csv.', required=True)
    parser.add_argument('--output_path', type=str,
                        help='Path to write kappa stats.', required=True)
    parser.add_argument('--first_20', type=bool, default=False,
        help='just run the first 20')
    args = parser.parse_args()

    main(args)
