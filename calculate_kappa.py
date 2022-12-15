"""
Script to generate kappa statistics from a (tensorflow or pytorch) model, thresholds, taxonomy, and
taxon_ranges the model, thresholds, and taxonomy must be local. The taxon_ranges are pulled from
iNaturalist and stored in a geojsons folder. The script outputs visualizations in a pngs folder and
kappa statistics in a csv
"""

import argparse
import requests
import os
import sys
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
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep
        
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

def plot_confusion_map(bp_h3, tr_h3, taxon_id, world, png_folder):
    
    tr_h3 = tr_h3.set_geometry(tr_h3.geometry.apply(push_right))
    fp_map = bp_h3[~bp_h3.h3index.isin(tr_h3.h3index)].h3.h3_to_geo_boundary()[['geometry']].copy()
    fp_map = fp_map.set_geometry(fp_map.geometry.apply(push_right))
    fp_map["score"] = 1
    tp_map = tr_h3[tr_h3.h3index.isin(bp_h3.h3index)][['geometry']].copy()
    tp_map["score"] = 2
    fn_map = tr_h3[~tr_h3.h3index.isin(bp_h3.h3index)][['geometry']].copy()
    fn_map["score"] = 3
    kappa_map = pd.concat([fp_map, tp_map, fn_map], axis=0)
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
    world.boundary.plot(ax=ax, alpha=0.7, color="pink")
    ax.set_xlim(minx - .1, maxx + .1)
    ax.set_ylim(miny - .1, maxy + .1)
    plt.savefig(png_folder + "/" + str(taxon_id) + ".png")
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
    if args.stop_after is not None:
        taxon_ids = taxon_ids.loc[0:args.stop_after]
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
    
    thresholds = pd.read_csv(args.threshold_path)
    if not os.path.exists('geojsons'):
       os.makedirs('geojsons')
    if not os.path.exists(args.png_folder):
       os.makedirs(args.png_folder)

    print("setting up the map...")
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    geoJson1 = {'type': 'Polygon', 'coordinates': [[[90,-180],[90,0],[-90,0],[-90,-180]]]}
    geoJson2 = {'type': 'Polygon', 'coordinates': [[[90,0],[90,180],[-90,180],[-90,0]]]}
    hexagons = list(h3.polyfill(geoJson1, args.h3_resolution)) + list(h3.polyfill(geoJson2, args.h3_resolution))
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
    if args.mask_land == True:
        land_shp_fname = shpreader.natural_earth(resolution='50m',category='physical', name='land')
        land_geom =  unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
        land_gpd = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[land_geom])
        gdf_h3 = land_gpd.h3.polyfill(args.h3_resolution)
        gdfb = gdfb.loc[gdf_h3['h3_polyfill'][0]]
    longitude = gdfb.centroid.x.values.astype('float32')
    latitude = gdfb.centroid.y.values.astype('float32')
    hexagon_count = gdfb.shape[0]
    
    gc.collect()
    
    output = []
    for taxon_id in tqdm(taxon_ids):
        taxon_id = int(taxon_id)
        try:
            if args.model_type == "tf":
                class_of_interest = mt.node_key_to_leaf_class_id[taxon_id]
            elif args.model_type == "pytorch":
                class_of_interest = net_params["params"]["class_to_taxa"].index(taxon_id)
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
            tr_h3 = util.h3index_column_to_geodataframe(vector.geodataframe_to_h3(tr, args.h3_resolution))
        except:
            continue
        
        gc.collect()
        
        if args.model_type == "tf":
            pred = tfgpm.eval_one_class(
                np.array(latitude), 
                np.array(longitude), 
                class_of_interest
            )
            gdfb["pred"] = pred[:,0]
        elif args.model_type == "pytorch":
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
            gdfb["pred"] = predictions
            
        #theshold map
        thres = thresholds[thresholds["taxon_id"]==taxon_id]["threshold"]
        bin_pred_h3 = gdfb[gdfb["pred"]>np.float32(thres)[0]].copy()
        bin_pred_h3["h3_hex"] =  bin_pred_h3.index.astype(str)
        bin_pred_h3["h3index"] = bin_pred_h3["h3_hex"].apply(int, base=16)
        
        #calculate kappa
        kappa_stat = get_kappa(hexagon_count, bin_pred_h3.h3index, tr_h3.h3index)
        
        #plot map
        plot_confusion_map(bin_pred_h3, tr_h3, taxon_id, world, args.png_folder)
        
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
               '   python calculate_kappa.py\n' + \
               '   --model_type tf \n' + \
               '   --model_path tf_geoprior_1_4_r6.h5 \n' + \
               '   --taxonomy_path taxonomy_1_4.csv\n' + \
               '   --threshold_path thresholds.csv\n' + \
               '   --output_path kappas.csv\n' + \
               '   --png_folder pngs\n' + \
               '   --h3_resolution 4\n' + \
               '   --stop_after 10\n' + \
               '   --mask_land True\n'
    
    parser = argparse.ArgumentParser(usage=info_str)
    parser.add_argument('--model_type', type=str,
                        help='Can be either "tf" or "pytorch".', required=True)
    parser.add_argument('--model_path', type=str,
                        help='Path to tf model.', required=True)
    parser.add_argument('--taxonomy_path', type=str,
                        help='Path to taxonomy csv.', required=True)
    parser.add_argument('--threshold_path', type=str,
                        help='Path to threshold csv.', required=True)
    parser.add_argument('--output_path', type=str,
                        help='Path to write kappa stats.', required=True)
    parser.add_argument('--png_folder', type=str,
                        help='Name of folder to write pngs to (will create if does not exist).', required=True)                    
    parser.add_argument('--h3_resolution', type=int, default=4,
        help='grid resolution from 0 - 15, lower numbers are coarser/faster. Recommend 3, 4, or 5')
    parser.add_argument('--stop_after', type=int,
        help='just run the first x taxa')
    parser.add_argument('--mask_land', type=bool, default=False,
        help='exclude oceans')
    args = parser.parse_args()

    main(args)
