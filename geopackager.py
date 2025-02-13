import yaml
import h3
import json
import geopandas as gpd
import io
import time
import antimeridian
import pandas as pd
import numpy as np
import os

from tqdm.auto import tqdm
from lib.inat_inferrer import InatInferrer


CONFIG = yaml.safe_load(open("config.yml"))["models"][0]
INFERRER = InatInferrer(CONFIG)
MODEL_VERSION = CONFIG["name"]

ANTIMERIDIAN_CROSSING_CELLS = INFERRER.geo_elevation_cells.query("abs(minx-maxx) > 90").index


def generate_joined_taxonomy():
    # the taxonomy file currently does not include rank as a string or iconic_taxon_id.
    # Until it does, there is a separate file that includes these attributes that we can join
    # with the original taxonomy data
    additions = pd.read_csv(
        CONFIG["taxonomy_additions_path"],
        dtype={
            "taxon_id": int,
            "parent_taxon_id": "Int64",
            "rank": pd.StringDtype(),
            "rank_level": float,
            "iconic_taxon_id": "Int64"
        }
    ).set_index("taxon_id", drop=False).sort_index()
    joined_taxonomy = INFERRER.taxonomy.df.join(additions[["rank", "iconic_taxon_id"]])
    joined_taxonomy = pd.merge(
        joined_taxonomy, INFERRER.taxonomy.df[["name"]],
        left_on="iconic_taxon_id", right_on="taxon_id"
    ).rename(
        columns={"name_y": "iconic_taxon_name"}
    ).rename(
        columns={"name_x": "name"}
    ).set_index("taxon_id", drop=False)
    return joined_taxonomy


JOINED_TAXONOMY = generate_joined_taxonomy()


def remove_antimeridian_crossing_cells(cells):
    return list(filter(lambda cell_key: cell_key not in ANTIMERIDIAN_CROSSING_CELLS, cells))


def taxon_h3_cells_to_geojson(taxon, cells):
    return {
        "type": "Feature",
        "properties": {
            "taxon_id": int(taxon["taxon_id"]),
            "parent_taxon_id": int(taxon["parent_taxon_id"]),
            "name": taxon["name"],
            "rank": taxon["rank"],
            "iconic_taxon_id": int(taxon["iconic_taxon_id"]),
            "iconic_taxon_name": taxon["iconic_taxon_name"],
            "geomodel_version": MODEL_VERSION
        },
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": cells
        }
    }


def geojson_for_taxon(taxon):
    geomodel_results = INFERRER.h3_04_geo_results_for_taxon(
        taxon["taxon_id"], bounds=None, thresholded=True, raw_results=True
    )
    if geomodel_results is None:
        return

    cells = h3.h3_set_to_multi_polygon(geomodel_results.index.astype(str), geo_json=True)
    geojson = taxon_h3_cells_to_geojson(taxon, cells)

    try:
        return antimeridian.fix_geojson(geojson, fix_winding=False)
    except AssertionError:
        # print(f"Problem splitting at antimeridian: {taxon['taxon_id']}")
        return geojson_for_taxon_pruned(taxon)


def geojson_for_taxon_pruned(taxon):
    geomodel_results = INFERRER.h3_04_geo_results_for_taxon(
        taxon["taxon_id"], bounds=None, thresholded=True, raw_results=True
    )
    if geomodel_results is None:
        return
    valid_cells = remove_antimeridian_crossing_cells(geomodel_results.index.values)
    cells = h3.h3_set_to_multi_polygon(valid_cells, geo_json=True)
    return taxon_h3_cells_to_geojson(taxon, cells)


def process_taxon_geojson(taxon_id):
    taxon = JOINED_TAXONOMY.loc[taxon_id]
    geojson = geojson_for_taxon(taxon)
    if geojson is None:
        return

    geojson_file = open(f"geopackages/geojson/{taxon_id}_{MODEL_VERSION}.geojson", "w")
    json.dump(geojson, geojson_file)
    geojson_file.write("\n")
    geojson_file.close()


def process_taxon_shapefile(taxon_id):
    taxon = JOINED_TAXONOMY.loc[taxon_id]
    geojson = geojson_for_taxon(taxon)
    if geojson is None:
        return

    geojson_file = io.BytesIO()
    geojson_file.write(json.dumps(geojson).encode())
    geojson_file.seek(0)

    gdf = gpd.read_file(geojson_file)
    gdf.to_file(f"geopackages/shapefiles/{taxon_id}_{MODEL_VERSION}.shp")


def process_taxon_geopackge(taxon_id, geopackage_name):
    taxon = JOINED_TAXONOMY.loc[taxon_id]
    geojson = geojson_for_taxon(taxon)
    if geojson is None:
        return

    geojson_file = io.BytesIO()
    geojson_file.write(json.dumps(geojson).encode())
    geojson_file.seek(0)

    # Load the GeoJSON file
    gdf = gpd.read_file(geojson_file)

    # Write the GeoDataFrame to a GPKG file
    gdf.to_file(
        f"geopackages/{geopackage_name}.gpkg",
        layer=f"{taxon['name']} ({taxon_id})",
        driver="GPKG"
    )


def process_taxa_featureset_geopackage(taxon_ids, geopackage_name):
    features = []
    # for taxon_id in taxon_ids:
    for taxon_id in tqdm(taxon_ids, maxinterval=10.0, miniters=20):
        taxon = JOINED_TAXONOMY.loc[taxon_id]
        geojson = geojson_for_taxon(taxon)
        if geojson is None:
            continue

        # save the generated GeoJSON separately
        geojson_file = open(f"geopackages/geojsons/{taxon_id}_{MODEL_VERSION}.geojson", "w")
        json.dump(geojson, geojson_file)
        geojson_file.write("\n")
        geojson_file.close()

        features.append(geojson)

    geojson_file = io.BytesIO()
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    geojson_file.write(json.dumps(geojson).encode())
    geojson_file.seek(0)
    gdf = gpd.read_file(geojson_file)

    gdf.to_file(
        f"geopackages/{geopackage_name}.gpkg",
        driver="GPKG"
    )


def process_taxa_featureset_geojson(taxon_ids, geojson_name):
    features = []
    for taxon_id in taxon_ids:
        taxon = JOINED_TAXONOMY.loc[taxon_id]
        geojson = geojson_for_taxon(taxon)
        if geojson is None:
            continue

        features.append(geojson)

    geojson_file = io.BytesIO()
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    geojson_file = open(f"geopackages/geojson_{geojson_name}.geojson", "w")
    json.dump(geojson, geojson_file)
    geojson_file.write("\n")
    geojson_file.close()


# create output directories
if not os.path.exists("geopackages"):
    os.makedirs("geopackages")
if not os.path.exists("geopackages/geojsons"):
    os.makedirs("geopackages/geojsons")


start_time = time.time()
iconic_taxon_ids = np.array(JOINED_TAXONOMY["iconic_taxon_id"].unique())
iconic_taxon_ids.sort()
# loop through each iconic taxon
for iconic_taxon_id in iconic_taxon_ids:
    iconic_taxon = JOINED_TAXONOMY.loc[iconic_taxon_id]
    # fetch leaf taxon_ids within this iconic taxon
    clade_taxon_ids = np.array(JOINED_TAXONOMY.query(
        f"leaf_class_id.notnull() and iconic_taxon_id == {iconic_taxon_id}")["taxon_id"].values
    )
    clade_taxon_ids.sort()

    print(f"{iconic_taxon['name']}: {clade_taxon_ids.size}")
    batch_size = 5000
    # split taxon_ids into batches of at most 5000
    batches = np.split(clade_taxon_ids, np.arange(batch_size, len(clade_taxon_ids), batch_size))
    for idx, batch_ids in enumerate(batches):
        # if there are multiple batches, include the ID range in the filename,
        # otherwise name the file after the iconic taxon and the model version
        if len(batches) > 1:
            filename = f"{iconic_taxon['name']}_{batch_ids[0]}_{batch_ids[-1]}"
        else:
            filename = iconic_taxon["name"]
        filename = f"iNaturalist_geomodel_{MODEL_VERSION}_{filename}"
        print(f"    {filename}")
        process_taxa_featureset_geopackage(batch_ids, filename)

print("Total Time: %0.2fs" % (time.time() - start_time))
