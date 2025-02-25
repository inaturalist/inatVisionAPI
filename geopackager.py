import yaml
import h3
import json
import geopandas as gpd
import io
import time
import antimeridian
import numpy as np
import os

from tqdm.auto import tqdm
from lib.inat_inferrer import InatInferrer


CONFIG = yaml.safe_load(open("config.yml"))["models"][0]
# do not use synonym mappings when generating geomodel ranges. Use the
# original taxonomy and taxa that the model was trained with
CONFIG.pop("synonyms_path", None)
CONFIG.pop("synonyms_taxonomy_path", None)
INFERRER = InatInferrer(CONFIG)
MODEL_VERSION = CONFIG["name"]

ANTIMERIDIAN_CROSSING_CELLS = INFERRER.geo_elevation_cells.query("abs(minx-maxx) > 90").index

ICONIC_TAXON_IDS = np.array(INFERRER.taxonomy.df["iconic_taxon_id"].dropna().unique()).astype(int)
ICONIC_TAXON_IDS.sort()
iconic_taxa_array = list(map(
    lambda taxon_id: INFERRER.taxonomy.df.loc[taxon_id], ICONIC_TAXON_IDS
))
ICONIC_TAXA = {taxon["taxon_id"]: taxon for taxon in iconic_taxa_array}


def remove_antimeridian_crossing_cells(cells):
    return list(filter(lambda cell_key: cell_key not in ANTIMERIDIAN_CROSSING_CELLS, cells))


def taxon_h3_cells_to_geojson(taxon, cells):
    iconic_taxon = ICONIC_TAXA[taxon["iconic_taxon_id"]] if \
        taxon["iconic_taxon_id"] in ICONIC_TAXA else None
    if iconic_taxon is None:
        print(f"Problem: {taxon['iconic_taxon_id']} has no matching iconic taxon")
    return {
        "type": "Feature",
        "properties": {
            "taxon_id": int(taxon["taxon_id"]),
            "parent_taxon_id": int(taxon["parent_taxon_id"]),
            "name": taxon["name"],
            "rank": taxon["rank"],
            "iconic_taxon_id": None if iconic_taxon is None else int(taxon["iconic_taxon_id"]),
            "iconic_taxon_name": None if iconic_taxon is None else iconic_taxon["name"],
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


def process_taxon_geopackge(taxon_id, geopackage_name):
    taxon = INFERRER.taxonomy.df.loc[taxon_id]
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
        taxon = INFERRER.taxonomy.df.loc[taxon_id]
        geojson = geojson_for_taxon(taxon)
        if geojson is None:
            continue

        # save the generated GeoJSON separately
        geojson_file = open(f"geopackages/geojsons/{taxon_id}.geojson", "w")
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
    return len(features)


# create output directories
if not os.path.exists("geopackages"):
    os.makedirs("geopackages")
if not os.path.exists("geopackages/geojsons"):
    os.makedirs("geopackages/geojsons")


metadata = {
    "version": MODEL_VERSION,
    "ranges": 0,
    "collections": {}
}

batch_size = 5000
range_count = 0
start_time = time.time()
# loop through each iconic taxon
for iconic_taxon_id in ICONIC_TAXON_IDS:
    iconic_taxon = INFERRER.taxonomy.df.loc[iconic_taxon_id]
    # fetch leaf taxon_ids within this iconic taxon
    clade_taxon_ids = np.array(INFERRER.taxonomy.df.query(
        f"leaf_class_id.notnull() and iconic_taxon_id == {iconic_taxon_id}")["taxon_id"].values
    )
    clade_taxon_ids.sort()

    iconic_taxon_name = "OtherAnimalia" if iconic_taxon_id == 1 else iconic_taxon["name"]
    print(f"{iconic_taxon_name}: {clade_taxon_ids.size}")

    # split taxon_ids into batches of at most `batch_size`
    batches = np.split(clade_taxon_ids, np.arange(batch_size, len(clade_taxon_ids), batch_size))
    iconic_taxon_range_count = 0
    for idx, batch_ids in enumerate(batches):
        # if there are multiple batches, include the ID range in the filename,
        # otherwise name the file after the iconic taxon and the model version
        if len(batches) > 1:
            filename = f"{iconic_taxon_name}_{idx + 1}"
        else:
            filename = iconic_taxon_name
        filename = f"iNaturalist_geomodel_{filename}"
        print(f"    {filename}")
        # increment the clade range count by the number of ranges actually added
        iconic_taxon_range_count += process_taxa_featureset_geopackage(batch_ids, filename)

    # increment the count of total ranges
    range_count += iconic_taxon_range_count
    # add metadata for this iconic taxon
    metadata["collections"][iconic_taxon_name] = {
        "ranges": iconic_taxon_range_count,
        "archives": len(batches)
    }

# write the metadata to file
metadata["ranges"] = range_count
metadata_file = open("geopackages/metadata.json", "w")
json.dump(metadata, metadata_file, indent=2)
metadata_file.write("\n")

# write the taxonomy to file
taxonomy_columns_to_export = [
    "taxon_id",
    "parent_taxon_id",
    "name",
    "rank_level",
    "rank",
    "iconic_taxon_id",
    "is_leaf"
]
# convert rank_levels that do not have decimals to strings without the decimal value: 30.0 => 30
INFERRER.taxonomy.df["rank_level"] = INFERRER.taxonomy.df["rank_level"].apply(
    lambda x: str(int(x)) if x == int(x) else str(x)
)
# add a column with a boolean value indicating if the taxon is a model leaf node
INFERRER.taxonomy.df["is_leaf"] = INFERRER.taxonomy.df["leaf_class_id"].notnull()
INFERRER.taxonomy.df[taxonomy_columns_to_export].to_csv("geopackages/taxonomy.csv", index=False)


print("Total Time: %0.2fs" % (time.time() - start_time))
