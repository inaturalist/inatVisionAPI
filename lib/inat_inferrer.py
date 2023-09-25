import time
import magic
import tensorflow as tf
import pandas as pd
import h3
import h3pandas  # noqa: F401
import math
import os
import tifffile
import numpy as np
from PIL import Image
from lib.tf_gp_elev_model import TFGeoPriorModelElev
from lib.vision_inferrer import VisionInferrer
from lib.model_taxonomy_dataframe import ModelTaxonomyDataframe

# TODO: better way to address the SettingWithCopyWarning warning?
pd.options.mode.chained_assignment = None

MINIMUM_GEO_SCORE = 0.005


class InatInferrer:

    def __init__(self, config):
        self.config = config
        self.setup_taxonomy(config)
        self.setup_vision_model(config)
        self.setup_elevation_dataframe(config)
        self.setup_geo_model(config)
        self.upload_folder = "static/"

    def setup_taxonomy(self, config):
        self.taxonomy = ModelTaxonomyDataframe(
            config["taxonomy_path"], config["tf_elev_thresholds"]
        )

    def setup_vision_model(self, config):
        self.vision_inferrer = VisionInferrer(config["vision_model_path"], self.taxonomy)

    def setup_elevation_dataframe(self, config):
        # load elevation data stored at H3 resolution 4
        if "elevation_h3_r4" in config:
            self.geo_elevation_cells = pd.read_csv(config["elevation_h3_r4"]). \
                sort_values("h3_04").set_index("h3_04").sort_index()
            self.geo_elevation_cells = InatInferrer.add_lat_lng_to_h3_geo_dataframe(self.geo_elevation_cells)

    def setup_elevation_dataframe_from_worldclim(self, config, resolution):
        # preventing from processing at too high a resolution
        if resolution > 5:
            return
        if "wc2.1_5m_elev_2.tif" in config:
            im = tifffile.imread(config["wc2.1_5m_elev_2.tif"])
            im_df = pd.DataFrame(im)
            im_df.index = np.linspace(90, -90, 2160)
            im_df.columns = np.linspace(-180, 180, 4320)
            im_df = im_df.reset_index()
            im_df = im_df.melt(id_vars=["index"])
            im_df.columns = ["lat", "lng", "elevation"]
            elev_dfh3 = im_df.h3.geo_to_h3(resolution)
            elev_dfh3 = elev_dfh3.drop(columns=["lng", "lat"]).groupby(f'h3_0{resolution}').mean()

    def setup_geo_model(self, config):
        if "tf_geo_elevation_model_path" in config and self.geo_elevation_cells is not None:
            self.geo_elevation_model = TFGeoPriorModelElev(config["tf_geo_elevation_model_path"])
            self.geo_model_features = self.geo_elevation_model.features_for_one_class_elevation(
                latitude=list(self.geo_elevation_cells.lat),
                longitude=list(self.geo_elevation_cells.lng),
                elevation=list(self.geo_elevation_cells.elevation)
            )

    def prepare_image_for_inference(self, file_path):
        mime_type = magic.from_file(file_path, mime=True)
        # attempt to convert non jpegs
        if mime_type != "image/jpeg":
            im = Image.open(file_path)
            image = im.convert("RGB")
        else:
            image = tf.io.read_file(file_path)
            image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.central_crop(image, 0.875)
        image = tf.image.resize(image, [299, 299], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.expand_dims(image, 0)

    def vision_predict(self, image, debug=False):
        if debug:
            start_time = time.time()
        vision_scores = self.vision_inferrer.process_image(image)
        if debug:
            print("Vision Time: %0.2fms" % ((time.time() - start_time) * 1000.))
        return vision_scores

    def geo_model_predict(self, lat, lng, debug=False):
        if debug:
            start_time = time.time()
        if lat is None or lat == "" or lng is None or lng == "":
            return None
        if self.geo_elevation_model is None:
            return None
        # lookup the H3 cell this lat lng occurs in
        h3_cell = h3.geo_to_h3(float(lat), float(lng), 4)
        h3_cell_centroid = h3.h3_to_geo(h3_cell)
        # get the average elevation of the above H3 cell
        elevation = self.geo_elevation_cells.loc[h3_cell].elevation
        geo_scores = self.geo_elevation_model.predict(
            h3_cell_centroid[0], h3_cell_centroid[1], float(elevation))
        if debug:
            print("Geo Time: %0.2fms" % ((time.time() - start_time) * 1000.))
        return geo_scores

    def lookup_taxon(self, taxon_id):
        if taxon_id is None:
            return None
        try:
            return self.taxonomy.df.loc[taxon_id]
        except Exception as e:
            print(f'taxon `{taxon_id}` does not exist in the taxonomy')
            raise e

    def predictions_for_image(self, file_path, lat, lng, filter_taxon, score_without_geo=False,
                              debug=False):
        if debug:
            start_time = time.time()
        image = self.prepare_image_for_inference(file_path)
        raw_vision_scores = self.vision_predict(image, debug)
        raw_geo_scores = self.geo_model_predict(lat, lng, debug)
        top100 = self.combine_results(raw_vision_scores, raw_geo_scores, filter_taxon,
                                      score_without_geo, debug)
        if debug:
            print("Prediction Time: %0.2fms" % ((time.time() - start_time) * 1000.))
        return top100

    def combine_results(self, raw_vision_scores, raw_geo_scores, filter_taxon,
                        score_without_geo=False, debug=False):
        if debug:
            start_time = time.time()
        no_geo_scores = (raw_geo_scores is None)

        # make a copy of the model taxonomy leaf nodes to be used for storing results. Skip any
        # filtering at this stage as the taxonomy dataframe needs to have the same number of
        # leaf taxa and in the same order as the raw scores
        leaf_scores = self.taxonomy.leaf_df.copy()
        # add a column for vision scores
        leaf_scores["vision_score"] = raw_vision_scores
        # add a column for geo scores
        leaf_scores["geo_score"] = 0 if no_geo_scores else raw_geo_scores
        # set a lower limit for geo scores if there are any
        leaf_scores["normalized_geo_score"] = 0 if no_geo_scores \
            else leaf_scores["geo_score"].clip(MINIMUM_GEO_SCORE, None)

        # if filtering by a taxon, restrict results to that taxon and its descendants
        if filter_taxon is not None:
            # using nested set left and right values, select the filter taxon and its descendants
            leaf_scores = leaf_scores.query(
                f'left >= {filter_taxon["left"]} and right <= {filter_taxon["right"]}'
            )
            # normalize the vision scores so they add up to 1 after filtering
            sum_of_vision_scores = leaf_scores["vision_score"].sum()
            leaf_scores["normalized_vision_score"] = leaf_scores["vision_score"] / sum_of_vision_scores
        else:
            # when not filtering by a taxon, the normalized vision score is the same as the original
            leaf_scores["normalized_vision_score"] = leaf_scores["vision_score"]

        if no_geo_scores or score_without_geo:
            # if there are no geo scores, or it was requested to not use geo scores to affect
            # the final combined score, set the combined scores to be the same as the vision scores
            leaf_scores["combined_score"] = leaf_scores["normalized_vision_score"]
        else:
            # the combined score is simply the normalized vision score
            # multipliedby the normalized geo score
            leaf_scores["combined_score"] = leaf_scores["normalized_vision_score"] * \
                leaf_scores["normalized_geo_score"]
        if debug:
            print("Score Combining Time: %0.2fms" % ((time.time() - start_time) * 1000.))
        return leaf_scores

    def aggregate_results(self, leaf_scores, filter_taxon, score_without_geo=False, debug=False):
        if debug:
            start_time = time.time()

        no_geo_scores = (leaf_scores["geo_score"].max() == 0)
        # make a copy of the full taxonomy including non-leaves to be used for storing results
        if filter_taxon is not None:
            # using nested set left and right values, select the filter taxon,
            # its descendants, and its ancestors
            all_node_scores = self.taxonomy.df.query(
                f'(left >= {filter_taxon["left"]} and right <= {filter_taxon["right"]}) or' +
                f'(left < {filter_taxon["left"]} and right > {filter_taxon["right"]})'
            ).copy().reset_index(drop=True)
        else:
            all_node_scores = self.taxonomy.df.copy().reset_index(drop=True)

        # copy the score columns from the already-calculated leaf scores
        all_node_scores = pd.merge(all_node_scores, leaf_scores[[
            "taxon_id", "vision_score", "normalized_vision_score", "geo_score",
            "normalized_geo_score"]], on="taxon_id", how="left").set_index("taxon_id", drop=False)

        # calculate the highest combined score
        top_combined_score = leaf_scores.sort_values(
            "combined_score", ascending=False).head(1)["combined_score"].values[0]
        lower_cutoff_threshold = 0.0001
        # determine a lower-bound cutoff where results with combined scores below this cutoff
        # will be ignored. This isn't necessary for scoring, but it improves performance
        # TODO: evaluate this
        cutoff = max([0.00001, top_combined_score * lower_cutoff_threshold])

        aggregated_scores = {}
        # restrict score aggregation to results where the combined score is above the cutoff
        scores_to_aggregate = leaf_scores.query(f'combined_score > {cutoff}')
        # loop through all results where the combined score is above the cutoff
        for taxon_id, vision_score, geo_score, geo_threshold in zip(
            scores_to_aggregate["taxon_id"],
            scores_to_aggregate["normalized_vision_score"],
            scores_to_aggregate["normalized_geo_score"],
            scores_to_aggregate["geo_threshold"]
        ):
            # loop through the pre-calculated ancestors of this result's taxon
            for ancestor_taxon_id in self.taxonomy.taxon_ancestors[taxon_id]:
                # set default values for the ancestor the first time it is referenced
                if ancestor_taxon_id not in aggregated_scores:
                    aggregated_scores[ancestor_taxon_id] = {}
                    aggregated_scores[ancestor_taxon_id]["aggregated_vision_score"] = 0
                    if not no_geo_scores:
                        aggregated_scores[ancestor_taxon_id]["aggregated_geo_score"] = 0
                        aggregated_scores[ancestor_taxon_id]["aggregated_geo_threshold"] = 100
                # aggregated vision score is a sum of descendant scores
                aggregated_scores[ancestor_taxon_id]["aggregated_vision_score"] += vision_score
                if not no_geo_scores and geo_score > aggregated_scores[ancestor_taxon_id]["aggregated_geo_score"]:
                    # aggregated geo score is the max of descendant geo scores
                    aggregated_scores[ancestor_taxon_id]["aggregated_geo_score"] = geo_score
                if not no_geo_scores and \
                    aggregated_scores[ancestor_taxon_id]["aggregated_geo_threshold"] != 0 and \
                        geo_score > geo_threshold:
                    # aggregated geo threshold is set to 0 if any descendants are above their threshold
                    aggregated_scores[ancestor_taxon_id]["aggregated_geo_threshold"] = 0

        # turn the aggregated_scores dict into a data frame
        scores_df = pd.DataFrame.from_dict(aggregated_scores, orient="index")
        # merge the aggregated scores into the scoring taxonomy
        all_node_scores = all_node_scores.join(scores_df)

        # the aggregated scores of leaves will be NaN, so populate them with their original scores
        all_node_scores["aggregated_vision_score"].fillna(
            all_node_scores["normalized_vision_score"], inplace=True)
        if no_geo_scores:
            all_node_scores["aggregated_geo_score"] = 0
            all_node_scores["aggregated_geo_threshold"] = 0
        else:
            all_node_scores["aggregated_geo_score"].fillna(
                all_node_scores["normalized_geo_score"], inplace=True)
            all_node_scores["aggregated_geo_threshold"].fillna(
                all_node_scores["geo_threshold"], inplace=True)

        if (no_geo_scores or score_without_geo):
            # if there are no geo scores, or it was requested to not use geo scores to affect
            # the final combined score, set the combined scores to be the same as the vision scores
            all_node_scores["aggregated_combined_score"] = all_node_scores["aggregated_vision_score"]
        else:
            # the combined score is simply the normalized vision score
            # multipliedby the normalized geo score
            all_node_scores["aggregated_combined_score"] = all_node_scores["aggregated_vision_score"] * \
                all_node_scores["aggregated_geo_score"]

        # calculate a normalized combined score so all values add to 1, to be used for thresholding
        sum_of_root_node_aggregated_combined_scores = all_node_scores.query(
            "parent_taxon_id.isnull()")["aggregated_combined_score"].sum()
        all_node_scores["normalized_aggregated_combined_score"] = all_node_scores[
            "aggregated_combined_score"] / sum_of_root_node_aggregated_combined_scores

        if debug:
            print("Aggregation Time: %0.2fms" % ((time.time() - start_time) * 1000.))
            thresholded_results = all_node_scores.query("normalized_aggregated_combined_score > 0.05")
            print("\nTree of aggregated results:")
            ModelTaxonomyDataframe.print(thresholded_results, display_taxon_lambda=(
                lambda row: f'{row.name}    [' +
                            f'V:{round(row.aggregated_vision_score, 4)}, ' +
                            f'G:{round(row.aggregated_geo_score, 4)}, ' +
                            f'C:{round(row.aggregated_combined_score, 4)}, ' +
                            f'NC:{round(row.normalized_aggregated_combined_score, 4)}]'))
            print("")
        return all_node_scores

    def h3_04_geo_results_for_taxon(self, taxon_id, bounds=[], thresholded=False):
        if (self.geo_elevation_cells is None) or (self.geo_elevation_model is None):
            return
        try:
            taxon = self.taxonomy.df.loc[taxon_id]
        except Exception as e:
            print(f'taxon `{taxon_id}` does not exist in the taxonomy')
            raise e
        if math.isnan(taxon["leaf_class_id"]):
            return

        geo_scores = self.geo_elevation_model.eval_one_class_elevation_from_features(
            self.geo_model_features, int(taxon["leaf_class_id"]))
        geo_score_cells = self.geo_elevation_cells.copy()
        geo_score_cells["geo_score"] = tf.squeeze(geo_scores).numpy()
        if thresholded:
            geo_score_cells = geo_score_cells.query(f'geo_score > {taxon["geo_threshold"]}')
        else:
            # return scores more than 10% of the taxon threshold, or more than 0.0001, whichever
            # is smaller. This reduces data needed to be redendered client-side for the Data Layer
            # mapping approach, and maybe can be removed once switching to map tiles
            lower_bound_score = np.array([0.0001, taxon["geo_threshold"] / 10]).min()
            geo_score_cells = geo_score_cells.query(f'geo_score > {lower_bound_score}')

        if bounds:
            min = geo_score_cells["geo_score"].min()
            max = geo_score_cells["geo_score"].max()
            geo_score_cells = InatInferrer.filter_geo_dataframe_by_bounds(geo_score_cells, bounds)
            # perform a log transform on the scores based on the min/max score for the unbounded set
            geo_score_cells["geo_score"] = \
                (np.log10(geo_score_cells["geo_score"]) - math.log10(min)) / \
                (math.log10(max) - math.log10(min))

        return dict(zip(geo_score_cells.index.astype(str), geo_score_cells["geo_score"]))

    def h3_04_taxon_range(self, taxon_id, bounds=[]):
        taxon_range_path = os.path.join(self.config["taxon_ranges_path"], f'{taxon_id}.csv')
        if not os.path.exists(taxon_range_path):
            return None
        taxon_range_df = pd.read_csv(taxon_range_path, names=["h3_04"], header=None). \
            sort_values("h3_04").set_index("h3_04").sort_index()
        taxon_range_df = InatInferrer.add_lat_lng_to_h3_geo_dataframe(taxon_range_df)
        if bounds:
            taxon_range_df = InatInferrer.filter_geo_dataframe_by_bounds(taxon_range_df, bounds)
        taxon_range_df["value"] = 1
        return dict(zip(taxon_range_df.index.astype(str), taxon_range_df["value"]))

    def h3_04_taxon_range_comparison(self, taxon_id, bounds=[]):
        geomodel_results = self.h3_04_geo_results_for_taxon(taxon_id, bounds, True) or {}
        taxon_range_results = self.h3_04_taxon_range(taxon_id, bounds) or {}
        combined_results = {}
        for cell_key in geomodel_results:
            if cell_key in taxon_range_results:
                combined_results[cell_key] = 0.5
            else:
                combined_results[cell_key] = 0
        for cell_key in taxon_range_results:
            if cell_key not in geomodel_results:
                combined_results[cell_key] = 1
        return combined_results

    @staticmethod
    def add_lat_lng_to_h3_geo_dataframe(geo_df):
        geo_df = geo_df.h3.h3_to_geo()
        geo_df["lng"] = geo_df["geometry"].x
        geo_df["lat"] = geo_df["geometry"].y
        geo_df.pop("geometry")
        return geo_df

    @staticmethod
    def filter_geo_dataframe_by_bounds(geo_df, bounds):
        # this is querying on the centroid, but cells just outside the bounds may have a
        # centroid outside the bounds while part of the polygon is within the bounds. Add
        # a small buffer to ensure this returns any cell whose polygon is
        # even partially within the bounds
        buffer = 0.6

        # similarly, the centroid may be on the other side of the antimedirian, so lookup
        # cells that might be just over the antimeridian on either side
        antimedirian_condition = ""
        if bounds[1] < -179:
            antimedirian_condition = "or (lng > 179)"
        elif bounds[3] > 179:
            antimedirian_condition = "or (lng < -179)"

        # query for cells wtihin the buffered bounds, and potentially
        # on the other side of the antimeridian
        query = f'lat >= {bounds[0] - buffer} and lat <= {bounds[2] + buffer} and ' + \
            f' ((lng >= {bounds[1] - buffer} and lng <= {bounds[3] + buffer})' + \
            f' {antimedirian_condition})'
        return geo_df.query(query)
