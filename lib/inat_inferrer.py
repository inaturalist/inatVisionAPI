import time
import tensorflow as tf
import pandas as pd
import h3
import h3pandas  # noqa: F401
import math
import os
import tifffile
import numpy as np
import urllib
import hashlib
import magic
import aiohttp
import aiofiles
import aiofiles.os
import asyncio

from PIL import Image
from lib.tf_gp_elev_model import TFGeoPriorModelElev
from lib.vision_inferrer import VisionInferrer
from lib.model_taxonomy_dataframe import ModelTaxonomyDataframe

pd.options.mode.copy_on_write = True


class InatInferrer:

    MINIMUM_GEO_SCORE = 0.005
    COMMON_ANCESTOR_CUTOFF_RATIO = 0.01
    COMMON_ANCESTOR_WINDOW = 15
    SYNONYMS_CHECK_FREQUENCY = 60

    def __init__(self, config):
        self.config = config
        self.setup_taxonomy()
        self.setup_synonyms()
        self.setup_vision_model()
        self.setup_elevation_dataframe()
        self.setup_geo_model()
        self.upload_folder = "static/"

    def setup_taxonomy(self):
        self.taxonomy = ModelTaxonomyDataframe(
            self.config["taxonomy_path"],
            self.config["tf_elev_thresholds"] if "tf_elev_thresholds" in self.config else None
        )

    def check_for_modified_synonyms(self):
        # only run the refresh check again if `SYNONYMS_CHECK_FREQUENCY` seconds have passed
        if not hasattr(self, "synonym_refresh_check_time") or (
            time.time() - self.synonym_refresh_check_time > InatInferrer.SYNONYMS_CHECK_FREQUENCY
        ):
            self.refresh_synonyms_if_modified()

    def refresh_synonyms_if_modified(self):
        self.synonym_refresh_check_time = time.time()
        # only process the synonyms file if it has changed since last being processed
        if os.path.exists(self.config["synonyms_path"]) and (
            not hasattr(self, "synonyms_path_updated_at") or  # noqa: W504
            os.path.getmtime(self.config["synonyms_path"]) != self.synonyms_path_updated_at
        ):
            self.setup_synonyms()

    def setup_synonyms(self):
        if "synonyms_path" not in self.config:
            self.synonyms = None
            return

        if not os.path.exists(self.config["synonyms_path"]):
            self.synonyms = None
            return

        self.synonyms = pd.read_csv(
            self.config["synonyms_path"],
            dtype={
                "model_taxon_id": int,
                "parent_taxon_id": "Int64",
                "taxon_id": "Int64",
                "rank_level": float,
                "name": pd.StringDtype()
            }
        )

        # create a dict indexed by model_taxon_id for efficient synonym mappings at inference time
        self.synonyms_by_model_taxon_id = {}
        for synonym in self.synonyms.to_dict("records"):
            if not synonym["model_taxon_id"] in self.synonyms_by_model_taxon_id:
                self.synonyms_by_model_taxon_id[synonym["model_taxon_id"]] = []
            self.synonyms_by_model_taxon_id[synonym["model_taxon_id"]].append(synonym)

        # record when the synonyms file was updated to know later when to refresh it
        self.synonyms_path_updated_at = os.path.getmtime(self.config["synonyms_path"])
        self.setup_synonym_taxonomy()

    def setup_synonym_taxonomy(self):
        if self.synonyms is None:
            return

        if "synonyms_taxonomy_path" not in self.config:
            return

        synonym_taxonomy = ModelTaxonomyDataframe(
            self.config["synonyms_taxonomy_path"],
            self.config["tf_elev_thresholds"] if "tf_elev_thresholds" in self.config else None
        )
        # ensure the leaf_class_ids from the synonym taxonomy are identical
        # to the taxonomy generated at data export time
        if not self.taxonomy.leaf_df.index.equals(synonym_taxonomy.leaf_df.index):
            error = "Synonym taxonomy does not match the model taxonomy"
            print(error)
            return

        synonym_taxon_ids = np.unique(pd.array(self.synonyms["taxon_id"].dropna().values))
        synonym_taxonomy_taxon_ids = np.unique(
            pd.array(synonym_taxonomy.df[
                synonym_taxonomy.df.taxon_id.isin(synonym_taxon_ids)
            ]["taxon_id"].values)
        )
        synonym_taxon_ids_not_present_in_taxonomy = np.setdiff1d(
            synonym_taxon_ids, synonym_taxonomy_taxon_ids
        )
        # ensure all taxa referenced in the synonym mappings file are present in the
        # updated taxonomy that should include all original taxa plus all synonyms
        if synonym_taxon_ids_not_present_in_taxonomy.size > 0:
            error = "There are taxa in the synonyms file not present in the synonyms " + \
                f"taxonomy:  {synonym_taxon_ids_not_present_in_taxonomy}"
            print(error)
            return

        synonym_taxonomy.leaf_df["has_synonyms"] = False
        # mark taxa that should be replaced or removed as having synonyms
        for index, taxon in self.taxonomy.leaf_df[self.taxonomy.leaf_df["taxon_id"].isin(
            self.synonyms["model_taxon_id"]
        )].iterrows():
            synonym_taxonomy.leaf_df.loc[taxon["leaf_class_id"], "has_synonyms"] = True

        # replace the originally exported taxonomy with the updated taxonomy that includes synonyms
        self.taxonomy = synonym_taxonomy

    def setup_vision_model(self):
        self.vision_inferrer = VisionInferrer(
            self.config["vision_model_path"]
        )

    def setup_elevation_dataframe(self):
        self.geo_elevation_cells = None
        if "elevation_h3_r4" not in self.config:
            return

        # load elevation data stored at H3 resolution 4
        self.geo_elevation_cells = pd.read_csv(self.config["elevation_h3_r4"]). \
            sort_values("h3_04").set_index("h3_04").sort_index()
        self.geo_elevation_cells = InatInferrer.add_lat_lng_to_h3_geo_dataframe(
            self.geo_elevation_cells
        )
        self.geo_elevation_cell_indices = {
            index: idx for idx, index in enumerate(self.geo_elevation_cells.index)
        }

    def setup_elevation_dataframe_from_worldclim(self, resolution):
        # preventing from processing at too high a resolution
        if resolution > 5:
            return

        if "wc2.1_5m_elev_2.tif" in self.config:
            im = tifffile.imread(self.config["wc2.1_5m_elev_2.tif"])
            im_df = pd.DataFrame(im)
            im_df.index = np.linspace(90, -90, 2160)
            im_df.columns = np.linspace(-180, 180, 4320)
            im_df = im_df.reset_index()
            im_df = im_df.melt(id_vars=["index"])
            im_df.columns = ["lat", "lng", "elevation"]
            elev_dfh3 = im_df.h3.geo_to_h3(resolution)
            elev_dfh3 = elev_dfh3.drop(columns=["lng", "lat"]).groupby(f"h3_0{resolution}").mean()

    def setup_geo_model(self):
        self.geo_elevation_model = None
        self.geo_model_features = None
        if "tf_geo_elevation_model_path" not in self.config:
            return

        if self.geo_elevation_cells is None:
            return

        self.geo_elevation_model = TFGeoPriorModelElev(self.config["tf_geo_elevation_model_path"])
        self.geo_model_features = self.geo_elevation_model.features_for_one_class_elevation(
            latitude=list(self.geo_elevation_cells.lat),
            longitude=list(self.geo_elevation_cells.lng),
            elevation=list(self.geo_elevation_cells.elevation)
        )

    def vision_predict(self, image, debug=False):
        if debug:
            start_time = time.time()
        results = self.vision_inferrer.process_image(image)
        if debug:
            print("Vision Time: %0.2fms" % ((time.time() - start_time) * 1000.))
        return results

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
            print(f"taxon `{taxon_id}` does not exist in the taxonomy")
            raise e

    def predictions_for_image(self, file_path, lat, lng, filter_taxon, debug=False):
        if debug:
            start_time = time.time()
        image = InatInferrer.prepare_image_for_inference(file_path)
        vision_model_results = self.vision_predict(image, debug)
        raw_vision_scores = vision_model_results["predictions"]
        raw_geo_scores = self.geo_model_predict(lat, lng, debug)
        combined_scores = self.combine_results(
            raw_vision_scores, raw_geo_scores, filter_taxon, debug
        )
        combined_scores = self.map_result_synonyms(combined_scores, debug)
        # for any taxon that doesn't have a geo threshold, set it to 1 which is the highest
        # possible value, and thus all its taxa will not be considered "expected nearby"
        combined_scores["geo_threshold"] = combined_scores["geo_threshold"].fillna(1)
        if debug:
            print("Prediction Time: %0.2fms" % ((time.time() - start_time) * 1000.))
        return {
            "combined_scores": combined_scores,
            "features": vision_model_results["features"]
        }

    def combine_results(self, raw_vision_scores, raw_geo_scores, filter_taxon, debug=False):
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
            else leaf_scores["geo_score"].clip(InatInferrer.MINIMUM_GEO_SCORE, None)

        # if filtering by a taxon, restrict results to that taxon and its descendants
        if filter_taxon is not None:
            # using nested set left and right values, select the filter taxon and its descendants
            leaf_scores = leaf_scores.query(
                f'left >= {filter_taxon["left"]} and right <= {filter_taxon["right"]}'
            )
            # normalize the vision scores so they add up to 1 after filtering
            sum_of_vision_scores = leaf_scores["vision_score"].sum()
            leaf_scores["normalized_vision_score"] = \
                leaf_scores["vision_score"] / sum_of_vision_scores
        else:
            # when not filtering by a taxon, the normalized vision score is the same as the original
            leaf_scores["normalized_vision_score"] = leaf_scores["vision_score"]

        if no_geo_scores:
            # if there are no geo scores, or it was requested to not use geo scores to affect
            # the final combined score, set the combined scores to be the same as the vision scores
            leaf_scores["combined_score"] = leaf_scores["normalized_vision_score"]
        else:
            # the combined score is simply the normalized vision score
            # multipliedby the normalized geo score
            leaf_scores["combined_score"] = leaf_scores["normalized_vision_score"] * \
                leaf_scores["normalized_geo_score"]

        sum_of_root_node_aggregated_combined_scores = leaf_scores["combined_score"].sum()
        if sum_of_root_node_aggregated_combined_scores > 0:
            leaf_scores["normalized_combined_score"] = leaf_scores[
                "combined_score"] / sum_of_root_node_aggregated_combined_scores
        else:
            leaf_scores["normalized_combined_score"] = 0

        if debug:
            print("Score Combining Time: %0.2fms" % ((time.time() - start_time) * 1000.))
        leaf_scores.reset_index(drop=True, inplace=True)
        return leaf_scores

    def map_result_synonyms(self, leaf_scores, debug=False):
        if self.synonyms is None or "has_synonyms" not in leaf_scores.columns:
            return leaf_scores

        if debug:
            start_time = time.time()
        # loop through the taxa in leaf_scores that have synonym mappings
        leaf_taxa = leaf_scores[
            leaf_scores.taxon_id.isin(self.synonyms["model_taxon_id"])
        ].to_dict("records")
        replacements = {}
        for taxon in leaf_taxa:
            if taxon["taxon_id"] not in self.synonyms_by_model_taxon_id:
                continue
            for synonym in self.synonyms_by_model_taxon_id[taxon["taxon_id"]]:
                # the taxon has been removed, so there is no replacement
                if pd.isna(synonym["taxon_id"]):
                    continue

                # replace some attributes of the leaf_scores taxon, while keeping
                # all other columns, like the scores, untouched
                replacement = taxon.copy()
                replacement["parent_taxon_id"] = synonym["parent_taxon_id"]
                replacement["taxon_id"] = synonym["taxon_id"]
                replacement["rank_level"] = synonym["rank_level"]
                replacement["name"] = synonym["name"]
                replacement["left"] = np.nan
                replacement["right"] = np.nan
                # add the replacement taxon to the synonyms dataframe
                replacements[replacement["taxon_id"]] = replacement
        # remove all taxa from leaf scores that had synonym mappings
        leaf_scores = leaf_scores.query("has_synonyms == False")
        if replacements:
            # inject the synonym replacements into leaf_scores
            leaf_scores = pd.concat([
                leaf_scores,
                pd.DataFrame.from_dict(replacements, orient="index")
            ], axis=0)
        if debug:
            print("Synonym Mapping Time: %0.2fms" % ((time.time() - start_time) * 1000.))
        return leaf_scores

    def aggregate_results(self, leaf_scores, debug=False,
                          score_ratio_cutoff=0.001,
                          max_leaf_scores_to_consider=None,
                          column_for_cutoff="combined_score"):
        if debug:
            start_time = time.time()

        # start with a copy of the whole taxonomy
        all_node_scores = self.taxonomy.df.copy().reset_index(drop=True)

        # copy columns from the already calculated leaf scores including scores
        # and class_id columns which will not be populated for synonyms in the taxonomy
        all_node_scores = pd.merge(all_node_scores, leaf_scores[[
            "taxon_id", "vision_score", "normalized_vision_score", "geo_score", "combined_score",
            "normalized_geo_score", "leaf_class_id", "iconic_class_id", "spatial_class_id"]],
            on="taxon_id",
            how="left",
            suffixes=["_x", None]
        ).set_index("taxon_id", drop=False)

        # calculate the highest combined score from leaf_scores
        top_combined_score = leaf_scores.sort_values(
            column_for_cutoff, ascending=False
        ).head(1)[column_for_cutoff].values[0]
        # define some cutoff based on a percentage of the top combined score. Taxa with
        # scores below the cutoff will be ignored when aggregating scores up the taxonomy
        cutoff = top_combined_score * score_ratio_cutoff

        # restrict score aggregation to results where the combined score is above the cutoff
        scores_to_aggregate = leaf_scores.query(
            f"{column_for_cutoff} > {cutoff}"
        )
        if max_leaf_scores_to_consider is not None:
            scores_to_aggregate = scores_to_aggregate.sort_values(
                column_for_cutoff, ascending=False
            ).head(max_leaf_scores_to_consider)

        # loop through all results where the combined score is above the cutoff
        aggregated_scores = {}
        for taxon_id, vision_score, geo_score, combined_score, geo_threshold in zip(
            scores_to_aggregate["taxon_id"],
            scores_to_aggregate["normalized_vision_score"],
            scores_to_aggregate["geo_score"],
            scores_to_aggregate["combined_score"],
            scores_to_aggregate["geo_threshold"]
        ):
            # loop through the pre-calculated ancestors of this result's taxon
            for ancestor_taxon_id in self.taxonomy.taxon_ancestors[taxon_id]:
                # set default values for the ancestor the first time it is referenced
                if ancestor_taxon_id not in aggregated_scores:
                    aggregated_scores[ancestor_taxon_id] = {}
                    aggregated_scores[ancestor_taxon_id]["aggregated_vision_score"] = 0
                    aggregated_scores[ancestor_taxon_id]["aggregated_combined_score"] = 0
                    aggregated_scores[ancestor_taxon_id]["aggregated_geo_score"] = 0
                    aggregated_scores[ancestor_taxon_id][
                        "aggregated_geo_threshold"
                    ] = geo_threshold if (ancestor_taxon_id == taxon_id) else 1.0
                # aggregated vision and combined scores are sums of descendant scores
                aggregated_scores[ancestor_taxon_id]["aggregated_vision_score"] += vision_score
                aggregated_scores[ancestor_taxon_id]["aggregated_combined_score"] += combined_score

                # aggregated geo score is the max of descendant geo scores
                if geo_score > aggregated_scores[ancestor_taxon_id]["aggregated_geo_score"]:
                    aggregated_scores[ancestor_taxon_id]["aggregated_geo_score"] = geo_score

                # aggregated geo threshold is the min of descendant geo thresholds
                if ancestor_taxon_id != taxon_id and geo_threshold < aggregated_scores[
                    ancestor_taxon_id
                ]["aggregated_geo_threshold"]:
                    aggregated_scores[ancestor_taxon_id][
                        "aggregated_geo_threshold"
                    ] = geo_threshold

        # turn the aggregated_scores dict into a data frame
        scores_df = pd.DataFrame.from_dict(aggregated_scores, orient="index")
        # merge the aggregated scores into the scoring taxonomy
        all_node_scores = all_node_scores.join(scores_df).query(
            "aggregated_combined_score.notnull()"
        )

        # calculate normalized scores so all values add to 1, to be used for thresholding
        sum_of_root_node_aggregated_vision_scores = all_node_scores.query(
            "parent_taxon_id.isnull()")["aggregated_vision_score"].sum()
        all_node_scores["normalized_aggregated_vision_score"] = all_node_scores[
            "aggregated_vision_score"] / sum_of_root_node_aggregated_vision_scores
        sum_of_root_node_aggregated_combined_scores = all_node_scores.query(
            "parent_taxon_id.isnull()")["aggregated_combined_score"].sum()
        all_node_scores["normalized_aggregated_combined_score"] = all_node_scores[
            "aggregated_combined_score"] / sum_of_root_node_aggregated_combined_scores

        if debug:
            print("Aggregation Time: %0.2fms" % ((time.time() - start_time) * 1000.))
            # InatInferrer.print_aggregated_scores(all_node_scores)
        return all_node_scores

    def h3_04_geo_results_for_taxon_and_cell(self, taxon_id, lat, lng):
        if lat is None or lng is None:
            return None
        try:
            lat_float = float(lat)
            lng_float = float(lng)
        except ValueError:
            return None

        try:
            taxon = self.taxonomy.df.loc[taxon_id]
        except KeyError:
            return None

        if pd.isna(taxon["leaf_class_id"]) or pd.isna(taxon["geo_threshold"]):
            return None

        h3_cell = h3.geo_to_h3(lat_float, lng_float, 4)
        return float(self.geo_elevation_model.eval_one_class_elevation_from_features(
            [self.geo_model_features[self.geo_elevation_cell_indices[h3_cell]]],
            int(taxon["leaf_class_id"])
        )[0][0]) / taxon["geo_threshold"]

    def h3_04_geo_results_for_taxon(self, taxon_id, bounds=[],
                                    thresholded=False, raw_results=False):
        if (self.geo_elevation_cells is None) or (self.geo_elevation_model is None):
            return
        try:
            taxon = self.taxonomy.df.loc[taxon_id]
        except Exception as e:
            print(f"taxon `{taxon_id}` does not exist in the taxonomy")
            raise e
        if pd.isna(taxon["leaf_class_id"]):
            return

        geo_scores = self.geo_elevation_model.eval_one_class_elevation_from_features(
            self.geo_model_features, int(taxon["leaf_class_id"]))
        geo_score_cells = self.geo_elevation_cells.copy()
        geo_score_cells["geo_score"] = tf.squeeze(geo_scores).numpy()
        if thresholded:
            geo_score_cells = geo_score_cells.query(f'geo_score >= {taxon["geo_threshold"]}')
        else:
            # return scores more than 10% of the taxon threshold, or more than 0.0001, whichever
            # is smaller. This reduces data needed to be redendered client-side for the Data Layer
            # mapping approach, and maybe can be removed once switching to map tiles
            lower_bound_score = np.array([0.0001, taxon["geo_threshold"] / 10]).min()
            geo_score_cells = geo_score_cells.query(f"geo_score > {lower_bound_score}")

        if bounds:
            min = geo_score_cells["geo_score"].min()
            max = geo_score_cells["geo_score"].max()
            geo_score_cells = InatInferrer.filter_geo_dataframe_by_bounds(geo_score_cells, bounds)
            if min == max:
                # all scores are the same, so no transform is needed and all cells get the max value
                geo_score_cells["geo_score"] = 1
            else:
                # perform a log transform based on the min/max score for the unbounded set
                geo_score_cells["geo_score"] = \
                    (np.log10(geo_score_cells["geo_score"]) - math.log10(min)) / \
                    (math.log10(max) - math.log10(min))

        if raw_results:
            return geo_score_cells
        return dict(zip(geo_score_cells.index.astype(str), geo_score_cells["geo_score"]))

    def h3_04_taxon_range(self, taxon_id, bounds=[]):
        taxon_range_path = os.path.join(self.config["taxon_ranges_path"], f"{taxon_id}.csv")
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
        geomodel_results = self.h3_04_geo_results_for_taxon(
            taxon_id, bounds, thresholded=True
        ) or {}
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

    def h3_04_bounds(self, taxon_id):
        geomodel_results = self.h3_04_geo_results_for_taxon(
            taxon_id, bounds=None, thresholded=True, raw_results=True)
        if geomodel_results is None:
            return
        swlat = geomodel_results["lat"].min()
        swlng = geomodel_results["lng"].min()
        nelat = geomodel_results["lat"].max()
        nelng = geomodel_results["lng"].max()
        # when the the bounds edges have the same values, add a small buffer
        if swlat == nelat:
            swlat -= 0.3
            nelat += 0.3
        if swlng == nelng:
            swlng -= 0.3
            nelng += 0.3
        return {
            "swlat": swlat,
            "swlng": swlng,
            "nelat": nelat,
            "nelng": nelng
        }

    def common_ancestor_from_leaf_scores(
        self, leaf_scores, debug=False, score_to_use="combined_score", disallow_humans=False
    ):
        aggregated_scores = self.aggregate_results(
            leaf_scores,
            debug=debug,
            score_ratio_cutoff=InatInferrer.COMMON_ANCESTOR_CUTOFF_RATIO,
            max_leaf_scores_to_consider=InatInferrer.COMMON_ANCESTOR_WINDOW,
            column_for_cutoff=score_to_use
        )
        return self.common_ancestor_from_aggregated_scores(
            aggregated_scores,
            debug=debug,
            score_to_use=score_to_use,
            disallow_humans=disallow_humans
        )

    def common_ancestor_from_aggregated_scores(
        self, aggregated_scores, debug=False, score_to_use="combined_score", disallow_humans=False
    ):
        aggregated_score_to_use = "normalized_aggregated_vision_score" if \
            score_to_use == "vision_score" else "normalized_aggregated_combined_score"
        # if using combined scores to aggregate, and there are taxa expected nearby,
        # then add a query filter to only look at nearby taxa as common ancestor candidates
        nearby_query_filter = ""
        if aggregated_score_to_use == "normalized_aggregated_combined_score" and not \
           aggregated_scores.query("aggregated_geo_score >= aggregated_geo_threshold").empty:
            nearby_query_filter = " and aggregated_geo_score >= aggregated_geo_threshold"
        common_ancestor_candidates = aggregated_scores.query(
            f"{aggregated_score_to_use} > 0.78 and rank_level >= 20 and rank_level <= 33"
            f"{nearby_query_filter}"
        ).sort_values(
            by=["rank_level"]
        )
        if common_ancestor_candidates.empty:
            return None

        common_ancestor = common_ancestor_candidates.iloc[0]
        if disallow_humans and self.taxonomy.human_taxon is not None and \
                common_ancestor["taxon_id"] == self.taxonomy.human_taxon["parent_taxon_id"]:
            return None

        return common_ancestor

    def limit_leaf_scores_that_include_humans(self, leaf_scores):
        if self.taxonomy.human_taxon is None:
            return leaf_scores

        top_results = leaf_scores.sort_values(
            "combined_score",
            ascending=False
        ).reset_index(drop=True)
        human_results = top_results.query(f"taxon_id == {self.taxonomy.human_taxon['taxon_id']}")
        # there is only 1 result, or humans aren't in the top results
        if human_results.empty or top_results.index.size == 1:
            return leaf_scores

        # at this point there are multiple results, and humans is one of them
        human_result_index = human_results.index[0]
        # if humans is first and has a substantially higher score than the next, return only humans
        if human_result_index == 0:
            human_score_margin = top_results.iloc[0]["combined_score"] / \
                top_results.iloc[1]["combined_score"]
            if human_score_margin > 1.5:
                return top_results.head(1)

        # otherwise return no results
        return leaf_scores.head(0)

    async def embeddings_for_photos(self, photos):
        response = {}
        async with aiohttp.ClientSession() as session:
            queue = asyncio.Queue()
            workers = [asyncio.create_task(self.embeddings_worker_task(queue, response, session))
                       for _ in range(5)]
            for photo in photos:
                queue.put_nowait(photo)
            await queue.join()
            for worker in workers:
                worker.cancel()
        return response

    async def embeddings_worker_task(self, queue, response, session):
        while not queue.empty():
            photo = await queue.get()
            try:
                embedding = await self.embedding_for_photo(photo["url"], session)
                response[photo["id"]] = embedding
            finally:
                queue.task_done()

    async def embedding_for_photo(self, url, session):
        if url is None:
            return

        try:
            cache_path = await self.download_photo_async(url, session)
            if cache_path is None:
                return
            return self.signature_for_image(cache_path)
        except urllib.error.HTTPError:
            return

    def signature_for_image(self, image_path, debug=False):
        if debug:
            start_time = time.time()
        image = InatInferrer.prepare_image_for_inference(image_path)
        signature = self.vision_inferrer.process_image(image)["features"]
        if debug:
            print("Signature Time: %0.2fms" % ((time.time() - start_time) * 1000.))
        if signature is None:
            return
        return signature.numpy().tolist()

    async def download_photo_async(self, url, session):
        checksum = hashlib.md5(url.encode()).hexdigest()
        cache_path = os.path.join(self.upload_folder, "download-" + checksum) + ".jpg"
        if await aiofiles.os.path.exists(cache_path):
            return cache_path
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    f = await aiofiles.open(cache_path, mode="wb")
                    await f.write(await resp.read())
                    await f.close()
        except asyncio.TimeoutError as e:
            print("`download_photo_async` timed out")
            print(e)
        if not os.path.exists(cache_path):
            return
        mime_type = magic.from_file(cache_path, mime=True)
        if mime_type != "image/jpeg":
            im = Image.open(cache_path)
            rgb_im = im.convert("RGB")
            rgb_im.save(cache_path)
        return cache_path

    @staticmethod
    def prepare_image_for_inference(file_path):
        image = Image.open(file_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = tf.image.convert_image_dtype(image, tf.float32)

        eventual_size = 299
        central_crop_factor = 0.875
        resize_min_dimension = eventual_size / central_crop_factor

        height, width = image.shape[0], image.shape[1]
        resize_ratio = min(height, width) / resize_min_dimension
        new_height = math.ceil(height / resize_ratio)
        new_width = math.ceil(width / resize_ratio)
        # resize the image so we can take a central crop without needing to resample again
        image = tf.image.resize(
            image,
            [new_height, new_width],
            method=tf.image.ResizeMethod.AREA,
            preserve_aspect_ratio=True
        )
        # determine the upper-left corner that needs to be used to grab the square crop
        offset_height = math.floor((new_height - eventual_size) / 2)
        offset_width = math.floor((new_width - eventual_size) / 2)
        # take a square crop out of the resized image
        image = tf.image.crop_to_bounding_box(
            image, offset_height, offset_width, eventual_size, eventual_size
        )
        return tf.expand_dims(image, 0)

    @staticmethod
    def add_lat_lng_to_h3_geo_dataframe(geo_df):
        h3_cells_df = pd.DataFrame(index=geo_df.index)
        # add h3 cell centroids as lat, lng
        h3_centroids_df = h3_cells_df.copy().h3.h3_to_geo()
        h3_centroids_df["lng"] = h3_centroids_df["geometry"].x
        h3_centroids_df["lat"] = h3_centroids_df["geometry"].y
        h3_centroids_df.pop("geometry")
        geo_df = geo_df.join(h3_centroids_df)

        # add h3 cell bounds as minx, miny, maxx, maxy
        h3_bounds_df = h3_cells_df.copy().h3.h3_to_geo_boundary()
        geo_df = geo_df.join(h3_bounds_df["geometry"].bounds)
        return geo_df

    @staticmethod
    def filter_geo_dataframe_by_bounds(geo_df, bounds):
        # this is querying on the centroid, but cells just outside the bounds may have a
        # centroid outside the bounds while part of the polygon is within the bounds. Add
        # a small buffer to ensure this returns any cell whose polygon is
        # even partially within the bounds
        buffer = 1.3

        # similarly, the centroid may be on the other side of the antimedirian, so lookup
        # cells that might be just over the antimeridian on either side
        antimedirian_condition = ""
        if bounds[1] < -179:
            antimedirian_condition = "or (lng > 179)"
        elif bounds[3] > 179:
            antimedirian_condition = "or (lng < -179)"

        # query for cells wtihin the buffered bounds, and potentially
        # on the other side of the antimeridian
        query = f"lat >= {bounds[0] - buffer} and lat <= {bounds[2] + buffer} and " + \
            f" ((lng >= {bounds[1] - buffer} and lng <= {bounds[3] + buffer})" + \
            f" {antimedirian_condition})"
        return geo_df.query(query)

    @staticmethod
    def print_aggregated_scores(aggregated_scores):
        thresholded_results = aggregated_scores.query(
            "normalized_aggregated_combined_score > 0.005"
        )
        print("\nTree of aggregated results:")
        ModelTaxonomyDataframe.print(thresholded_results, display_taxon_lambda=(
            lambda row: f"{row.name}    ["
                        f"ID:{row.taxon_id}, "
                        f"V:{round(row.aggregated_vision_score, 4)}, "
                        f"NV:{round(row.normalized_aggregated_vision_score, 4)}, "
                        f"G:{round(row.aggregated_geo_score, 4)}, "
                        f"GT:{round(row.aggregated_geo_threshold, 4)}, "
                        f"C:{round(row.aggregated_combined_score, 4)}, "
                        f"NC:{round(row.normalized_aggregated_combined_score, 4)}]"
        ))
        print("")
