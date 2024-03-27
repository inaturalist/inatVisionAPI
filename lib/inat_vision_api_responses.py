import numpy as np
import pandas as pd
from lib.model_taxonomy_dataframe import ModelTaxonomyDataframe


class InatVisionAPIResponses:
    @staticmethod
    def legacy_dictionary_response(leaf_scores):
        leaf_scores = InatVisionAPIResponses.limit_leaf_scores_for_response(leaf_scores)
        leaf_scores = InatVisionAPIResponses.update_leaf_scores_scaling(leaf_scores)
        top_taxon_combined_scores = leaf_scores[
            ["taxon_id", "combined_score"]
        ].to_dict(orient="records")
        return {x["taxon_id"]: x["combined_score"] for x in top_taxon_combined_scores}

    @staticmethod
    def array_response(leaf_scores):
        leaf_scores = InatVisionAPIResponses.limit_leaf_scores_for_response(leaf_scores)
        leaf_scores = InatVisionAPIResponses.update_leaf_scores_scaling(leaf_scores)
        return InatVisionAPIResponses.array_response_columns(leaf_scores).to_dict(orient="records")

    @staticmethod
    def object_response(leaf_scores, inferrer):
        leaf_scores = InatVisionAPIResponses.limit_leaf_scores_for_response(leaf_scores)
        leaf_scores = InatVisionAPIResponses.update_leaf_scores_scaling(leaf_scores)
        results = InatVisionAPIResponses.array_response_columns(
            leaf_scores
        ).to_dict(orient="records")
        common_ancestor = inferrer.common_ancestor_from_leaf_scores(leaf_scores, debug=True)
        if common_ancestor is not None:
            common_ancestor_frame = pd.DataFrame([common_ancestor])
            common_ancestor_frame = InatVisionAPIResponses.update_aggregated_scores_scaling(
                common_ancestor_frame
            )
            common_ancestor = InatVisionAPIResponses.array_response_common_ancestor_columns(
                common_ancestor_frame
            ).to_dict(orient="records")[0]

        return {
            "common_ancestor": common_ancestor,
            "results": results
        }

    @staticmethod
    def aggregated_tree_response(aggregated_scores):
        top_leaf_combined_score = aggregated_scores.query(
            "leaf_class_id.notnull()"
        ).sort_values(
            "normalized_aggregated_combined_score",
            ascending=False
        ).head(1)["normalized_aggregated_combined_score"].values[0]
        # set a cutoff so results whose combined scores are
        # much lower than the best score are not returned
        aggregated_scores = aggregated_scores.query(
            f"normalized_aggregated_combined_score > {top_leaf_combined_score * 0.001}"
        )

        printable_tree = ModelTaxonomyDataframe.printable_tree(
            aggregated_scores,
            display_taxon_lambda=(
                lambda row: f"{row.name}\t\t["
                            f"ID:{row.taxon_id}, "
                            f"V:{round(row.aggregated_vision_score, 4)}, "
                            f"NV:{round(row.normalized_aggregated_vision_score, 4)}, "
                            f"G:{round(row.aggregated_geo_score, 4)}, "
                            f"C:{round(row.aggregated_combined_score, 4)}, "
                            f"NC:{round(row.normalized_aggregated_combined_score, 4)}]"
            )
        )
        return "<pre>" + "<br/>".join(printable_tree) + "</pre>"

    @staticmethod
    def aggregated_object_response(leaf_scores, aggregated_scores, inferrer):
        top_leaf_combined_score = aggregated_scores.query(
            "leaf_class_id.notnull()"
        ).sort_values(
            "normalized_aggregated_combined_score",
            ascending=False
        ).head(1)["normalized_aggregated_combined_score"].values[0]

        top_100_leaves = aggregated_scores.query(
            "leaf_class_id.notnull() and "
            f"normalized_aggregated_combined_score > {top_leaf_combined_score * 0.001}"
        ).sort_values(
            "normalized_aggregated_combined_score",
            ascending=False
        ).head(100)

        common_ancestor = inferrer.common_ancestor_from_leaf_scores(leaf_scores, debug=True)
        aggregated_scores = InatVisionAPIResponses.update_aggregated_scores_scaling(
            aggregated_scores
        )

        filter_taxa = np.array([])
        for leaf_taxon_id in top_100_leaves["taxon_id"].to_numpy(dtype=int):
            filter_taxa = np.append(filter_taxa, leaf_taxon_id)
            filter_taxa = np.append(filter_taxa,
                                    inferrer.taxonomy.taxon_ancestors[leaf_taxon_id])
        top_100_and_ancestors = aggregated_scores[aggregated_scores["taxon_id"].isin(filter_taxa)]

        final_results = InatVisionAPIResponses.aggregated_scores_response_columns(
            top_100_and_ancestors
        )

        if common_ancestor is not None:
            common_ancestor_frame = pd.DataFrame([common_ancestor])
            common_ancestor_frame = InatVisionAPIResponses.update_aggregated_scores_scaling(
                common_ancestor_frame
            )
            common_ancestor = InatVisionAPIResponses.aggregated_scores_response_columns(
                common_ancestor_frame
            ).to_dict(orient="records")[0]

        return {
            "common_ancestor": common_ancestor,
            "results": final_results.to_dict(orient="records")
        }

    @staticmethod
    def limit_leaf_scores_for_response(leaf_scores):
        top_combined_score = leaf_scores.sort_values(
            "combined_score",
            ascending=False
        ).head(1)["combined_score"].values[0]
        # set a cutoff so results whose combined scores are
        # much lower than the best score are not returned
        leaf_scores = leaf_scores.query(f"combined_score > {top_combined_score * 0.001}")
        return leaf_scores.sort_values("combined_score", ascending=False).head(100)

    @staticmethod
    def update_leaf_scores_scaling(leaf_scores):
        score_columns = [
            "combined_score",
            "geo_score",
            "normalized_vision_score",
            "geo_threshold"
        ]
        leaf_scores[score_columns] = leaf_scores[
            score_columns
        ].multiply(100, axis="index")
        return leaf_scores

    @staticmethod
    def update_aggregated_scores_scaling(aggregated_scores):
        score_columns = [
            "aggregated_combined_score",
            "normalized_aggregated_combined_score",
            "aggregated_geo_score",
            "aggregated_vision_score",
            "aggregated_geo_threshold"
        ]
        aggregated_scores[score_columns] = aggregated_scores[
            score_columns
        ].multiply(100, axis="index")
        return aggregated_scores

    @staticmethod
    def array_response_columns(leaf_scores):
        columns_to_return = [
            "combined_score",
            "geo_score",
            "taxon_id",
            "name",
            "normalized_vision_score",
            "geo_threshold"
        ]
        column_mapping = {
            "taxon_id": "id",
            "normalized_vision_score": "vision_score"
        }
        return leaf_scores[columns_to_return].rename(columns=column_mapping)

    @staticmethod
    def array_response_common_ancestor_columns(common_ancestor_dataframe):
        columns_to_return = [
            "aggregated_combined_score",
            "aggregated_geo_score",
            "taxon_id",
            "name",
            "normalized_aggregated_vision_score",
            "aggregated_geo_threshold"
        ]
        column_mapping = {
            "aggregated_combined_score": "combined_score",
            "aggregated_geo_score": "geo_score",
            "taxon_id": "id",
            "normalized_aggregated_vision_score": "vision_score",
            "aggregated_geo_threshold": "geo_threshold"
        }
        return common_ancestor_dataframe[columns_to_return].rename(columns=column_mapping)

    @staticmethod
    def aggregated_scores_response_columns(aggregated_scores):
        columns_to_return = [
            "aggregated_combined_score",
            "normalized_aggregated_combined_score",
            "aggregated_geo_score",
            "taxon_id",
            "parent_taxon_id",
            "name",
            "rank_level",
            "left",
            "right",
            "depth",
            "aggregated_vision_score",
            "aggregated_geo_threshold",
        ]
        column_mapping = {
            "taxon_id": "id",
            "parent_taxon_id": "parent_id",
            "aggregated_combined_score": "combined_score",
            "normalized_aggregated_combined_score": "normalized_combined_score",
            "aggregated_geo_score": "geo_score",
            "aggregated_vision_score": "vision_score",
            "aggregated_geo_threshold": "geo_threshold"
        }
        return aggregated_scores[columns_to_return].rename(columns=column_mapping)
