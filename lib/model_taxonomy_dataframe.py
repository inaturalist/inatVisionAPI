import pandas as pd
import math
import numba
import numpy as np

from dataclasses import dataclass


class ModelTaxonomyDataframe:

    def __init__(self, path, thresholds_path):
        self.load_mapping(path, thresholds_path)
        self.set_human_taxon()
        self.set_mammals_taxon()

    def load_mapping(self, path, thresholds_path):
        self.df = pd.read_csv(
            path,
            dtype={
                "parent_taxon_id": "Int64",
                "taxon_id": int,
                "rank_level": float,
                "leaf_class_id": "Int64",
                "iconic_class_id": "Int64",
                "spatial_class_id": "Int64",
                "name": pd.StringDtype(),
                "iconic_taxon_id": "Int64",
                "rank": pd.StringDtype()
            }
        ).set_index("taxon_id", drop=False).sort_index()
        # left and right will be used to store nested set indices
        self.taxon_children = {}
        self.taxon_row_mapping = {}
        self.taxon_ancestors = {}
        for taxon_id, parent_taxon_id in zip(self.df["taxon_id"], self.df["parent_taxon_id"]):
            parent_id = 0 if pd.isna(parent_taxon_id) else int(parent_taxon_id)
            if parent_id not in self.taxon_children:
                self.taxon_children[parent_id] = []
            self.taxon_children[parent_id].append(taxon_id)

        self.nested_set_values = {}
        self.assign_nested_values()
        nested_set_values_df = pd.DataFrame.from_dict(self.nested_set_values, orient="index")
        self.df = self.df.join(nested_set_values_df)

        if thresholds_path is not None:
            thresholds = pd.read_csv(thresholds_path)[["taxon_id", "thres"]]. \
                rename(columns={"thres": "geo_threshold"}).set_index("taxon_id").sort_index()
            # round thresholds down to 5 decimal places, as long as that won't make it 0
            thresholds["geo_threshold"] = thresholds["geo_threshold"].apply(
                lambda x: x if x < 0.00001 else math.floor(x * 100000) / 100000
            )
            self.df = self.df.join(thresholds)
            self.df["geo_threshold"] = self.df["geo_threshold"].fillna(1)

        self.aggregation_index = build_taxonomy_aggregation_index(
            self.df,
            left_column="left",
        )

        # create a data frame with just the leaf taxa using leaf_class_id as the index
        self.leaf_df = self.df.query("leaf_class_id.notnull()").set_index(
            "leaf_class_id", drop=False).sort_index()

        self.spatial_df = self.df.query("spatial_class_id.notnull()").set_index(
            "spatial_class_id", drop=False).sort_index()

    # calculate nested set left and right values. These can be later used for an efficient
    # way to calculate if a taxon is an ancestor or descendant of another
    def assign_nested_values(self, taxon_id=0, index=0, depth=0, ancestor_taxon_ids=[]):
        for child_id in self.taxon_children[taxon_id]:
            self.nested_set_values[child_id] = {}
            self.nested_set_values[child_id]["left"] = index
            self.nested_set_values[child_id]["depth"] = depth
            child_ancestor_taxon_ids = ancestor_taxon_ids + [child_id]
            self.taxon_ancestors[child_id] = child_ancestor_taxon_ids
            index += 1
            if child_id in self.taxon_children:
                index = self.assign_nested_values(
                    child_id, index, depth + 1, child_ancestor_taxon_ids
                )
            self.nested_set_values[child_id]["right"] = index
            index += 1
        return index

    def set_human_taxon(self):
        self.human_taxon = None
        human_rows = self.df.query("name == 'Homo sapiens'")
        if human_rows.empty:
            return

        self.human_taxon = human_rows.iloc[0]

    def set_mammals_taxon(self):
        self.mammals_taxon = None
        if self.human_taxon is None:
            return

        mammals_rows = self.df.query(
            f"name == 'Mammalia' and left < {self.human_taxon['left']} and "
            f"right > {self.human_taxon['right']}"
        )
        if mammals_rows.empty:
            return

        self.mammals_taxon = mammals_rows.iloc[0]

    @staticmethod
    def children(df, taxon_id):
        if taxon_id == 0:
            return df.query("parent_taxon_id.isnull()")
        return df.query(f"parent_taxon_id == {taxon_id}")

    @staticmethod
    def print(df, taxon_id=0, ancestor_prefix="", display_taxon_lambda=None):
        print("\n".join(ModelTaxonomyDataframe.printable_tree(
            df, taxon_id, ancestor_prefix, display_taxon_lambda
        )))

    @staticmethod
    def printable_tree(df, taxon_id=0, ancestor_prefix="", display_taxon_lambda=None):
        children = ModelTaxonomyDataframe.children(df, taxon_id)
        index = 0
        if "aggregated_combined_score" in children:
            children = children.sort_values("aggregated_combined_score", ascending=False)
        elif "normalized_combined_score" in children:
            children = children.sort_values("normalized_combined_score", ascending=False)
        else:
            children = children.sort_values("name")
        linesToPrint = []
        for row in children.itertuples():
            last_in_branch = (index == len(children) - 1)
            index += 1
            icon = "└──" if last_in_branch else "├──"
            prefixIcon = "   " if last_in_branch else "│   "
            lineToPrint = f"{ancestor_prefix}{icon}"
            if display_taxon_lambda is None:
                lineToPrint += f"{row.name} :: {row.left}:{row.right}"
            else:
                lineToPrint += display_taxon_lambda(row)
            linesToPrint.append(lineToPrint)
            if row.right != row.left + 1:
                linesToPrint += ModelTaxonomyDataframe.printable_tree(
                    df,
                    row.taxon_id,
                    f"{ancestor_prefix}{prefixIcon}",
                    display_taxon_lambda
                )
        return linesToPrint

    # an API request may contain a taxon_id parameter where the ID does not match any taxa
    # known to the model. In that case, the value should not be ignored and treated as if
    # there is no filter taxon as that would return all results. Rather we want some surrogate
    # taxon that will allow all logic that references the filter taxon to work, but ultimately
    # return no results as nothing will be a descendant. This method returns a dict where
    # left and right are 0, and when used to query for descendants, none will be returned
    @staticmethod
    def undefined_filter_taxon():
        return {
            "left": 0,
            "right": 0
        }


@dataclass(frozen=True)
class TaxonomyAggregationIndex:
    # Indexed by spatial_class_id:
    parent_position: np.ndarray
    # All nodes, children before parents:
    postorder: np.ndarray
    # All nodes, parents before children:
    preorder: np.ndarray


def build_taxonomy_aggregation_index(
    taxonomy: pd.DataFrame,
    *,
    id_column: str = "taxon_id",
    parent_id_column: str = "parent_taxon_id",
    spatial_class_id_column: str = "spatial_class_id",
    left_column: str = "left",
) -> TaxonomyAggregationIndex:
    """
    Build dense tree arrays where every array position is spatial_class_id.

    Assumptions:
    - spatial_class_id is unique
    - spatial_class_id contains every integer from 0 through N - 1
    - parent_id_column refers to taxon IDs, not spatial_class_ids
    - descending nested-set left order processes descendants before ancestors
    """
    node_count = len(taxonomy)

    spatial_class_ids = taxonomy[
        spatial_class_id_column
    ].to_numpy(dtype=np.int64, copy=False)

    if taxonomy[spatial_class_id_column].isna().any():
        raise ValueError(
            f"{spatial_class_id_column} cannot contain null values"
        )

    if taxonomy[spatial_class_id_column].duplicated().any():
        raise ValueError(
            f"{spatial_class_id_column} must be unique"
        )

    # Since raw_scores[i] represents spatial_class_id i, IDs must be dense.
    expected_ids = np.arange(node_count, dtype=np.int64)

    if not np.array_equal(np.sort(spatial_class_ids), expected_ids):
        raise ValueError(
            f"{spatial_class_id_column} must contain every integer "
            f"from 0 through {node_count - 1}"
        )

    if taxonomy[id_column].duplicated().any():
        raise ValueError(f"{id_column} must be unique")

    # Map taxon_id -> spatial_class_id. This is only needed during setup.
    taxon_id_to_position = pd.Series(
        spatial_class_ids,
        index=taxonomy[id_column],
    )

    # Arrays are indexed directly by spatial_class_id.
    parent_position = np.full(
        node_count,
        -1,
        dtype=np.int32,
    )

    has_parent = taxonomy[parent_id_column].notna()

    parent_positions = taxon_id_to_position.reindex(
        taxonomy.loc[has_parent, parent_id_column]
    )

    if parent_positions.isna().any():
        missing_parent_ids = (
            taxonomy.loc[has_parent, parent_id_column]
            .loc[parent_positions.isna().to_numpy()]
            .unique()
        )

        raise ValueError(
            "Parent taxon IDs are missing from the taxonomy: "
            f"{missing_parent_ids[:10]}"
        )

    child_positions = taxonomy.loc[
        has_parent,
        spatial_class_id_column,
    ].to_numpy(dtype=np.int64, copy=False)

    parent_position[child_positions] = parent_positions.to_numpy(
        dtype=np.int32,
    )

    left_values = taxonomy[left_column].to_numpy(
        dtype=np.int64,
        copy=False,
    )

    preorder_row_indices = np.argsort(
        left_values,
        kind="stable",
    )

    preorder = spatial_class_ids[
        preorder_row_indices
    ].astype(
        np.int32,
        copy=False,
    )

    postorder = preorder[::-1].copy()

    return TaxonomyAggregationIndex(
        parent_position=parent_position,
        postorder=postorder,
        preorder=preorder
    )


@numba.njit(cache=True, nogil=True)
def aggregate_taxonomy_up_inplace(
    parent_position: np.ndarray,
    postorder: np.ndarray,
    min_arrays: tuple,
    max_arrays: tuple,
    sum_arrays: tuple,
) -> None:
    """
    Aggregate each array over the node itself and all descendants.

    Arrays must be writable and indexed by spatial_class_id.
    Pass () for unused aggregation types.
    """
    for order_index in range(postorder.size):
        node = postorder[order_index]
        parent = parent_position[node]

        if parent < 0:
            continue

        if len(min_arrays) > 0:
            for min_values in min_arrays:
                if (np.isnan(min_values[parent]) or min_values[node] < min_values[parent]):
                    min_values[parent] = min_values[node]

        if len(max_arrays) > 0:
            for max_values in max_arrays:
                if max_values[node] > max_values[parent]:
                    max_values[parent] = max_values[node]

        if len(sum_arrays) > 0:
            for sum_values in sum_arrays:
                sum_values[parent] += sum_values[node]


@numba.njit(cache=True, nogil=True)
def propagate_max_down_tree_inplace(
    scores: np.ndarray,
    parent_position: np.ndarray,
    preorder: np.ndarray,
) -> None:
    """
    Afterward:

        scores[node] = max(
            raw score at node,
            raw scores of all ancestors
        )
    """
    for order_index in range(preorder.size):
        node = preorder[order_index]
        parent = parent_position[node]

        if parent >= 0 and scores[parent] > scores[node]:
            scores[node] = scores[parent]
