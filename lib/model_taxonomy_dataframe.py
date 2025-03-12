import pandas as pd
import math


class ModelTaxonomyDataframe:

    def __init__(self, path, thresholds_path):
        self.load_mapping(path, thresholds_path)
        self.set_human_taxon()

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

        # create a data frame with just the leaf taxa using leaf_class_id as the index
        self.leaf_df = self.df.query("leaf_class_id.notnull()").set_index(
            "leaf_class_id", drop=False).sort_index()

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
