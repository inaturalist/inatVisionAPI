import pandas as pd


class ModelTaxonomyDataframe:

    def __init__(self, path, thresholds_path):
        self.load_mapping(path, thresholds_path)

    def load_mapping(self, path, thresholds_path):
        self.df = pd.read_csv(
            path,
            usecols=[
                "parent_taxon_id",
                "taxon_id",
                "rank_level",
                "leaf_class_id",
                "iconic_class_id",
                "spatial_class_id",
                "name"
            ],
            dtype={
                "parent_taxon_id": "Int64",
                "taxon_id": int,
                "rank_level": float,
                "leaf_class_id": "Int64",
                "iconic_class_id": "Int64",
                "spatial_class_id": "Int64",
                "name": pd.StringDtype()
            }
        )
        # left and right will be used to store nested set indices
        self.df["left"] = pd.Series([], dtype=object)
        self.df["right"] = pd.Series([], dtype=object)
        self.taxon_children = {}
        self.taxon_row_mapping = {}
        self.taxon_ancestors = {}
        for index, taxon in self.df.iterrows():
            self.taxon_row_mapping[taxon["taxon_id"]] = index
            parent_id = 0 if pd.isna(taxon["parent_taxon_id"]) else int(taxon["parent_taxon_id"])
            if parent_id not in self.taxon_children:
                self.taxon_children[parent_id] = []
            self.taxon_children[parent_id].append(taxon["taxon_id"])
        self.assign_nested_values()
        self.df = self.df.set_index("taxon_id", drop=False).sort_index()
        if thresholds_path is not None:
            thresholds = pd.read_csv(thresholds_path)[["taxon_id", "thres"]]. \
                rename(columns={"thres": "geo_threshold"}).set_index("taxon_id").sort_index()
            self.df = self.df.join(thresholds)

        # create a data frame with just the leaf taxa using leaf_class_id as the index
        self.leaf_df = self.df.query("leaf_class_id.notnull()").set_index(
            "leaf_class_id", drop=False).sort_index()

    # calculate nested set left and right values. These can be later used for an efficient
    # way to calculate if a taxon is an ancestor or descendant of another
    def assign_nested_values(self, taxon_id=0, index=0, depth=0, ancestor_taxon_ids=[]):
        for child_id in self.taxon_children[taxon_id]:
            self.df.at[self.taxon_row_mapping[child_id], "left"] = index
            self.df.at[self.taxon_row_mapping[child_id], "depth"] = depth
            child_ancestor_taxon_ids = ancestor_taxon_ids + [child_id]
            self.taxon_ancestors[child_id] = child_ancestor_taxon_ids
            index += 1
            if child_id in self.taxon_children:
                index = self.assign_nested_values(
                    child_id, index, depth + 1, child_ancestor_taxon_ids
                )
            self.df.at[self.taxon_row_mapping[child_id], "right"] = index
            index += 1
        return index

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
