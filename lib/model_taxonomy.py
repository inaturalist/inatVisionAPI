import csv
from lib.taxon import Taxon


class ModelTaxonomy:

    def __init__(self, path):
        self.load_mapping(path)
        self.assign_nested_values()

    def load_mapping(self, path):
        self.node_key_to_leaf_class_id = {}
        self.leaf_class_to_taxon = {}
        # there is no taxon with ID 0, but roots of the taxonomy have a parent ID of 0,
        # so create a fake taxon of Life to represent the root of the entire tree
        self.taxa = {0: Taxon({"name": "Life", "depth": 0})}
        self.taxon_children = {}
        try:
            with open(path) as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=",")
                for row in csv_reader:
                    taxon_id = int(row["taxon_id"])
                    rank_level = float(row["rank_level"])
                    leaf_class_id = int(row["leaf_class_id"]) if row["leaf_class_id"] else None
                    parent_id = int(row["parent_taxon_id"]) if row["parent_taxon_id"] else 0
                    # some taxa are not leaves and aren't represented in the leaf layer
                    if leaf_class_id is not None:
                        self.node_key_to_leaf_class_id[taxon_id] = leaf_class_id
                        self.leaf_class_to_taxon[leaf_class_id] = taxon_id
                    self.taxa[taxon_id] = Taxon({
                        "id": taxon_id,
                        "name": row["name"],
                        "parent_id": parent_id,
                        "leaf_class_id": leaf_class_id,
                        "rank_level": rank_level
                    })
                    if parent_id not in self.taxon_children:
                        self.taxon_children[parent_id] = []
                    self.taxon_children[parent_id].append(taxon_id)
        except IOError as e:
            print(e)
            print(f"\n\nCannot open mapping file `{path}`\n\n")
            raise e

    # prints to the console a representation of this tree
    def print(self, taxon_id=0, ancestor_prefix=""):
        children = self.taxon_children[taxon_id]
        index = 0
        for child_id in children:
            last_in_branch = (index == len(children) - 1)
            index += 1
            icon = "└──" if last_in_branch else "├──"
            prefixIcon = "   " if last_in_branch else "│   "
            taxon = self.taxa[child_id]
            print(f'{ancestor_prefix}{icon}{taxon.name} :: {taxon.left}:{taxon.right}')
            if child_id in self.taxon_children:
                self.print(child_id, f"{ancestor_prefix}{prefixIcon}")

    # calculated nested set left and right values and depth representing how many nodes
    # down the taxon is from Life. These can be later used for an efficient way to calculate
    # if a taxon is a descendant of another
    def assign_nested_values(self, taxon_id=0, index=0, depth=1, ancestors=[]):
        for child_id in self.taxon_children[taxon_id]:
            self.taxa[child_id].set("left", index)
            self.taxa[child_id].set("depth", depth)
            self.taxa[child_id].set("ancestors", ancestors)
            index += 1
            if child_id in self.taxon_children:
                child_ancestors = ancestors + [child_id]
                index = self.assign_nested_values(child_id, index, depth + 1, child_ancestors)
            self.taxa[child_id].set("right", index)
            index += 1
        return index
