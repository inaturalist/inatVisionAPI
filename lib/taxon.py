# Taxon:
#   parent_taxon_id
#   taxon_id
#   rank_level
#   leaf_class_id
#   iconic_class_id
#   name
#   left
#   right
#   depth


class Taxon:

    def __init__(self, row):
        for key in row:
            setattr(self, key, row[key])

    def set(self, attr, val):
        setattr(self, attr, val)

    def is_or_descendant_of(self, taxon):
        if self.id == taxon.id:
            return True
        return self.descendant_of(taxon)

    # using the nested set left and right values, a taxon is a descendant of another
    # as long as its left is higher and its right is lower
    def descendant_of(self, taxon):
        return self.left > taxon.left and self.right < taxon.right
