from lib.taxon import Taxon


class TestTaxon:
    def test_initialization(self):
        taxon = Taxon({"id": 0, "name": "Life"})
        assert taxon.name == "Life"

    def test_is_or_descendant_of_self(self):
        taxon = Taxon({"id": 1})
        assert taxon.is_or_descendant_of(taxon)

    def test_is_or_descendant_of_taxon(self):
        parent_taxon = Taxon({"id": 1, "left": 0, "right": 3})
        child_taxon = Taxon({"id": 2, "left": 1, "right": 2})
        assert child_taxon.is_or_descendant_of(parent_taxon)
        assert not parent_taxon.is_or_descendant_of(child_taxon)
