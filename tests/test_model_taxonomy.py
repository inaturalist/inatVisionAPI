import pytest
import os
from lib.model_taxonomy import ModelTaxonomy


@pytest.fixture()
def taxonomy():
    yield ModelTaxonomy(
        os.path.realpath(os.path.dirname(__file__) + "/fixtures/taxonomy.csv")
    )


@pytest.fixture()
def taxon(request, taxonomy):
    yield next(v for k, v in taxonomy.taxa.items() if v.name == request.param)


class TestModelTaxonomyDataframe:
    def test_raise_error_on_missing_path(self):
        with pytest.raises(FileNotFoundError):
            ModelTaxonomy(
                os.path.realpath("nonsense")
            )

    @pytest.mark.parametrize("taxon", ["Aramus guarauna"], indirect=True)
    def test_loading_mapping(self, taxon):
        assert taxon.id == 7
        assert taxon.parent_id == 6
        assert taxon.rank_level == 10
        assert taxon.leaf_class_id == 1
        assert taxon.name == "Aramus guarauna"

    @pytest.mark.parametrize("taxon", ["Aramus guarauna"], indirect=True)
    def test_nested_set_assigning(self, taxon):
        assert taxon.left == 7
        assert taxon.right == 8

    def test_children_of_root(self, taxonomy):
        children = taxonomy.taxon_children[0]
        assert len(children) == 2
        assert taxonomy.taxa[children[0]].name == "Animalia"
        assert taxonomy.taxa[children[1]].name == "Plantae"

    @pytest.mark.parametrize("taxon", ["Animalia"], indirect=True)
    def test_children_of_taxon(self, taxonomy, taxon):
        children = taxonomy.taxon_children[taxon.id]
        assert len(children) == 1
        assert taxonomy.taxa[children[0]].name == "Chordata"

    def test_print(self, capsys, taxonomy):
        taxonomy.print()
        captured = capsys.readouterr()
        assert "├──Animalia :: 0:23" in captured.out
        assert "│   └──Chordata :: 1:22" in captured.out
