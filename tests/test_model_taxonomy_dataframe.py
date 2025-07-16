import pytest
from lib.model_taxonomy_dataframe import ModelTaxonomyDataframe


class TestModelTaxonomyDataframe:
    @pytest.mark.parametrize("taxon", ["Aramus guarauna"], indirect=True)
    def test_loading_mapping(self, taxon):
        assert taxon["taxon_id"] == 7
        assert taxon["parent_taxon_id"] == 6
        assert taxon["rank_level"] == 10
        assert taxon["leaf_class_id"] == 1
        assert taxon["iconic_class_id"] == 1
        assert taxon["spatial_class_id"] == 8
        assert taxon["name"] == "Aramus guarauna"
        assert taxon["geo_threshold"] == 0.1

    @pytest.mark.parametrize("taxon", ["Aramus guarauna"], indirect=True)
    def test_nested_set_assigning(self, taxon):
        assert taxon["left"] == 7
        assert taxon["right"] == 8

    @pytest.mark.parametrize("taxon", ["Aramus guarauna"], indirect=True)
    def test_geo_threshold_assigning(self, taxon):
        assert taxon["geo_threshold"] == 0.1

    def test_children_of_root(self, taxonomy):
        children = ModelTaxonomyDataframe.children(taxonomy.df, 0)
        assert len(children.index) == 2
        assert children.iloc[0]["name"] == "Animalia"
        assert children.iloc[1]["name"] == "Plantae"

    @pytest.mark.parametrize("taxon", ["Animalia"], indirect=True)
    def test_children_of_taxon(self, taxonomy, taxon):
        children = ModelTaxonomyDataframe.children(taxonomy.df, taxon["taxon_id"])
        assert len(children.index) == 1
        assert children.iloc[0]["name"] == "Chordata"

    def test_human_taxon(self, capsys, taxonomy):
        assert taxonomy.human_taxon["name"] == "Homo sapiens"

    def test_print(self, capsys, taxonomy):
        ModelTaxonomyDataframe.print(taxonomy.df)
        captured = capsys.readouterr()
        assert "├──Animalia :: 0:41" in captured.out
        assert "│   └──Chordata :: 1:40" in captured.out

    def test_print_with_aggregated_combined_score(self, capsys, taxonomy):
        taxonomy.df["aggregated_combined_score"] = 1
        ModelTaxonomyDataframe.print(taxonomy.df)
        captured = capsys.readouterr()
        assert "├──Animalia :: 0:41" in captured.out
        assert "│   └──Chordata :: 1:40" in captured.out

    def test_print_with_lambda(self, capsys, taxonomy):
        ModelTaxonomyDataframe.print(taxonomy.df, display_taxon_lambda=(
            lambda row: "customformat"
        ))
        captured = capsys.readouterr()
        assert "customformat" in captured.out
