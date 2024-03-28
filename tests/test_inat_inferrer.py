import tensorflow as tf
import pandas as pd
import os
import pytest
from unittest.mock import MagicMock
from lib.res_layer import ResLayer
from lib.model_taxonomy_dataframe import ModelTaxonomyDataframe


class TestInatInferrer:
    def test_initialization(self, inatInferrer):
        assert isinstance(inatInferrer.taxonomy, ModelTaxonomyDataframe)
        assert isinstance(inatInferrer.synonyms, pd.DataFrame)
        assert isinstance(inatInferrer.geo_elevation_cells, pd.DataFrame)
        tf.keras.models.load_model.assert_any_call(
            inatInferrer.config["vision_model_path"],
            compile=False
        )
        tf.keras.models.load_model.assert_any_call(
            inatInferrer.config["tf_geo_elevation_model_path"],
            custom_objects={"ResLayer": ResLayer},
            compile=False
        )

    def test_predictions_for_image(self, inatInferrer):
        test_image_path = \
            os.path.realpath(os.path.dirname(__file__) + "/fixtures/lamprocapnos_spectabilis.jpeg")
        scores = inatInferrer.predictions_for_image(
            file_path=test_image_path,
            lat=42,
            lng=-71,
            filter_taxon=None,
            debug=True
        )
        assert isinstance(scores, pd.DataFrame)
        assert "leaf_class_id" in scores.columns
        assert "parent_taxon_id" in scores.columns
        assert "taxon_id" in scores.columns
        assert "rank_level" in scores.columns
        assert "iconic_class_id" in scores.columns
        assert "vision_score" in scores.columns
        assert "geo_score" in scores.columns
        assert "normalized_vision_score" in scores.columns
        assert "normalized_geo_score" in scores.columns
        assert "combined_score" in scores.columns
        assert "geo_threshold" in scores.columns

    def test_geo_model_predict_with_no_location(self, inatInferrer):
        assert inatInferrer.geo_model_predict(lat=None, lng=None) is None
        assert inatInferrer.geo_model_predict(lat="", lng="") is None

    @pytest.mark.parametrize("taxon", ["Aramus guarauna"], indirect=True)
    def test_lookup_taxon(self, inatInferrer, taxon):
        assert inatInferrer.lookup_taxon(taxon["taxon_id"])["name"] == taxon["name"]

    def test_lookup_taxon_with_no_taxon(self, inatInferrer):
        assert inatInferrer.lookup_taxon(None) is None

    def test_lookup_taxon_with_invalid_taxon(self, inatInferrer):
        with pytest.raises(KeyError):
            assert inatInferrer.lookup_taxon(999999999) is None

    def test_aggregate_results(self, inatInferrer):
        test_image_path = \
            os.path.realpath(os.path.dirname(__file__) + "/fixtures/lamprocapnos_spectabilis.jpeg")
        scores = inatInferrer.predictions_for_image(
            file_path=test_image_path,
            lat=42,
            lng=-71,
            filter_taxon=None,
            debug=True
        )
        scores.normalized_vision_score = 0.5
        scores.normalized_geo_score = 0.5
        scores.combined_score = 0.25
        scores.geo_threshold = 0.001
        aggregated_scores = inatInferrer.aggregate_results(
            leaf_scores=scores,
            debug=True
        )
        assert "aggregated_vision_score" in aggregated_scores.columns
        assert "aggregated_geo_score" in aggregated_scores.columns
        assert "aggregated_geo_threshold" in aggregated_scores.columns
        assert "aggregated_combined_score" in aggregated_scores.columns
        assert "normalized_aggregated_combined_score" in aggregated_scores.columns

    @pytest.mark.parametrize("taxon", ["Aramus guarauna"], indirect=True)
    def test_h3_04_taxon_range_comparison(self, mocker, inatInferrer, taxon):
        inatInferrer.h3_04_geo_results_for_taxon = MagicMock(return_value={
            "aa": "0.1",
            "ab": "0.1"
        })
        inatInferrer.h3_04_taxon_range = MagicMock(return_value={
            "ab": "0.1",
            "bb": "0.1"
        })
        range_comparison_results = inatInferrer.h3_04_taxon_range_comparison(taxon["taxon_id"])
        assert range_comparison_results == {
            "aa": 0,
            "ab": 0.5,
            "bb": 1
        }
