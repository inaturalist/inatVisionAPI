import pytest
import os
import tensorflow as tf
from unittest.mock import MagicMock
from lib.inat_inferrer import InatInferrer
from lib.model_taxonomy_dataframe import ModelTaxonomyDataframe


@pytest.fixture()
def taxonomy():
    yield ModelTaxonomyDataframe(
        os.path.realpath(os.path.dirname(__file__) + "/fixtures/taxonomy.csv"),
        os.path.realpath(os.path.dirname(__file__) + "/fixtures/thresholds.csv")
    )


@pytest.fixture()
def taxon(request, taxonomy):
    results = taxonomy.df.query(f'name == "{request.param}"')
    yield results.iloc[0]


@pytest.fixture()
def mock_model():
    yield MagicMock(
        return_value=(
            tf.constant([[0.2, 0.3, 0.5]], dtype=tf.float32),
            tf.constant([[0.0]]),
        )
    )


@pytest.fixture()
def inatInferrer(request, mocker, mock_model):
    config = {
        "vision_model_path": "vision_model_path",
        "tf_geo_elevation_model_path": "tf_geo_elevation_model_path",
        "taxonomy_path":
            os.path.realpath(os.path.dirname(__file__) + "/fixtures/taxonomy.csv"),
        "elevation_h3_r4":
            os.path.realpath(os.path.dirname(__file__) + "/fixtures/elevation.csv"),
        "tf_elev_thresholds":
            os.path.realpath(os.path.dirname(__file__) + "/fixtures/thresholds.csv"),
        "taxon_ranges_path":
            os.path.realpath(os.path.dirname(__file__) + "/fixtures/taxon_ranges"),
        "synonyms_path":
            os.path.realpath(os.path.dirname(__file__) + "/fixtures/synonyms.csv")
    }
    mocker.patch("tensorflow.keras.models.load_model", return_value=MagicMock())
    mocker.patch("tensorflow.keras.Model", return_value=mock_model)
    return InatInferrer(config)
