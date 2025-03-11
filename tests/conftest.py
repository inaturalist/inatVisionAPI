import pytest
import os
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
def inatInferrer(request, mocker):
    config = {
        "vision_model_path": "vision_model_path.h5",
        "tf_geo_elevation_model_path": "tf_geo_elevation_model_path.h5",
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
    mocker.patch("tensorflow.keras.Model", return_value=MagicMock())
    return InatInferrer(config)
