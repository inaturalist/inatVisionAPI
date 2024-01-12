import sys
import yaml
import os
sys.path.append("./lib")
from inat_vision_api import InatVisionAPI  # noqa: E402

if "APP_SECRET" in os.environ:
    CONFIG = {
        "app_secret": os.environ["APP_SECRET"],
        "models": [{
            "name": os.environ["MODEL_NAME"],
            "vision_model_path": os.environ["VISION_MODEL_PATH"],
            "taxonomy_path": os.environ["TAXONOMY_PATH"],
            "tf_geo_elevation_model_path": os.environ["TF_GEO_MODEL_PATH"],
            "elevation_h3_r4": os.environ["ELEVATION_H3_R4_PATH"],
            "tf_elev_thresholds": os.environ["GEO_THRESHOLDS_PATH"],
            "taxon_ranges_path": os.environ["TAXON_RANGES_PATH"]
        }]
    }
    if "GEO_MIN" in os.environ:
        CONFIG["models"][0]["geo_min"] = os.environ["GEO_MIN"]
else:
    CONFIG = yaml.safe_load(open("config.yml"))

api = InatVisionAPI(CONFIG)
app = api.app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6006)
