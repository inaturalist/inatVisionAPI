import sys
import yaml
import os
sys.path.append("./lib")
from inat_vision_api import InatVisionAPI  # noqa: E402

if "APP_SECRET" in os.environ:
    CONFIG = {
        "app_secret": os.environ["APP_SECRET"],
        "models": [{
            "vision_model_path": os.environ["VISION_MODEL_PATH"],
            "taxonomy_path": os.environ["TAXONOMY_PATH"],
            "tf_geo_model_path": os.environ["TF_GEO_MODEL_PATH"]
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
