import torch
import numpy as np
import sys


class PTGeoPriorModel:

    def __init__(self, model_path, taxonomy):
        sys.path.append("../geo_prior_inat")
        from geo_prior import models
        self.taxonomy = taxonomy
        # initialize the geo model for inference
        net_params = torch.load(model_path, map_location="cpu")
        self.params = net_params["params"]
        self.params["device"] = "cpu"
        model_name = models.select_model(self.params["model"])
        self.model = model_name(
            num_inputs=self.params["num_feats"],
            num_classes=self.params["num_classes"],
            num_filts=self.params["num_filts"],
            num_users=self.params["num_users"],
            num_context=self.params["num_context"]
        ).to(self.params["device"])
        self.model.load_state_dict(net_params["state_dict"])
        self.model.eval()

    def predict(self, latitude, longitude, filter_taxon_id=None):
        from geo_prior import utils
        filter_taxon = None
        if filter_taxon_id is not None:
            try:
                filter_taxon = self.taxonomy.df.iloc[filter_taxon_id]
            except Exception as e:
                print(f"filter_taxon `{filter_taxon_id}` does not exist in the taxonomy")
                raise e
        location = np.array([longitude, latitude])[np.newaxis, ...]
        # we're not currently using date inference, so set default values for date
        date = np.ones(1) * 0.5
        location, date = utils.convert_loc_and_date(location, location, self.params["device"])
        inference_features = utils.generate_feats(location, date, self.params, None)

        with torch.no_grad():
            geo_pred = self.model(inference_features)[0, :]
        geo_pred = np.float64(geo_pred.cpu().numpy())

        # simpler approach to populating geo prediction mappings if we don't need filtering
        # geo_pred_dict = dict(zip(self.params["class_to_taxa"], geo_pred))

        geo_pred_dict = {}
        for index, pred in enumerate(geo_pred):
            # map the geo model index to a taxon_id
            taxon_id = self.params["class_to_taxa"][index]
            if filter_taxon is not None:
                if taxon_id not in self.taxonomy.taxa:
                    continue
                taxon = self.taxonomy.taxa[taxon_id]
                # the predicted taxon is not the filter_taxon or a descendant, so skip it
                if not taxon.is_or_descendant_of(filter_taxon):
                    continue
            geo_pred_dict[taxon_id] = pred
        return geo_pred_dict
