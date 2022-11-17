import tensorflow as tf
import numpy as np
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ResLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ResLayer, self).__init__()
        self.w1 = tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_initializer="he_normal"
        )
        self.w2 = tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_initializer="he_normal"
        )
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        x = self.w1(inputs)
        x = self.dropout(x)
        x = self.w2(x)
        x = self.add([x, inputs])
        return x

    def get_config(self):
        return {}


class TFGeoPriorModel:

    def __init__(self, model_path, taxonomy):
        self.taxonomy = taxonomy
        # initialize the geo model for inference
        self.gpmodel = tf.keras.models.load_model(
            model_path,
            custom_objects={'ResLayer': ResLayer},
            compile=False
        )

    def predict(self, latitude, longitude, filter_taxon_id=None):
        filter_taxon = None
        if filter_taxon_id is not None:
            try:
                filter_taxon = self.taxonomy.taxa[filter_taxon_id]
            except Exception as e:
                print(f'filter_taxon `{filter_taxon_id}` does not exist in the taxonomy')
                raise e

        norm_lat = np.array([float(latitude)]) / 90.0
        norm_lng = np.array([float(longitude)]) / 180.0
        norm_loc = tf.stack([norm_lng, norm_lat], axis=1)
        encoded_loc = tf.concat([
            tf.sin(norm_loc * math.pi),
            tf.cos(norm_loc * math.pi)
        ], axis=1)

        preds = self.gpmodel.predict([encoded_loc], verbose=0)[0]
        geo_pred_dict = {}
        for index, pred in enumerate(preds):
            if index not in self.taxonomy.leaf_class_to_taxon:
                continue
            taxon_id = self.taxonomy.leaf_class_to_taxon[index]
            if filter_taxon_id is not None:
                taxon = self.taxonomy.taxa[taxon_id]
                # the predicted taxon is not the filter_taxon or a descendent, so skip it
                if not taxon.is_or_descendant_of(filter_taxon):
                    continue

            geo_pred_dict[taxon_id] = pred

        return geo_pred_dict
