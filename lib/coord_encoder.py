import math

import numpy as np
import tensorflow as tf


class CoordEncoder:
    def __init__(self, encoding_strategy, raster=None):
        assert encoding_strategy in [
            "sinusoidal",
        ], "unsupported encoding strategy"

        self.encoding_strategy = encoding_strategy
        
        self.raster = raster

    def encode(self, locs, normalize=True):
        if normalize:
            locs = CoordEncoder.normalize_coords(locs)
        
        if self.encoding_strategy == "sinusoidal":
            loc_feats = CoordEncoder.encode_loc_sinusoidal(locs)
        else:
            assert False, "unsupported encoding strategy"
        
        if self.raster is not None:
            context_feats = CoordEncoder.bilinear_interpolate(locs, self.raster)
            loc_feats = np.concatenate((loc_feats, context_feats), 1)

        return loc_feats

    def num_input_feats(self):
        if self.encoding_strategy == "sinusoidal":
            coord_feats = 4
        else:
            assert False, "unsupported encoding strategy"

        if self.raster is not None:
            return coord_feats + self.raster.shape[-1]
        else:
            return coord_feats

    @staticmethod
    def normalize_coords(locs):
        return tf.stack([
            locs[:, 0] / 180.0,
            locs[:, 1] / 90.0,
        ], axis=1)

    @staticmethod
    def encode_loc_sinusoidal(loc_ip, concat_dim=1):
        return tf.concat([
            tf.sin(loc_ip * math.pi),
            tf.cos(loc_ip * math.pi),
        ], axis=concat_dim)

   
    @staticmethod
    def bilinear_interpolate(loc_ip, data, remove_nans_raster=True):
        """
        Perform bilinear interpolation on a raster using normalized 
        [-1, 1] lng, lat input.
        
        Args:
            loc_ip: [N x 2] tensor/array of [lng, lat] in [-1, 1] space
            data: [H x W x C] raster data
            remove_nans_raster: whether to replace NaNs in `data` with 0.0
        
        Returns:
            np.ndarray: [N x C] interpolated feats for each location
        """
        assert data is not None
        assert loc_ip.shape[1] == 2
        
        if remove_nans_raster:
            data = np.nan_to_num(data, nan=0.0)
        
        # normalize from [-1, 1] to [0, 1]
        loc = (loc_ip + 1.0) / 2.0

        # flip y-axis for raster top down layout
        x = loc[:, 0]
        y = 1.0 - loc[:, 1]
    
        # convert to pixel indices
        px = x * (data.shape[1] - 1)
        py = y * (data.shape[0] - 1)
    
        # corner integer indices
        x0 = tf.floor(px).numpy().astype(int)
        y0 = tf.floor(py).numpy().astype(int)
        x1 = np.clip(x0 + 1, 0, data.shape[1] - 1)
        y1 = np.clip(y0 + 1, 0, data.shape[0] - 1)

        # deltas for interpolation
        dx = np.expand_dims(px - x0, axis=1)
        dy = np.expand_dims(py - y0, axis=1)

        # fetch corner values
        top_left             = data[y0, x0, :]
        top_right            = data[y0, x1, :]
        bottom_left          = data[y1, x0, :]
        bottom_right         = data[y1, x1, :]
        
        # bilinear interpolation
        interp_value = (
            top_left * (1 - dx) * (1 - dy) +
            top_right * dx * (1 - dy) + 
            bottom_left * (1 - dx) * dy +
            bottom_right * dx * dy
        )

        return interp_value






