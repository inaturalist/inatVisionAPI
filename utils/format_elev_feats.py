import tifffile
import numpy as np

# gather tiff files
files = ["wc2.1_5m_elev.tif"]

ims = []
for ff in files:  # process into numpy array
    im = tifffile.imread(ff)
    im = im.astype(np.float64)

    print(f"max is {np.max(im)}")
    print(f"min is {np.min(im)}")

    # normalize
    im[im > 0] /= np.max(im)
    im[im < 0] /= np.min(im) * -1

    ims.append(im)

# want op to be H W C
ims_op = np.zeros((ims[0].shape[0], ims[0].shape[1], len(ims)), dtype=np.float16)
for ii in range(len(ims)):
    ims_op[:, :, ii] = ims[ii].astype(np.float16)
# save bioclimatic data as numpy array
np.save("elev_scaled", ims_op)
