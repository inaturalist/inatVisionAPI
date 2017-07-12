# computervision

We're doing some computer vision stuff at iNat.

#### ubuntu dependencies

- `apt-get update && apt-get install -y python-virtualenv`

#### os x dependencies
- `brew install libmagic`

#### python

- `virtualenv inatvision-venv`
- `source inatvision-venv/bin/activate`
- `pip install -U pip`
- `pip install -r requirements.txt`

#### installation

Here's a rough script for OS X assuming you already have Python and virtualenv installed.

```bash
# Get dependencies
brew install libmagic

# Get the repo
git clone git@github.com:inaturalist/inatVisionAPI.git
cd inatVisionAPI/

# Set up your python environment
virtualenv inatvision-venv
source inatvision-venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Get your model and taxon ID files and copy them into place. Note the file names are important.
cp /path/to/optimized_model-3.pb .
cp /path/to/taxa.txt .

# Copy your config file (and edit, of course)
cp config.yml.example config.yml

# Run the app
python app.py

```

Now you should be able to test at http://localhost:6006 via the browser.

##### Notes

If the device you're installing on has AVX extensions (check flags in /proc/cpuinfo), try compiling tensorflow for better performance:
https://www.tensorflow.org/install/install_sources
This is a good idea on AWS or bare metal, but won't make a difference on Rackspace due to them using an old hypervisor.
If you're not compiling, install tensorflow from pip: `pip install tensorflow`

If the device you're installing on has AVX2 or SSE4, install pillow-simd for faster image resizing:
`pip install pillow-simd` if you only have SSE4, or `CC="cc -mavx2" pip install pillow-simd` if you have AVX2. I saw a significant increase in performance from pillow to pillow-simd with SSE4, less of an increase for AVX2.
otherwise, install pillow from pip: `pip install pillow`

tensorflow seems to want to compile against your system copy of numpy regardless of the virtualenv, so if you see stupid errors like `ImportError: numpy.core.multiarray failed to import`, try running `deactivate` to get out the virtualenv, then `pip install -U numpy` or somesuch to update your system copy of numpy. Then `source inatvision-venv/bin/activate` to get back in your virtualend and try again.

Some performance data from my 15" MBP, 2.5GHz i7:

| task               | pip tensorflow | compiled tensorflow | compiled tensorflow + pillow-simd |
| ------------------ | -------------- | ------------------- | --------------------------------- |
| 100x medium.jpg    | 25 seconds     | 17 seconds          | 15 seconds                        |
| 100x iphone photos | 81 seconds     | 72 seconds          | 46 seconds                        | 

The larger the images coming into the pipeline, the more important optimized resize (like pillow-simd) is.
