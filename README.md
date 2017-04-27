# computervision

We're doing some computer vision stuff at iNat.

#### ubuntu dependencies

- `apt-get update && apt-get install -y python-virtualenv`

#### python

- `virtualenv inatvision-venv`
- `source inatvision-venv/bin/activate`
- `pip install —upgrade pip`
- `pip install -r requirements.txt`

#### install notes

If the device you're installing on has AVX extensions (check flags in /proc/cpuinfo), try compiling tensorflow for better performance:
https://www.tensorflow.org/install/install_sources
This is a good idea on AWS or bare metal, but won't make a difference on Rackspace due to them using an old hypervisor.
If you're not compiling, install tensorflow from pip: `pip install tensorflow`

If the device you're installing on has AVX2 or SSE4, install pillow-simd for faster image resizing:
`pip install pillow-simd` if you only have SSE4, or `CC="cc -mavx2" pip install pillow-simd` if you have AVX2. I saw a significant increase in performance from pillow to pillow-simd with SSE4, less of an increase for AVX2.
otherwise, install pillow from pip: `pip install pillow`

Install other requirements:
`pip install flask flask_wtf scipy numpy`

Copy the optimized model into place: `cp /tmp/optimized_model-3.pb tf-session-reuse/`


Run the app:
`python app.py`


Some performance data from my 15" MBP, 2.5GHz i7:

| task               | pip tensorflow | compiled tensorflow | compiled tensorflow + pillow-simd |
| ------------------ | -------------- | ------------------- | --------------------------------- |
| 100x medium.jpg    | 25 seconds     | 17 seconds          | 15 seconds                        |
| 100x iphone photos | 81 seconds     | 72 seconds          | 46 seconds                        | 

The larger the images coming into the pipeline, the more important optimized resize (like pillow-simd) is.
