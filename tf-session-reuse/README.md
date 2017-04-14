If the device you're installing on has AVX extensions (check flags in /proc/cpuinfo), try compiling tensorflow for better performance:
https://www.tensorflow.org/install/install_sources
This is a good idea on AWS or bare metal, but won't make a difference on Rackspace due to them using an old hypervisor.
If you're not compiling, install tensorflow from pip: `pip install tensorflow`

If the device you're installing on has AVX2 or SSE4, install pillow-simd for faster image resizing:
`pip install pillow-simd` if you only have SSE4, or `CC="cc -mavx2" pip install pillow-simd` if you have AVX2. I saw a significant increase in performance from pillow to pillow-simd with SSE4, less of an increase for AVX2.
otherwise, install pillow from pip: `pip install pillow`

Install other requirements
`pip install flask flask_wtf scipy numpy`

Run the app:
`python app.py`
