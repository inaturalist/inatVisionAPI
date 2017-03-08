# computervision

## tfserving

### tensorflow serving setup
- `mkdir ~/tf_imagerec_client`
- `cd ~/tf_imagerec_client`

#### bazel
- follow instructions at [https://bazel.build/versions/master/docs/install.html]

#### python
- `virtualenv tfserving-venv`
- `source tfserving-venv/bin/activate`
- `pip install —upgrade pip`
- `pip install grpcio`

#### ubuntu dependencies for tfserving
install some required packages:
- `apt-get update && apt-get install -y build-essential curl libcurl3-dev git libfreetype6-dev libpng12-dev libzmq3-dev pkg-config python-dev python-numpy python-pip software-properties-common swig zip zlib1g-dev`

#### install tfserving
- `git clone --recurse-submodules https://github.com/tensorflow/serving`
- `cd serving/tensorflow`
- `./configure`
- `bazel build tensorflow_serving/...`
- get some coffee or lunch. building tensorflow takes a while.

#### download the model exports
- TBD - assume they’re in `/home/inaturalist/model`

#### start tfserving rpc server
- `~~./bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port 9000 --model_name 'inception' --model_base_path '/home/inaturalist/model`

### inat node client
- `cd ~/tf_imagerec_client`
- `git clone git@github.com:inaturalist/computervision.git`
- `cd computervision/cv_tfserving_node`
- `npm install`
- `n use latest client.js`
