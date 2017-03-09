from grpc.beta import implementations
import predict_pb2
import prediction_service_pb2
import tensor_pb2
import tensor_shape_pb2
import tensorflow as tf
from tensorflow.python.util import compat
import numpy as np
import requests
import os

class InatClassifier:

    def __init__(self, taxa_file, tfserving_host, tfserving_port):
        self.taxa_file = taxa_file
        self.tfserving_host = tfserving_host
        self.tfserving_port = tfserving_port

        # the classifier was trained on the label ordinals
        # use the taxa labels to look up the actual label (taxon id)
        self.taxa = self._load_taxa_labels()

        # setup grcp channel
        self.channel = implementations.insecure_channel(
            self.tfserving_host,
            self.tfserving_port
        )

        # setup grpc prediction stub for tfserving
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)

    # the classifier was trained on the label ordinals. we
    # need to use the taxa labels file to look up the actual label,
    # which in this case is the taxonid
    def _load_taxa_labels(self):
        taxa = {}
        with open(self.taxa_file) as f:
            for line in f:
                iter, taxon = line.rstrip().split(": ")
                taxa[int(iter)] = int(taxon)
        return taxa

    # we're passing a single image to tfserving, so this
    # is the shape of the data. basically: { dim { size: 1 } }
    def _predict_shape(self):
        return tensor_shape_pb2.TensorShapeProto(dim=[
            tensor_shape_pb2.TensorShapeProto.Dim(
                size=1
            )
        ])

    # tensor.proto takes the image as a string
    # this format was a little fiddly, mostly stole this
    # helper method straight from tfserving's inception_client.py
    def _flatten_to_string(self, nested_strings):
        if isinstance(nested_strings, (list, tuple)):
            for inner in nested_strings:
                for flattened_string in flatten_to_string(inner):
                    yield flattened_string
        else:
            yield nested_strings

    # tensor.proto takes the image as a string
    # this format was a little fiddly, mostly stole this
    # and _flatten_to_string() straight from tfserving's
    # inception_client.py
    def _image_as_string(self, image_data):
        data_as_string = np.array(image_data, None).tostring()
        proto_values = self._flatten_to_string(data_as_string)
        return [compat.as_bytes(x) for x in proto_values]

    # construct a tensor.proto containing our image data
    def _predict_tensor_proto(self, image_data):
        # dtype 7 is String
        tensor_proto = tensor_pb2.TensorProto(
            dtype=7,
            tensor_shape=self._predict_shape()
        )
        tensor_proto.string_val.extend(self._image_as_string(image_data))

        return tensor_proto

    # construct a prediction request for tfserving
    # the prediction request contains the tensor.proto with the
    # image data embedded, courtesy of _predict_tensor_proto()
    def _predict_request(self, image_data):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'inception'
        request.model_spec.signature_name = 'predict_images'
        request.inputs['images'].CopyFrom(self._predict_tensor_proto(image_data))
        return request

    # public method: classify image data.
    # image_data should be binary data, like from:
    # with open('image_file.jpg', 'rb') as f:
    #       image_data = f.read()
    #       classify_data(image_data)
    def classify_data(self, image_data):
        # make a prediction request
        request = self._predict_request(image_data)
        # send it to the server, with a 60 second timeout
        result = self.stub.Predict(request, 60.0)

        scores = result.outputs['scores'].float_val
        classes = [self.taxa[i] for i in result.outputs['classes'].int_val]

        # return all results
        """
        results = dict(zip(classes, scores))
        return results
        """

        # return top n results
        results = {}
        for i in range(10):
            results[classes[i]] = scores[i]
        return results

    # classify an iNaturalist observation by the observation id
    # this only downloads the first observation photo. it will fail
    # disastrously if there is no photo on the observation in question.
    def classify_observation(self, observation_id):
        print("*** fetching observation")
        obs_url = 'https://api.inaturalist.org/v1/observations/{}'.format(observation_id)

        # should be one and only one result in this request
        obs = requests.get(obs_url).json()['results'][0]

        # only use the first photo, assume the photo exists
        first_photo = obs['observation_photos'][0]

        # download and classify the medium res photo
        url = first_photo['photo']['url'].replace('square', 'medium')
        print("*** downloading first photo of observation")
        photo = requests.get(url)

        # run the classifier on the bytes in the get request
        if photo.status_code == 200 and len(photo.content) > 0:
            return self.classify_data(photo.content)
        else:
            print("*** error downloading {}: status code {}".format(
                url, photo.status_code
            ))
            return {}
