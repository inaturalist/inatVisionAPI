const fs = require('fs');
const grpc = require('grpc');

// read in our grpc service description and associated data protobuf files
const protoDescriptor = grpc.load(__dirname + '/proto/prediction_service.proto');
// service description for tfserving
var tfserving = protoDescriptor.tensorflow.serving;

// our tensorflow stub
const stub = new tfserving.PredictionService(
  'localhost:9000',
  grpc.credentials.createInsecure()
);

// our image to predict
const IMAGE_PATH = '/Users/alex/Downloads/IMG_20170306_203609450.jpg';
const buffer = fs.readFileSync(IMAGE_PATH);

// the prediction service wants an array of images
// but i haven't been able to get ths (or the python client)
// to offer predictions on more than one file, yet. the service
// only returns one prediction, no matter how many files are 
// passed in. may need to enabled --batching on tfserving? or
// something in the grpc service definition?
var buffers;
if (buffer.constructor === Array) {
  buffers = buffer;
} else {
  buffers = [buffer];
}

// build PredictRequest proto message
// using the dynamic, not generated, ptoto descriptors
const msg = {
  model_spec: {
    name: "inception",
    signature_name: "predict_images"
  },
  inputs: {
    images: {
      dtype: 'DT_STRING',
      tensor_shape: {
        dim: {
          size: buffers.length
        }
      },
      string_val: buffers
    }
  }
};

// read in the taxa mappings
var lineReader = require('readline').createInterface({
  input: require('fs').createReadStream('taxa.txt')
});
// i think this is how to you read a file into a
// dictionary in javascript? should probably just convert
// taxa.txt to json and read it in that way.
var taxa = {}
lineReader.on('line', function (line) {
  var arr = line.split(": ");
  taxa[arr[0]] = arr[1];
});
lineReader.on('close', function () {
  // done reading in the taxa mappings, make the prediction request
  stub.predict( msg, function(err, result) {
    if (err) {
      console.log(err)
    } else {
      // log them to the console?
      for (var i = 0; i < 10; i++) {
        klass = result.outputs.classes.int_val[i]
        score = result.outputs.scores.float_val[i]
        console.log(taxa[klass] + ": " + score)
      }
    }
  });
});
