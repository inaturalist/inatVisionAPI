from client.classifier import InatClassifier
import argparse

def parse_args():

    parser = argparse.ArgumentParser(
        description="Classify observations from iNat"
    )

    parser.add_argument('--obsid', dest='obsid',
        help='observation id to classify (first photo only)',
        type=int, required=False
    )

    parser.add_argument('--file', dest='file',
        help='image file to classify',
        type=str, required=False
    )

    parser.add_argument('--taxafile', dest='taxafile',
        help='taxa file',
        type=str, required=True
    )

    parser.add_argument('--tfserving_host', dest='tfserving_host',
        help='hostname of tensorflow-serving grpc server',
        type=str, required=True
    )

    parser.add_argument('--tfserving_port', dest='tfserving_port',
        help='port of tensorflow-serving grpc server',
        type=int, required=True
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    classifier = InatClassifier(
        args.taxafile,
        args.tfserving_host,
        args.tfserving_port
    )
    if args.obsid:
        print(classifier.classify_observation(args.obsid))
    elif args.file:
        with open(args.file, 'rb') as f:
            data = f.read()
            print(classifier.classify_data(data))
    else:
        print("nothing to do")
