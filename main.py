import argparse
parser = argparse.ArgumentParser(description='Deep learning weather program')

def CommandArgs():
    parser.add_argument('-out', help='selects output for files (optional)')
    parser.add_argument('-in', help='selects directory containing model (optional)')

    parser.add_argument('-fetch', help='fetch api data', required=False)
    parser.add_argument('--query', help='user location, format: city, country', default='Oslo, Norge', required=False)
    parser.add_argument('-build', help='build model if you dont have yet', required=False)
    parser.add_argument('-predict', help='runs model', required=False)
    args = parser.parse_args()
    return args

def Main():
    args = CommandArgs()
    if args.fetch is not None:
        apiData()

#api

#dataprocessing
#buildmodel

#model

if __name__ == "__main__":