import argparse
import apiData
import DataProcessing
import Model
import BuildModel
parser = argparse.ArgumentParser(description='Deep learning weather program')

def CommandArgs():
    parser.add_argument('-out', help='selects output for files (optional)')
    parser.add_argument('-in', help='selects directory containing model (optional)')

    parser.add_argument('-fetch', action='store_true', help='fetch api data', required=False)
    parser.add_argument('-processData', action='store_true', help='fetch api data', required=False)
    parser.add_argument('--query', help='user location provided for api, format: city, country', default='Oslo, Norge', required=False)
    parser.add_argument('-build', action='store_true', help='build model if you dont have yet', required=False)
    parser.add_argument('-predict', action='store_true', help='runs model', required=False)
    args = parser.parse_args()
    return args

def main():
    args = CommandArgs()
    if args.fetch is True:
        if args.query is not None:
            apiData.main(args.query)
        else: apiData.main()
    if args.processData is True:
        DataProcessing.main()
    if args.build is True:
        BuildModel.main()
    if args.predict is True:
        Model.main()



#api

#dataprocessing
#buildmodel

#model

if __name__ == "__main__":
    main()