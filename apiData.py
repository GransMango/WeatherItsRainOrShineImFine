import os

import requests
import json
import re

def splitTime(data, month, day, hour):
    for i in range(len(data['time'])):
        split_time = re.split(r'[-T:]+', data['time'][i])
        month.append(int(split_time[1]))
        day.append(int(split_time[2]))
        hour.append(int(split_time[3]))

def checkForNull(data):
    for i in range(len(data['hour'])):
        for key in data.keys():
            if data[key][i] == 'null':
                for key in data.keys():
                    data[key].remove(data[key][i])
                    print('test')

# query = 'Oslo+Norge' # Fetch from user input website

def main(query="Oslo+Norge"):
    # API request, retry 3 times if Timeout
    json_dir = os.getcwd() + '/Data/' + query + '.json'
    for i in range(3):
        try:
            location_request = requests.get('https://nominatim.openstreetmap.org/?addressdetails=1&q=' + query + '&format=json&limit=1')
        except location_request.exceptions.Timeout:
            print("Request timeout")
            continue
        except location_request.exceptions.TooManyRedirects:
            print("Bad url")
        except location_request.exceptions.RequestException as e:
            print("Bad error")
            raise SystemExit(e)
        break

    if location_request.status_code != 200:
        print("Status code: ", location_request.status_code)


    location_json = location_request.json()[0]
    latitude = str(location_json['lat'])
    longitude = str(location_json['lon'])

    # Retry 3 times if timeout
    for i in range(3):
        try:
            data_request = requests.get('https://archive-api.open-meteo.com/v1/archive?latitude=' + latitude + '&longitude=' + longitude + '&start_date=1991-01-01&end_date=2022-12-14&hourly=temperature_2m,relativehumidity_2m,pressure_msl,precipitation,windspeed_10m,winddirection_10m')
        except data_request.exceptions.Timeout:
            print("Request timeout")
            continue
        except data_request.exceptions.TooManyRedirects:
            print("Bad url")
        except data_request.exceptions.RequestException as e:
            print("Bad error")
            raise SystemExit(e)
        break

    if data_request.status_code != 200:
        print("Status code: ", data_request.status_code)

    file_json = data_request.json()

    # Loading in weather data from json file
    data = file_json['hourly']

    # creating lists to split time from json file into
    month = []
    day = []
    hour = []

    # Splitting time
    splitTime(data, month, day, hour)

    # Adding month day and hour keys into json
    data['month'] = month
    data['day'] = day
    data['hour'] = hour

    # Deleting time
    del data['time']

    # Checking json for null
    checkForNull(data)

    with open(json_dir, 'w+') as file:
        json.dump(data, file, indent=4)

# if __name__ == "__main__":
#     main()
