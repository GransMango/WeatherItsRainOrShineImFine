import os

import requests
import json

query = 'Oslo+Norge' # Fetch from user input website

# API request, retry 3 times if Timeout
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

print(location_request.status_code)

location_json = location_request.json()[0]
latitude = str(location_json['lat'])
longitude = str(location_json['lon'])

# Retry 3 times if timeout
for i in range(3):
    try:
        data_request = requests.get('https://archive-api.open-meteo.com/v1/archive?latitude=' + latitude + '&longitude=' + longitude + '&start_date=2021-01-01&end_date=2022-12-14&hourly=temperature_2m,relativehumidity_2m,pressure_msl,precipitation,windspeed_10m,winddirection_10m')
    except data_request.exceptions.Timeout:
        print("Request timeout")
        continue
    except data_request.exceptions.TooManyRedirects:
        print("Bad url")
    except data_request.exceptions.RequestException as e:
        print("Bad error")
        raise SystemExit(e)
    break


print(data_request.status_code)


file_json = data_request.json()
print(file_json.keys())
for keys in file_json:
    print(type(file_json[keys]))

print(type(file_json['hourly']))

with open(os.getcwd() + '\\Data\\' + query + '.json', 'w+') as file:
    json.dump(file_json, file, indent=4)
