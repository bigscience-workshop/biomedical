import json

with open('C:/Users/franc/Desktop/dataset/data.json') as json_file:
    data = json.load(json_file)


print(data['data'][0])