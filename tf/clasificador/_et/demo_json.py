import json

data = dict()

data['hola'] = 'mundo'

with open('data.json', 'w') as outfile:
    json.dump(data, outfile, ensure_ascii=False)
