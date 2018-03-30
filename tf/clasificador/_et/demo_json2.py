import json

data = dict()
aux = []

data['image'] = 'xxxx'
data['peso1'] = 0.998
data['peso2'] = 0.002

aux.append(data)

data['image'] = 'yyyyy'
data['peso1'] = 0.9
data['peso2'] = 0.1

aux.append(data)

with open('data.json', 'w') as outfile:
    json.dump(aux, outfile, ensure_ascii=False)
