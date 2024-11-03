import json 

with open('after.json', 'r') as file:
    data = json.loads(file)
# js = json.load('after.json')
print(data)