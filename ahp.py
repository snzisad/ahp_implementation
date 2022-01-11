import json
from AHP.AHPModel import parse


with open('benefit.json') as json_model:
    # model can also be a python dictionary
    model = json.load(json_model)

ahp_model = parse(model)
priorities = ahp_model.get_priorities()

print("\nFinal Ranking:")
print(priorities)