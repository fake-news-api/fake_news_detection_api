import requests
import numpy as np


BASE = "http://127.0.0.1:4000/"
# BASE = "https://capstone-2021.herokuapp.com/"

# response = requests.patch(BASE + "2", {"txt": "whatthefuck"})
# print(response.json())
BASE = 'http://hoangtranminerva.pythonanywhere.com/'

import pandas as pd
import random

data = pd.read_csv("./test_data/cleaned_train.csv")

print("Test 1: put response")

test_case = 10
for i in range(test_case):
    entry = {}
    entry["text"] = data.text[i]
    if random.random() > 0.5:
        entry["author"] = data.author[i]
    if random.random() > 0.5:
        entry["title"] = data.title[i]

    response = requests.put(BASE + "running", entry)
    print(f"Expected: {['Real', 'Fake'][data.label[i]]}", response.json())
    print("")

# print("Test 2: get response")

# response = requests.get(BASE + "1")
# print(response.json())

# print("Test 3: not existed")

# response = requests.get(BASE + "9")
# print(response.json())

# print("Test 4: delete")

# response = requests.delete(BASE + "0")
# print(response)

