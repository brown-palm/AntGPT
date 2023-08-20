import json
import pickle


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(dic, file_path, indent=4):
    with open(file_path, 'w') as fp:
        json.dump(dic, fp, indent=indent)
 
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, file_path):
    with open(file_path, 'wb') as fp:
        pickle.dump(data, fp)

