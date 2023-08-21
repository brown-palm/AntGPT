import pickle

def replace_to_recognition(file_path,data)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        f.close()

    val_data_recognition = data.copy()
    for i in val_data_recognition:
        i['hist_verb'] = [index_to_verb[data[j]['verb'].softmax(dim=-1).argmax().item()] for j in i['hist_clip_aid']]
        i['hist_noun'] = [index_to_noun[data[j]['noun'].softmax(dim=-1).argmax().item()] for j in i['hist_clip_aid']]
    
    return val_data_recognition