import json

def remove_word_synonyms(text):
    '''
    This function handles the text from Ego4d taxonomy.
    Due to duplicate texts, We will give up a few classes when map back to indexes.
    '''
    return text.split("_")[0]

def build_dictionary(path):
    '''
    We can always construct our own index_to_word dictionary
    '''
    with open(path) as f:
        d = json.load(f)

    new_d = {}
    for i in d:
        _ = []
        for j in d[i]:
            _.append(remove_word_synonyms(j))
        new_d.update({i:_})
    
    index_to_verb = dict(zip(list(range(len(new_d['verbs']))),new_d['verbs']))
    index_to_noun = dict(zip(list(range(len(new_d['nouns']))),new_d['nouns']))
    
    verb_to_index = dict()
    for k,v in index_to_verb.items():
        if v not in verb_to_index.keys():
            verb_to_index.update({v:k})
            
    noun_to_index = dict()
    for k,v in index_to_noun.items():
        if v not in noun_to_index.keys():
            noun_to_index.update({v:k})
    
    #adding back the synonyms as the output of LLM is not strictly limited to the words in training
    for i in d:
        for j in d[i]:
            if j.find("(") != -1:
                prime = j.split("_")[0]
                syno = j[j.find("(")+1:j.find(")")].replace(",_","_").replace("/","_").split("_")
                if i == 'verbs':
                    value = verb_to_index[prime]
                else:
                    value = noun_to_index[prime]
                for k in syno:
                    if i == 'verbs' and k not in verb_to_index.keys():
                        verb_to_index.update({k:value})
                    elif i == 'nouns' and k not in noun_to_index.keys():
                        noun_to_index.update({k:value})
                        
    #these duplicate words need to manully map back to their indexes   
    noun_to_index.update({"vacuum":447})
    noun_to_index.update({"nail":264})
    noun_to_index.update({"tape":415})
    return index_to_verb, index_to_noun, verb_to_index, noun_to_index