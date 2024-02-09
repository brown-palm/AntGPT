import numpy as np
import time
import openai

def query_ft_model(apikey,df,batch_size=20):
    openai.api_key = apikey
    responses=[]
    i=0
    batch_size = batch_size
    while i < len(df['prompt']): #loop over target dataset
        start = i
        end = i + batch_size
        if end > len(df['prompt']): end = None
        p = df['prompt'].values.tolist()[start:end] #batched prompt
        response = openai.Completion.create(
                                        model='your_finetuned_model_name',
                                        prompt=p,
                                        max_tokens = 100,
                                        stop = " ###",
                                        n = 5 #top-5 responses
                                        )
        responses.append(np.array([j['text'] for j in response['choices']]).reshape(-1,5))
        if i%500 == 0: time.sleep(30) #accomodate openai token per minute limit
        i+=batch_size

    #strip the less important information but keep only the text in the responses
    res = np.concatenate(responses).tolist()
    return res