This folder contains the environment, data, scripts and results files for NeurIPS submission #3880: 
**Can Large Language Models Help Long-term Action Anticipation from Videos?**
In details, this part covers fine-tuning LLM for the quantitative results shown in table 1, 2 and figure 1.

# Environment
```
pip install pandas editdistance rapidfuzz openai
```

# Dataset
+ `fho_lta_taxonomy.json`: the official index2word dictionary provided by openai
+ `train_words_nseg8.jsonl`: sample raw data file sampled from Ego4d annotations
+ `val_words_nseg8.jsonl`: sample raw data file sampled from Ego4d annotations.
+ `train_nseg8.jsonl`: sample fine-tuning dataset with nseg=8, "prompt" correspond to ground-truth history actions, "completion" corresponds to ground-truth prediction.
+ `val_nseg8.jsonl`: sample fine-tuning dataset with nseg=8, "prompt" correspond to ground-truth history actions, "completion" corresponds to ground-truth prediction.

# Scripts

+ `preprocessing.py`: building the dictionaries, remove synonyms.
+ `build_finetune_ds.py`: building fine_tune datasets.
+ `recognition_utils.py`: replace the raw data with recognition results. In testing, ground truth annotations are not available.
+ `openai_utils.py`: query the fine-tuned model for responses.
+ `post_processing.py`: process the language model responses into index matrixes.
+ `edit_distance.py`: compute the edit distance with respect to groung truth.

To see a demonstration of overall workflow: see ``workflow_illustration.ipynb``.

# Existing Responses:
Existing responses are stored in `responses/cg/`:
For FT evaluation:
```
RESPONSE_FILE = [
    responses_nseg8_recognition.json,             
    responses_nseg1_recognition.json,        
    responses_nseg3_recognition.json,        
    responses_nseg5_recognition.json,  
    responses_nseg8_gt.json,                # response file with ground-truth actions as input
]
```
