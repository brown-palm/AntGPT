# Environment
```
pip install pandas editdistance rapidfuzz openai
```

# Scripts

+ `preprocessing.py`: building the dictionaries, remove synonyms.
+ `build_finetune_ds.py`: building fine_tune datasets.
+ `recognition_utils.py`: replace the raw data with recognition results. In testing, ground truth annotations are not available.
+ `openai_utils.py`: query the fine-tuned model for responses.
+ `post_processing.py`: process the language model responses into index matrixes.
+ `edit_distance.py`: compute the edit distance with respect to groung truth.

To see a demonstration of overall workflow: see ``workflow_illustration.ipynb``.