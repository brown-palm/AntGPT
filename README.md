<div align="center">

# AntGPT: Can Large Language Models Help Long-term Action Anticipation from Videos?

ICLR2024

[[Website]](https://brown-palm.github.io/AntGPT/)
[[Arxiv]](https://arxiv.org/abs/2307.16368)
[[PDF]](https://arxiv.org/pdf/2307.16368.pdf)

![](assets/main.gif)
</div>

Can we better anticipate an actor’s future actions (e.g. mix eggs) by knowing what commonly happens after his/her current action (e.g. crack eggs)? What if we also know the longer-term goal of the actor (e.g. making egg fried rice)? We hypothesize that large language models (LLMs), which have been pretrained on procedure text data (e.g. recipes, how-tos), have the potential to help LTA from both perspectives. It can help provide the prior knowledge on the possible next actions, and infer the goal given the observed part of a procedure, respectively. 

AntGPT is the proposed framework in our [paper](https://arxiv.org/abs/2307.16368) to leverage LLMs in video-based long-term action anticipation. AntGPT achieves state-of-the-art performance on the Ego4D LTA v1 and v2 benchmarks, EPIC-Kitchens-55, as well as EGTEA GAZE+ by the time of publication.

# Contents
- [Setup Environment](#Setup-Environment)
- [Prepare Data](#Prepare-Data)
  - [Datasets](#Datasets)
  - [Features](#Features)
  - [Data Folder Structure](#Data-Folder-Structure)
- [Running Experiments](#Running-Experiments)
  - [Download Outputs and Checkpoints](#Download-Outputs-and-Checkpoints)
  - [Evalutation on Ego4D LTA](#Evalutation-on-Ego4D-LTA)
  - [Inference on Ego4D LTA](#Inference-on-Ego4D-LTA)
  - [Transformer Experiments](#Transformer-Experiments)
  - [GPT Experiments](#GPT-Experiments)
  - [Llama2 Experiments](#Llama2-Experiments)
- [Our Paper](#Our-Paper)
- [License](#License)

# Setup Environment

Clone this repository.

```bash
git clone git@github.com:brown-palm/AntGPT.git
cd AntGPT
```

Set up python (3.9) virtual environment. Install pytorch with the right CUDA version. 

```bash
python3 -m venv venv/forecasting
source venv/forecasting/bin/activate
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Install CLIP.
```bash
pip install git+https://github.com/openai/CLIP.git
```

Install other packages.

```bash
pip install -r requirements.txt 
```

Install llama-recipe packages following instructions [here](https://github.com/facebookresearch/llama-recipes).

# Prepare Data 

In our experiments, we used data from Ego4D, Epic-Kitchens-55, and EGTEA GAZE+. For Epic-Kitchens-55 and EGTEA GAZE+, we also used the data annotation and splits of EGO-TOPO. First start a data folder in the root directory.
```bash
mkdir data
```

### Datasets

Download Ego4D dataset, annotations and pretrained models from [here](https://github.com/EGO4D/forecasting). <br>
Download Epic-Kitchens 55 [dataset](https://github.com/epic-kitchens/epic-kitchens-download-scripts) and [annotations](https://github.com/epic-kitchens/epic-kitchens-55-annotations). <br>
Download EGTEA GAZE+ dataset from [here](https://cbs.ic.gatech.edu/fpv/). <br>
Download data annotations from [EGO-TOPO](https://github.com/facebookresearch/ego-topo/tree/main). Please refer to their instructions. 

### Preprocessed Files 
You can find our preprocessed file including text prompts, goal features, etc [here](https://drive.google.com/drive/folders/1dPxJyAVBmd5k9i5fYnSoDFRGKY_wsRwN). <br>
Downloaded and unzip both folders. <br>
Place the `goal_features` under `data` folder. <br>
Place the `dataset` folder under `Llama2_models` folder. <br>
Make a symlink in the `ICL` subfolder of the `Llama2_models` folder.
```bash
ln -s {path_to_dataset} AntGPT/Llama2_models/ICL
```

### Features
We used [CLIP](https://github.com/openai/CLIP) to extract features from these datasets.
You can use the feature extraction file under transformer_models to extract the features.
```bash
python -m transformer_models.generate_clip_img_embedding
```

### Data Folder Structure
We have a data folder structure like illustrated below. Feel free to use your own setup and remember to adjust the path configs accordingly.
```
data
├── ego4d 
│   └── annotations
|   │   ├── fho_lta_taxonomy.json
|   │   ├── fho_test_unannotated.json
│   │   ├── ...
│   │
│   └── clips
│       ├── 0a7a74bf-1564-41dc-a516-f5f1fa7f75d1.mp4
│       ├── 0a975e6e-4b13-426d-be5f-0ef99b123358.mp4
│       ├── ...
│
├── ek 
│   └── annotations
|   │   ├── EPIC_many_shot_verbs.csv
│   │   ├── ...
│   │
│   └── clips
│       ├── rgb
│       ├── obj
│       └── flow
│
├── gaze 
│   └── annotations
|   │   ├── action_list_t+v.csv
│   │   ├── ...
│   │
│   └── clips
│       ├── OP01-R01-PastaSalad.mp4
│       ├── ...
│
├── goal_features
│    ├── ego4d_feature_gt_val.pkl 
│    ├── ...
│
├── output_CLIP_img_embedding_ego4d
│
...
```

# Running Experiments
Our codebase consists of three parts: the transformer experiments, the GPT experiments, and the Llama2 experiments. Implementation of each modules are located in the `transformer_models` folder, `GPT_models`, and `Llama2_models` folder respectively.

### Download Outputs and Checkpoints
You can find our model checkpoints and output files for Ego4D LTA [here](https://drive.google.com/drive/folders/1dPxJyAVBmd5k9i5fYnSoDFRGKY_wsRwN).
Unzip both folders.
Place the `ckpt` folder under the `llama_recipe` subfolder of the `Llama2_models` folder.
Place the `ego4d_outputs` folder under the `llama_recipe` subfolder of the `Llama2_models` folder.

### Evalutation on Ego4D LTA
Submit the output files to [leaderboard](https://eval.ai/web/challenges/challenge-page/1598/leaderboard).

### Inference on Ego4D LTA
```bash
cd Llama2_models/Finetune
```
```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_lta.py --model_name {your llama checkpoint path} --peft_model {pretrained model path} --prompt_file ../dataset/test_nseg8_recog_egovlp.jsonl --response_path {output file path}
```

### Transformer Experiments
To run an experiment on the transformer models, please use the following command

```bash
python -m transformer_models.run --cfg transformer_models/configs/ego4d_image_pred_in8.yaml --exp_name ego4d_lta/clip_feature_in8
```

### GPT Experiments

To run a GPT experiment, please use one of the workflow illustration [notebooks](llm_models/Finetuning/workflow_illustration.ipynb).


### Llama2 Experiments

To run a Llama2 experiment, please refer to the instructions in that folder.

# Our Paper 

Our paper is available on [Arxiv](https://arxiv.org/abs/2307.16368). If you find our work  useful, please consider citing us. 
```bibtex
@article{zhao2023antgpt,
  title   = {AntGPT: Can Large Language Models Help Long-term Action Anticipation from Videos?},
  author  = {Qi Zhao and Shijie Wang and Ce Zhang and Changcheng Fu and Minh Quan Do and Nakul Agarwal and Kwonjoon Lee and Chen Sun},
  journal = {ICLR},
  year    = {2024}
}
```

# License

This project is released under the MIT license.
