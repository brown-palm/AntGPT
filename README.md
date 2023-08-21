<div align="center">

# AntGPT: Can Large Language Models Help Long-term Action Anticipation from Videos?

[[Website]](https://brown-palm.github.io/AntGPT/)
[[Arxiv Paper]](https://arxiv.org/abs/2307.16368)
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
- [Getting Started](#Getting-Started)
  - [Transformer Experiments](#Transformer-Experiments)
  - [Language Models Experiments](#Language-Models-Experiments)
- [Our Paper](#Our-Paper)
- [License](#License)

# Setup Environment

If you are using OSCAR (Brown University's cluster): 

```bash
module load python/3.9.0 ffmpeg/4.0.1 gcc/10.2
```

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

Install other packages.

```bash
pip install -r requirements.txt 
```

# Prepare Data 

In our experiments, we used data from Ego4D, Epic-Kitchens-55, and EGTEA GAZE+. For Epic-Kitchens-55 and EGTEA GAZE+, we also used the data annotation and splits of EGO-TOPO. We used [CLIP](https://github.com/openai/CLIP) to extract features from these datasets.

## Datasets

Download Ego4D dataset, annotations and pretrained models from [here](https://github.com/EGO4D/forecasting). <br>
Download Epic-Kitchens 55 [dataset](https://github.com/epic-kitchens/epic-kitchens-download-scripts) and [annotations](https://github.com/epic-kitchens/epic-kitchens-55-annotations). <br>
Download EGTEA GAZE+ dataset from [here](https://cbs.ic.gatech.edu/fpv/). <br>
Download data annotations from [EGO-TOPO](https://github.com/facebookresearch/ego-topo/tree/main). Please refer to their instructions. 
 
## Features

Coming Soon!

## On Brown CCV

Most features are linked or pointed to the feature directory in the codebase already. Ensure the data folder structure is like this.
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
│   └── data_full  # (EK55)
│       ├── rgb
│       ├── obj
│       └── flow
│
├── gaze 
│   └── annotations
|   │   ├── action_list_t+v.csv
│   │   ├── ...
│   │
│   └── data_full
│       ├── rgb
│       ├── obj
│       └── flow
│
└── text_features
    ├── ego4d_feature_gt_val.pkl 
    ├── ...

```

# Getting Started
Our codebase consists of two part the transformer based experiments and the language model based experiments. Implementation of each modules are located in the `transformer_models` folder and `llm_models` folder respectively.

## Transformer Experiments

To run a transformer based experiment, please use the following command from the root directory

```bash
python -m transformer_models.run --cfg transformer_models/configs/ek_clip_feature_best.yaml --exp_name ek_lta/clip_feature
```

## Language Models Experiments

To run a LLM based experiment, please use one of the workflow illustration [notebooks](llm_models/Finetuning/workflow_illustration.ipynb).

# Our Paper 

Our paper is available on [Arxiv](https://arxiv.org/abs/2307.16368). If you find our work or code useful, please consider citing us. 
```bibtex
@article{zhao2023antgpt,
  title   = {AntGPT: Can Large Language Models Help Long-term Action Anticipation from Videos?},
  author  = {Qi Zhao and Ce Zhang and Shijie Wang and Changcheng Fu and Nakul Agarwal and Kwonjoon Lee and Chen Sun},
  journal = {arXiv preprint arXiv:2307.16368},
  year    = {2023}
}
```

# Lincense

This project is released under the [TBD] license.
