<div align="center">

# AntGPT: Can Large Language Models Help Long-term Action Anticipation from Videos?

[[Website]](https://brown-palm.github.io/AntGPT/)
[[Arxiv Paper]](https://arxiv.org/abs/2307.16368)
[[PDF]](https://arxiv.org/pdf/2307.16368.pdf)

![](assets/main.gif)
</div>

Can we better anticipate an actor’s future actions (e.g. mix eggs) by knowing what commonly happens after his/her current action (e.g. crack eggs)? What if we also know the longer-term goal of the actor (e.g. making egg fried rice)? The long-term action anticipation (LTA) task aims to predict an actor’s future behavior from video observations in the form of verb and noun sequences, and it is crucial for human-machine interaction. We propose to formulate the LTA task from two perspectives: a bottom-up approach that predicts the next actions autoregressively by modeling temporal dynamics; and a top-down approach that infers the goal of the actor and “plans” the needed procedure to accomplish the goal. We hypothesize that large language models (LLMs), which have been pretrained on procedure text data (e.g. recipes, how-tos), have the potential to help LTA from both perspectives. It can help provide the prior knowledge on the possible next actions, and infer the goal given the observed part of a procedure, respectively. To leverage the LLMs, we propose a two-stage framework, AntGPT. It first recognizes the actions already performed in the observed videos and then asks an LLM to predict the future actions via conditioned generation, or to infer the goal and plan the whole procedure by chain-of-thought prompting. Empirical results on the Ego4D LTA v1 and v2 benchmarks, EPIC-Kitchens-55, as well as EGTEA GAZE+ demonstrate the effectiveness of our proposed approach. AntGPT achieves state-of-the-art performance on all above benchmarks, and can successfully infer the goal and thus perform goal-conditioned “counterfactual” prediction via qualitative analysis.

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

# Prepare Data 

## Datasets

Coming Soon!

## Features

Coming Soon!

# Getting Started

## Transformer Experiments

## Language Models Experiments

Coming Soon!

# Our Paper 

Our paper is available on [Arxiv](https://arxiv.org/abs/2307.16368). If you find our work or code useful, please consider citing us. 
```bibtex
@article{zhao2023antgpt,
  title     = {AntGPT: Can Large Language Models Help Long-term Action Anticipation from Videos?},
  author    = {Qi Zhao and Ce Zhang and Shijie Wang and Changcheng Fu and Nakul Agarwal and Kwonjoon Lee and Chen Sun},
  journal={arXiv preprint arXiv:2307.16368},
  year      = {2023}
}
```
# Lincense

This project is released under the [TBD] license.
