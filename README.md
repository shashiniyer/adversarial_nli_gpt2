# Adversarially Robust Natural Language Inference using GPT-2

Code repo corresponding to my Text Mining project - Adversarially Robust Natural Language Inference using GPT-2

## Abstract
Resourced primarily by private sector concerns, large language models (LLMs) have been at the de facto centre of contemporary discourse on Natural Language Understanding for several years now. A number of these models are being deployed in real-life textual applications despite several reasonable questions as to their suitability for these applications, applications in which humans perform well naturally. At the heart of this issue is the fact that large models are ostensibly 'overfitted' to training data. Such models perform well on data that very closely resemble training data but poorly on data that for models may be 'adversarial' but none too different from training data for humans. In this project, I aim to make the LLM GPT-2 adversarially robust in the Natural Language Inference task by attempting to remove spurious artefacts (biases) in the Stanford Natural Language Inference (SNLI) dataset, through an algorithm called AFLite (which stands for Lightweight Adversarial Filtering). Because of resource constraints (compute and time), however, I do not succeed entirely. I do, however, make publicly available a hitherto unavailable repository of code that can further the objective, and point to some recent research for additional directions through which to progress towards adversarially robust NLI.

## Structure of the Code Repo
The main files for implementing the experiments reported on are organised in two folders - `gpt2-small` and `gpt2-medium`. Most of these files are interactive notebooks in which comments in natural language are made alongside code segments to explain what the respective code segments achieve. A folder called `raw_data` consists data coresponding to Stress Tests ([Naik et al. 2018](https://aclanthology.org/C18-1198/)), which were the only data that could not be downloaded from [HuggingFace](https://huggingface.co/) :hugs:. Other files in the repo include files containing codes pertaining to utility functions used by the interactive notebooks mentioned above and some auxiliary files such as a copy of the project report and experiment results in Excel format.
