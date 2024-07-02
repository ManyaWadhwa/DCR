# Learning to Refine with Fine-Grained Natural Language Feedback

This repo contains code and instructions for reproducing experiments in the paper "Learning to Refine with Fine-Grained Natural Language Feedback". We propose a new method - Detect, Critique and Refine (DCR) for post-hoc editing document grounded summaries and making them more factual.

## Run end to end refinement with DCR 
To run end to end editing with DCR you can run our code with the following command and arguments:

```
from run_end_to_end_refinement.dcr import DCR
document_instruction = '' # source document with the summarization instruction 
initial_response = '' # initial response 
model = "llama3-ft" # critique and refinement model: could be any HF model or GPT-4
dcr = DCR(cuda_id=0, model_name=model, path_to_minicheck="/home/mwadhwa/code/MiniCheck/",cache_dir="/data/users/mwadhwa/")
refinement = dcr.refine(source_text=document_instruction, initial_response=initial_response)
print(refinement)
```

## Models 
Our fine-tuned feedback and refinement models are available on HuggingFace 🤗:
1. Critique Model: [Llama2-7b-Chat Fine-Tuned](https://huggingface.co/wadhma/Critique-L2-FT-DCR) / [Llama3-8b-Instruct Fine-Tuned](https://huggingface.co/wadhma/Critique-L3-FT-DCR)
2. Refinement Model: [Llama2-7b-Chat Fine-Tune](https://huggingface.co/wadhma/Refine-L2-FT-DCR) / [Llama3-8b-Instruct Fine-Tuned](https://huggingface.co/wadhma/Refine-L3-FT-DCR) 

## Data for fine-tuning 
The fine-tuning data distilled from GPT-4 is available on HuggingFace: https://huggingface.co/datasets/wadhma/dcr_data

## Setup
You need to setup the folloiwng:
1. pip install -r requirements.txt
2. Setup MiniCheck [here](https://github.com/Liyan06/MiniCheck/tree/main)

## Evaluation

We use the following metrics for evaluation:
1. AlignScore ([here](https://github.com/yuh-zha/AlignScore))
2. GPT-4 Likert Score on a scale of 1-5 
3. GPT-4 pairwise score 
