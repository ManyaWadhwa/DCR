# reference code: https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html
# reference code: https://colab.research.google.com/drive/1Zmaceu65d7w4Tcd-cfnZRb6k_Tcv2b8g?usp=sharing#scrollTo=qf1qxbiF-x6p
# reference code: https://huggingface.co/docs/trl/en/sft_trainer
# reference code: https://huggingface.co/blog/4bit-transformers-bitsandbytes
# https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
import wandb
import sys
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"

import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,  # TODO use this
    TrainingArguments,
    pipeline,
    EarlyStoppingCallback
)
import random
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from training_config import *

def main():
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError(
            "need CUDA_VISIBLE_DEVICES to be set before running. If you are running with max seq lenght of 2048 you might need three GPUs")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="Path to input jsonl file")
    parser.add_argument("--val", type=str, help="Path to input jsonl file")
    parser.add_argument("--output", type=str, help="Path to output jsonl file")
    parser.add_argument("--lora_r", default=8)
    parser.add_argument("--lora_alpha", default=8)
    parser.add_argument("--epochs", default=3)
    parser.add_argument("--lr", default = 0.0002)
    parser.add_argument("--model_name",default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--wandb_project")
    parser.add_argument("--task_type")
    parser.add_argument("--random_seed",default=10,type=int)

    use_4bit = False
    args = parser.parse_args()
    train_file = args.train
    val_file = args.val
    base_path = args.output
    lora_r = int(args.lora_r)
    lora_alpha = int(args.lora_alpha)
    num_train_epochs = int(args.epochs)
    wandb_project = args.wandb_project
    task_type = args.task_type
    learning_rate = float(args.lr)
    model_name = args.model_name
    seed = args.random_seed
    random.seed(seed)  # change this if you want to try different seeds!
    print(args)

    # Load datasets
    # config_name = f"overfit_run_seeing_if_it_fully_learns_one_set"
    config_name = f"{task_type}_{wandb_project}_{lora_r}_{lora_alpha}_{learning_rate}_{num_train_epochs}_{seed}"
    print(base_path + "/" + config_name)
    assert not os.path.isdir(base_path + "/" + config_name)

    print(config_name)
    output_dir = base_path + "/" +config_name

    print("starting training..")

    wandb.init(
        project=wandb_project,
        dir=wandb_dir,
        name=config_name
    )
    train_dataset = load_dataset('json', data_files=train_file, split="train")
    train_dataset.shuffle()
    valid_dataset = load_dataset('json', data_files=val_file, split="train")
    # Preprocess datasets
    train_dataset = train_dataset.map(lambda examples: {
        'text': [prompt + response for prompt, response in zip(examples['prompt'], examples['response'])]},
                                      batched=True)
    valid_dataset = valid_dataset.map(lambda examples: {
        'text': [prompt + response for prompt, response in zip(examples['prompt'], examples['response'])]},
                                      batched=True)
    print(valid_dataset)
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir, device_map='auto',use_auth_token=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    if use_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            cache_dir=model_cache_dir,
            quantization_config=bnb_config,
            device_map='auto',
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            cache_dir=model_cache_dir,
            device_map='auto',
            # use_auth_token=True
        )

    model.resize_token_embeddings(len(tokenizer))
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
    )

    if model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
        response_template_with_context = "<|end_header_id|>\n"
        response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[1:]
    else:
        response_template_with_context = "\n\n[/INST]\n"
        response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[3:]
    if collate_data:
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer)

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)

    training_arguments = TrainingArguments(
        report_to='wandb',
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy="steps",
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        eval_steps=eval_steps,
        seed = seed
        # include_inputs_for_metrics = True if generate_with_predict else None
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,  # Pass validation dataset here
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # compute_metrics=compute_metrics,
        data_collator=collator,
        callbacks=[early_stopping_callback]
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir +"/" +new_model)

    print("trained and saved lora config!!")
    print("loading base model again..")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        cache_dir=model_cache_dir,
        device_map='auto'
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, output_dir + "/"+new_model)
    model = model.merge_and_unload()
    print("merged!!!!")
    model.save_pretrained(output_dir + "/" + new_model + "_merged")
    tokenizer.save_pretrained(output_dir + "/"+ new_model + "_merged")
    print("saved merged model!!!")

    config = variable_dict
    output_file = output_dir + "/config.json"
    with open(output_file, "w") as f_:
        f_.write(json.dumps(config))
    print("saved config!")

    # run evaluation and generate data from this model
    print("running on validation data...")

    if model_name=="meta-llama/Meta-Llama-3-8B-Instruct":
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        gen = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=2048, eos_token_id=terminators, model_kwargs={"torch_dtype": torch.bfloat16})
    else:
        gen = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=2048)

    output_file = output_dir + "/validation_completions_from_ft_model.jsonl"

    validation_data = open(val_file).readlines()
    with open(output_file, 'w') as f_:
        for i, data in enumerate(validation_data):
            print(i)
            data = json.loads(data)
            prompt = data['prompt']
            completion_from_llama = gen(prompt)
            data['completion_from_fine_tuned_llama'] = completion_from_llama
            f_.write(json.dumps(data))
            f_.write("\n")
    print("done with the validation step too!")


if __name__ == "__main__":

    main()
