lr_scheduler_type =  "linear" #"constant"#"linear"#"constant"

wandb_dir = ""
max_seq_length = 2048
collate_data = True

new_model = "llama-2-7b-error_detection"
model_cache_dir = ""
hf_token = ""

# quantization config
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# lora config

lora_dropout = 0.1

# training config

per_device_train_batch_size = 2 # can change this
per_device_eval_batch_size = 2 # can change this
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
weight_decay = 0.001
optim = "paged_adamw_32bit" # TODO explore this and maybe change to normal optimizer

max_steps = -1
warmup_ratio = 0.05
group_by_length = True
save_steps = 100
eval_steps = 100
logging_steps = 100
packing = False
fp16 = False
bf16 = False
# change this to True for A100
# this is set to be true so compute metrics in the middle can use the generated text for evaluation instead of the model sending logits and albels
generate_with_predict = True

variable_dict = {key: value for key, value in locals().items() if key != 'variable_dict' and key[:2]!="__"}