# Learning to Refine with Fine-Grained Natural Language Feedback

To be able to run feedback and refinements you need to install the following requirements:
1. pip install -r requirements.txt 
2. Setup AlignScore installation [here](https://github.com/yuh-zha/AlignScore)
3. Setup MiniCheck [here](https://github.com/Liyan06/MiniCheck/tree/main)


## Running end to end refinement 
To run end to end refinement with DCR

```
python run_end_to_end.py --type three_step --dataset [dataset] --input_file [file_name.jsonl] --cuda_id [cuda_id_for_minicheck] \ 
--cache_dir [cache_dir] \ --feedback_model [feedback_model] --refinement_model [refinement_model] --output_file [output_file.jsonl]
```

`dataset`: tofueval / ultrachat \\ 
`file_name.json` : input file name, in JSONL format. Each JSON object should have 