import argparse
import json
import time
from pathlib import Path
from feedback_templates import templates
from openai import OpenAI


def call_gpt(prompt: str, client: OpenAI, model_name: str = 'gpt-4-0613') -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=1
    )
    returned = response.choices[0].message.content
    return returned


def get_feedback(prompt: str, values: dict, output_type: str, client: OpenAI, model:str) -> dict:
    final_prompt = prompt.format(**values)
    if output_type == "string":
        attempts = 0
        response = None
        while attempts<2 and not response:
            try:
                response = call_gpt(prompt=final_prompt, model_name=model, client=client)
            except:
                time.sleep(30)
                response = call_gpt(prompt=final_prompt, model_name=model, client=client)
            attempts +=1
        return {
            "feedback":response,
            "feedback_prompt":final_prompt,
            "attempts":attempts
        }
    elif output_type == 'json':
        found = False
        count = 0
        feed = None
        while not found and count < 2:
            try:
                feed = call_gpt(prompt=final_prompt, model_name=model, client=client)
                try:
                    feed_json = json.loads(feed.replace("json","").replace("`","").strip())
                    if len(feed_json) > 0:
                        found = True
                    else:
                        print("no feedback, retry")
                except:
                    print("no json")
            except:
                print('gpt error')
                time.sleep(10)
            count += 1
        if feed is not None:
            print("Found feedback!")
        return {
            "feedback": feed,
            "feedback_prompt": final_prompt,
            "attempts":count}


def get_gpt4_feedback(prompt_id: int, input_file: Path, output_file: Path, api_key: str, model:str) -> None:
    template = templates[prompt_id]
    prompt_output_type = templates[prompt_id]['output_type']
    prompt = template['prompt']

    client = OpenAI(api_key=api_key)

    input_data = open(input_file).readlines()
    with open(output_file,"w") as f_output:
        for inp in input_data:
            inp_json = json.loads(inp)
            print(inp_json.keys())
            assert inp_json['summary_factual'] == False
            if prompt_id == "tofueval":
                document = inp_json['source_doc']
                response = inp_json['completions'][0] if 'completions' in inp_json else inp_json['response']
                aspect = inp_json['topic']
            elif prompt_id == "ultrachat":
                document = inp_json['instruction']
                aspect = ""
                response = inp_json['completions'][0]
            # document = inp_json.get('input',inp_json.get("document",""))
            values = {
                "document": document,
                "aspect": aspect,
                "summary": response
            }
            feedback = get_feedback(prompt=prompt, values=values, output_type=prompt_output_type,
                                                  client=client, model=model)
            inp_json['feedback'] = feedback['feedback']
            inp_json['feedback_prompt'] = feedback['feedback_prompt']
            inp_json['attempts'] = feedback['attempts']
            inp_json['prompt_id'] = prompt_id
            f_output.write(json.dumps(inp_json))
            f_output.write("\n")

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_id",
                        type=str,
                        help="Prompt string identifier from feedback_templates.py indicating which prompt to use")

    parser.add_argument("--input", type=Path, help="Path to input jsonl file")
    parser.add_argument("--output", type=Path, help="Path to output jsonl file")
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--model", type=str)

    args = parser.parse_args()
    print(args)

    get_gpt4_feedback(
        prompt_id=args.prompt_id,
        input_file=args.input,
        output_file=args.output,
        api_key=args.api_key,
        model=args.model
    )

if __name__=="__main__":
    main()


