import argparse
import json
import sys
from pathlib import Path

from revise_with_gpt4 import get_minimum_revision

ultrachat_prompt = """{document}

Response:
{summary}

Feedback for the above response: 
{feedback}

Edit the summary such that the refinement doesn't have any errors mentioned in the feedback. Make the minimum number of changes when doing the refinement.
"""

tofueval_prompt = """I summarized the following document on the topic: '{aspect}': 
{document}

Summary of the above document on topic: '{aspect}':
{summary}

Feedback for the above summary: 
{feedback}

Edit the summary such that the refinement doesn't have any errors mentioned in the feedback. Make the minimum number of changes when doing the refinement.
"""


def refine(prompt,
           instruction: str,
           response: str,
           feedback: list,
           few_shot: bool,
           api_key: str,
           feedback_type: str,
           include_inconsistency: bool,
           model: str,
           aspect):
    print(type(feedback))
    if feedback_type == 'str' and not isinstance(feedback, list):
        feedback_str = feedback
    else:
        feedback_str = ""
        for i, feed in enumerate(feedback):
            # There's one instance in the ultrachat dataset where GPT-4 JSON feedback has
            # `'    inconsistency'` instead of `'inconsistency'`.
            feed = {k.strip(): v for k, v in feed.items()}
            if include_inconsistency:
                if 'fix' in feed:
                    if len(feed["fix"]) == 0:
                        fix = "To fix this, consider removing the information from the summary"
                    else:
                        fix = f"To fix this, consider changing the span to '{feed['fix']}'"
                    feedback_str += f"{i + 1}. For sentence/span in the summary: '{feed['inconsistency']}', feedback is: '{feed['feedback']}'. {fix}\n"
                else:
                    print(feed)
                    fix = "To fix this, consider removing the information from the summary"
                    feedback_str += f"{i + 1}. For sentence/span in the summary: '{feed['inconsistency']}', feedback is: '{feed['feedback']}'. {fix}\n"
            else:
                feedback_str += f"{i + 1}. {feed['feedback']}\n"

    feedback_str = feedback_str.strip()
    print(feedback_str)
    print('----')
    prompt = prompt.format(**{"document": instruction, "aspect": aspect, "summary": response, "feedback": feedback_str})
    revised = get_minimum_revision(
        final_prompt=prompt,
        instruction=instruction,
        response=response,
        feedback=feedback_str,
        few_shot=few_shot,
        api_key=api_key,
        model=model
    )
    revised["sample_num"] = int(revised["sample_num"])
    refinement = revised["minimum_edit_response"]
    output = {
        "prompt": prompt,
        "intermediate_refinements": revised,
        "refinement": refinement,
    }
    return output


def run_refinement(
        input_file: Path,
        few_shot: bool,
        api_key: str,
        output_file: Path,
        feedback_type: str,
        include_inconsistency: bool,
        model: str,
        dataset: str
):
    infile  = open(input_file).readlines()
    with open(output_file, "w") as outfile:
        for index, inp in enumerate(infile):
            print(index)
            inp = json.loads(inp)
            if isinstance(inp, str):
                inp = json.loads(inp)
            print(inp.keys())
            aspect = ""
            if "completions" in inp:
                response = inp['completions'][0]
            else:
                response = inp['response']
            if dataset == "ultrachat":
                instruction_columns = [c for c in ["instruction", "input", "document"] if c in inp]
                instruction = inp[instruction_columns[0]]
                # response = inp.get("response", inp["summary"])
                feedback = inp["feedback"]
                prompt = ultrachat_prompt
            elif dataset == "tofueval":
                instruction = inp["source_doc"]
                aspect = inp['topic']
                feedback = inp['feedback']
                prompt = tofueval_prompt
                try:
                    assert response  == " ".join(inp['sent_summary_with_filler_sentences']).strip()
                except:
                    print("skipping this instance!")
                    print(response)
                    feedback = ""
                    inp['feedback'] = feedback
            #feedback = feedback.replace("`", "").replace("json", "")
            if type(feedback) == str:
                try:
                    feedback = feedback.replace("`", "").replace("json", "")
                    feedback = eval(feedback)
                except:
                    print("not a json")
            if isinstance(feedback, dict): # sometimes gpt4 gives one error as a json; so make it a list of errors
                feedback = [feedback]
            if len(feedback) == 0:
                print("summary correct!")
                inp['refinement'] = ""
            else:
                output = refine(
                        prompt=prompt,
                        instruction=instruction,
                        response=response,
                        feedback=feedback,
                        few_shot=few_shot,
                        api_key=api_key,
                        feedback_type=feedback_type,
                        include_inconsistency=include_inconsistency,
                        model=model,
                        aspect=aspect
                )
                    # print(output)
                inp["refinement_prompt"] = output["prompt"]
                inp['refinement'] = output['refinement']
                print(output['refinement'])
            outfile.write(json.dumps(inp))
            outfile.write("\n")
            print("")
            print("********")

def main() -> int:
    print("parsing args")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        help="Path to jsonl file with each row containing the fields ['instruction', 'response', 'feedback']. The 'feedback' column is expected to contain a list of feedback.")
    parser.add_argument("--few_shot", action="store_true", default=False)
    parser.add_argument("--iterative", action="store_true", default=False,
                        help="If True, then iteratively refine using one feedback item at a time.")
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--output", type=Path, help="Path to output jsonl file")
    parser.add_argument("--feedback_type", type=str, help="")
    parser.add_argument("--include_inconsistency", action="store_true", default=False, help="")
    parser.add_argument("--model", default="gpt-4-0613")
    parser.add_argument("--dataset", type=str, help="'tofueval' or 'ultrachat'")
    args = parser.parse_args()
    print(args)

    valid_datasets = ["tofueval", "ultrachat"]
    if args.dataset not in valid_datasets:
        raise ValueError(f"--dataset must be one of {valid_datasets}, but found '{args.dataset}'")

    run_refinement(
        input_file=args.input,
        few_shot=args.few_shot,
        api_key=args.openai_api_key,
        output_file=args.output,
        feedback_type=args.feedback_type,
        include_inconsistency=args.include_inconsistency,
        model=args.model,
        dataset=args.dataset
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
