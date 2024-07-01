import copy
import os
import gc
import json
import argparse
import re
import sys
from pathlib import Path
from utils import load_model

REPO_ROOT = Path(__file__).absolute().parent.parent
print(f"Adding {REPO_ROOT} to PATH")
sys.path.append(str(REPO_ROOT))

import torch
import transformers
import numpy as np
from nltk.tokenize import sent_tokenize
from fine_tuning.final_prompts import prompts
from run_evaluate_revision import edit_distance_ngram_operations

methods = {
    "single_step": {"detect": None, "feedback": None, "refinement": prompts["single_step"]},
    "two_step": {
        "detect": None,
        "feedback": prompts["feedback_with_correct"],
        "refinement": prompts["refinement_without_correct"],
    },
    "two_step_with_minicheck": {
        "detect": None,
        "feedback": None,
        "refinement": prompts["two_step_with_minicheck"],
    },
    "three_step": {
        "detect": None,
        "feedback": prompts["feedback_without_correct"],
        "refinement": prompts["refinement_without_correct"],
    },
}


def clean_llama2_refinement(s):
    """Without fine-tuning, llama2-7b-chat tends to output more than just the refinement."""
    split_lines = s.strip().split("\n")
    if split_lines[0].strip().endswith(":"):
        split_lines = split_lines[1:]
    processed_lines = []
    for l in split_lines:
        cleaned_l = l.strip().lower()
        if cleaned_l.startswith("summary of the above document on topic"):
            processed_lines = []

        if cleaned_l.startswith("feedback") and cleaned_l.endswith(":"):
            break

        match_str = r"i made.*change.*to the.*summary"
        if (
                cleaned_l == "changes made:"
                or cleaned_l == "refinements made:"
                or cleaned_l == "refinement made:"
                or cleaned_l.endswith("made the following changes:")
                or re.match(match_str, cleaned_l)
        ):
            break

        processed_lines.append(l)
    return "\n".join(processed_lines).strip()


def post_process_refinement(refinement):
    phrases = ["Of course! Here is the refined response:", "Sure, here is the refined response:"]
    refinement_fixed = copy.deepcopy(refinement).strip()
    found = False
    for p in phrases:
        if p in refinement_fixed:
            found = True
            refinement_fixed = refinement_fixed.replace(p, "")
            break
    refinement_fixed = refinement_fixed.strip().split("\n\n")[0] if found else refinement.strip()
    return refinement_fixed





def unload_model(generator):
    if not isinstance(generator, transformers.Pipeline):
        print(f"Cannot unload {generator}")
    else:
        generator.model.to("cpu")
        print(f"{generator.device=}")
        del generator
        gc.collect()
        torch.cuda.empty_cache()
        import time

        print("sleeping for 10s...")
        time.sleep(30)


def get_dataset_fields(djson, dataset):
    if dataset == "tofueval":
        instruction = djson["source_doc"]
        response = djson["summary"]
        topic = djson["topic"]
        sentences = sent_tokenize(response)
    elif dataset == "ultrachat":
        instruction = djson["instruction"]
        response = djson["completions"][0]
        sentences = sent_tokenize(response)
        topic = ""
    return instruction, response, sentences, topic


def run_single(dataset, single_method, refinement_model, djson, refinement_model_path, samples):
    refinement_prompt = single_method["refinement"]

    instruction, response, _, topic = get_dataset_fields(djson, dataset)

    refinement_values = {
        "document": instruction,
        "summary": response,
        "aspect": topic,
    }
    refinement_prompt = refinement_prompt[dataset].format(**refinement_values)
    sys_prompt = make_final_prompt(refinement_model_path, refinement_prompt)
    sampled_refinements = sample_and_get_minimum_edits(
        generator=refinement_model,
        prompt=sys_prompt,
        model_name=refinement_model_path,
        original=response,
        samples=samples
    )
    return {
        "sampled_refinements": sampled_refinements["sampled_refinements"],
        "refinement": sampled_refinements["minimum_edit_refinement"],
        "refine_prompts": sys_prompt,
    }


def run_feedback(
        dataset,
        djson,
        use_sentence_wise_labels,
        feedback_prompt,
        feedback_model,
        feedback_model_path,
):
    instruction, response, sentences, topic = get_dataset_fields(djson, dataset)

    if feedback_model_path in ['ultracm', 'shepherd']:
        print(feedback_model_path)
        print("Running on the entire summary!")
        sys_prompt = ""
        if all(djson["sentence_wise_labels"]):
            print("all correct!")
            feedback_str = ""
        else:
            if dataset == "tofueval":
                new_instruction = f"{instruction}\n Summarize the provided document focusing on '{topic}'. The summary should be less than 50 words in length."
            else:
                new_instruction = instruction
            sys_prompt = make_final_prompt(feedback_model_path, instruction=new_instruction,
                                           user_message=response)
            feedback_str = run_inference(feedback_model, sys_prompt, feedback_model_path)
            if feedback_model_path=="shepherd":
                feedback_str_broken = feedback_str.split("</s>")[:-1]
                unique_feedback= "\n".join(list(set(feedback_str_broken))).strip()
                feedback_str = unique_feedback
        all_sentence_feedback = []
        feedback_prompts = [sys_prompt]
        result = {
            "sentence_wise_feedback": all_sentence_feedback,
            "feedback_prompts": feedback_prompts,
            "feedback": feedback_str,
        }
        return result
    all_sentence_feedback = []
    feedback_prompts = []
    feedback_str = ""
    count = 0
    if use_sentence_wise_labels:
        if all(djson["sentence_wise_labels"]):
            return {
                "sentence_wise_feedback": [],
                "feedback_prompts": [],
                "feedback": "",
            }
        else:
            for snum, (s, slabel) in enumerate(zip(sentences,djson['sentence_wise_labels'])):
                print(snum)
                if slabel == 0:
                    values = {"document": instruction, "summary": response, "aspect": topic, "sentence": s}
                    prompt = feedback_prompt[dataset].format(**values)
                    sys_prompt = make_final_prompt(model_path=feedback_model_path, user_message=prompt)
                    feedback = run_inference(feedback_model, sys_prompt, feedback_model_path)
                    feedback_prompts.append(sys_prompt)
                    all_sentence_feedback.append(feedback)
                    feedback_str += f"{count + 1}. {feedback}\n"
                    count += 1
                else:
                    print("no feedback cause sentence is correct!")
    else:
        for snum, s in enumerate(sentences):
            print(snum)
            values = {"document": instruction, "summary": response, "aspect": topic, "sentence": s}
            prompt = feedback_prompt[dataset].format(**values)
            sys_prompt = make_final_prompt(model_path=feedback_model_path, user_message=prompt)
            feedback = run_inference(feedback_model, sys_prompt, feedback_model_path)
            feedback_prompts.append(sys_prompt)
            all_sentence_feedback.append(feedback)
            feedback_str += f"{snum + 1}. {feedback}\n"

    print(feedback_str)
    result = {
        "sentence_wise_feedback": all_sentence_feedback,
        "feedback_prompts": feedback_prompts,
        "feedback": feedback_str,
    }
    return result


def run_feedback_step(
        input_file,
        output_file,
        dataset,
        feedback_prompt,
        feedback_model_path,
        use_sentence_wise_labels=False,
):
    data = open(input_file).readlines()
    print(f"running feedback step with {use_sentence_wise_labels=}")
    print(len(data))

    print(f"Loading models...", feedback_model_path)
    feedback_model = load_model(feedback_model_path)

    with open(output_file, "w") as f_:
        for i, djson in enumerate(data):
            print("instance: ", i)
            if isinstance(djson, str):
                djson = json.loads(djson)

            result = run_feedback(
                dataset=dataset,
                djson=djson,
                use_sentence_wise_labels=use_sentence_wise_labels,
                feedback_prompt=feedback_prompt,
                feedback_model=feedback_model,
                feedback_model_path=feedback_model_path,
            )
            djson.update(result)

            djson["end_to_end_feedback_response"] = result
            f_.write(json.dumps(djson))
            f_.write("\n")

    unload_model(feedback_model)


def run_refinement(
        dataset,
        djson,
        use_sentence_wise_labels: bool,
        refinement_model,
        refinement_prompt,
        refinement_model_path,
        samples
):
    instruction, response, sentences, topic = get_dataset_fields(djson, dataset)
    feedback_str = djson["feedback"]
    print(feedback_str)
    print("*************")
    if use_sentence_wise_labels and all(djson["sentence_wise_labels"]):
        return {
            "sampled_refinements": [],
            "refinement_prompts": "",
            "refinement_raw": response,
            "refinement": response,
        }

    refinement_values = {
        "document": instruction,
        "summary": response,
        "aspect": topic,
        "feedback": feedback_str,
    }
    refinement_prompt = refinement_prompt[dataset].format(**refinement_values)
    sys_prompt = make_final_prompt(model_path=refinement_model_path, user_message=refinement_prompt)
    # refinement = run_inference(refinement_model, sys_prompt, refinement_model_path)
    sampled_refinements  = {}
    try:
        sampled_refinements = sample_and_get_minimum_edits(
            generator=refinement_model,
            prompt=sys_prompt,
            model_name=refinement_model_path,
            original=response,
            samples=samples
        )

        refinement = sampled_refinements["minimum_edit_refinement"]
        # refinement = run_inference(refinement_model, sys_prompt, refinement_model_path)
        # print(refinement)

        if dataset == "ultrachat":
            refinement_fixed = refinement
        elif refinement_model_path == "llama2":
            refinement_fixed = clean_llama2_refinement(refinement)
        else:
            refinement_fixed = post_process_refinement(refinement)
    except Exception as e:
        print(f"Error: {e}")
        refinement_fixed = ""
        refinement = ""


    print(feedback_str)
    print("")
    print(refinement)
    print("---")
    print(refinement_fixed)
    print(response == refinement_fixed)
    print("")

    result = {
        "sampled_refinements": sampled_refinements.get("sampled_refinements",[]),
        "refinement_prompts": sys_prompt,
        "refinement_raw": refinement,
        "refinement": refinement_fixed,
    }
    return result


def run_refinement_step(
        input_file: Path,
        output_file: Path,
        dataset,
        refinement_prompt,
        refinement_model_path,
        samples,
        use_sentence_wise_labels=False,
):
    data = open(input_file).readlines()
    print(f"running refinement step with {use_sentence_wise_labels=}")
    print(len(data))

    print(f"Loading models...", refinement_model_path)
    refinement_model = load_model(refinement_model_path)

    with open(output_file, "w") as f_:
        for i, djson in enumerate(data):
            print(i)
            if isinstance(djson, str):
                djson = json.loads(djson)

            result = run_refinement(
                dataset=dataset,
                djson=djson,
                use_sentence_wise_labels=use_sentence_wise_labels,
                refinement_model=refinement_model,
                refinement_prompt=refinement_prompt,
                refinement_model_path=refinement_model_path,
                samples=samples
            )
            djson.update(result)

            djson["end_to_end_refinement_response"] = result
            f_.write(json.dumps(djson))
            f_.write("\n")

    unload_model(refinement_model)


def run_two_step(
        input_file: Path,
        output_file: Path,
        dataset,
        two_method,
        feedback_model_path,
        refinement_model_path,
        samples
):
    feedback_prompt = two_method["feedback"]
    refinement_prompt = two_method["refinement"]

    feedback_file = output_file.parent / f"{output_file.stem}_feedback.jsonl.tmp"

    run_feedback_step(
        input_file,
        feedback_file,
        dataset,
        feedback_prompt,
        feedback_model_path,
    )

    run_refinement_step(
        feedback_file,
        output_file,
        dataset,
        refinement_prompt,
        refinement_model_path,
        samples=samples
    )
    return output_file


def run_minicheck_step(
        input_file: Path,
        output_file: Path,
        dataset,
        cuda_id: int,
        cache_dir: Path,
):
    from minicheck.minicheck import MiniCheck

    detect_scorer = MiniCheck(
        model_name="flan-t5-large", device=f"cuda:{cuda_id}", cache_dir=cache_dir
    )

    data = open(input_file).readlines()
    print("running minicheck step")
    print(len(data))

    with open(output_file, "w") as f_:
        for i, djson in enumerate(data):
            if isinstance(djson, str):
                djson = json.loads(djson)

            instruction, response, sentences, topic = get_dataset_fields(djson, dataset)
            sentence_wise_labels, prob, _, _ = detect_scorer.score(
                docs=[instruction] * len(sentences), claims=sentences
            )

            djson["sentence_wise_labels"] = sentence_wise_labels

            f_.write(json.dumps(djson))
            f_.write("\n")
    detect_scorer.model.model.to("cpu")
    del detect_scorer
    gc.collect()
    torch.cuda.empty_cache()


def run_two_step_with_minicheck(
        dataset, two_method, detector, refinement_model, djson, refinement_model_path, samples
):
    refinement_prompt = two_method["refinement"]

    instruction, response, sentences, topic = get_dataset_fields(djson, dataset)

    sentence_wise_labels, prob, _, _ = detector.score(
        docs=[instruction] * len(sentences), claims=sentences
    )
    if all(sentence_wise_labels):
        return {
            "sentence_wise_labels": sentence_wise_labels,
            "refinement_prompts": "",
            "refinement": response,
        }
    else:
        refinement_values = {
            "document": instruction,
            "summary": response,
            "aspect": topic,
        }
        refinement_prompt = refinement_prompt[dataset].format(**refinement_values)
        sys_prompt = make_final_prompt(refinement_model_path, refinement_prompt)
        sampled_refinements = sample_and_get_minimum_edits(
            generator=refinement_model,
            prompt=sys_prompt,
            model_name=refinement_model_path,
            original=response,
            samples=samples
        )
    return {
        "sentence_wise_labels": sentence_wise_labels,
        "sampled_refinements": sampled_refinements["sampled_refinements"],
        "refinement": sampled_refinements["minimum_edit_refinement"],
        "refine_prompts": sys_prompt,
    }


def run_three_step(
        input_file: Path,
        output_file: Path,
        dataset,
        three_method,
        feedback_model_path,
        refinement_model_path,
        cuda_id: int,
        cache_dir: Path,
        samples: int,
        refinement_only: bool = False,
):
    feedback_prompt = three_method["feedback"]
    refinement_prompt = three_method["refinement"]

    if not refinement_only:
        minicheck_file = output_file.parent / f"{output_file.stem}_minicheck.jsonl.tmp"
        run_minicheck_step(
            input_file=input_file,
            output_file=minicheck_file,
            dataset=dataset,
            cuda_id=cuda_id,
            cache_dir=cache_dir,
        )

        feedback_file = output_file.parent / f"{output_file.stem}_feedback.jsonl.tmp"
        run_feedback_step(
            minicheck_file,
            feedback_file,
            dataset,
            feedback_prompt,
            feedback_model_path,
            use_sentence_wise_labels=True,
        )


    run_refinement_step(
        input_file=input_file,
        output_file=output_file,
        dataset=dataset,
        refinement_prompt=refinement_prompt,
        refinement_model_path=refinement_model_path,
        use_sentence_wise_labels=True,
        samples=samples
    )

    return output_file


def sample_and_get_minimum_edits(
        generator, prompt: str, model_name: str, original: str, samples=3
) -> dict:
    if model_name == "gpt4":
        samples = 1

    sample_refinements = []
    sample_edits = []
    for i in range(samples):
        print("sampling: ", i)
        refinement = run_inference(generator, prompt, model_name, do_sample=True)
        print(refinement)
        print("-----**------")
        edits = edit_distance_ngram_operations(paragraph1=original, paragraph2=refinement)
        sample_edits.append(edits["edits"])
        sample_refinements.append(refinement)

    min_edits = np.argmin(sample_edits)

    return {
        "sampled_refinements": sample_refinements,
        "minimum_edit_refinement": sample_refinements[min_edits],
    }


def run(args):
    input_file = args.input_file
    output_file = args.output_file
    dataset = args.dataset
    method_type = args.type
    refinement_model = None
    detect_scorer = None
    # based on the args passed this logic loads the relevant model
    if method_type == "single_step":
        print("doing single step!")
        refinement_model = load_model(args.refinement_model)
    elif method_type == "two_step_with_minicheck":
        from minicheck.minicheck import MiniCheck

        cuda_id = args.cuda_id
        cache_dir = args.cache_dir
        detect_scorer = MiniCheck(
            model_name="flan-t5-large", device=f"cuda:{cuda_id}", cache_dir=cache_dir
        )
        refinement_model = load_model(args.refinement_model)
    else:
        raise ValueError(
            "Method type not in : single_step, two_step, two_step_with_minicheck, three_step"
        )

    data = open(input_file).readlines()
    print(method_type)
    print(len(data))

    with open(output_file, "w") as f_:
        for i, d in enumerate(data):
            print("Instance number: ", i)
            djson = json.loads(d)
            print(djson.keys())
            if method_type == "two_step_with_minicheck":
                response = run_two_step_with_minicheck(
                    dataset=dataset,
                    two_method=methods[method_type],
                    detector=detect_scorer,
                    refinement_model=refinement_model,
                    djson=djson,
                    refinement_model_path=args.refinement_model,
                    samples=args.samples
                )
            elif method_type == "single_step":
                response = run_single(
                    dataset=dataset,
                    single_method=methods[method_type],
                    refinement_model=refinement_model,
                    djson=djson,
                    refinement_model_path=args.refinement_model,
                    samples=args.samples
                )

            djson["refinement"] = response["refinement"]
            djson["feedback"] = response.get("feedback", "")
            djson["end_to_end_response"] = response
            f_.write(json.dumps(djson))
            f_.write("\n")
    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset type from tofueval, ultrachat",
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        help="Path to input file",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help="Path to output file",
    )
    parser.add_argument(
        "--type", type=str, help="type of refinement to run: single_step, two_step, three_step"
    )
    parser.add_argument("--cuda_id", type=int, help="cuda id for running minicheck")
    parser.add_argument("--cache_dir", type=Path, help="cache for running minicheck")
    parser.add_argument("--samples", type=int, default=1, help="number of samples to get while refining")
    parser.add_argument("--feedback_model", type=str, help="feedback model used to run feedback!")
    parser.add_argument(
        "--refinement_model", type=str, help="refinement model used to run refinement"
    )
    parser.add_argument("--eval", action="store_true", default=False, help="to run eval or not!")
    parser.add_argument(
        "--run_gpt4_eval",
        action="store_true",
        default=False,
        help="whether to run GPT-4 eval in addition to edits and alignscore",
    )
    parser.add_argument("--minicheck_path", type=Path, default=Path("/home/mwadhwa/code/MiniCheck"))

    args = parser.parse_args()
    print(args)
    sys.path.append(str(args.minicheck_path))
    samples = int(args.samples)
    # assert args.type in methods.keys()

    if args.type == "two_step":
        output_file = run_two_step(
            input_file=args.input_file,
            output_file=args.output_file,
            dataset=args.dataset,
            two_method=methods[args.type],
            feedback_model_path=args.feedback_model,
            refinement_model_path=args.refinement_model,
            samples=samples
        )
    elif args.type == "three_step":
        output_file = run_three_step(
            input_file=args.input_file,
            output_file=args.output_file,
            dataset=args.dataset,
            three_method=methods[args.type],
            feedback_model_path=args.feedback_model,
            refinement_model_path=args.refinement_model,
            cuda_id=args.cuda_id,
            cache_dir=args.cache_dir,
            samples=samples
        )
    elif args.type == "three_step_refinement_only":
        output_file = run_three_step(
            input_file=args.input_file,
            output_file=args.output_file,
            dataset=args.dataset,
            three_method=methods["three_step"],
            feedback_model_path=args.feedback_model,
            refinement_model_path=args.refinement_model,
            cuda_id=args.cuda_id,
            cache_dir=args.cache_dir,
            refinement_only=True,
            samples=samples
        )
    else:
        # Old code path
        output_file = run(args)

    if args.eval:
        from run_evaluate_revision import run_evaluate

        eval_output_path = output_file.parent / f"{output_file.stem}_eval.jsonl"
        print(f"Running eval and writing to {eval_output_path}")
        run_evaluate(
            input_file=output_file,
            output_file=eval_output_path,
            openai_key=os.environ["OPENAI_API_KEY"],
            dataset=args.dataset,
            model_name="gpt-4-0613",#"gpt-4-0125-preview",
            random_pick=True,
            run_gpt4_eval=args.run_gpt4_eval,
        )


if __name__ == "__main__":
    main()
