import argparse
import json
import os
import sys
from pathlib import Path

import evaluate as hf_evaluate
from openai import OpenAI
from alignscore import AlignScore


from evaluate_revision import edit_distance_ngram_operations, get_gpt4_eval, get_gpt4_pairwise_eval

def get_align_score_model():
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise EnvironmentError("Please set CUDA_VISIBLE_DEVICES first")
    scorer = AlignScore(model='roberta-large',
                        batch_size=32,
                        device="cuda",
                        ckpt_path='/data/users/mwadhwa/models/AlignScore-large.ckpt',
                        evaluation_mode='nli_sp')
    return scorer


def run_evaluate(
        input_file: Path,
        output_file: Path,
        openai_key: str,
        dataset: str,
        model_name: str,
        random_pick,
        run_gpt4_eval=False
):
    client = OpenAI(api_key=openai_key)
    rouge = hf_evaluate.load("rouge")
    scorer = get_align_score_model()

    try:
        assert input_file != output_file
    except:
        raise ValueError("input and output file are the same!")

    input_data = open(input_file).readlines()
    with open(output_file, "w") as outfile:
        for line_id, line in enumerate(input_data):
            print(line_id)
            inp = json.loads(line)
            if isinstance(inp, str):
                inp= json.loads(inp)
            print(line_id, inp.keys())
            if dataset == "tofueval":
                document = inp.get("input", inp.get("document", inp.get("source_doc", "")))
                aspect = inp['aspect'] if "aspect" in inp else inp['topic']
                if "response" in inp:
                    summary = inp['response']
                elif "summary" in inp:
                    summary = inp['summary']
                else:
                    raise ValueError("original response not found in input file")
                if "refinement" in inp:
                    refinement = inp['refinement']
                elif 'revision' in inp:
                    refinement = inp['revision']
                elif 'refinement_response' in inp:
                    refinement = inp['refinement_response']
                # create instruction for GPT4eval
                instruction = f"Summarize the following document with respect to the topic: '{aspect}': \n {document}"
                original_response = f"Summary of the document with respect to the topic: '{aspect}': \n {summary}"
                refined_response = f"Summary of the document with respect to the topic: '{aspect}': \n {refinement}"

            elif dataset == 'ultrachat':
                document = inp.get("document", inp.get("instruction", ""))
                summary = inp['summary'] if "summary" in inp else inp['completions'][0]
                refinement = inp.get("refine_response", inp.get("refinement_response", inp.get("refinement","")))  # ['refine_response']
                # create instruction for GPT4eval
                instruction = document
                original_response = summary
                refined_response = refinement

            refinement = refinement.strip()
            if len(refinement) == 0:
                outfile.write(json.dumps(inp))
                outfile.write("\n")
                continue

            if "no error" in refinement.lower():
                refinement = summary

            rougeL = rouge.compute(
                predictions=[summary, refinement],
                references=[document, document],
                use_aggregator=False
            )["rougeL"]
            inp["original_rougeL"], inp["refinement_rougeL"] = rougeL

            try:
                score = scorer.score(
                    contexts=[document] * 2,
                    claims=[summary, refinement],
                )
            except Exception as e:
                print(e)
                score = [0, 0]
            inp["response_alignscore"] = score[0]
            inp["refinement_alignscore"] = score[1]

            edit_distance_metrics = edit_distance_ngram_operations(
                summary, refinement
            )
            inp.update(edit_distance_metrics)

            if run_gpt4_eval:
                original_score = get_gpt4_eval(instruction=instruction, response=original_response, client=client,
                                               model_name=model_name)
                refinement_score = get_gpt4_eval(instruction=instruction, response=refined_response, client=client,
                                                 model_name=model_name)

                inp['gpt4_original'] = original_score["score"]
                inp['gpt4_refined'] = refinement_score["score"]
                import random
                if random_pick:
                    print("random pick!!!")
                    number = random.uniform(0, 1)
                    responses = [original_response, refined_response]
                    indexes = []
                    if number < 0.5:
                        # original response gets scored first
                        indexes.append(0)
                        indexes.append(1)
                    else:
                        # original response gets scored second
                        indexes.append(1)
                        indexes.append(0)
                    pairwise_scores = get_gpt4_pairwise_eval(
                        instruction=instruction, response1=responses[indexes[0]], response2=responses[indexes[1]],
                        client=client, model_name=model_name)

                    score_original = pairwise_scores['score1'] if indexes[0] == 0 else pairwise_scores['score2']
                    score_refined = pairwise_scores['score2'] if indexes[0] == 0 else pairwise_scores['score1']

                    inp['flip_number'] = number
                    inp['response1_for_eval'] = "original" if indexes[0] == 0 else "refined"
                    inp['response2_for_eval'] = "refined" if indexes[0] == 0 else "original"
                    inp["gpt4_pair_ab_original"] = -1
                    inp["gpt4_pair_ab_refined"] = -1
                    inp["gpt4_pair_ba_original"] = -1
                    inp["gpt4_pair_ba_refined"] = -1
                else:
                    print("no random pick!!!")
                    pairwise_scores_original_first = get_gpt4_pairwise_eval(
                        instruction=instruction, response1=original_response, response2=refined_response, client=client,
                        model_name=model_name
                    )
                    inp["gpt4_pair_ab_original"] = pairwise_scores_original_first["score1"]
                    inp["gpt4_pair_ab_refined"] = pairwise_scores_original_first["score2"]

                    pairwise_scores_refined_first = get_gpt4_pairwise_eval(
                        instruction=instruction, response1=refined_response, response2=original_response, client=client,
                        model_name=model_name
                    )
                    inp["gpt4_pair_ba_original"] = pairwise_scores_refined_first["score2"]
                    inp["gpt4_pair_ba_refined"] = pairwise_scores_refined_first["score1"]
                    score_original = (pairwise_scores_original_first['score1'] + pairwise_scores_refined_first[
                        'score2']) / 2.0 if pairwise_scores_original_first['score1'] and pairwise_scores_refined_first[
                        'score2'] else 0
                    score_refined = (pairwise_scores_original_first['score2'] + pairwise_scores_refined_first[
                        'score1']) / 2.0 if pairwise_scores_original_first['score2'] and pairwise_scores_refined_first[
                        'score1'] else 0

                win_rate = None
                if score_refined > score_original:
                    win_rate = 1
                elif score_refined == score_original:
                    win_rate = 0
                else:
                    win_rate = -1
                inp['gpt4_pairwise_score_original'] = score_original
                inp['gpt4_pairwise_score_refined'] = score_refined
                inp['win_rate'] = win_rate
                inp['random_eval'] = random_pick
            else:
                print("not running GPT4 eval!")

            outfile.write(json.dumps(inp))
            outfile.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Path to input jsonl file")
    parser.add_argument("--output", type=Path, help="Path to output jsonl file")
    parser.add_argument("--api_key", type=str, help="")
    parser.add_argument("--dataset", type=str, help="dataset name being run - tofueval or ultrachat")
    parser.add_argument("--model_name", type=str, help="gpt4 api to use")
    parser.add_argument("--random_pick", action="store_true", default=False,
                        help="whether to do random order eval or two pairwise evals where you switch the order, for small datasets this should be False, otherwise this should be true")
    parser.add_argument("--run_gpt4_eval", action="store_true", default=False,
                        help="whether or not to run GPT4 eval")
    args = parser.parse_args()
    print(args)

    if args.run_gpt4_eval:
        valid_api = ["gpt-4-0613", "gpt-4-0125-preview"]

        if not args.model_name in valid_api:
            raise ValueError("model name not valid should be one of: 'gpt-4-0613','gpt-4-0125-preview' ")

    run_evaluate(input_file=args.input, output_file=args.output, openai_key=args.api_key, dataset=args.dataset,
                 model_name=args.model_name, random_pick=args.random_pick, run_gpt4_eval=args.run_gpt4_eval)

    return 0


if __name__ == "__main__":
    sys.exit(main())
