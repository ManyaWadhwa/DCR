import argparse
import copy
import json
import random
import sys
from pathlib import Path
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize


def obtain_dialogue_mediasum(speakers, utts):
    transcript = ""
    for speaker, utt in zip(speakers, utts):
        transcript += f"{speaker}: {utt}\n"
    transcript = transcript.strip()
    return transcript

def run_minicheck(
    dataset: str,
    input_path: Path,
    output_path: Path,
    cuda_id: int,
    cache_dir: Path,
    minicheck_repo_path: str
):
    """
    The input file is either ultrachat or tofueval test set or mediasum completion file
    :param dataset:
    :param input_path:
    :param output_path:
    :param cuda_id:
    :param cache_dir:
    :param minicheck_repo_path:
    :return:
    """
    sys.path.append(minicheck_repo_path)
    from minicheck.minicheck import MiniCheck
    """Runs MiniCheck"""
    scorer = MiniCheck(model_name="flan-t5-large", device=f"cuda:{cuda_id}", cache_dir=cache_dir)

    data = open(input_path).readlines()
    with open(output_path, "w") as f_:
        for i, d in enumerate(data):
            print(i)
            djson = json.loads(d)
            print(djson.keys())
            if dataset == "tofueval":
                if "source_doc" in djson:
                    source_document = djson["source_doc"]
                else:
                    source_document = obtain_dialogue_mediasum(djson["speaker"], djson["utt"])
                if "completions" in djson:
                    summary = djson['completions'][0]
                else:
                    summary = djson["response"] if "response" in djson else djson["summary"]
            elif dataset == "ultrachat":
                source_document = djson.get("instruction", djson.get("document", ""))
                if "completions" in djson:
                    summary = djson["completions"][0]
                else:
                    summary = djson["response"] if "response" in djson else djson["summary"]
            else:
                raise ValueError(
                    f"{dataset=} is not a valid option. Supported: ['tofueval', 'ultrachat']"
                )

            if 'sent_summary' not in djson:
                sentences = sent_tokenize(summary)
                djson["sent_summary"] = sentences
            else:
                sentences = djson['sent_summary']
            labels, prob, _, _ = scorer.score(
                docs=[source_document] * len(sentences), claims=sentences
            )
            assert len(labels) == len(sentences)
            labels_fixed = []
            for s, l in zip(sentences, labels):
                if len(word_tokenize(s))<3:
                    labels_fixed.append(1)
                else:
                    labels_fixed.append(l)
            assert len(labels_fixed) == len(sentences)
            # run the filtering where if there is a sentence with less than a few tokens then that is automatically factual!
            djson["sent_wise_labels_pre_filtering"] = labels
            djson["sent_wise_labels"] = labels_fixed
            djson['summary_factual'] = all(labels_fixed)
            f_.write(json.dumps(djson))
            f_.write("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="'tofueval' or 'ultrachat'",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output jsonl",
    )
    parser.add_argument(
        "--cuda_id",
        type=int,
        help="Model will be run on device 'cuda:{cuda_id}'",
    )
    parser.add_argument("--cache_dir", type=str, help="huggingface_hub cache directory")
    parser.add_argument("--path_to_minicheck", type=str, help="path to minicheck repo on your local")


    args = parser.parse_args()
    minicheck_path = args.path_to_minicheck


    run_minicheck(
        dataset=args.dataset,
        input_path=args.input,
        output_path=args.output,
        cuda_id=args.cuda_id,
        cache_dir=args.cache_dir,
        minicheck_repo_path=minicheck_path
    )


if __name__ == "__main__":
    main()
