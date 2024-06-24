"""
To prepare data:
1. Generate completions file (using initial_response_generation_with_llama.py)
2. Run MiniCheck on completions (using minicheck_filter.py or --run_minicheck) to get the fields:
    - "sent_summary": The response as a list of sentences using `nltk.sent_tokenize()`
    - "sent_wise_labels": MiniCheck output {1 = factual, 0 = non-factual} label for each sentence
3. For UltraChat, filter out completions where len(completion) > len(instruction)
4. Identify filler text (first and last sentences) and remove.
5. Add summary_factual label based on whether MiniCheck marked any sentence as non-factual
"""

import argparse
import copy
import sys
from pathlib import Path
from nltk.tokenize import sent_tokenize
from typing import Optional

import nltk
import pandas as pd


def filter_completions_on_prompt_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter any responses where the instruction is shorter than the completion This tends to happen for UltraChat since the summarization instruction does not enforce any length on the generation
    :param df: input dataframe with columns: completions, instruction
    :return: filtered df where length completion < length instruction
    """
    df["len_completion"] = [len(s[0]) for s in df["completions"]]
    df["len_instruction"] = [len(s) for s in df["instruction"]]

    return df[df["len_completion"] < df["len_instruction"]]


def filter_completion_filler_text(df: pd.DataFrame, dataset) -> None:
    """
    Modifies input dataframe in place, adding columns to DataFrame:
    - "sent_summary_cleaned": sentences with first- and last- sentence filler text removed
    - "sent_wise_labels_cleaned": labels for remaining sentences in "sent_summary_cleaned"
    """

    if "sent_summary" not in df.columns:
        if 'completions' in df.columns:
            df['sent_summary'] = df['completions'].apply(lambda x: sent_tokenize(x[0]))
        elif 'response' in df.columns:
            df['sent_summary'] = df['response'].apply(lambda x: sent_tokenize(x))

    first_sentences = [s[0] for s in df["sent_summary"]]
    duplicate_first_sentences = pd.Series(first_sentences).value_counts()
    duplicate_first_sentences = set(duplicate_first_sentences[duplicate_first_sentences > 1].index)
    print(f"Removing {len(duplicate_first_sentences)} duplicate first sentences as filler text")
    print(duplicate_first_sentences)

    last_sentences = [s[-1] for s in df["sent_summary"]]
    duplicate_last_sentences = pd.Series(last_sentences).value_counts()
    duplicate_last_sentences = set(duplicate_last_sentences[duplicate_last_sentences > 1].index)
    print(f"Removing {len(duplicate_last_sentences)} duplicate last sentences as filler text")
    print(duplicate_last_sentences)

    def remove_filler(sent_summary):
        if sent_summary[0] in duplicate_first_sentences:
            sent_summary = sent_summary[1:]
        if sent_summary[-1] in duplicate_last_sentences:
            sent_summary = sent_summary[:-1]
        return sent_summary

    cleaned = [remove_filler(s) for s in df["sent_summary"]]
    df['sent_summary_with_filler_sentences'] = copy.deepcopy(df['sent_summary'])
    df["sent_summary"] = [s for s in cleaned]


def mark_short_sentences_as_factual(df: pd.DataFrame) -> None:
    """
    There are some responses in list form, that result in short sentences like "5.", and sometimes
    MiniCheck marks these as non-factual. We manually enforce that any sentences with n_tokens < 3
    must be factual.
    """
    df["sent_summary_cleaned_tokens"] = [
        [len(nltk.word_tokenize(s)) for s in summary] for summary in df["sent_summary_cleaned"]
    ]
    sent_wise_labels_cleaned_filter_tokens = []
    for row in df.to_dict(orient="records"):
        # Sentences with <3 tokens should be automatically marked as factual
        labels = [
            1 if n < 3 else l
            for n, l in zip(row["sent_summary_cleaned_tokens"], row["sent_wise_labels_cleaned"])
        ]
        sent_wise_labels_cleaned_filter_tokens.append(labels)
    df["sent_wise_labels_cleaned_filter_tokens"] = sent_wise_labels_cleaned_filter_tokens


def prepare_data(
        dataset: str,
        input_path: Path,
        output_path: Path,
) -> None:
    supported_datasets = ["tofueval", "ultrachat"]
    if dataset not in supported_datasets:
        raise ValueError("{dataset=} not in {supported_datasets=}")

    df = pd.read_json(input_path, lines=True)

    if dataset == "ultrachat":
        df = filter_completions_on_prompt_length(df)

    filter_completion_filler_text(df=df, dataset=dataset)

    # df["summary_factual"] = [all(l) for l in df["sent_wise_labels_cleaned_filter_tokens"]]
    # print(df["summary_factual"].value_counts())

    with open(output_path, "w+") as outfile:
        outfile.write(df.to_json(orient="records", lines=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="'tofueval' or 'ultrachat'")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to input jsonl file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output jsonl file")

    parser.add_argument(
        "--run_minicheck",
        action="store_true",
        help="Optional: Only provide if running MiniCheck is needed",
    )
    parser.add_argument(
        "--cuda_id",
        type=int,
        default=None,
        help="Optional: Only provide if running MiniCheck is needed",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Optional: Only provide if running MiniCheck is needed; huggingface_hub cache directory.",
    )

    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    supported_datasets = ["tofueval", "ultrachat"]
    if dataset not in supported_datasets:
        raise ValueError("{dataset=} not in {supported_datasets=}")

    input_path = args.input
    output_path = args.output

    prepare_data(
        dataset=dataset,
        input_path=input_path,
        output_path=output_path,
    )

    if args.run_minicheck:
        if args.cuda_id is None:
            raise ValueError("{args.cuda_id=}. Please provide cuda_id to run MiniCheck")

        from minicheck_filtering import run_minicheck
        input_path = output_path
        print(f"minicheck input path: {input_path}")
        print(f"cuda id: {args.cuda_id}")
        minicheck_output_path = output_path.parent / f"{output_path.stem}_minicheck.jsonl"
        print(f"minicheck output path: {minicheck_output_path}")
        run_minicheck(
            dataset=dataset,
            input_path=input_path,
            output_path=minicheck_output_path,
            cuda_id=int(args.cuda_id),
            cache_dir=args.cache_dir,
        )
        # input_path = minicheck_output_path

    return 0


if __name__ == "__main__":
    sys.exit(main())
