from openai import OpenAI
import random
import time
import numpy

random.seed(10)



def edit_distance_ngram_operations(paragraph1, paragraph2, n=1):
    ngrams1 = [tuple(paragraph1.split()[i:i + n]) for i in range(len(paragraph1.split()) - n + 1)]
    ngrams2 = [tuple(paragraph2.split()[i:i + n]) for i in range(len(paragraph2.split()) - n + 1)]

    len_ngrams1 = len(ngrams1)
    len_ngrams2 = len(ngrams2)

    # Create a 2D array to store the edit distances
    dp = [[0] * (len_ngrams2 + 1) for _ in range(len_ngrams1 + 1)]

    # Initialize the first row and column
    for i in range(len_ngrams1 + 1):
        dp[i][0] = i
    for j in range(len_ngrams2 + 1):
        dp[0][j] = j

    # Fill the array using dynamic programming
    for i in range(1, len_ngrams1 + 1):
        for j in range(1, len_ngrams2 + 1):
            cost = 0 if ngrams1[i - 1] == ngrams2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

    # Backtrack to find operations
    i, j = len_ngrams1, len_ngrams2
    insertions, deletions, substitutions = 0, 0, 0

    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            if ngrams1[i - 1] != ngrams2[j - 1]:
                substitutions += 1
            i -= 1
            j -= 1

    return {
        "edits": dp[len_ngrams1][len_ngrams2],
        "adds": insertions,
        "deletes": deletions,
        "sub": substitutions,
        "len1": len_ngrams1,
        "len2": len_ngrams2
    }


def call_gpt(prompt, model_name, client):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=1
    )
    returned = response.choices[0].message.content
    # returned = response['choices'][0]['message']['content']
    return returned


def get_minimum_revision(final_prompt,
                         response: str,
                         few_shot: bool = False,
                         api_key: str = None,
                         model: str = "gpt-4-0613") -> dict:
    """
    :param instruction:
    :param response:
    :param feedback:
    :return:
    "sample_num": sample index with the least numbber of edits
   "minimum_edit_response": revised response with the least number of edits
    "all_sampled_responses": all sampled revisions
    """
    if api_key is None:

        raise ValueError("openai api key not provided")

    client = OpenAI(
        api_key=api_key,
    )
    samples = 1 # number of samples to run! TODO: make this an arg
    responses = {i: [] for i in range(samples)}
    raw_output = {i: [] for i in range(samples)}
    edits = []
    prompts = []
    count = 0
    phrase = "refined response without error:"
    # if few_shot:
    #     pass
    for i in range(samples):
        try:
            response_string = call_gpt(prompt=final_prompt,
                                       model_name=model,
                                       client=client)
        except:
            print("waiting")
            time.sleep(10)
            response_string = call_gpt(prompt=final_prompt,
                                       model_name=model,
                                       client=client)
        if few_shot:
            response_phrase = response_string.split(phrase)[-1][
                              1:-1]  # geting the string after the refined respone without error
        else:
            response_phrase = response_string
        all_edit_types = edit_distance_ngram_operations(response, response_phrase)
        edits.append(all_edit_types['edits'])
        responses[i].append(response_phrase)
        raw_output[i].append(response_string)
        count += 1
    if count % 10 == 0 and count != 0:
        print("waiting")
        time.sleep(20)

    minimum_edits_index = numpy.argmin(edits)
    minimum_edit_response = responses[minimum_edits_index]
    return {
        "sample_num": minimum_edits_index,
        "minimum_edit_response": minimum_edit_response[0],
        "all_sampled_responses": responses
    }
