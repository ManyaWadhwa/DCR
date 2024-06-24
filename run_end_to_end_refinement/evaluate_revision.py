import copy
import time

MAX_API_RETRY=2

def get_eval(user_prompt: str, client, model_name, max_tokens: int = 500):
    for _ in range(MAX_API_RETRY):
        try:
            response = client.chat.completions.create(**{
                "model": model_name, #"gpt-4-0125-preview", # "gpt-4-0613", ## TODO change this to an argument
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
                "top_p": 0.6,
                "presence_penalty": 0,
                "frequency_penalty": 0
            })
            content = response.choices[0].message.content
        except Exception as e:
            print(e)
            time.sleep(10)
        else:
            break
    return content


def get_score(prediction:str)->int:
    lines = prediction.split(".")
    score_line = None
    for l in lines:
        if "therefore, the score is:" in l.lower().strip():
            score_line = l.lower().strip()
    if score_line:
        try:
            score = int(score_line.split("therefore, the score is:")[-1].replace(".","").strip()[0]) ## TO DO check if we need to make this more complex
        except:
            return None
        return score
    return None


def get_gpt4_eval(instruction:str, response:str, client, model_name)->dict:
    uf_prompt = """Determine whether the provided summary is consistent with the corresponding document. Consistency in this context implies that all information presented in the response is substantiated by the document. If not, it should be considered inconsistent.

{instruction}

{response}

The response can have one or more of the following errors:
1. Extrinsic Information: the response contains new information not grounded in the source material
2. Mis-Referencing: a property or an event in the response can be found in the source material, but are associated with the wrong entity 
3. Stating Opinion As Fact: the response entails a proposition that's mentioned in the source material not as a fact, but as someone's opinion 
4. Reasoning Error: the response makes one or more wrong inferences from the information in the source material 
5. Tense/modality Error: the tense or modal (eg: can, may, must) used in the response sentence does not match the tense/modality of the source material 
6. Contradiction: the response contradicts the source material 
7. Nuanced Meaning Shift: the response twists information from the source material in a subtle way 

Given the error categories, rate the above response on a scale of 1 to 5 based on extent of factual consistency:
5. ** completely consistent **: the response is completely factually consistent with the source material.
4. ** insignificant inconsistencies **:  the response is mostly factually consistent, with slight inconsistencies not affecting main points. 
3. ** partially inconsistent **: overall factually consistent, with a few inconsistencies with the source material.
2. ** severe inconsistencies **: nearly half response is factually inconsistent, with severe deviation from main points.
1. ** completely inconsistent **: the entire response is factually inconsistent with the source material.

First output a list of errors that the summary makes, then conclude the response with a score in the following format: "therefore, the score is:"
    """
    format_input = {"instruction":instruction, "response":response}
    found = False
    max_attempt = 5
    count = 0
    while not found and count < max_attempt:
        response = get_eval(user_prompt=uf_prompt.format(**format_input), client=client, model_name=model_name)
        if "therefore, the score is:" in response.lower():
            found = True
        count += 1
    import copy
    raw_response = copy.deepcopy(response)
    score = get_score(raw_response)
    return {
        "score":score,
        "response":response
    }


def get_gpt4_pairwise_eval(instruction: str, response1: str, response2: str, client, model_name) -> dict:
    uf_prompt = """Determine whether the provided summary is consistent with the corresponding document. Consistency in this context implies that all information presented in the response is substantiated by the document. If not, it should be considered inconsistent.

{instruction}

## Response 1
{response1}

## Response 2
{response2}

A response can have one or more of the following errors:
1. Extrinsic Information: the response contains new information not grounded in the source material
2. Mis-Referencing: a property or an event in the response can be found in the source material, but are associated with the wrong entity
3. Stating Opinion As Fact: the response entails a proposition that's mentioned in the source material not as a fact, but as someone's opinion
4. Reasoning Error: the response makes one or more wrong inferences from the information in the source material
5. Tense/modality Error: the tense or modal (eg: can, may, must) used in the response sentence does not match the tense/modality of the source mater    ial
6. Contradiction: the response contradicts the source material
7. Nuanced Meaning Shift: the response twists information from the source material in a subtle way

Given the error categories, rate each response on a scale of 1 to 5 based on extent of factual consistency:
5. ** completely consistent **: the response is completely factually consistent with the source material.
4. ** insignificant inconsistencies **:  the response is mostly factually consistent, with slight inconsistencies not affecting main points.
3. ** partially inconsistent **: overall factually consistent, with a few inconsistencies with the source material.
2. ** severe inconsistencies **: nearly half response is factually inconsistent, with severe deviation from main points.
1. ** completely inconsistent **: the entire response is factually inconsistent with the source material.

For each response, first output a list of errors that the summary makes, then conclude the response with a score in the following format: "therefore, the score is:"

Output Format:

## Response 1
...

## Response 2
...
"""
    format_input = {"instruction": instruction, "response1": response1, "response2": response2}
    found = False
    max_attempt = 2
    count = 0
    while not found and count < max_attempt:
        response = get_eval(user_prompt=uf_prompt.format(**format_input), client=client, model_name=model_name)
        response_parts = response.split("## Response")[1:]
        if len(response_parts) == 2 and all(["therefore, the score is:" in r.lower() for r in response_parts]):
            found = True
        count += 1
    if found:
        raw_response1 = copy.deepcopy(response_parts[0])
        score1 = get_score(raw_response1)
        raw_response2 = copy.deepcopy(response_parts[1])
        score2 = get_score(raw_response2)
    else:
        raw_response1 = ""
        score1 = 0
        raw_response2 = ""
        score2 = 0
    return {
        "score1": score1,
        "response1": response1,
        "score2": score2,
        "response2": response2,
    }


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
