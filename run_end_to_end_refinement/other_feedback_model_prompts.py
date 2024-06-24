ultracm_system_prompt = "User: A one-turn chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, very detailed, and polite answers to the user's questions.</s>"

ultracm_instruction_template = """Given my answer to an instruction, your role is to provide specific and constructive feedback for me. You should find the best way for me to learn from your feedback and improve my performance. 

You should consider multiple aspects of my answer, including helpfulness, truthfulness, honesty, and to what extent the answer follows instructions.
---

### Instruction
{instruction}

### Answer
{completion}
---

Please act as a teacher and provide specific and constructive feedback. Besides describing the weaknesses of the answer, you should also provide specific suggestions to guide me toward understanding how to improve. Please note, however, that your suggestions should help me better complete the instructions, but you should not introduce new requirements that are not mentioned in the instructions. Your feedback should focus on enhancing my ability to think critically and respond accurately. However, never explicitly provide the reference answer, nor do polite phrases be required. Only respond with concise feedback in chat style. Finally, score the overall quality of the answer from 1 to 10, where 1 is the worst and 10 is the best.

*Format*
### Feedback
Overall Score: [1-10]
[Your feedback]

---

### Feedback
Overall Score: 
"""


shepherd_prompt = """### System:
Please try to critique the answer to the question below.

### User:
<|Start of the Question|>
{question}
<|End of the Question|>

<|Start of the Answer|>
{answer}
<|End of the Answer|>

### Assistant:
"""

def make_shepherd_prompt(question, answer):
    shepherd_prompt = f"""### System:
Please try to critique the answer to the question below.

### User:
<|Start of the Question|>
{question}
<|End of the Question|>

<|Start of the Answer|>
{answer}
<|End of the Answer|>

### Assistant:
    """
    return shepherd_prompt


def make_ultracm_prompt(instruction, completion):
    conv = [ultracm_system_prompt, ultracm_system_prompt]
    conv.append("User: " + ultracm_instruction_template.format(
        instruction=instruction,
        completion=completion,
    ) + "</s>")
    conv.append("Assistant: ")
    prompt = "\n".join(conv)
    return prompt
