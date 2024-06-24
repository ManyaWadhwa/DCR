prompts = {
    "single_step":{
        "tofueval":"""I summarized the document below, on the topic '{aspect}'
{document}

The summary on the '{aspect}' is:
{summary}

If there are any factual inconsistencies in the summary then edit the summary such that the refinement doesn't have any inconsistencies. Consistency in this context implies that all information presented in the summary is substantiated by the document.If the summary is consistent, then just the copy the same summary with no changes. When refining, make the minimum number of changes.        
""",
        "ultrachat":"""Document:
{document}

Response:
{summary}

If there are any factual inconsistencies in the response then edit the response such that the refinement doesn't have any inconsistencies. Consistency in this context implies that all information presented in the response is substantiated by the document.If the response is consistent, then just the copy the same response with no changes. When refining, make the minimum number of changes.        
"""
    },
"two_step_with_minicheck":{
        "tofueval":"""I summarized the document below, on the topic '{aspect}'
{document}

The summary on the '{aspect}' is:
{summary}

Edit the summary such that the refinement doesn't have any factual inconsistencies. Consistency in this context implies that all information presented in the summary is substantiated by the document. When refining, make the minimum number of changes.        
""",
        "ultrachat":"""Document:
{document}

Response:
{summary}

Edit the response such that the refinement doesn't have any factual inconsistencies. Consistency in this context implies that all information presented in the response is substantiated by the document. When refining, make the minimum number of changes.        
"""
    },
"feedback_with_correct":{
    "tofueval":"""I summarized the document below, on the topic '{aspect}':
{document}


The summary of the above conversation, on the topic: '{aspect}' is:
{summary}

For the following sentence in the summary:
{sentence}

reason if there is any factually inconsistent span in the sentence. A span is factually inconsistent if it cannot be substantiated by the document. If there is no inconsistency, then end your answer with "no error". Otherwise if there is a factual inconsistency, then give reasons for it, point to the error span by stating "The error span: <span from sentence> and end your answer with a suggested fix to the summary. 
""",
    "ultrachat":"""Document:
{document}


Summary:
{summary}

For the following sentence in the summary:
{sentence}

reason if there is any factually inconsistent span in the sentence. A span is factually inconsistent if it cannot be substantiated by the document. If there is no inconsistency, then end your answer with "no error". Otherwise if there is a factual inconsistency, then give reasons for it, point to the error span by stating "The error span: <span from sentence> and end your answer with a suggested fix to the summary. 
""",
},
    "feedback_without_correct":{

        "tofueval":"""I summarized the document below, on the topic '{aspect}':
{document}


The summary of the above conversation, on the topic: '{aspect}' is:
{summary}

For the following sentence in the summary:
{sentence}

reason if there is any factually inconsistent span in the sentence. A span is factually inconsistent if it cannot be substantiated by the document. Give reasons for the factual inconsistency, point to the error span by stating "The error span: <span from sentence> and end your answer with a suggested fix to the summary.        
""",
        "ultrachat":"""{document}

summary:
{summary}

For the following sentence in the summary:
{sentence}

reason if there is any factually inconsistent span in the sentence. A span is factually inconsistent if it cannot be substantiated by the document. Give reasons for the factual inconsistency, point to the error span by stating "The error span: <span from sentence> and end your answer with a suggested fix to the summary.
"""
    },
    "feedback_without_correct_only_span":{

        "tofueval":"""I summarized the document below, on the topic '{aspect}':
{document}


The summary of the above conversation, on the topic: '{aspect}' is:
{summary}

For the following sentence in the summary:
{sentence}

reason if there is any factually inconsistent span in the sentence. A span is factually inconsistent if it cannot be substantiated by the document. Point to the error span by stating "The error span: <span from sentence>.        
""",
        "ultrachat":"""{document}

summary:
{summary}

For the following sentence in the summary:
{sentence}

reason if there is any factually inconsistent span in the sentence. A span is factually inconsistent if it cannot be substantiated by the document. Point to the error span by stating "The error span: <span from sentence>.
"""
    },
"feedback_without_correct_only_span_feedback":{

        "tofueval":"""I summarized the document below, on the topic '{aspect}':
{document}


The summary of the above conversation, on the topic: '{aspect}' is:
{summary}

For the following sentence in the summary:
{sentence}

reason if there is any factually inconsistent span in the sentence. A span is factually inconsistent if it cannot be substantiated by the document. Give reasons for the factual inconsistency and point to the error span by stating "The error span: <span from sentence>.        
""",
        "ultrachat":"""{document}

summary:
{summary}

For the following sentence in the summary:
{sentence}

reason if there is any factually inconsistent span in the sentence. A span is factually inconsistent if it cannot be substantiated by the document. Give reasons for the factual inconsistency and point to the error span by stating "The error span: <span from sentence>.
"""
    },

    "refinement_without_correct":{
        "tofueval":"""I summarized the document below, on the topic '{aspect}'
{document}

Summary of the above document on topic: '{aspect}':
{summary}

Feedback for the above summary: 
{feedback}

Edit the user response such that the refinement doesn't have any errors mentioned in the feedback. Make the minimum number of changes when doing the refinement. Do not include a preamble.
""",
        "ultrachat":"""{document}

Response:
{summary}

Feedback for the above response: 
{feedback}

Edit the user response such that the refinement doesn't have any errors mentioned in the feedback. Make the minimum number of changes. Do not include a preamble.
"""
    }
}