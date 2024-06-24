templates = {
    "tofueval":{
        "prompt":"""I summarized the following document on the topic: '{aspect}': 
{document}

Summary on topic: '{aspect}'

{summary}

-----

The provided summary is factually inconsistent with the corresponding document. This implies that there is information in the summary that is NOT substantiated by the document. Factual inconsistencies can be of the following types:
1. Mis-Referencing: a property or an event in the summary can be found in the document, but are associated with the wrong entity 
2. Stating Opinion As Fact: the summary entails a proposition that's mentioned in the document not as a fact, but as someone's opinion 
3. Reasoning Error: the summary makes one or more wrong inferences from the information in the document
4. Tense/modality Error: the tense or modal (eg: can, may, must) used in the summary does not match the tense/modality of the document
5. Extrinsic Information: the summary contains new information not grounded in the source document
6. Contradiction: the summary contradicts the document
7. Nuanced Meaning Shift: the summary twists information from the document in a subtle way

Identify factually inconsistent information in the form of a JSON where you return a list with the following keys:
1. inconsistency: <span from the summary that is factually inconsistent>
2. inconsistency type: <the inconsistency type from the above list of types>
3. feedback: <explanation of the error and how it can be fixed>
4. fix: <correct span that fixes the inconsistency>
""",
        "output_type": "json",
        "description": "feedback generation prompt for tofueval; this specifies the topic separately"
    },
    "ultrachat":{
        "prompt":"""{document}

Response to the above question:
{summary}

-----

The provided summary is factually nconsistent with the corresponding document. This implies that there is information in the summary that is NOT substantiated by the document. Factual inconsistencies can be of the following types:
1. Mis-Referencing: a property or an event in the summary can be found in the document, but are associated with the wrong entity 
2. Stating Opinion As Fact: the summary entails a proposition that's mentioned in the document not as a fact, but as someone's opinion 
3. Reasoning Error: the summary makes one or more wrong inferences from the information in the document
4. Tense/modality Error: the tense or modal (eg: can, may, must) used in the summary does not match the tense/modality of the document
5. Extrinsic Information: the summary contains new information not grounded in the source document
6. Contradiction: the summary contradicts the document
7. Nuanced Meaning Shift: the summary twists information from the document in a subtle way

Identify factually inconsistent information in the form of a JSON where you return a list with the following keys:
1. inconsistency: <span from the summary that is factually inconsistent>
2. inconsistency type: <the inconsistency type from the above list of types>
3. feedback: <explanation of the error and how it can be fixed>
4. fix: <correct span that fixes the inconsistency>
""",
    "output_type": "json",
        "description": "ultrachat feedback generation prompt"
    }
}
