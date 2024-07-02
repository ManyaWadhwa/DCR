import os
import sys
os.environ['HF_HOME'] = "/data/users/mwadhwa/"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
sys.path.append("../")
from typing import List
from run_end_to_end_refinement.utils import load_model, make_final_prompt, run_inference
from fine_tuning.final_prompts import prompts
from nltk.tokenize import sent_tokenize
import time
model_mapping = {
    "llama2-ft": {"feedback": "wadhma/Critique-L2-FT-DCR", "refine": "wadhma/Refine-L2-FT-DCR"},
    "llama3-ft": {"feedback": "wadhma/Critique-L3-FT-DCR", "refine": "wadhma/Refine-L3-FT-DCR"},
}


def resize_embedding(model):
    model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.tokenizer.padding_side = "right"
    model.model.resize_token_embeddings(len(model.tokenizer))
    return model

class DCR:
    def __init__(self, cuda_id, model_name, path_to_minicheck, cache_dir)-> None:
        try:
            assert os.path.isdir(path_to_minicheck)
        except:
            raise ValueError(f"Path to minicheck: {path_to_minicheck} is not a valid directory")
        try:
                assert os.path.isdir(cache_dir)
        except:
            raise ValueError(f"Path to cache: {cache_dir} is not a valid directory")
        print(f"Setting model cache to: {cache_dir}...")
        os.environ['HF_HOME'] = cache_dir
        sys.path.append(path_to_minicheck)
        from minicheck.minicheck import MiniCheck
        if model_name == 'GPT4':
            try:
                assert 'OPENAI_API_KEY' in os.environ
            except:
                raise ValueError("OPENAI_API_KEY should be in os.environ to run refinement with GPT4")
        if model_name in ['llama2', 'llama3']:
            try:
                assert 'HF_TOKEN' in os.environ
            except:
                raise ValueError("HF_TOKEN should be in os.environ to run llama2 and llama3 models")
        self.model_name = model_name
        self.detect_scorer = MiniCheck(model_name="flan-t5-large", device=f"cuda:{cuda_id}", cache_dir=cache_dir)
        self.feedback_model = load_model(model_mapping[model_name]['feedback']) if model_name in model_mapping else load_model(model_name)
        self.refinement_model = load_model(model_mapping[model_name]['refine']) if model_name in model_mapping else load_model(model_name)
        # if model_name in model_mapping:
        #     self.feedback_model = resize_embedding(self.feedback_model)
        # if model_name in model_mapping:
        #     self.refinement_model = resize_embedding(self.refinement_model)

    def detect(self, source_text: str, initial_response: str) -> [List[str], List[float]]:
        """

        :param source_text:
        :param initial_response:
        :return:
        """
        print("Running detect...")
        sentences = sent_tokenize(initial_response, "english")
        sentence_wise_labels, prob, _, _ = self.detect_scorer.score(
            docs=[source_text] * len(sentences), claims=sentences
        )
        print(f"Found {sum(sentence_wise_labels)} correct sentences out of {len(sentences)} sentences.")
        return sentences, sentence_wise_labels

    def critique(self, source_text, initial_response, run_detect=True):
        """

        :param source_text:
        :param initial_response:
        :param sentences:
        :param sentencewise_labels:
        :return:
        """
        if run_detect:
            sentences, sentence_wise_labels = self.detect(source_text, initial_response)
        else:
            sentences = sent_tokenize(initial_response)
            sentence_wise_labels = [0 for s in sentences]
        if all(sentence_wise_labels):
            return ["" for s in sentences]
        print("Running feedback generation..")
        feedback_prompt = prompts["feedback_without_correct"]['ultrachat']
        feedback = []
        count = 0
        feedback_str_all = ""
        for s, label in zip(sentences, sentence_wise_labels):
            if label == 1:
                feedback.append("")
            else:
                prompt = feedback_prompt.format(**{"document": source_text, "summary": initial_response, "sentence": s})
                sys_prompt = make_final_prompt(self.model_name, instruction=source_text, user_message=prompt)
                feedback_str = run_inference(self.feedback_model, sys_prompt, self.model_name)
                feedback.append(feedback_str)
                feedback_str_all += f"{count+1}. {feedback_str}\n"
                count += 1
        print(f"Critiqued: {count} sentences.")
        return feedback, feedback_str_all

    def refine(self, source_text, initial_response) -> str:
        """
        :param source_text:
        :param initial_response:
        :return:
        """
        feedback, feedback_str_all = self.critique(source_text, initial_response)
        print(feedback_str_all)
        print(f"Running Refinement...")
        refinement_prompt = prompts["refinement_without_correct"]['ultrachat']
        refinement_prompt = refinement_prompt.format(**{"document": source_text, "summary": initial_response, "feedback": feedback_str_all})
        sys_prompt = make_final_prompt(model_path=self.model_name, user_message=refinement_prompt)
        response = run_inference(self.refinement_model, sys_prompt, self.model_name)
        print(f"Refined response:\n{response}")
        return response


if __name__ == '__main__':
    source_document = """MICHELE NORRIS, host: To learn a little bit more about the Quds Force that President Bush referred to so often in his press conference, we turn now to Karim Sadjadpour. He's an analyst on Iran for the International Crisis Group. And he's spent a lot of time studying the influence of Iran in Iraq.
MICHELE NORRIS, host: And Karim Sadjadpour joins us now in the studio. So glad that you're with us.
Mr. KARIM SADJADPOUR (International Crisis Group): My pleasure, Michele.
MICHELE NORRIS, host: First, Karim, tell us about this branch of the Iranian Revolutionary Guard known as the Quds Force. Who are they and what are their responsibilities?
Mr. KARIM SADJADPOUR (International Crisis Group): The Quds Force - Quds in Persian or in Arabic means Jerusalem. And it's a very elite branch of the Revolutionary Guards. We don't know exactly how many they are in number, but I think we're talking about hundreds, not thousands. Revolutionary Guards altogether are about 150,000 troops. And they're a very elite branch, I think they have military activities that would compare to something like the Navy Seals, very elitely trained.
Mr. KARIM SADJADPOUR (International Crisis Group): On the other hand, they're also conducting intelligence operations that maybe are akin to more along the lines of the CIA, or the FBI. And the Quds Forces in Iraq are not just operating in terms of the military power and flexing their muscles in that sense, but they're also operating a lot under the scenes, behind the scenes in terms of intelligence, buying support in Iraq, conducting kind of social capital experiments, funding mosques, funding clinics, things like that. So they're very much a versatile force in Iraq.
MICHELE NORRIS, host: Almost sounds like - you mentioned the Navy Seals. Almost sounds a bit like the CIA if you were trying to think of an American equivalent.
Mr. KARIM SADJADPOUR (International Crisis Group): Well, they're a combination of a lot of these things, because they - I think they are trained militarily, but at the same time they're trained to conduct these intelligence operations as well. So I think something akin to an elite fighting force which also has this intelligence sound to it as well.
MICHELE NORRIS, host: There is also the question of control. We heard the president today say it's not clear exactly who picked up the phone and told them to do what they did. And some U.S. intelligence officials say that the Quds Force would not be doing this kind of thing unless they had approval from top leaders in Tehran. Does that make sense to you?
Mr. KARIM SADJADPOUR (International Crisis Group): What's an extent that does make sense? I mean if we look at the Iranian Constitution similar to the U.S. Constitution, the president of the United States is commander in chief of the U.S. military. Similarly in Iran, the Supreme Leader Ayatollah Khamenei, has constitutional jurisdiction over the Revolutionary Guards. And the Quds Forces are a branch of the Revolutionary Guards, so hierarchically if you look at it, the Quds Forces ultimately do not report to Supreme Leader Ayatollah Khamenei.
Mr. KARIM SADJADPOUR (International Crisis Group): But also, we have to be clear that this is a very dysfunctional regime in Tehran. You have many different types of personalities, many different types of institutions. It's not a dictatorship like Saddam Hussein's Iraq was, where you have one person who's ruling by decree. Iran - the way decisions are made in Iran is very much a consensus building process, so there's an element of dysfunctionality that different institutions of the regime are conducting.
MICHELE NORRIS, host: If Iran does want to exercise broad influence throughout the region, where else is the Quds Forces are operating right now?
Mr. KARIM SADJADPOUR (International Crisis Group): Well, in Lebanon, as well. There's been a lot of accusations in the past that Quds Forces has trained Hezbollah. This was an accusation that the Israelis made during the Lebanon war in the summer of 2006. And there's also, you know, concern that Iran has these Quds Forces and Revolutionary Guards well placed in some of the Shiite communities throughout the Persian Gulf. So for example, Saudi Arabia is a very, you know, key country, and the bulk of Saudi Arabia's energy reserves are in the eastern province.
Mr. KARIM SADJADPOUR (International Crisis Group): And the eastern province happens to have a Shiite majority population. And there is a concern, for example, that if Saudi Arabia somehow complicit in taking on Iran, if Saudi Arabia joins the United States in confronting Iran's nuclear posture, that Iran have capability to really make life difficult in Saudi's eastern province through these Quds Forces collaborating with local Shiite Saudi population.
MICHELE NORRIS, host: Karim, thanks so much for coming in to talk to us.
Mr. KARIM SADJADPOUR (International Crisis Group): Thank you, Michele.
MICHELE NORRIS, host: Karim Sadjadpour is an analyst on Iran for the International Crisis Group.

Summarize the above on the topic: 'Control and decision-making in Tehran'.
"""
    initial_response = """The Iranian regime is dysfunctional, with the Supreme Leader Ayatollah Khamenei having constitutional jurisdiction over the Revolutionary Guards, including the Quds Force. The Quds Force reports to the Revolutionary Guards, but it is not clear who specifically authorized their actions in Iraq. The Quds Force is involved in intelligence operations and social capital experiments in Iraq."""
    model = "llama2"
    start = time.time()
    dcr = DCR(cuda_id=0, model_name=model, path_to_minicheck="/home/mwadhwa/code/MiniCheck/",cache_dir="/data/users/mwadhwa/")
    to_load = time.time()
    refinement = dcr.refine(source_text=source_document, initial_response=initial_response)
    to_refine = time.time()
    print(f"Time to load models: {to_load - start}")
    print(f"Time to refine with models: {to_refine - to_load}")
