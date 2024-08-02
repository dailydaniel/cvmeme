from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import fitz
import math
from io import BytesIO
from datetime import datetime

from prompts import scale, prompt_rerank, cv_scale, base_prompt, format_example


load_dotenv()


def extract_text_from_pdf(file, max_pages=5, _bytes=True) -> tuple[bool, str]:
    if _bytes:
        document = fitz.open(stream=file, filetype="pdf")
    else:
        document = fitz.open(file)

    if len(document) > max_pages:
        # raise ValueError(f"Document has {len(document)} pages, but the limit is {max_pages} pages.")
        print(f"Document has {len(document)} pages, but the limit is {max_pages} pages.")
        return False, ""

    text = ""

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()

    return True, text


def parse_json_scale(txt: str, return_full=False) -> tuple[bool, dict | list[int]]:
    splitted = txt.split('\n')

    if splitted[0] == '```json' and splitted[-1] == '```':
        parsed = json.loads('\n'.join(splitted[1:-1]))

        if return_full:
            try:
                return True, {k: int(v) for k, v in parsed.items()}
            except:
                # raise ValueError("Invalid json format")
                print("Invalid json format")
                return False, {}
        else:
            try:
                return True, [int(v) for _, v in parsed.items()]
            except:
                print("Invalid json format")
                return False, []
    elif splitted[0] == '```json' and splitted[-1] == '' and splitted[-2].strip() == '```':
        parsed = json.loads('\n'.join(splitted[1:-2]))

        if return_full:
            try:
                return True, {k: int(v) for k, v in parsed.items()}
            except:
                print("Invalid json format")
                return False, {}
        else:
            try:
                return True, [int(v) if isinstance(v, int) else 3 for _, v in parsed.items()]
            except:
                # raise ValueError(f"Invalid json format {parsed}")
                print(f"Invalid json format {parsed}")
                return False, []
    else:
        # raise ValueError(f"Invalid json format {txt}")
        print(f"Invalid json format {txt}")
        return False, {}


def match_score(meme, resume) -> float:
    return math.sqrt(sum((m - r) ** 2 for m, r in zip(meme, resume)))


class CVMEME:
    def __init__(
            self,
            scores_path: str = '../data/meme/scores.json',
            descriptions_path: str = '../data/meme/descriptions.json',
            scale_prompt: str = scale,
            rerank_prompt: str = prompt_rerank,
            base_prompt: str = base_prompt,
            format_example_prompt: str = format_example,
            cv_scale_prompt: str = cv_scale,
            memes_base_dir: str = '../data/meme',
            log_path: str = '../data/log.txt',
            # model_name: str = "gpt-4o-mini"
            model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ):
        self.scale_prompt = scale_prompt
        self.rerank_prompt = rerank_prompt
        self.cv_scale_prompt = cv_scale_prompt
        self.base_prompt = base_prompt
        self.format_example_prompt = format_example_prompt

        self.log_path = log_path

        self.memes_base_dir = memes_base_dir

        self.model = model_name

        # self.client = OpenAI(
        #     api_key=os.environ['OPENAI_API_KEY'],
        # )

        self.client = OpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            base_url=os.environ['LLM_HOST']
        )

        with open(scores_path, "r", encoding="utf-8") as json_file:
            self.scores = json.load(json_file)

        with open(descriptions_path, "r", encoding="utf-8") as json_file:
            self.descriptions = json.load(json_file)

    def get_other_respond(self, txt: str, log: bool = True, user_id: int = 0) -> str:

        if log:
            with open(self.log_path, "a") as log_file:
                interaction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"User ID: {user_id} => Got question ({interaction_time})\n\n")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # {
                #     "role": "system",
                #     "content": [
                #         {"type": "text", "text": self.base_prompt},
                #     ],
                # },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.base_prompt+'\n\nСообщение:\n'+txt},
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.3,
        )

        return response.choices[0].message.content

    def get_cv_scale(self, txt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.cv_scale_prompt.format(
                            cv=txt,
                            scale=self.scale_prompt
                        ) + self.format_example_prompt},
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.0,
        )

        return response.choices[0].message.content

    def get_meme_rerank(self, cv: str, memes: list[tuple]) -> str:
        meme1, meme2, meme3 = memes

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.rerank_prompt.format(
                            cv=cv,
                            meme1_name=meme1[0],
                            meme1_desc=meme1[1],
                            meme1_worker=meme1[2],
                            meme2_name=meme2[0],
                            meme2_desc=meme2[1],
                            meme2_worker=meme2[2],
                            meme3_name=meme3[0],
                            meme3_desc=meme3[1],
                            meme3_worker=meme3[2],
                        )},
                    ],
                }
            ],
            max_tokens=128,
            temperature=0.3,
        )

        return response.choices[0].message.content

    def process_cv(self, file: str | BytesIO, _bytes=True) -> tuple[bool, tuple[str, str]]:
        success, text = extract_text_from_pdf(file, _bytes=_bytes)

        if success:
            response = self.get_cv_scale(text)
            if response:
                return True, (text, response)
            else:
                return False, (text, "")
        else:
            return False, ("", "")

    def get_scores(self, cv_scaled: str, return_scores: bool = False) -> list[str | tuple[str, float]]:
        success, cv_scale_parsed = parse_json_scale(cv_scaled)

        if not success:
            return []

        scores_by_meme = {
            meme_name: [int(v) for _, v in scores.items()]
            for meme_name, scores in self.scores.items()
        }

        try:
            scored_memes = [
                (meme, match_score(scores, cv_scale_parsed))
                for meme, scores in scores_by_meme.items()]
        except:
            return []

        if return_scores:
            return sorted(scored_memes, key=lambda x: x[1])[:3]
        else:
            return [n for n, s in sorted(scored_memes, key=lambda x: x[1])[:3]]

    def rerank_memes(self, cv: str, memes: list[str]) -> tuple[bool, str]:
        memes_descriptions = [self.descriptions[meme] for meme in memes]
        memes_short = []

        for meme in memes:
            path = f"{self.memes_base_dir}/txt/{meme}.txt"
            with open(path, "r", encoding="utf-8") as file:
                memes_short.append(file.read())

        response = self.get_meme_rerank(
            cv,
            [(memes[i], memes_short[i], memes_descriptions[i])
             for i in range(len(memes))]
        )
        response = response.split('\n')[0].strip()

        if response in memes:
            return True, response
        else:
            return False, response

    def get_meme_img_path(self, meme: str) -> str:
        return f"{self.memes_base_dir}/img/{meme}.jpg"

    def __call__(
            self,
            file: str | BytesIO,
            log: bool = True,
            user_id: int = 0
    ) -> tuple[bool, str]:
        if log:
            with open(self.log_path, "a") as log_file:
                interaction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"User ID: {user_id} => Start ({interaction_time})\n\n")

        success, (cv, cv_processed) = self.process_cv(file)

        if not success:
            return False, "Error in processing cv"

        if log:
            with open(self.log_path, "a") as log_file:
                interaction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"User ID: {user_id} => Process CV ({interaction_time})\n\n")

        cv_scores = self.get_scores(cv_processed)

        if not cv_scores:
            return False, "Error in getting scores"

        if log:
            with open(self.log_path, "a") as log_file:
                interaction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"User ID: {user_id} => Got Scores ({cv_scores}) ({interaction_time})\n\n")

        success, meme = self.rerank_memes(cv, cv_scores)

        if not success:
            return False, f"Error in reranking memes ({meme})"

        if log:
            with open(self.log_path, "a") as log_file:
                interaction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"User ID: {user_id} => Got Meme ({meme}) ({interaction_time})\n\n")

        return True, self.get_meme_img_path(meme)
