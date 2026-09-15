"""LLM-as-judge scoring using gpt-5-nano via a chat-completions endpoint.

Ported from agent_infer/src/qwen_max_score.py (which called qwen-max via dashscope).
Prompt templates are unchanged. Only the model-call path is swapped.

NEW_API detail: uses bare model name `gpt-5-nano` (not `openai/gpt-5-nano`) and
standard `temperature` + `max_tokens` params (same as the curl example the user shared).
"""
JUDGE_PROMPT_GAIA = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

Question: {query}

Labeled Answer: {reference_answer}

Predicted Answer: {generated_answer}

Did the model give an answer **equivalent** to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""

JUDGE_PROMPT_QA = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".
    - Predicted answers "120k", "124k", and 115k" are all CORRECT.
    - Predicted answers "100k" and "113k" are INCORRECT.
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```

Question: {query}
Gold target: {reference_answer}
Predicted answer: {generated_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

import argparse
import concurrent.futures
import json
import logging
import os
import time
from typing import Union

import requests
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

NEW_API_KEY = os.getenv("NEW_API")
NEW_API_URL = os.getenv("NEW_API_URL", "https://openrouter.ai/api/v1/chat/completions")
# gpt-5-nano on NEW_API: bare name, uses standard temperature + max_tokens (see curl reference).
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5-nano")

if not NEW_API_KEY:
    raise ValueError("NEW_API not set in environment (expected in src/.env)")

PROMPT = JUDGE_PROMPT_QA


def is_correct(text: str, style: str) -> int:
    if not isinstance(text, str):
        return 0
    t = text.strip()
    if style == "qa":
        return int(t.startswith("A"))
    if style == "gaia":
        return int(t.lower().startswith("correct"))
    raise ValueError("--style: [qa, gaia]")


def load_jsonl(path_: str):
    out = []
    with open(path_, "r") as f:
        for i, line in enumerate(f.readlines()):
            try:
                out.append(json.loads(line))
            except Exception:
                logging.warning(f"line {i} json.loads error: {line[:200]}")
    return out


def call_judge(messages, max_retries: int = 10, max_tokens: int = 2048, temperature: float = 0.7) -> str:
    """POST to NEW_API chat/completions (non-streaming).

    gpt-5-nano is a reasoning model — with a small max_tokens the entire budget is
    consumed by reasoning tokens and `content` comes back empty (finish_reason=length).
    So we keep max_tokens generous (same 2048 as the user's curl reference).
    """
    payload = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {NEW_API_KEY}", "Content-Type": "application/json"}

    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(NEW_API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                return content
            last_exc = RuntimeError(f"Empty content: {data}")
        except Exception as e:
            last_exc = e
            logging.warning(f"[Judge] attempt {attempt+1}/{max_retries}: {e}")
        time.sleep(min(1.0 * (attempt + 1), 5.0))
    logging.error(f"[Judge] all retries exhausted: {last_exc}")
    return ""


def prompt_judge_single(query: str, answers: Union[list, str], pred: str) -> str:
    user = PROMPT.format(query=query, reference_answer=answers, generated_answer=pred)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user},
    ]
    return call_judge(messages)


def _smoke_test():
    """Quick end-to-end check that the judge endpoint is reachable and grades correctly."""
    print(f"[smoke] endpoint={NEW_API_URL}  model={JUDGE_MODEL}")
    hello = call_judge(
        [{"role": "user", "content": "Say 'pong' and nothing else."}],
        max_tokens=2048, temperature=0.7,
    )
    print(f"[smoke] ping reply: {hello!r}")

    grade = prompt_judge_single(
        query="What is the capital of France?",
        answers="Paris",
        pred="The capital of France is Paris.",
    )
    print(f"[smoke] grade for a correct answer: {grade!r}  (expect starts with 'A')")

    grade_wrong = prompt_judge_single(
        query="What is the capital of France?",
        answers="Paris",
        pred="Berlin",
    )
    print(f"[smoke] grade for a wrong answer: {grade_wrong!r}  (expect starts with 'B')")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--overwrite", type=int, default=0)
    parser.add_argument("--style", type=str, default="qa", choices=["qa", "gaia"])
    parser.add_argument("--mp", type=int, default=8)
    parser.add_argument("--test", action="store_true", help="Run a quick smoke test and exit.")
    args = parser.parse_args()

    if args.test or not args.data_path:
        _smoke_test()
        return

    if not args.output_path:
        args.output_path = args.data_path.replace(".jsonl", "_gpt5nano_eval.jsonl")
    assert args.output_path != args.data_path, "output_path == data_path — would overwrite input"

    logging.info(f"Loading {args.data_path} (judge model: {JUDGE_MODEL})")
    data_ = load_jsonl(args.data_path)

    finished = 0
    if os.path.exists(args.output_path):
        if args.overwrite:
            os.remove(args.output_path)
        else:
            with open(args.output_path, "r") as f:
                finished = len(f.readlines())
            if finished == len(data_):
                logging.info("Already scored — proceeding to summary only.")

    def process_item(d):
        new = d.copy()
        query = d.get("question") or d.get("prompt")
        answers = d.get("answers")
        pred = d.get("response_answer")
        new["gpt5_nano_score"] = prompt_judge_single(query, answers, pred)
        return new

    todo = data_[finished:]
    with open(args.output_path, "a", buffering=1, encoding="utf-8") as g:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.mp) as ex:
            futs = [ex.submit(process_item, d) for d in todo]
            with tqdm(total=len(futs), desc="gpt-5-nano scoring...", ncols=100, leave=False) as pbar:
                for fut in concurrent.futures.as_completed(futs):
                    try:
                        res = fut.result()
                        g.write(json.dumps(res, ensure_ascii=False) + "\n")
                    except Exception as e:
                        logging.error(f"scoring error: {e}")
                    pbar.update(1)

    data_ = load_jsonl(args.output_path)
    correct = sum(is_correct(d.get("gpt5_nano_score", ""), style=args.style) for d in data_)
    total = len(data_)
    score = correct / total if total else 0.0
    msg = f"gpt-5-nano score: {score:.4f}  ({correct}/{total})"
    print(msg)
    txt_path = args.output_path.replace(".jsonl", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.write(f"# samples : {total}\n")


if __name__ == "__main__":
    main()
