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

import datetime
import json
import os
import random
import re
import sys
import time
import copy
from typing import Dict, List, Tuple, Union
from urllib.parse import urlencode
import argparse
from tqdm import tqdm
import concurrent.futures
from dotenv import load_dotenv

import dashscope
import requests
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

QWEN_SERVER = os.getenv('QWEN_SERVER', default='dashscope')
QWEN_MODEL = os.getenv('QWEN_MODEL', default='qwen-max')

PROMPT = JUDGE_PROMPT_QA

MESSAGE_TEMPLETE = [{"role": "system", "content": "You are a helpful assistant."}, \
    {"role": "user", "content": PROMPT}]

def is_correct(text, style):
    if style == 'qa':
        return int(text.startswith('A'))
    elif style == 'gaia':
        return int(text.lower().startswith('correct'))
    else:
        raise ValueError('--style: [qa, gaia]')
def load_jsonl(path_):
    with open(path_, 'r') as f:
        ret = []
        for idx, line in enumerate(f.readlines()):
            try:
                ret.append(json.loads(line))
            except:
                print(idx, 'json.loads error:')
                print(line)
                print('='*30)
        return ret

def call_qwen_dash(model, message, use_raw_prompt, stop_words, top_k):
    retry_limit = 10
    count = 0
    text = ''
    
    kwargs = {}
    kwargs['debug'] = True
    kwargs['headers'] = {'X-DashScope-DataInspection': 'disable'}
    
    while count < retry_limit:
        time.sleep(0.8)
        try:
            # print("dashscope.Generation.call")
            response = dashscope.Generation.call(model=model,
                messages=message,
                # prompt=prompt,
                use_raw_prompt=use_raw_prompt,
                stop_words=stop_words,
                top_k=top_k,
                **kwargs)
            # print(response)
            # print("dashscope.Generation.call done")
            if response.output is None:
                count += 1
                logging.warning(f"response.output is None. Continuing {count}")
                logging.warning("response:", response)
                continue
            text = response.output.text
            if isinstance(text, str) and text:
                break
        except Exception as e:
            logging.error(e)

        count += 1
        logging.error('Generation.call failed. Retrying...%d' % count, file=sys.stderr)
        continue
    return text

def prompt_qwen_single(query: str, answers: Union[list, str], pred: str):
    message = copy.deepcopy(MESSAGE_TEMPLETE)
    message[-1]['content'] = message[-1]['content'].format(query=query, reference_answer=answers, generated_answer=pred)
    # print(message)

    response = call_qwen_dash(
        model=QWEN_MODEL,
        message=message,
        # prompt=prompt,
        use_raw_prompt=False,
        stop_words=[{
            'stop_str': 'Observation:',
            'mode': 'exclude'
        }],
        top_k=1,
    )

    return response


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data_path', default='')
    parser.add_argument('--output_path', default='')
    parser.add_argument('--overwrite', type=int, default=0)
    parser.add_argument('--style', type=str, default='qa')
    parser.add_argument('--mp', type=int, default=2)
    args = parser.parse_args()

    if args.output_path == '':
        args.output_path = args.data_path.replace('.jsonl', '_qwen_max_eval.jsonl')
    logging.info(f"Loading {args.data_path}")

    assert args.output_path != args.data_path, print("Over write data_path!")
    data_ = load_jsonl(args.data_path)

    num_lines_finished = 0
    if os.path.exists(args.output_path):
        if args.overwrite:
            os.remove(args.output_path)
        else:
            with open(args.output_path, 'r') as f:
                num_lines_finished = len(f.readlines())
                if num_lines_finished == len(data_): print("Already scored!")


    with open(args.output_path, 'a', buffering=1, encoding='utf-8') as g:

        data_to_process = data_[num_lines_finished:]
        
        def process_item(data):
            new_data = data.copy()
            query = data.get('question') or data.get('prompt')
            answers = data.get('answers')
            pred = data.get('response_answer')
            
            qwen_eval = prompt_qwen_single(query, answers, pred)
            new_data['qwen_max_score'] = qwen_eval
            return new_data
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.mp) as executor:
            future_to_idx = {
                executor.submit(process_item, data): i 
                for i, data in enumerate(data_to_process)
            }
            with tqdm(total=len(data_to_process), desc='qwen-max scoring...', ncols=100, leave=False) as pbar:
                for future in concurrent.futures.as_completed(future_to_idx):
                    try:
                        result = future.result()
                        g.writelines(json.dumps(result, ensure_ascii=False) + '\n')
                        pbar.update(1)
                    except Exception as e:
                        logging.error(f"Error processing item: {e}")
                        pbar.update(1)


    single_thread = '''\
    with open(args.output_path, 'a', buffering=1, encoding='utf-8') as g:
        for i, data in enumerate(tqdm(data_, desc='qwen-max scoring...', ncols=50)):
            
            if i < num_lines_finished:
                continue

            new_data = data.copy()

            query = data.get('question') or data.get('prompt')
            answers = data.get('answers')
            pred = data.get('response_answer')

            qwen_eval = prompt_qwen_single(query, answers, pred)

            new_data['qwen_max_score'] = qwen_eval
            g.writelines(json.dumps(new_data, ensure_ascii=False)+'\n')
'''

    # print average score; write to txt
    with open(args.output_path, 'r', encoding='utf-8') as f, open(args.output_path.replace('.jsonl', '.txt'), 'w', encoding='utf-8') as f_txt:
        data_ = load_jsonl(args.output_path)
        correct_num = sum([is_correct(data['qwen_max_score'], style=args.style) for data in data_])
        print(f'qwen-max score: {correct_num/len(data_):.4f}')
        
        f_txt.write(f'qwen-max score: {correct_num/len(data_):.4f}\n')
        f_txt.write(f'# samples : {len(data_)}\n')