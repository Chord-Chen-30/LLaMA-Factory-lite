import json
import os
import sys
sys.path.append('./agent_infer/src')
from prompt import prompt_ins3, sys_prompt
from tqdm import tqdm

try:
    from rich import print
except ImportError:
    from pprint import pprint as print

def extract_question_image(data):
    str_ = data['messages'][0]['content']
    try:
        # 1. 以 "Input Question:" 为分隔符，取第二部分
        # 这部分会包含问题和图片链接
        temp_str = str_.split("Input Question:")[1]

        # 2. 以 "Input image:" 为分隔符，分割上一步的结果
        parts = temp_str.split("Input image:")

        # 3. 第一部分是问题，第二部分是图片链接，使用 strip() 清理空白字符
        question = parts[0].strip()
        image_url = parts[1].strip()
        return question, image_url
    except Exception as e:
        print(f"Error: {e}")
        print(str_)
        breakpoint()
        return False, False


subq_cnt = 0
think_cnt = 0
VLImageSearch_cnt = 0
multi_image_cnt = 0

with open(os.getenv("SOURCE_JSON", "data/toolcall_3more_ocr_code_6200.json")) as f, \
open('./data/toolcall_3more_ocr_code_6200_cz_cleaned_1203.json', 'w') as f_out:
    data = json.load(f)

    print(len(data))

    for d in tqdm(data):
        d['system'] = sys_prompt

        if len(d['images']) != 1:
            # print(d)
            multi_image_cnt += 1

        if d['messages'][0]['role'] != 'user':
            print(d['messages'][0])
            print('=====First message is not user!======')
            exit()
        question, image_url = extract_question_image(d)
        if not question or not image_url:
            continue

        d['messages'][0]['content'] = prompt_ins3.replace("{Question}", question).replace("{ImageUrl}", image_url)
        
        
        for m in d['messages']:
            if '<Sub-Question>' in m['content']:
                subq_cnt += 1
                # print(m['content'])
                m['content'] = m['content'].replace('<Sub-Question>', "Sub-problem:")
                # breakpoint()
            
            # if '<image>' in m['content']:
                # m['content'] = m['content'].replace('<image>', '')
            
            if '<think>' in m['content'].lower():
                think_cnt += 1
                # break
                
            # md不小心写了个bug，replace VLImageSearch, 应该是VLSearchImage
            # 在另外的jupyter中处理这个。这里留作log记录
            if '\"VLSearchImage\"' in m['content']:
                m['content'] = m['content'].replace('\"VLImageSearch\"', '\"image_search\"')
                VLImageSearch_cnt += 1
            
    print(subq_cnt)
    print(think_cnt)
    print(VLImageSearch_cnt)
    print(multi_image_cnt)

    f_out.write(json.dumps(data, indent=2, ensure_ascii=False))