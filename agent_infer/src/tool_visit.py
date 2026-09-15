# 目前支持：单条、多条url，仅通过jina。
# 不支持：本地起summary model +vllm 调用模型；访问online server （代码框架均有，需要修改细节） cz

import json
import os
import threading
from typing import List, Union
import requests
import os 
from openai import OpenAI
import random
# from concurrent.futures import ThreadPoolExecutor, as_completed

# from topsdk.client import TopApiClient,TopException
# from topsdk.defaultability.defaultability import Defaultability
# from topsdk.defaultability.request.alibaba_aidata_aignite_application_run_request import AlibabaAidataAigniteApplicationRunAigniteApplicationExecuteReqDTO,AlibabaAidataAigniteApplicationRunRequest
# from pdf_parser import download_pdf, parse_pdf

from urllib.parse import urlparse, unquote
import time 
import random
from transformers import AutoTokenizer
from dotenv import load_dotenv

from oss_summary_model import *
from prompt import EXTRACTOR_PROMPT
import logging
logger = logging.getLogger(__name__)


VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))

JINA_READER_URL_PREFIX = "https://r.jina.ai/"

load_dotenv()
JINA_API_KEYS = [
    os.getenv("JINA_KEY1"), # from zhangzhen 2025.11.3
]

_visit_tokenizer = None

def _get_visit_tokenizer():
    global _visit_tokenizer
    if _visit_tokenizer is not None:
        return _visit_tokenizer
    model_path = "gpt-oss-120b"
    try:
        _visit_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"[visit] Failed to load tokenizer '{model_path}': {e}")
        _visit_tokenizer = None
    return _visit_tokenizer

def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    # No need for truncation if letters not even close
    if len(text) <= 100000:
        return text
    tokenizer = _get_visit_tokenizer()
    if tokenizer is None or not text:
        return text
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        token_ids = token_ids[:max_tokens]
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except Exception as e:
        print(f"[visit] Tokenization failed, falling back to raw text trim: {e}")
        return text


OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""


class Visit:
    """
    A tool to visit web pages, fetch their content, and generate a summary
    based on a specified goal using a Large Language Model.
    """
    def __init__(self):
        """Initializes the WebpageSummarizer."""
        os.makedirs("log", exist_ok=True)

        # Test
        logger.info(f"---- Testing Visit ----")
        _res = self.call({"url": ["https://arxiv.org/abs/2404.02068"], "goal": "Give 1-2 sentence TLDR of the webpage"})

        if _res.startswith("[Error]"):
            logger.error(f"--- Test Visit Failed: {_res} ---")
            exit(-1)
        else:
            logger.info(f"Visit 初始化完成，Output: {_res[:300]}...")
            logger.info("-----------------------")

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except:
            return "[Error] [Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        start_time = time.time()

        if isinstance(url, str):
            response = self.readpage(url, goal)
        else:
            response = []
            assert isinstance(url, List)
            start_time = time.time()
            for u in url: 
                if time.time() - start_time > 900:
                    cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    cur_response += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    cur_response += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                else:
                    try:
                        cur_response = self.readpage(u, goal)
                    except Exception as e:
                        cur_response = f"[Error] Error fetching {u}: {str(e)}"
                response.append(cur_response)
            response = "\n=======\n".join(response)
        
        print(f'Summary Length {len(response)}; Summary Content[:500] {response[:500]}')
        return response.strip()
        
    def call_oss(self, msgs, max_retries=2):
        question = [msg for msg in msgs if msg['role'] == 'user'][0]['content']
        start_time = time.time()

        try:
            response = run_single_question(question, reasoning_effort="medium", developer_prompt=OSS_JSON_FORMAT).split('<|message|>')[-1].strip()
        except Exception as e:
            print(e)
            response = None
        logger.info(f"Obtained gpt-oss response after {time.time() - start_time:2f} seconds")
        return response

    def call_local_server(self, msgs, max_retries=2):
        "TODO: Implement a local server to call summary. Just a template now. cz"
        # 设置 OpenAI 的 API 密钥和 API 基础 URL 使用 vLLM 的 API 服务器。
        openai_api_key = "EMPTY"
        openai_api_base = "http://127.0.0.1:6002/v1" #TODO
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        for attempt in range(max_retries):
            try:
                chat_response = client.chat.completions.create(
                    # TODO: [Important] If you change the summary model, you need to change the model path correspondingly.
                    model=os.getenv("SUMMARY_MODEL_PATH", "Qwen/Qwen2.5-72B-Instruct"),
                    messages=msgs,
                    temperature=0.7
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string 
                        left = content.find('{')
                        right = content.rfind('}') 
                        if left != -1 and right != -1 and left <= right: 
                            content = content[left:right+1]
                    return content
            except Exception as e:
                # print(e)
                if attempt == (max_retries - 1):
                    return ""
                continue
    
    def call_server(self, msgs, max_retries=2): 
        url_llm = os.getenv("VISIT_SUMMARY_URL", "")
        token = os.getenv("VISIT_SUMMARY_TOKEN", "")
        if not url_llm or not token:
            raise RuntimeError("Set VISIT_SUMMARY_URL and VISIT_SUMMARY_TOKEN")

        headers = {
            'Content-Type': 'application/json',
            'Authorization': token,
        }

        payload = json.dumps({
            "model": "Qwen3-235B-A22B-Instruct-2507",
            "messages": msgs,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 16000,
            "presence_penalty": 1.5,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        })
        
        response = None
        for attempt in range(max_retries):
            try:
                response = requests.request("POST", url_llm, headers=headers, data=payload, timeout=VISIT_SERVER_TIMEOUT)
                
                # Check if we got a valid HTTP response
                if response.status_code != 200:
                    print(f"HTTP Error: {response.status_code} - {response.text}")
                    continue
                
                data = response.json()
                
                # Check for API-level errors
                if "error" in data and data["error"]["message"] == "Provider returned error":
                    print("error in message")
                    continue
                
                # Check if we have the expected response structure
                if "choices" not in data or not data["choices"] or "message" not in data["choices"][0] or "content" not in data["choices"][0]["message"]:
                    print("Invalid response structure")
                    continue
                
                content = data["choices"][0]["message"]["content"]
                try:
                    content = json.loads(content)
                except: 
                    try:
                        left_pos, right_pos = content.index("{"), content.rindex("}")
                        if left_pos >= 0 and right_pos > left_pos:
                            content = content[left_pos:right_pos+1]
                    except ValueError:  # Catch the "substring not found" error
                        print(f"No JSON braces found in response: {content}...")
                        return ""

                # If we reach here, it's a successful response
                return content
                
            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1}")
                continue
            except Exception as e:
                print("Visit Tool Error", e)
                if response:
                    print(response.text)
                continue
        
        return ""

    def scraper_readpage(self, url: str) -> str:
        keys = [k.strip() for k in os.getenv("SCRAPERAPI_KEY", "").split(",") if k.strip()]
        if not keys:
            raise RuntimeError("Set SCRAPERAPI_KEY")
        payload_key = random.choice(keys)
        payload = {'api_key': payload_key, 
                    'url': url, 
                    'output_format': 'markdown',
                    'country_code': 'us' }
        max_retries = 2
        for attempt in range(max_retries):
            try:
                r = requests.get('https://api.scraperapi.com/', params=payload, timeout=30)
                content = r.text
                return content
            except requests.exceptions.Timeout:
                # 超时情况下返回默认内容
                content = "[visit] Failed to read page."
            except Exception as e:
                content ="[visit] Failed to read page."
        return content

    def jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina service.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        max_retries = 10
        timeout = 50
        
        for attempt in range(max_retries):
            headers = {
                "Authorization": f"Bearer {random.choice(JINA_API_KEYS)}",
            }
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=timeout
                )
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content
                else:
                    print(f"response.status_code = {response.status_code}\n", response.text)
                    raise ValueError("jina readpage error")
            except Exception as e:
                time.sleep(0.5)
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."
                
            time.sleep(0.5)
                
        return "[visit] Failed to read page."


    def html_readpage(self, url: str) -> str:

        content = self.jina_readpage(url)
        service = "jina"
        logger.info(f"Visit service: {service}. URL: {url}")

        # if ('wikipedia.org' in url or 'wikipedia.com' in url) and attempt <= 0:
        #     service = 'self-wiki'
        #     content = self.query_wiki_dict_service(url)
        # elif attempt <= 2:
        #     service = "aidata-cache"
        #     content = self.aidata_readpage(url, only_cache=True)
        # elif attempt <= 5:
        #     service = "scraper"
        #     content = self.scraper_readpage(url)
        # else: 
        #     content = self.jina_readpage(url)
        #     service = "jina"

        if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
            if isinstance(content, str):
                logger.info(f"Visit SUCCESS (str): {content[:100]}")
            else:
                logger.info(f"Visit SUCCESS ({type(content)}): {cotent}")
            return content

        return "[visit] Failed to read page."

    def readpage(self, url: str, goal: str) -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        # use_local_summary_model = int(os.getenv('USE_LOCAL_SUMMARY_MODEL', 0))
        # if use_local_summary_model:
        #     summary_page_func = self.call_local_server
        # else:
        #     summary_page_func = self.call_server
        summary_page_func = self.call_oss
        
        max_retries = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 1))
        content = None
        sevice = None
        
        content = self.html_readpage(url)
        if (not content) or (content.startswith("[visit] Failed")) or (content == "[visit] Empty content.") or (content.startswith("[document_parser]")):
            ret = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            ret += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            ret += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            print("Readpage Result:", content)
            return ret
        
        # Truncate by transformer tokenizer to 95,000 tokens
        content = truncate_to_tokens(content, max_tokens=95000)

        raw_summary = None
        for i in range(4): # 1 initial attempt + 3 retries with truncation
            messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
            raw_summary = summary_page_func(messages, max_retries=1)

            if raw_summary and len(raw_summary) > 10:
                break # Successful summary, exit loop

            if i < 3:
                new_len = int(0.7 * len(content))
                print(f"[Summary Retry {i+1}/3] Summary empty. Truncating content from {len(content)} to {new_len} chars.")
                content = content[:new_len]
            else:
                print("[Summary] Could not generate a valid summary after multiple retries.")
                print("Conent:", content)
                return "[Error] [Summary] Could not generate a valid summary after multiple retries."

        parsed_summary = None
        if raw_summary:
            try:
                parsed_summary = json.loads(raw_summary)
            except json.JSONDecodeError:
                print(f"[Summary Parse] Failed to parse JSON from summary response: {raw_summary}")
        
        if parsed_summary and isinstance(parsed_summary, dict) and "evidence" in parsed_summary and "summary" in parsed_summary:
            useful_information = f"The useful information in {url} for user goal {goal} as follows: \n\n"
            useful_information += "Evidence in page: \n" + str(parsed_summary.get("evidence", "N/A")) + "\n\n"
            useful_information += "Summary: \n" + str(parsed_summary.get("summary", "N/A")) + "\n\n"
            return useful_information
        else:
            return "[Error] The summary model failed to extract relevant information in the expected format."


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    visit = Visit()
    result = visit.call(
        {
            # "url": ["https://muse.jhu.edu/pub/3/article/240795/pdf"],
            # "url": ["https://arxiv.org/abs/2404.02068"],
            "url": ["https://arxiv.org/pdf/2511.21631"],
            "goal": "Give 1-2 sentence TLDR of the webpage"
        }
    )

    print(result)


    # base_urls = [
    #     'https://arxiv.org/pdf/2505.16700', 
    #     'https://github.com/TZWwww?tab=repositories', 
    #     'https://www.json.cn/', 
    #     'https://www.quiverquant.com/congresstrading/politician/Nancy%20Pelosi-P000197'
    # ]
    # result = visit.call({
    #     "url": base_urls,
    #     "goal": "Give 1-2 sentence TLDR of the webpage"
    # })

    # print(result)