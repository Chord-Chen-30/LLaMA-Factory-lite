import json
import os
import requests
from dotenv import load_dotenv
import time
from typing import List, Union
import logging

logger = logging.getLogger(__name__)
load_dotenv()

# --- 配置常量 ---
# 注意：这是一个内部API端点，在外部网络可能无法访问。
# 您可能需要将其替换为您自己的搜索服务API。
SEARCH_API_URL = os.getenv("SEARCH_API_URL", "")
GOOGLE_SEARCH_KEY = os.getenv('GOOGLE_SEARCH_KEY') # 注意：您需要提供一个有效的API密钥 -> .env


class SearchTool:
    """
    一个独立的网页搜索工具，可以执行单个或批量的搜索查询。
    """
    name = "search"
    description = "执行网页搜索。可以提供单个查询字符串或一个查询字符串数组。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "需要搜索的查询词数组。"
            },
        },
        "required": ["query"],
    }

    def __init__(self):
        """初始化工具并重置统计信息。"""
        self.reset_stats()

        # Test
        logger.info("--- Testing WebSearch ---")
        _res = self.call({"query": "When is 2026 Chinese New Year?"})
        if _res.startswith("[Error]"):
            logger.error(f"--- Test WebSearch Failed: {_res} ---")
            exit(-1)
        else:
            logger.info(f"WebSearch 初始化完成. Output:\n{_res[:300]}...")
            logger.info("-------------------------")
    def get_stats(self) -> dict:
        """获取工具运行的统计信息。"""
        return {
            "total_requests": self.total_requests,
            "api_calls": self.api_calls,
        }

    def reset_stats(self):
        """重置统计信息。"""
        self.total_requests = 0
        self.api_calls = 0

    def _google_search_api(self, query: str, total: int=10) -> str:
        """
        调用搜索API执行搜索的核心函数。

        该函数包含：
        1. 根据查询语言（中文/英文）选择不同的请求参数。
        2. 带超时的重试机制。
        3. 解析和格式化返回的JSON结果。
        """
        self.api_calls += 1

        def contains_chinese(text: str) -> bool:
            return any('\u4E00' <= char <= '\u9FFF' for char in text)

        if contains_chinese(query):
            headers = {'X-AK': GOOGLE_SEARCH_KEY, 'Content-Type': 'application/json', 'Accept-Language': 'zh-CN,zh;q=0.9'}
            data = {"query": query, "num": 10, "extendParams": {"hl": "zh-CN", "gl": "CN", "lr": "lang_zh-CN", "page": 1}, "platformInput": {"model": "google-search"}}
        else:
            headers = {'X-AK': GOOGLE_SEARCH_KEY, 'Content-Type': 'application/json'}
            data = {"query": query, "num": 10, "extendParams": {"country": "en", "page": 1}, "platformInput": {"model": "google-search"}}

        # 重试 total 次
        for i in range(total):
            try:
                response = requests.post(SEARCH_API_URL, headers=headers, data=json.dumps(data), timeout=30)
                if response.status_code == 200:
                    results = response.json()

                    # 解析结果
                    if "data" not in results:
                        return f"[Error] query: {query}.\n - \"data\" key not found in the results: {results}"
                    if results.get("data") is None:
                        return f"[Error] query: {query}.\n - \"data\" key is None in the results: {results}"
                    if "originalOutput" not in results.get("data", {}):
                        return f"[Error] query: {query}.\n - \"originalOutput\" key not found in the results[\"data\"]: {results}"
                    if "organic" not in results.get("data", {}).get("originalOutput", {}):
                        return f"[Error] query: {query}.\n - \"organic\" key not found in the results[\"data\"][\"originalOutput\"]: {results}"

                    # Normal case
                    web_snippets = []
                    for idx, page in enumerate(results["data"]["originalOutput"]["organic"]):
                        date_published = f"\nDate published: {page['date']}" if "date" in page else ""
                        source = f"\nSource: {page['source']}" if "source" in page else ""
                        snippet = f"\n{page.get('snippet', '')}"
                        redacted_version = f"{idx + 1}. [{page.get('title', 'No Title')}]({page.get('link', '#')}){date_published}{source}{snippet}"
                        redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                        web_snippets.append(redacted_version)
                    
                    if not web_snippets:
                        return f"No results found for '{query}'. Try with a more general query."

                    return f"A search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)

                else:
                    error_message = f"[Error] API Error for query '{query}': Status {response.status_code} - {response.text}"
                    logger.error(error_message)
                    if i == (total-1): return error_message
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for query '{query}' (Attempt {i+1}/{total}): {e}")
                if i == (total-1): return f"[Error] Search request timeout for query '{query}' after {total} attempts. Please try again later."

            time.sleep(1)
        return f"[Error] Failed to get a result for '{query}' after all retries."

    def _search_single_query(self, query: str) -> str:
        """
        执行单次查询。
        流程: 增加请求计数 -> 调用API -> 返回结果
        """
        self.total_requests += 1
        # print(f"  - Calling API for query: '{query}'...")
        return self._google_search_api(query)

    def call(self, params: dict) -> str:
        """
        工具的公共入口方法。
        :param params: 一个字典，必须包含 'query' 键。
                       'query' 的值可以是单个字符串或一个字符串列表。
        :return: 格式化后的搜索结果字符串。
        """
        if GOOGLE_SEARCH_KEY == 'YOUR_API_KEY_HERE':
            return "[Error] Please set the GOOGLE_SEARCH_KEY variable in the script."
        
        try:
            query = params.get("query", False) or params.get("queries", False)
            assert query != False
        except (TypeError, KeyError):
            return "[Error] Invalid input format. Input must be a dictionary like {'query': 'search term'} or {'query': ['term1', 'term2']}."

        if isinstance(query, str):
            # 单个查询
            return self._search_single_query(query)
        elif isinstance(query, list):
            # 批量查询
            responses = [self._search_single_query(q) for q in query]
            return "\n\n=======\n\n".join(responses)
        else:
            return "[Error] The 'query' field must be a string or a list of strings."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 实例化工具
    search_tool = SearchTool()

    print("--- 1. 测试单个查询 ---")
    single_query_params = {
        "query": "What is the latest version of Python?"
    }
    single_result = search_tool.call(single_query_params)
    print("--------------")
    print(single_result)
    print("--------------\n")

    print("--- 2. 测试批量查询 (包含中文和英文) ---")
    batch_query_params = {
        "query": [
            "欧阳修的主要文学成就",
            "Latest news about OpenAI Sora"
        ]
    }
    batch_result = search_tool.call(batch_query_params)
    print("--------------")
    print(batch_result)
    print("--------------\n")
    
    # 打印最终统计信息
    print("\n--- 最终统计信息 ---")
    print(json.dumps(search_tool.get_stats(), indent=2))
