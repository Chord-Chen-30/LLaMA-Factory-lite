"""Tool-call wrapper: dispatches legacy prompt_ins3 tool names to the new tools.py impls.

prompt_ins3 describes these tools to the model:
    web_search, image_search, visit, code_interpreter

image_search routes to tools.image_search (Google Lens via Serper, local paths
auto-uploaded to imgbb and cached in image_url_map.json).
"""
import json
import logging
import re
import time

from tools import search, visit_url, run_python, image_search

logger = logging.getLogger(__name__)


def _extract_wrapped(s: str):
    m = re.search(r"<(tools|tool_call)>(.*?)</\1>", s, re.DOTALL)
    return m.group(2) if m else None


class ToolCallWrapper:
    def call_tools(self, params):
        start = time.time()

        if isinstance(params, str):
            try:
                inner = _extract_wrapped(params)
                params = json.loads(inner)
            except json.JSONDecodeError:
                return "Invalid JSON"
            except Exception as e:
                logger.error(f"Error processing <tool_call> content: {e}")
                return "Error processing <tool_call> content"

        tool_name = (params.get("name") or params.get("tool_name") or "").strip()
        tool_params = params.get("arguments") or params.get("parameters") or {}

        if not tool_name:
            return "\"name\" field is required. Specify a tool name"

        name = tool_name.lower()
        try:
            if name in ("web_search", "search"):
                query = (
                    tool_params.get("queries")
                    or tool_params.get("query")
                    or tool_params.get("q")
                )
                if not query:
                    ret = "[web_search] Missing 'queries' argument."
                else:
                    num = tool_params.get("num", 10)
                    ret = search(query, num=num)

            elif name in ("visit", "visit_url", "open_url"):
                url = tool_params.get("url") or tool_params.get("urls")
                goal = (
                    tool_params.get("goal")
                    or tool_params.get("info_to_extract")
                    or tool_params.get("description")
                    or ""
                )
                if not url:
                    ret = "[visit] Missing 'url' argument."
                else:
                    ret = visit_url(url, info_to_extract=goal)

            elif name in ("code_interpreter", "run_python", "pythoninterpreter", "run_python_code"):
                code = tool_params.get("code") or tool_params.get("code_block")
                if not code:
                    ret = "[code_interpreter] Missing 'code' argument."
                else:
                    ret = run_python(code)

            elif name in ("image_search", "vlsearchimage"):
                images = (
                    tool_params.get("images")
                    or tool_params.get("image_urls")
                    or tool_params.get("urls")
                    or tool_params.get("image")
                    or tool_params.get("image_url")
                    or tool_params.get("url")
                )
                if not images:
                    ret = "[image_search] Missing 'images' argument."
                else:
                    ret = image_search(images)

            else:
                ret = (
                    f"Invalid tool name {tool_name!r}. "
                    "Available tools: web_search, visit, code_interpreter, image_search."
                )
        except Exception as e:
            logger.exception("call_tools failed")
            ret = f"[{tool_name}] tool execution error: {e}"

        if not isinstance(ret, str):
            ret = json.dumps(ret, ensure_ascii=False)

        logger.info(f"call_tools({tool_name}) took {time.time()-start:.2f}s; ret[:100]={ret[:100]!r}")
        return ret


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    wrapper = ToolCallWrapper()
    print(wrapper.call_tools({"name": "web_search", "arguments": {"queries": ["Who is Chord Chen?"]}})[:500])
    print("---")
    print(wrapper.call_tools({"name": "code_interpreter", "arguments": {"code": "print(2+2)"}}))
    print("---")
    # image_search with a local path (mimicking how infer_w_tools.py passes file:// URLs back from the model)
    print(wrapper.call_tools({
        "name": "image_search",
        "arguments": {"images": ["file://./agent_infer_sg/data/livevqa_files/0a5o7ZLh8b.jpeg"]},
    })[:800])
