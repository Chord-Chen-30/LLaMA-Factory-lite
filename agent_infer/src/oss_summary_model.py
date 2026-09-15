import jinja2
import datetime
import json
import aiohttp
import asyncio
import os
import argparse
import random

# use_eas = os.getenv("USE_EAS_OSS_SERVICE", "false") == "true"

url_eas = os.getenv("EAS_COMPLETIONS_URL", "")
api_key = os.getenv("EAS_API_KEY", "")

pairs = [
    # ["http://localhost:6002/v1/completions", "rand"],
    [url_eas, api_key]
]

def strftime_now_function(fmt):
    return datetime.datetime.now().strftime(fmt)

async def get_stream_response(payload, print_stream=True):
    """Get streaming response from the LLM API using aiohttp"""
    full_response = ""
    pair = random.choice(pairs)
    url = pair[0]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {pair[1]}"}
    
    timeout = aiohttp.ClientTimeout(total=300)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, data=json.dumps(payload)) as response:
            response.raise_for_status()
            
            async for chunk in response.content.iter_any():
                if chunk:
                    decoded_line = chunk.decode('utf-8')
                    
                    # Handle multiple lines in a single chunk
                    for line in decoded_line.strip().split('\n'):
                        if line.startswith('data: '):
                            json_str = line[6:].strip()
                        elif line.startswith('data:'):
                            json_str = line[5:].strip()
                        else:
                            continue

                        if not json_str or json_str == '[DONE]':
                            continue

                        try:
                            data = json.loads(json_str)
                            text_chunk = data['choices'][0]['text']
                            full_response += text_chunk
                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                            print(e)
                            pass

    return full_response

async def get_llm_response(question, 
                          template_file_path=os.getenv("EAS_CHAT_TEMPLATE", "chat_template.jinja"),
                          developer_prompt="You are a helpful assistant.",
                          max_tokens=32768,
                          reasoning_effort="low",
                          stream=True):
    
    # Setup headers
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer"}
    
    # Load and setup Jinja template
    with open(template_file_path, 'r', encoding='utf-8') as f:
        template_string = f.read()
    
    env = jinja2.Environment()
    env.globals['strftime_now'] = strftime_now_function
    env.filters['tojson'] = json.dumps
    template = env.from_string(template_string)
    
    # Prepare messages
    messages = [
        {"role": "developer", "content": developer_prompt},
        {"role": "user", "content": question}
    ]
    
    # Render prompt (no tools)
    prompt = template.render(
        messages=messages, 
        reasoning_effort=reasoning_effort, 
        tools=[],
        add_generation_prompt=True
    )
    
    # Prepare payload
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": stream,
        "skip_special_tokens": False,
    }
    
    # Get response
    try:
        response = await get_stream_response(payload, print_stream=stream)
        print("Obtained valid response")
    except Exception as e:
        response = f"Error getting visit response: {e}"
    return response.strip()

async def _process_single_question(question_id, question, semaphore, **kwargs):
    """Helper function to process a single question with rate limiting"""
    async with semaphore:  # Limit concurrent requests
        try:
            response = await get_llm_response(question, **kwargs)
            return question_id, response
        except Exception as e:
            return question_id, f"Error processing question {question_id}: {e}"

async def get_batch_llm_responses(questions_dict, max_concurrent=10, **kwargs):
    """
    Send batched requests using asyncio
    
    Args:
        questions_dict (dict): Dictionary with keys as IDs and values as questions
        max_concurrent (int): Maximum number of concurrent requests (default: 10)
        **kwargs: Additional arguments to pass to get_llm_response
    
    Returns:
        dict: Dictionary with same keys as input, values are LLM responses
    """
    if not questions_dict:
        return {}
    
    print("Sending requests to batched processor")
    
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for all questions
    tasks = [
        _process_single_question(qid, question, semaphore, **kwargs)
        for qid, question in questions_dict.items()
    ]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Convert results back to dictionary
    return {question_id: response for question_id, response in results}

# Example usage functions for backward compatibility
def run_single_question(question, **kwargs):
    """Wrapper to run a single question synchronously"""
    return asyncio.run(get_llm_response(question, **kwargs))

def run_batch_questions(questions_dict, **kwargs):
    """Wrapper to run batch questions synchronously"""
    return asyncio.run(get_batch_llm_responses(questions_dict, **kwargs))