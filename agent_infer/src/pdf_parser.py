from alibabacloud_docmind_api20220711.client import Client as docmind_api20220711Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_docmind_api20220711 import models as docmind_api20220711_models
from alibabacloud_tea_util.client import Client as UtilClient
from alibabacloud_tea_util import models as util_models
from alibabacloud_credentials.client import Client as CredClient
# import subprocess
import os
from datetime import datetime
import json
import base64
import time
import uuid
import requests
import aiohttp
import asyncio
import aiofiles
from io import BytesIO


def _require_env(name: str) -> str:
    val = os.getenv(name, "")
    if not val:
        raise RuntimeError(f"{name} is not set")
    return val


def _docmind_client():
    config = open_api_models.Config(
        access_key_id=_require_env("ALIBABA_CLOUD_ACCESS_KEY_ID"),
        access_key_secret=_require_env("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
    )
    config.endpoint = "docmind-api.cn-hangzhou.aliyuncs.com"
    return docmind_api20220711Client(config)


def parse_pdf(pdf_path):
    client = _docmind_client()
    request = docmind_api20220711_models.SubmitDocParserJobAdvanceRequest(
        file_url_object=open(pdf_path, "rb"),
        file_name=pdf_path.split('/')[-1],
        file_name_extension='pdf',
        formula_enhancement=True
    )
    runtime = util_models.RuntimeOptions()
    try:
        response = client.submit_doc_parser_job_advance(request, runtime)
    except Exception as error:
        UtilClient.assert_as_string(error.message)
        print("*************************")
        print(error)
        print("*************************")

    jobid = response.body.data.id
    
    status = 'processing'
    process_pagenum = 0
    time_start  = time.time()
    while(status != 'success' and process_pagenum < 100):
        request = docmind_api20220711_models.QueryDocParserStatusRequest(
            id=jobid
        )
        try:
            response = client.query_doc_parser_status(request)
            status = response.body.data.status

        except Exception as error:
            UtilClient.assert_as_string(error.message)  

        request = docmind_api20220711_models.GetDocParserResultRequest(
            id=jobid,
            layout_step_size=3000,
            layout_num=0
        )
        try:
            response = client.get_doc_parser_result(request)
            if len(response.body.data['layouts']) > 0:
                process_pagenum = response.body.data['layouts'][-1]['pageNum']

        except Exception as error:
            print(error)
            raise error

    time_end  = time.time()
    total_time = time_end - time_start
    return '\n'.join([elem['markdownContent'] for elem in response.body.data['layouts']])


async def parse_pdf_async(pdf_path):
    client = _docmind_client()

    async with aiofiles.open(pdf_path, "rb") as file:
        file_url_object = await file.read()
    file_url_object = BytesIO(file_url_object)

    request = docmind_api20220711_models.SubmitDocParserJobAdvanceRequest(
        file_url_object=file_url_object,
        file_name=pdf_path.split('/')[-1],
        file_name_extension='pdf',
        formula_enhancement=True
    )

    runtime = util_models.RuntimeOptions()

    try:
        response = await asyncio.to_thread(client.submit_doc_parser_job_advance, request, runtime)
    except Exception as error:
        UtilClient.assert_as_string(error.message)

    jobid = response.body.data.id

    status = 'processing'
    process_pagenum = 0
    time_start = time.time()

    while status != 'success' and process_pagenum < 100:
        request = docmind_api20220711_models.QueryDocParserStatusRequest(id=jobid)
        try:
            response = await asyncio.to_thread(client.query_doc_parser_status, request)
            status = response.body.data.status
        except Exception as error:
            UtilClient.assert_as_string(error.message)

        request = docmind_api20220711_models.GetDocParserResultRequest(
            id=jobid,
            layout_step_size=3000,
            layout_num=0
        )

        try:
            response = await asyncio.to_thread(client.get_doc_parser_result, request)
            if len(response.body.data['layouts']) > 0:
                process_pagenum = response.body.data['layouts'][-1]['pageNum']
        except Exception as error:
            print(error)
            raise error

    time_end = time.time()
    total_time = time_end - time_start
    return '\n'.join(elem['markdownContent'] for elem in response.body.data['layouts'])


proxy_num = 0
scriper_num = 0
jina_num = 0

def download_pdf(url, timeout=60):
    global proxy_num, scriper_num, jina_num
    output_dir = "pdf"
    os.makedirs(output_dir, exist_ok=True)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"document_{current_time}_{str(uuid.uuid4())}.pdf"
    file_path = os.path.join(output_dir, filename)
    
    max_retries = 3
    def try_scraper_download():
        scraper_payload = {'api_key': _require_env("SCRAPERAPI_KEY"), 'url': url}
        
        for attempt in range(max_retries):
            try:
                response = requests.get('https://api.scraperapi.com/', params=scraper_payload, timeout=60)
                response.raise_for_status()

                with open(file_path, 'wb') as f:
                    f.write(response.content)

                return file_path

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"ScraperAPI download failed: {str(e)}")
                    return None
                time.sleep(1)

    result = try_scraper_download()
    if result:
        scriper_num += 1
        print(f"proxy_num | scriper_num | jina_num: {proxy_num} | {scriper_num} | {jina_num}")
        return result

    jina_num += 1
    print(f"proxy_num | scriper_num | jina_num: {proxy_num} | {scriper_num} | {jina_num}")
    return None


async def download_pdf_async(url, timeout=60):
    global proxy_num, scriper_num, jina_num
    output_dir = "pdf"
    os.makedirs(output_dir, exist_ok=True)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"document_{current_time}_{str(uuid.uuid4())}.pdf"
    file_path = os.path.join(output_dir, filename)
    file_path = os.path.abspath(file_path)
    
    max_retries = 100

    async def try_scraper_download():
        scraper_payload = {'api_key': _require_env("SCRAPERAPI_KEY"), 'url': url}
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_retries):
                try:
                    async with session.get('https://api.scraperapi.com/', params=scraper_payload, timeout=timeout) as response:
                        response.raise_for_status()
                        content = await response.read()

                        with open(file_path, 'wb') as f:
                            f.write(content)

                        return file_path

                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"ScraperAPI download failed: {str(e)}")
                        return None
                    await asyncio.sleep(1)  # Use asyncio.sleep for non-blocking sleep

    result = await try_scraper_download()
    if result:
        scriper_num += 1
        print(f"proxy_num | scriper_num | jina_num: {proxy_num} | {scriper_num} | {jina_num}")
        return result

    jina_num += 1
    print(f"proxy_num | scriper_num | jina_num: {proxy_num} | {scriper_num} | {jina_num}")
    return None


async def google_scholar(url):
    path = await download_pdf_async(url)
    doc = await parse_pdf_async(path)
    return doc


if __name__ == "__main__":
    async def main():
        results = await google_scholar('https://arxiv.org/pdf/1706.03762')
        print(results)

    asyncio.run(main())