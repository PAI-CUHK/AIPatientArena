import json
import os
import re
import time

import requests
from openai import OpenAI


class LLM_Models:
    def __init__(self):
        """
        Initialize the processor using configuration from the config file.
        """
        self.max_tokens = 4086
        self.temperature = 1.0
        self.secrete_file = "../secrets.txt"
        # Load OpenAI API key and base_url from the secret file
        self.open_ai_key, self.base_url = self._load_openai_client(self.secrete_file)

    def _load_openai_client(self, secret_file):
        """
        Load the OpenAI client using the API key from the secret file.
        :param secret_file: Path to the file containing the OpenAI API key.
        :return: A tuple of (open_ai_key, base_url)
        """
        open_ai_key = None
        base_url = None  # 默认值
        
        with open(secret_file) as f:
            lines = f.readlines()
            for line in lines:
                if line.split(',')[0].strip() == "open_ai_key":
                    open_ai_key = line.split(',')[1].strip()
                elif line.split(',')[0].strip() == "base_url":
                    base_url = line.split(',')[1].strip()

        return open_ai_key, base_url

    def run_gpt(self, messages, model='gpt-5', retry_interval=10):
        """
        调用 GPT API
        :param messages: 消息列表
        :param model: 模型名称
        :param retry_interval: 失败后的等待时间（秒）
        :return: 成功的 response 内容
        """
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.open_ai_key
        )

        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        # 成功返回
        return response.choices[0].message.content

    def run_vllm(self, messages=None, temperature=0.7, max_tokens=5000,
                 vllm_port=8000, vllm_model="mistralai/Mistral-7B-Instruct-v0.3"):
        """
        调用本地 vLLM (OpenAI Chat API 格式)
        """
        VLLM_BASE_URL = f"http://localhost:{vllm_port}"
        chat_url = f"{VLLM_BASE_URL}/v1/chat/completions"

        model_name = os.path.join("../huggingface", vllm_model)

        data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        }

        response = requests.post(chat_url, json=data, timeout=120)
        response.raise_for_status()
        response_data = response.json()

        response_text = response_data["choices"][0]["message"]["content"]
        print(response_text)
        cleaned_output = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

        return cleaned_output