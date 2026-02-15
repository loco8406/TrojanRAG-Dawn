# filepath: /rds/project/rds-kqoCForZB1s/TrojanRAG-Dawn/evaluation/LLMs/GPT.py
import os
from openai import OpenAI
from LLMs.Model import Model


class GPT(Model):
    def __init__(self, config):
        super().__init__(config)
        
        # Try environment variable first, then fall back to config
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            # Fall back to config file if env var not set
            api_keys = config["api_key_info"]["api_keys"]
            api_pos = int(config["api_key_info"]["api_key_use"])
            assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
            api_key = api_keys[api_pos]
        
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        
        if 'base_url' in config:
            self.base_url = config["base_url"]
            self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def query(self, msg, sys_prompt="You are a helpful assistant."):
        try:
            completion = self.client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": msg}
                ],
            )
            response = completion.choices[0].message.content

        except Exception as e:
            print(e)
            response = ""

        return response
