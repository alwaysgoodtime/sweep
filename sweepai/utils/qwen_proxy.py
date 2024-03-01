import requests
from loguru import logger

class Qwen():
    def call_qwen(self,message):

        server_llm_url = "http://192.168.198.33:8000/v1/chat/completions"
        payload = {
            "model": "Qwen-14B-Chat-Int4",
            "messages": message
        }
        headers = {"Content-Type": "application/json"}


        response = requests.post(server_llm_url, json=payload, headers=headers)
        logger.info("得到通义千问结果")
        # 检查响应状态
        if response.status_code == 200:
            # 成功响应
            data = response.json()
            generated_text = data.get('choices', [])[0].get('message', {}).get('content', "")
            generated_text = generated_text.strip('\n')
            generated_text = generated_text.replace("\\begin{code}", "")
            generated_text = generated_text.replace("\\end{code}", "")
            logger.info('通义千问response: ' + generated_text)
            return generated_text

