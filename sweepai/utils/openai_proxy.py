import os

from loguru import logger
from openai import AzureOpenAI, OpenAI

from sweepai.config.server import (
    BASERUN_API_KEY,
    OPENAI_API_KEY,
    OPENAI_API_TYPE,
    OPENAI_API_VERSION,
)
from sweepai.logn.cache import file_cache
from sweepai.utils.qwen_proxy import Qwen

if BASERUN_API_KEY is not None:
    pass

OPENAI_TIMEOUT = 60  # one minute

OPENAI_EXCLUSIVE_MODELS = [
    "gpt-4-0125-preview",
    "gpt-3.5-turbo-1106",
]
SEED = 100

USE_QWEN = os.environ.get("USE_QWEN",True)


class OpenAIProxy:
    @file_cache(ignore_params=[])
    def call_openai(self, model, messages, max_tokens, temperature) -> str:
        try:
            model = "gpt-3.5-turbo"
            response = self.set_openai_default_api_parameters(
                model, messages, max_tokens, temperature
            )
            return response
            # return response.choices[0].message.content
        except SystemExit:
            raise SystemExit
        except Exception as e:
            if OPENAI_API_KEY:
                try:
                    response = self.set_openai_default_api_parameters(
                        model, messages, max_tokens, temperature
                    )
                    return response.choices[0].message.content
                except SystemExit:
                    raise SystemExit
                except Exception as _e:
                    logger.error(f"OpenAI API Key found but error: {_e}")
            logger.error(f"OpenAI API Key not found and Azure Error: {e}")
            # Raise exception to report error
            raise e

    def determine_openai_engine(self, model):
        engine = None
        if model in OPENAI_EXCLUSIVE_MODELS and OPENAI_API_TYPE != "azure":
            logger.info(f"Calling OpenAI exclusive model. {model}")
        elif (
            model == "gpt-4"
            or model == "gpt-4-0613"
            or model == "gpt-4-1106-preview"
            or model == "gpt-4-0125-preview"
        ):
            engine = model
        elif model == "gpt-4-32k" or model == "gpt-4-32k-0613":
            engine = model

        return engine

    def create_openai_chat_completion(
        self, engine, base_url, api_key, model, messages, max_tokens, temperature
    ):
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=OPENAI_API_VERSION,
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=OPENAI_TIMEOUT,
        )
        return response

    def set_openai_default_api_parameters(
        self, model, messages, max_tokens, temperature
    ):
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=OPENAI_TIMEOUT,
            seed=SEED,
        )
        logger.info("chatGpt3.5-turbo request message")
        logger.info(messages)
        logger.info("chatGpt3.5-turbo response")
        logger.info(response)

        # 修改为通义千问版本，并做比对
        if USE_QWEN:
            client = Qwen()
            return client.call_qwen(messages)