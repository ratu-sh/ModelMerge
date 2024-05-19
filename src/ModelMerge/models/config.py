import os
import json
import httpx
import requests
from pathlib import Path
from collections import defaultdict


from ..utils import prompt

PLUGINS = {
    "SEARCH": (os.environ.get('SEARCH', "True") == "False") == False,
    "URL": True,
    "CODE": True,
    "IMAGE": (os.environ.get('IMAGE', "False") == "False") == False,
    "DATE": False,
    "VERSION": False,
    "TARVEL": (os.environ.get('TARVEL', "False") == "False") == False,
}

LANGUAGE = os.environ.get('LANGUAGE', 'Simplified Chinese')

class BaseAPI:
    def __init__(
        self,
        api_url: str = (os.environ.get("API_URL", None) or "https://api.openai.com/v1/chat/completions"),
    ):
        from urllib.parse import urlparse, urlunparse
        if api_url is None:
            raise Exception("API_URL is not set")
        self.source_api_url: str = api_url
        parsed_url = urlparse(self.source_api_url)
        if parsed_url.path != '/':
            before_v1 = parsed_url.path.split("/v1")[0]
        else:
            before_v1 = ""
        self.base_url: str = urlunparse(parsed_url[:2] + (before_v1,) + ("",) * 3)
        self.v1_url: str = urlunparse(parsed_url[:2]+ (before_v1 + "/v1",) + ("",) * 3)
        self.chat_url: str = urlunparse(parsed_url[:2] + (before_v1 + "/v1/chat/completions",) + ("",) * 3)
        self.image_url: str = urlunparse(parsed_url[:2] + (before_v1 + "/v1/images/generations",) + ("",) * 3)

class BaseLLM:
    def __init__(
        self,
        api_key: str,
        engine: str = os.environ.get("GPT_ENGINE") or "gpt-3.5-turbo",
        api_url: str = (os.environ.get("API_URL", None) or "https://api.openai.com/v1/chat/completions"),
        system_prompt: str = prompt.chatgpt_system_prompt,
        proxy: str = None,
        timeout: float = 600,
        max_tokens: int = None,
        temperature: float = 0.5,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        reply_count: int = 1,
        truncate_limit: int = None,
    ) -> None:
        self.api_key: str = api_key
        self.engine: str = engine
        self.api_url: str = BaseAPI(api_url or "https://api.openai.com/v1/chat/completions")
        self.system_prompt: str = system_prompt
        self.max_tokens: int = max_tokens
        self.truncate_limit: int = truncate_limit
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.presence_penalty: float = presence_penalty
        self.frequency_penalty: float = frequency_penalty
        self.reply_count: int = reply_count
        self.timeout: float = timeout
        self.proxy = proxy
        self.session = requests.Session()
        self.session.proxies.update(
            {
                "http": proxy,
                "https": proxy,
            },
        )
        if proxy := (
            proxy or os.environ.get("all_proxy") or os.environ.get("ALL_PROXY") or None
        ):
            if "socks5h" not in proxy:
                self.aclient = httpx.AsyncClient(
                    follow_redirects=True,
                    proxies=proxy,
                    timeout=timeout,
                )
        else:
            self.aclient = httpx.AsyncClient(
                follow_redirects=True,
                proxies=proxy,
                timeout=timeout,
            )

        self.conversation: dict[str, list[dict]] = {
            "default": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ],
        }
        self.tokens_usage = defaultdict(int)
        self.function_calls_counter = {}
        self.function_call_max_loop = 10

    def add_to_conversation(
        self,
        message: list,
        role: str,
        convo_id: str = "default",
        function_name: str = "",
    ) -> None:
        """
        Add a message to the conversation
        """
        pass

    def __truncate_conversation(self, convo_id: str = "default") -> None:
        """
        Truncate the conversation
        """
        pass

    def truncate_conversation(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = None,
        pass_history: bool = True,
        **kwargs,
    ) -> None:
        """
        Truncate the conversation
        """
        pass

    def extract_values(self, obj):
        pass

    def get_token_count(self, convo_id: str = "default") -> int:
        """
        Get token count
        """
        pass

    def get_message_token(self, url, json_post):
        pass

    def get_post_body(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = None,
        pass_history: bool = True,
        **kwargs,
    ):
        pass

    def get_max_tokens(self, convo_id: str) -> int:
        """
        Get max tokens
        """
        pass

    def ask_stream(
        self,
        prompt: list,
        role: str = "user",
        convo_id: str = "default",
        model: str = None,
        pass_history: bool = True,
        function_name: str = "",
        **kwargs,
    ):
        """
        Ask a question
        """
        response = self.session.post(
            self.api_url.chat_url,
            headers={"Authorization": f"Bearer {kwargs.get('api_key', self.api_key)}"},
            json={
                "model": model or self.engine,
                "messages": [{"role": "system","content": self.system_prompt},{"role": role, "content": prompt}],
                "stream": True,
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "presence_penalty": kwargs.get(
                    "presence_penalty",
                    self.presence_penalty,
                ),
                "frequency_penalty": kwargs.get(
                    "frequency_penalty",
                    self.frequency_penalty,
                ),
                "n": kwargs.get("n", self.reply_count),
                "user": role,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            },
            timeout=kwargs.get("timeout", self.timeout),
            stream=True,
        )
        if response.status_code != 200:
            raise Exception(f"{response.status_code} {response.reason} {response.text}")
        full_response: str = ""
        for line in response.iter_lines():
            line = line.strip()
            if not line:
                continue
            line = line.decode("utf-8")[6:]
            # print("line", line)
            if line == "[DONE]":
                break
            resp: dict = json.loads(line)
            if "error" in resp:
                raise Exception(f"{resp['error']}")
            choices = resp.get("choices")
            if not choices:
                continue
            delta: dict[str, str] = choices[0].get("delta")
            if not delta:
                continue
            if "content" in delta:
                content: str = delta["content"]
                full_response += content
                yield content

    async def ask_async(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = None,
        pass_history: bool = True,
        **kwargs,
    ) -> str:
        """
        Non-streaming ask
        """
        pass

    def ask(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = None,
        pass_history: bool = False,
        **kwargs,
    ) -> str:
        """
        Non-streaming ask
        """
        response = self.ask_stream(
            prompt=prompt,
            role=role,
            convo_id=convo_id,
            model=model,
            pass_history=pass_history,
            **kwargs,
        )
        full_response: str = "".join(response)
        return full_response

    def rollback(self, n: int = 1, convo_id: str = "default") -> None:
        """
        Rollback the conversation
        """
        for _ in range(n):
            self.conversation[convo_id].pop()

    def reset(self, convo_id: str = "default", system_prompt: str = None) -> None:
        """
        Reset the conversation
        """
        self.conversation[convo_id] = [
            {"role": "system", "content": system_prompt or self.system_prompt},
        ]

    def save(self, file: str, *keys: str) -> None:
        """
        Save the Chatbot configuration to a JSON file
        """
        pass

    def load(self, file: Path, *keys_: str) -> None:
        """
        Load the Chatbot configuration from a JSON file
        """
        pass