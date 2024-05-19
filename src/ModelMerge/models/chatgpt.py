import os
import re
import json
import copy
from pathlib import Path
from typing import AsyncGenerator, Set
from .config import BaseLLM, PLUGINS, LANGUAGE

import httpx
import requests
import tiktoken

# import config
from ..utils.scripts import check_json, cut_message
from ..utils.prompt import search_key_word_prompt
from ..tools.chatgpt import function_call_list
from ..plugins import *

def get_filtered_keys_from_object(obj: object, *keys: str) -> Set[str]:
    """
    Get filtered list of object variable names.
    :param keys: List of keys to include. If the first key is "not", the remaining keys will be removed from the class keys.
    :return: List of class keys.
    """
    class_keys = obj.__dict__.keys()
    if not keys:
        return set(class_keys)

    # Remove the passed keys from the class keys.
    if keys[0] == "not":
        return {key for key in class_keys if key not in keys[1:]}
    # Check if all passed keys are valid
    if invalid_keys := set(keys) - class_keys:
        raise ValueError(
            f"Invalid keys: {invalid_keys}",
        )
    # Only return specified keys that are in class_keys
    return {key for key in keys if key in class_keys}

class chatgpt(BaseLLM):
    """
    Official ChatGPT API
    """

    def __init__(
        self,
        api_key: str,
        engine: str = os.environ.get("GPT_ENGINE") or "gpt-3.5-turbo",
        api_url: str = (os.environ.get("API_URL") or "https://api.openai.com/v1/chat/completions"),
        system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
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
        """
        Initialize Chatbot with API key (from https://platform.openai.com/account/api-keys)
        """
        super().__init__(api_key, engine, api_url, system_prompt, proxy, timeout, max_tokens, temperature, top_p, presence_penalty, frequency_penalty, reply_count, truncate_limit)
        self.max_tokens: int = max_tokens or (
            4096
            if "gpt-4-1106-preview" in engine or "gpt-4-0125-preview" in engine or "gpt-4-turbo" in engine or "gpt-3.5-turbo-1106" in engine or "claude" in engine or "gpt-4o" in engine
            else 31000
            if "gpt-4-32k" in engine
            else 7000
            if "gpt-4" in engine
            else 16385
            if "gpt-3.5-turbo-16k" in engine
            # else 99000
            # if "claude-2.1" in engine
            else 4000
        )
        self.truncate_limit: int = truncate_limit or (
            127500
            if "gpt-4-1106-preview" in engine or "gpt-4-0125-preview" in engine or "gpt-4-turbo" in engine or "gpt-4o" in engine
            else 30500
            if "gpt-4-32k" in engine
            else 6500
            if "gpt-4" in engine
            else 14500
            if "gpt-3.5-turbo-16k" in engine or "gpt-3.5-turbo-1106" in engine
            else 98500
            if "claude-2.1" in engine
            else 3500
        )
        self.conversation: dict[str, list[dict]] = {
            "default": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ],
        }
        self.function_calls_counter = {}
        self.function_call_max_loop = 3
        # self.encode_web_text_list = []

        if self.get_token_count("default") > self.max_tokens:
            raise Exception("System prompt is too long")

    def add_to_conversation(
        self,
        message: list,
        role: str,
        convo_id: str = "default",
        function_name: str = "",
        total_tokens: int = 0,
    ) -> None:
        """
        Add a message to the conversation
        """
        if convo_id not in self.conversation:
            self.reset(convo_id=convo_id)
        if function_name == "" and message and message != None:
            self.conversation[convo_id].append({"role": role, "content": message})
        elif function_name != "" and message and message != None:
            self.conversation[convo_id].append({"role": role, "name": function_name, "content": message})
        else:
            print('\033[31m')
            print("error: add_to_conversation message is None or empty")
            print("role", role, "function_name", function_name, "message", message)
            print('\033[0m')
        if total_tokens:
            self.tokens_usage[convo_id] += total_tokens

    def __truncate_conversation(self, convo_id: str = "default") -> None:
        """
        Truncate the conversation
        """
        while True:
            if (
                self.get_token_count(convo_id) > self.truncate_limit
                and len(self.conversation[convo_id]) > 1
            ):
                # Don't remove the first message
                mess = self.conversation[convo_id].pop(1)
                print("Truncate message:", mess)
            else:
                break

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
        while True:
            json_post = self.get_post_body(prompt, role, convo_id, model, pass_history, **kwargs)
            url = self.api_url.chat_url
            # if "gpt-4" in self.engine or "claude" in self.engine or (CUSTOM_MODELS and self.engine in CUSTOM_MODELS):
            message_token = {
                "total": self.get_token_count(convo_id),
            }
            # else:
            #     message_token = self.get_message_token(url, json_post)
            print("message_token", message_token, "truncate_limit", self.truncate_limit)
            if (
                message_token["total"] > self.truncate_limit
                and len(self.conversation[convo_id]) > 1
            ):
                # Don't remove the first message
                mess = self.conversation[convo_id].pop(1)
                print("Truncate message:", mess)
            else:
                break
        return json_post, message_token

    def extract_values(self, obj):
        if isinstance(obj, dict):
            for value in obj.values():
                yield from self.extract_values(value)
        elif isinstance(obj, list):
            for item in obj:
                yield from self.extract_values(item)
        else:
            yield obj
    # def clear_function_call(self, convo_id: str = "default"):
    #     self.conversation[convo_id] = [item for item in self.conversation[convo_id] if '@Trash@' not in item['content']]
    #     function_call_items = [item for item in self.conversation[convo_id] if 'function' in item['role']]
    #     function_call_num = len(function_call_items)
    #     if function_call_num > 50:
    #         for i in range(function_call_num - 25):
    #             self.conversation[convo_id].remove(function_call_items[i])

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def get_token_count(self, convo_id: str = "default") -> int:
        """
        Get token count
        """
        encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        for message in self.conversation[convo_id]:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 5
            for key, value in message.items():
                values = list(self.extract_values(value))
                if "image_url" in values:
                    continue
                if values:
                    for value in values:
                        num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += 5  # role is always required and always 1 token
        num_tokens += 5  # every reply is primed with <im_start>assistant
        return num_tokens

    def get_message_token(self, url, json_post):
        json_post["max_tokens"] = 17000
        headers = {"Authorization": f"Bearer {os.environ.get('API', None)}"}
        response = requests.Session().post(
            url,
            headers=headers,
            json=json_post,
            timeout=None,
        )
        if response.status_code != 200:
            print(response.text)
            json_response = json.loads(response.text)
            string = json_response["error"]["message"]
            # print(json_response)
            try:
                string = re.findall(r"\((.*?)\)", string)[0]
            except:
                if "You exceeded your current quota" in json_response:
                    raise Exception("当前账号余额不足！")
            numbers = re.findall(r"\d+\.?\d*", string)
            numbers = [int(i) for i in numbers]
            if len(numbers) == 2:
                return {
                    "messages": numbers[0],
                    "total": numbers[0],
                }
            elif len(numbers) == 3:
                return {
                    "messages": numbers[0],
                    "functions": numbers[1],
                    "total": numbers[0] + numbers[1],
                }
            else:
                raise Exception(json_post, json_response)
        # print("response.text", response.text)
        return {
            "messages": 0,
            "total": 0,
        }


    def get_post_body(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = None,
        pass_history: bool = True,
        **kwargs,
    ):
        json_post_body = {
            "model": model or self.engine,
            "messages": self.conversation[convo_id] if pass_history else [{"role": "system","content": self.system_prompt},{"role": role, "content": prompt}],
            "max_tokens": 5000,
            "stream": True,
        }
        body = {
            # kwargs
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
        }
        json_post_body.update(copy.deepcopy(body))
        if all(value == False for value in PLUGINS.values()):
            return json_post_body
        json_post_body.update(copy.deepcopy(function_call_list["base"]))
        for item in PLUGINS.keys():
            try:
                # print(item, PLUGINS[item])
                if PLUGINS[item]:
                    json_post_body["functions"].append(function_call_list[item])
            except:
                pass

        return json_post_body

    def get_max_tokens(self, convo_id: str) -> int:
        """
        Get max tokens
        """
        # print("self.max_tokens, self.get_token_count(convo_id)", self.max_tokens, self.get_token_count(convo_id))
        return self.max_tokens - self.get_token_count(convo_id)

    def ask_stream(
        self,
        prompt: list,
        role: str = "user",
        convo_id: str = "default",
        model: str = None,
        pass_history: bool = True,
        function_name: str = "",
        total_tokens: int = 0,
        **kwargs,
    ):
        """
        Ask a question
        """
        # Make conversation if it doesn't exist
        if convo_id not in self.conversation or pass_history == False:
            self.reset(convo_id=convo_id, system_prompt=self.system_prompt)
        self.add_to_conversation(prompt, role, convo_id=convo_id, function_name=function_name, total_tokens=total_tokens)
        json_post, message_token = self.truncate_conversation(prompt, role, convo_id, model, pass_history, **kwargs)
        # print(self.conversation[convo_id])
        model_max_tokens = kwargs.get("max_tokens", self.max_tokens)
        print("model_max_tokens", model_max_tokens)
        json_post["max_tokens"] = model_max_tokens
        print("api_url", self.api_url.chat_url)
        for _ in range(2):
            print(json.dumps(json_post, indent=4, ensure_ascii=False))
            try:
                response = self.session.post(
                    self.api_url.chat_url,
                    headers={"Authorization": f"Bearer {kwargs.get('api_key', self.api_key)}"},
                    json=json_post,
                    timeout=kwargs.get("timeout", self.timeout),
                    stream=True,
                )
            except ConnectionError:
                print("连接错误，请检查服务器状态或网络连接。")
                return
            except requests.exceptions.ReadTimeout:
                print("请求超时，请检查网络连接或增加超时时间。{e}")
                return
            except Exception as e:
                print(f"发生了未预料的错误: {e}")
                return
            # print("response.text", response.text)
            if response.status_code == 400:
                del json_post["function_call"]
                del json_post["functions"]
                continue
            if response.status_code == 200:
                break
        if response.status_code != 200:
            raise Exception(f"{response.status_code} {response.reason} {response.text}")
        response_role: str = None
        full_response: str = ""
        function_full_response: str = ""
        function_call_name: str = ""
        need_function_call: bool = False
        total_tokens = 0
        for line in response.iter_lines():
            if not line or line.decode("utf-8").startswith(':'):
                continue
            # print("line.decode('utf-8')", line.decode("utf-8"))
            if line.decode("utf-8").startswith('data:'):
                line = line.decode("utf-8")[6:]
            else:
                # print("line", line.decode("utf-8"))
                full_response = json.loads(line.decode("utf-8"))["choices"][0]["message"]["content"]
                yield full_response
                break
            if line == "[DONE]":
                break
            resp: dict = json.loads(line)
            # print("resp", resp)
            usage = resp.get("usage")
            if usage:
                total_tokens = usage.get("total_tokens") or 0
                print("\n\rtotal_tokens", total_tokens)
            choices = resp.get("choices")
            if not choices:
                continue
            delta = choices[0].get("delta")
            if not delta:
                continue
            if "role" in delta and response_role == None:
                response_role = delta["role"]
            if "content" in delta and delta["content"]:
                need_function_call = False
                content = delta["content"]
                full_response += content
                yield content
            if "function_call" in delta:
                # print(delta["function_call"]["arguments"], end="")
                need_function_call = True
                function_call_content = delta["function_call"]["arguments"]
                if "name" in delta["function_call"]:
                    function_call_name = delta["function_call"]["name"]
                function_full_response += function_call_content
                if function_full_response.count("\\n") > 2 or "}" in function_full_response:
                    break
        if need_function_call:
            function_full_response = check_json(function_full_response)
            print("function_full_response", function_full_response)
            function_response = ""
            # print(self.function_calls_counter)
            if not self.function_calls_counter.get(function_call_name):
                self.function_calls_counter[function_call_name] = 1
            else:
                self.function_calls_counter[function_call_name] += 1
            if self.function_calls_counter[function_call_name] <= self.function_call_max_loop:
                function_call_max_tokens = self.truncate_limit - message_token["total"] - 1000
                if function_call_max_tokens <= 0:
                    function_call_max_tokens = int(self.truncate_limit / 2)
                print("\033[32m function_call", function_call_name, "max token:", function_call_max_tokens, "\033[0m")
                if function_call_name == "get_search_results":
                    prompt = json.loads(function_full_response)["prompt"]
                    yield "🌐 正在搜索您的问题，提取关键词..."
                    llm = BaseLLM(api_key=self.api_key, api_url=self.api_url.source_api_url , engine=self.engine, system_prompt=self.system_prompt)
                    keywords = llm.ask(search_key_word_prompt.format(source=prompt)).split("\n")
                    function_response = yield from eval(function_call_name)(prompt, keywords)
                    function_call_max_tokens = 32000
                    function_response, text_len = cut_message(function_response, function_call_max_tokens, self.engine)
                    if function_response:
                        function_response = (
                            f"You need to response the following question: {prompt}. Search results is provided inside <Search_results></Search_results> XML tags. Your task is to think about the question step by step and then answer the above question in {LANGUAGE} based on the Search results provided. Please response in {LANGUAGE} and adopt a style that is logical, in-depth, and detailed. Note: In order to make the answer appear highly professional, you should be an expert in textual analysis, aiming to make the answer precise and comprehensive. Directly response markdown format, without using markdown code blocks"
                            "Here is the Search results, inside <Search_results></Search_results> XML tags:"
                            "<Search_results>"
                            "{}"
                            "</Search_results>"
                        ).format(function_response)
                    else:
                        function_response = "无法找到相关信息，停止使用 tools"
                    # user_prompt = f"You need to response the following question: {prompt}. Search results is provided inside <Search_results></Search_results> XML tags. Your task is to think about the question step by step and then answer the above question in {config.LANGUAGE} based on the Search results provided. Please response in {config.LANGUAGE} and adopt a style that is logical, in-depth, and detailed. Note: In order to make the answer appear highly professional, you should be an expert in textual analysis, aiming to make the answer precise and comprehensive. Directly response markdown format, without using markdown code blocks"
                    # self.add_to_conversation(user_prompt, "user", convo_id=convo_id)
                if function_call_name == "get_url_content":
                    url = json.loads(function_full_response)["url"]
                    print("\n\nurl", url)
                    # function_response = jina_ai_Web_crawler(url)
                    function_response = Web_crawler(url)
                    function_response, text_len = cut_message(function_response, function_call_max_tokens, self.engine)
                if function_call_name == "get_city_tarvel_info":
                    city = json.loads(function_full_response)["city"]
                    function_response = eval(function_call_name)(city)
                    function_response, text_len = cut_message(function_response, function_call_max_tokens, self.engine)
                    function_response = (
                        f"You need to response the following question: {city}. Tarvel infomation is provided inside <infomation></infomation> XML tags. Your task is to think about the question step by step and then answer the above question in {LANGUAGE} based on the tarvel infomation provided. Please response in {LANGUAGE} and adopt a style that is logical, in-depth, and detailed. Note: In order to make the answer appear highly professional, you should be an expert in textual analysis, aiming to make the answer precise and comprehensive."
                        "Here is the tarvel infomation, inside <infomation></infomation> XML tags:"
                        "<infomation>"
                        "{}"
                        "</infomation>"
                    ).format(function_response)
                if function_call_name == "generate_image":
                    prompt = json.loads(function_full_response)["prompt"]
                    function_response = eval(function_call_name)(prompt)
                    function_response, text_len = cut_message(function_response, function_call_max_tokens, self.engine)
                if function_call_name == "run_python_script":
                    prompt = json.loads(function_full_response)["prompt"]
                    function_response = eval(function_call_name)(prompt)
                    function_response, text_len = cut_message(function_response, function_call_max_tokens, self.engine)
                if function_call_name == "get_date_time_weekday":
                    function_response = eval(function_call_name)()
                    function_response, text_len = cut_message(function_response, function_call_max_tokens, self.engine)
                if function_call_name == "get_version_info":
                    function_response = eval(function_call_name)()
                    function_response, text_len = cut_message(function_response, function_call_max_tokens, self.engine)
            else:
                function_response = "无法找到相关信息，停止使用 tools"
            response_role = "function"
            # print(self.conversation[convo_id][-1])
            if self.conversation[convo_id][-1]["role"] == "function" and self.conversation[convo_id][-1]["name"] == "get_search_results":
                mess = self.conversation[convo_id].pop(-1)
                # print("Truncate message:", mess)
            yield from self.ask_stream(function_response, response_role, convo_id=convo_id, function_name=function_call_name, total_tokens=total_tokens)
        else:
            if self.conversation[convo_id][-1]["role"] == "function" and self.conversation[convo_id][-1]["name"] == "get_search_results":
                mess = self.conversation[convo_id].pop(-1)
            self.add_to_conversation(full_response, response_role, convo_id=convo_id, total_tokens=total_tokens)
            self.function_calls_counter = {}
            # self.clear_function_call(convo_id=convo_id)
            # self.encode_web_text_list = []
            # total_tokens = self.get_token_count(convo_id)

    async def ask_stream_async(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = None,
        pass_history: bool = True,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Ask a question
        """
        # Make conversation if it doesn't exist
        if convo_id not in self.conversation or pass_history == False:
            self.reset(convo_id=convo_id, system_prompt=self.system_prompt)
        self.add_to_conversation(prompt, "user", convo_id=convo_id)
        self.__truncate_conversation(convo_id=convo_id)
        if self.engine == "gpt-4-1106-preview" or "gpt-4-0125-preview" in self.engine or "gpt-4-turbo" in self.engine:
            model_max_tokens = kwargs.get("max_tokens", self.max_tokens)
        else:
            model_max_tokens = min(self.get_max_tokens(convo_id=convo_id) - 500, kwargs.get("max_tokens", self.max_tokens))
        # Get response
        async with self.aclient.stream(
            "post",
            self.api_url.chat_url,
            headers={"Authorization": f"Bearer {kwargs.get('api_key', self.api_key)}"},
            json={
                "model": model or self.engine,
                "messages": self.conversation[convo_id] if pass_history else [{"role": "system","content": self.system_prompt},{"role": role, "content": prompt}],
                "stream": True,
                # kwargs
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
                "max_tokens": model_max_tokens,
                # "max_tokens": min(
                #     self.get_max_tokens(convo_id=convo_id),
                #     kwargs.get("max_tokens", self.max_tokens),
                # ),
            },
            timeout=kwargs.get("timeout", self.timeout),
        ) as response:
            if response.status_code != 200:
                await response.aread()
                raise Exception(f"{response.status_code} {response.reason_phrase} {response.text}")
            response_role: str = ""
            full_response: str = ""
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                # Remove "data: "
                line = line[6:]
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
                if "role" in delta:
                    response_role = delta["role"]
                if "content" in delta:
                    content: str = delta["content"]
                    full_response += content
                    yield content
        self.add_to_conversation(full_response, response_role, convo_id=convo_id)
        print("total tokens:", self.get_token_count(convo_id))

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
        response = self.ask_stream_async(
            prompt=prompt,
            role=role,
            convo_id=convo_id,
            **kwargs,
        )
        full_response: str = "".join([r async for r in response])
        return full_response

    def ask(
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
        with open(file, "w", encoding="utf-8") as f:
            data = {
                key: self.__dict__[key]
                for key in get_filtered_keys_from_object(self, *keys)
            }
            # saves session.proxies dict as session
            # leave this here for compatibility
            data["session"] = data["proxy"]
            del data["aclient"]
            json.dump(
                data,
                f,
                indent=2,
            )

    def load(self, file: Path, *keys_: str) -> None:
        """
        Load the Chatbot configuration from a JSON file
        """
        with open(file, encoding="utf-8") as f:
            # load json, if session is in keys, load proxies
            loaded_config = json.load(f)
            keys = get_filtered_keys_from_object(self, *keys_)

            if (
                "session" in keys
                and loaded_config["session"]
                or "proxy" in keys
                and loaded_config["proxy"]
            ):
                self.proxy = loaded_config.get("session", loaded_config["proxy"])
                self.session = httpx.Client(
                    follow_redirects=True,
                    proxies=self.proxy,
                    timeout=self.timeout,
                    cookies=self.session.cookies,
                    headers=self.session.headers,
                )
                self.aclient = httpx.AsyncClient(
                    follow_redirects=True,
                    proxies=self.proxy,
                    timeout=self.timeout,
                    cookies=self.session.cookies,
                    headers=self.session.headers,
                )
            if "session" in keys:
                keys.remove("session")
            if "aclient" in keys:
                keys.remove("aclient")
            self.__dict__.update({key: loaded_config[key] for key in keys})