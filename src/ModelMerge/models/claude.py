import os
import json
import copy
import tiktoken
import requests

from .config import BaseLLM, PLUGINS, LANGUAGE
from ..tools.claude import claude_tools_list
from ..utils.scripts import check_json, cut_message
from ..utils.prompt import search_key_word_prompt
from ..plugins import *



class claudeConversation(dict):
    def Conversation(self, index):
        conversation_list = super().__getitem__(index)
        return "\n\n" + "\n\n".join([f"{item['role']}:{item['content']}" for item in conversation_list]) + "\n\nAssistant:"

class claude(BaseLLM):
    def __init__(
        self,
        api_key: str,
        engine: str = os.environ.get("GPT_ENGINE") or "claude-2.1",
        api_url: str = "https://api.anthropic.com/v1/complete",
        system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
        temperature: float = 0.5,
        top_p: float = 0.7,
        timeout: float = 20,
    ):
        super().__init__(api_key, engine, api_url, system_prompt, timeout=timeout, temperature=temperature, top_p=top_p)
        # self.api_url = api_url
        self.conversation = claudeConversation()

    def add_to_conversation(
        self,
        message: str,
        role: str,
        convo_id: str = "default",
        pass_history: bool = True,
        total_tokens: int = 0,
    ) -> None:
        """
        Add a message to the conversation
        """

        if convo_id not in self.conversation or pass_history == False:
            self.reset(convo_id=convo_id)
        self.conversation[convo_id].append({"role": role, "content": message})
        if total_tokens:
            self.tokens_usage[convo_id] += total_tokens

    def reset(self, convo_id: str = "default", system_prompt: str = None) -> None:
        """
        Reset the conversation
        """
        self.conversation[convo_id] = claudeConversation()

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
                self.conversation[convo_id].pop(1)
            else:
                break

    def get_token_count(self, convo_id: str = "default") -> int:
        """
        Get token count
        """
        tiktoken.model.MODEL_TO_ENCODING["claude-2.1"] = "cl100k_base"
        encoding = tiktoken.encoding_for_model(self.engine)

        num_tokens = 0
        for message in self.conversation[convo_id]:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 5
            for key, value in message.items():
                if value:
                    num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += 5  # role is always required and always 1 token
        num_tokens += 5  # every reply is primed with <im_start>assistant
        return num_tokens

    def ask_stream(
        self,
        prompt: str,
        role: str = "Human",
        convo_id: str = "default",
        model: str = None,
        pass_history: bool = True,
        model_max_tokens: int = 4096,
        **kwargs,
    ):
        pass_history = True
        if convo_id not in self.conversation or pass_history == False:
            self.reset(convo_id=convo_id)
        self.add_to_conversation(prompt, role, convo_id=convo_id)
        # self.__truncate_conversation(convo_id=convo_id)
        # print(self.conversation[convo_id])

        url = self.api_url
        headers = {
            "accept": "application/json",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": f"{kwargs.get('api_key', self.api_key)}",
        }

        json_post = {
            "model": model or self.engine,
            "prompt": self.conversation.Conversation(convo_id) if pass_history else f"\n\nHuman:{prompt}\n\nAssistant:",
            "stream": True,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "max_tokens_to_sample": model_max_tokens,
        }

        try:
            response = self.session.post(
                url,
                headers=headers,
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

        if response.status_code != 200:
            print(response.text)
            raise BaseException(f"{response.status_code} {response.reason} {response.text}")
        response_role: str = "Assistant"
        full_response: str = ""
        for line in response.iter_lines():
            if not line or line.decode("utf-8") == "event: completion" or line.decode("utf-8") == "event: ping" or line.decode("utf-8") == "data: {}":
                continue
            line = line.decode("utf-8")[6:]
            # print(line)
            resp: dict = json.loads(line)
            content = resp.get("completion")
            if content:
                full_response += content
                yield content
        self.add_to_conversation(full_response, response_role, convo_id=convo_id)

class claude3(BaseLLM):
    def __init__(
        self,
        api_key: str,
        engine: str = os.environ.get("GPT_ENGINE") or "claude-3-opus-20240229",
        api_url: str = "https://api.anthropic.com/v1/messages",
        system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
        temperature: float = 0.5,
        timeout: float = 20,
        top_p: float = 0.7,
    ):
        super().__init__(api_key, engine, api_url, system_prompt, timeout=timeout, temperature=temperature, top_p=top_p)
        self.api_url = api_url
        self.conversation: dict[str, list[dict]] = {
            "default": [],
        }

    def add_to_conversation(
        self,
        message: str,
        role: str,
        convo_id: str = "default",
        pass_history: bool = True,
        total_tokens: int = 0,
        tools_id= "",
        function_name: str = "",
        function_full_response: str = "",
    ) -> None:
        """
        Add a message to the conversation
        """

        if convo_id not in self.conversation or pass_history == False:
            self.reset(convo_id=convo_id)
        if role == "user":
            self.conversation[convo_id].append({"role": role, "content": message})
        # if role == "function":
        if role == "assistant" and function_full_response:
            print("function_full_response", function_full_response)
            function_dict = {
                "type": "tool_use",
                "id": f"{tools_id}",
                "name": f"{function_name}",
                "input": json.loads(function_full_response)
                # "input": json.dumps(function_full_response, ensure_ascii=False)
            }
            self.conversation[convo_id].append({"role": role, "content": [function_dict]})
            function_dict = {
                "type": "tool_result",
                "tool_use_id": f"{tools_id}",
                "content": f"{message}",
                # "is_error": true
            }
            self.conversation[convo_id].append({"role": "user", "content": [function_dict]})
        # index = len(self.conversation[convo_id]) - 2
        # if index >= 0 and self.conversation[convo_id][index]["role"] == self.conversation[convo_id][index + 1]["role"]:
        #     self.conversation[convo_id][index]["content"] += self.conversation[convo_id][index + 1]["content"]
        #     self.conversation[convo_id].pop(index + 1)
        if total_tokens:
            self.tokens_usage[convo_id] += total_tokens

    def reset(self, convo_id: str = "default", system_prompt: str = None) -> None:
        """
        Reset the conversation
        """
        self.conversation[convo_id] = list()

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
                self.conversation[convo_id].pop(1)
            else:
                break

    def get_token_count(self, convo_id: str = "default") -> int:
        """
        Get token count
        """
        tiktoken.model.MODEL_TO_ENCODING["claude-2.1"] = "cl100k_base"
        encoding = tiktoken.encoding_for_model(self.engine)

        num_tokens = 0
        for message in self.conversation[convo_id]:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 5
            for key, value in message.items():
                if value:
                    num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += 5  # role is always required and always 1 token
        num_tokens += 5  # every reply is primed with <im_start>assistant
        return num_tokens

    def ask_stream(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = None,
        pass_history: bool = True,
        model_max_tokens: int = 4096,
        tools_id: str = "",
        total_tokens: int = 0,
        function_name: str = "",
        function_full_response: str = "",
        **kwargs,
    ):
        pass_history = True
        if convo_id not in self.conversation or pass_history == False:
            self.reset(convo_id=convo_id)
        self.add_to_conversation(prompt, role, convo_id=convo_id, tools_id=tools_id, total_tokens=total_tokens, function_name=function_name, function_full_response=function_full_response)
        # self.__truncate_conversation(convo_id=convo_id)
        # print(self.conversation[convo_id])

        url = self.api_url
        headers = {
            "content-type": "application/json",
            "x-api-key": f"{kwargs.get('api_key', self.api_key)}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "tools-2024-05-16"
        }

        json_post = {
            "model": model or self.engine,
            "messages": self.conversation[convo_id] if pass_history else [{
                "role": "user",
                "content": prompt
            }],
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "max_tokens": model_max_tokens,
            "stream": True,
        }
        if self.system_prompt:
            json_post["system"] = self.system_prompt
        if all(value == False for value in PLUGINS.values()) == False:
            json_post.update(copy.deepcopy(claude_tools_list["base"]))
            for item in PLUGINS.keys():
                try:
                    if PLUGINS[item]:
                        json_post["tools"].append(claude_tools_list[item])
                except:
                    pass

        print(json.dumps(json_post, indent=4, ensure_ascii=False))

        try:
            response = self.session.post(
                url,
                headers=headers,
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

        if response.status_code != 200:
            print(response.text)
            raise BaseException(f"{response.status_code} {response.reason} {response.text}")
        response_role: str = "assistant"
        full_response: str = ""
        need_function_call: bool = False
        function_call_name: str = ""
        function_full_response: str = ""
        total_tokens = 0
        tools_id = ""
        for line in response.iter_lines():
            if not line or line.decode("utf-8")[:6] == "event:" or line.decode("utf-8") == "data: {}":
                continue
            # print(line.decode("utf-8"))
            # if "tool_use" in line.decode("utf-8"):
            #     tool_input = json.loads(line.decode("utf-8")["content"][1]["input"])
            # else:
            #     line = line.decode("utf-8")[6:]
            line = line.decode("utf-8")[6:]
            # print(line)
            resp: dict = json.loads(line)
            message = resp.get("message")
            if message:
                tokens_use = resp.get("usage")
                if tokens_use:
                    total_tokens = tokens_use["input_tokens"] + tokens_use["output_tokens"]
                    print("\n\rtotal_tokens", total_tokens)
            tool_use = resp.get("content_block")
            if tool_use and "tool_use" == tool_use['type']:
                # print("tool_use", tool_use)
                tools_id = tool_use["id"]
                need_function_call = True
                if "name" in tool_use:
                    function_call_name = tool_use["name"]
            delta = resp.get("delta")
            # print("delta", delta)
            if not delta:
                continue
            if "text" in delta:
                content = delta["text"]
                full_response += content
                yield content
            if "partial_json" in delta:
                function_call_content = delta["partial_json"]
                function_full_response += function_call_content
        # print("function_full_response", function_full_response)
        # print("function_call_name", function_call_name)
        # print("need_function_call", need_function_call)
        if need_function_call:
            function_full_response = check_json(function_full_response)
            print("function_full_response", function_full_response)
            function_response = ""
            if function_call_name == "get_search_results":
                prompt = json.loads(function_full_response)["prompt"]
                yield "🌐 正在搜索您的问题，提取关键词..."
                llm = claude3(api_key=self.api_key, api_url=self.api_url, engine=self.engine, system_prompt=self.system_prompt)
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
            response_role = "assistant"
            if self.conversation[convo_id][-1]["role"] == "function" and self.conversation[convo_id][-1]["name"] == "get_search_results":
                mess = self.conversation[convo_id].pop(-1)
            yield from self.ask_stream(function_response, response_role, convo_id=convo_id, function_name=function_call_name, total_tokens=total_tokens, tools_id=tools_id, function_full_response=function_full_response)
        else:
            if self.conversation[convo_id][-1]["role"] == "function" and self.conversation[convo_id][-1]["name"] == "get_search_results":
                mess = self.conversation[convo_id].pop(-1)
            self.add_to_conversation(full_response, response_role, convo_id=convo_id, total_tokens=total_tokens)
            self.function_calls_counter = {}





        # self.add_to_conversation(full_response, response_role, convo_id=convo_id)