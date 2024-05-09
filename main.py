import os
from datetime import datetime

from ModelMerge.utils import prompt
from ModelMerge.models import chatgpt, claude3, gemini

LANGUAGE = os.environ.get('LANGUAGE', 'Simplified Chinese')
GPT_ENGINE = os.environ.get('GPT_ENGINE', 'gpt-4-turbo-2024-04-09')

API = os.environ.get('API', None)
API_URL = os.environ.get('API_URL', None)

CLAUDE_API = os.environ.get('CLAUDE_API', None)
GOOGLE_AI_API_KEY = os.environ.get('GOOGLE_AI_API_KEY', None)

current_date = datetime.now()
Current_Date = current_date.strftime("%Y-%m-%d")

systemprompt = os.environ.get('SYSTEMPROMPT', prompt.system_prompt.format(LANGUAGE, Current_Date))
bot = chatgpt(api_key=API, api_url=API_URL , engine=GPT_ENGINE, system_prompt=systemprompt)
bot = claude3(api_key=CLAUDE_API, engine=GPT_ENGINE, system_prompt=systemprompt)
bot = gemini(api_key=GOOGLE_AI_API_KEY, engine=GPT_ENGINE, system_prompt=systemprompt)
# for text in bot.ask_stream("今天的微博热搜有哪些？"):
for text in bot.ask_stream("python 包duckduckgo-search怎么使用？"):
    print(text, end="")