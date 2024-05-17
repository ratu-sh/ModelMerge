import os
from ModelMerge.models import dalle3

API = os.environ.get('API', None)
API_URL = os.environ.get('API_URL', None)

def generate_image(text):
    dallbot = dalle3(api_key=f"{API}", api_url=f"{API_URL}")
    for data in dallbot.generate(text):
        return data