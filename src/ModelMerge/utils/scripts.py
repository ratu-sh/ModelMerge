import os
import json
import base64
import tiktoken
import requests
import urllib.parse

def get_encode_text(text, model_name):
    tiktoken.get_encoding("cl100k_base")
    model_name = "gpt-3.5-turbo"
    encoding = tiktoken.encoding_for_model(model_name)
    encode_text = encoding.encode(text)
    return encoding, encode_text

def get_text_token_len(text, model_name):
    encoding, encode_text = get_encode_text(text, model_name)
    return len(encode_text)

def cut_message(message: str, max_tokens: int, model_name: str):
    if type(message) != str:
        message = str(message)
    encoding, encode_text = get_encode_text(message, model_name)
    if len(encode_text) > max_tokens:
        encode_text = encode_text[:max_tokens]
        message = encoding.decode(encode_text)
    encode_text = encoding.encode(message)
    return message, len(encode_text)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_doc_from_url(url):
    filename = urllib.parse.unquote(url.split("/")[-1])
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    return filename

def get_encode_image(image_url):
    filename = get_doc_from_url(image_url)
    image_path = os.getcwd() + "/" + filename
    base64_image = encode_image(image_path)
    prompt = f"data:image/jpeg;base64,{base64_image}"
    os.remove(image_path)
    return prompt

def get_image_message(image_url, message, engine = None):
    if image_url:
        base64_image = get_encode_image(image_url)
        if "gpt-4" in engine or (os.environ.get('claude_api_key', None) is None and "claude-3" in engine):
            message.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image
                    }
                }
            )
        if os.environ.get('claude_api_key', None) and "claude-3" in engine:
            message.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image.split(",")[1],
                    }
                }
            )
        if os.environ.get('GOOGLE_AI_API_KEY', None) and "gemini" in engine:
            message.append(
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": base64_image.split(",")[1],
                    }
                }
            )
    return message

def Document_extract(docurl, docpath=None, engine = None):
    filename = docpath
    text = None
    if docpath and docurl and "paper.pdf" != docpath:
        filename = get_doc_from_url(docurl)
        docpath = os.getcwd() + "/" + filename
    if filename and filename[-3:] == "pdf":
        from pdfminer.high_level import extract_text
        text = extract_text(docpath)
    if filename and (filename[-3:] == "txt" or filename[-3:] == ".md" or filename[-3:] == ".py"):
        with open(docpath, 'r') as f:
            text = f.read()
    if text:
        prompt = (
            "Here is the document, inside <document></document> XML tags:"
            "<document>"
            "{}"
            "</document>"
        ).format(text)
    if filename and filename[-3:] == "jpg" or filename[-3:] == "png" or filename[-4:] == "jpeg":
        prompt = get_image_message(docurl, [], engine)
    if os.path.exists(docpath):
        os.remove(docpath)
    return prompt

def check_json(json_data):
    while True:
        try:
            json.loads(json_data)
            break
        except json.decoder.JSONDecodeError as e:
            print("JSON error：", e)
            print("JSON body", repr(json_data))
            if "Invalid control character" in str(e):
                json_data = json_data.replace("\n", "\\n")
            if "Unterminated string starting" in str(e):
                json_data += '"}'
            if "Expecting ',' delimiter" in str(e):
                json_data += '}'
            if "Expecting value: line 1 column 1" in str(e):
                if json_data.startswith("prompt: "):
                    json_data = json_data.replace("prompt: ", "")
                json_data = '{"prompt": ' + json.dumps(json_data) + '}'
    return json_data

def is_surrounded_by_chinese(text, index):
    left_char = text[index - 1]
    if 0 < index < len(text) - 1:
        right_char = text[index + 1]
        return '\u4e00' <= left_char <= '\u9fff' or '\u4e00' <= right_char <= '\u9fff'
    if index == len(text) - 1:
        return '\u4e00' <= left_char <= '\u9fff'
    return False

def replace_char(string, index, new_char):
    return string[:index] + new_char + string[index+1:]

def claude_replace(text):
    Punctuation_mapping = {",": "，", ":": "：", "!": "！", "?": "？", ";": "；"}
    key_list = list(Punctuation_mapping.keys())
    for i in range(len(text)):
        if is_surrounded_by_chinese(text, i) and (text[i] in key_list):
            text = replace_char(text, i, Punctuation_mapping[text[i]])
    return text

if __name__ == "__main__":
    os.system("clear")