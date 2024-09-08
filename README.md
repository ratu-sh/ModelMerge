# modelmerge

[English](./README.md) | [Chinese](./README-zh.md)

modelmerge is a powerful library designed to simplify and unify the usage of different large language models, including GPT-3.5/4/4 Turbo/4o/4o mini, DALL-E 3, Claude2/3/3.5, Gemini1.5 Pro/Flash, Vertex AI (Claude, Gemini), DuckDuckGo, and Groq. The library supports GPT-format function calls and has built-in Google search and URL summarization features, greatly enhancing the practicality and flexibility of the models.

## Characteristics

- **Multi-model support**: Integrate various latest large language models.
- **Real-time Interaction**: Supports real-time query streams, real-time model response retrieval.
- **Function Expansion**: With built-in function call support, the model's capabilities can be easily extended, such as performing web searches or content summarization.
- **Simple Interface**: Provides a concise API interface, making it easy to call and manage models.

## Quick Start

The following is a guide on how to quickly integrate and use modelmerge in your Python project.

### Install

First, you need to install modelmerge. It can be installed directly via pip:

```bash
pip install modelmerge
```

### Usage example

The following is a simple example demonstrating how to use modelmerge to request the GPT-4 model and handle the returned streaming data:

```python
from ModelMerge import chatgpt

# Initialize the model, set the API key and selected model
bot = chatgpt(api_key="{YOUR_API_KEY}", engine="gpt-4o")

# Get the answer
result = bot.ask("python list use")

# Send the request and get the streaming response in real-time
for text in bot.ask_stream("python list use"):
    print(text, end="")
```

## Configuration

You can adjust the parameters of different models by modifying the configuration file, including API key, model selection, etc.

## Supported models

- GPT-3.5/4/4 Turbo/4o/4o mini
- DALL-E 3
- Claude2/3/3.5
- Gemini1.5 Pro/Flash
- Vertex AI (Claude, Gemini)
- Groq
- DuckDuckGo(gpt-4o-mini, claude-3-haiku, Meta-Llama-3.1-70B, Mixtral-8x7B)

## License

This project is licensed under the MIT License.

## Contribution

Welcome to contribute improvements by submitting issues or pull requests through GitHub.

## Contact Information

If you have any questions or need assistance, please contact us at [yym68686@outlook.com](mailto:yym68686@outlook.com).