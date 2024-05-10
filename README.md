# modelmerge

[中文](./README-zh.md)

modelmerge is a powerful library designed to simplify and unify the use of various large language models, including GPT-4, GPT-3.5, Claude3, Claude2, Gemini1.5 Pro, DALL-E 3, and Groq. This library supports GPT-style function calls and comes with built-in Google search and URL summarization features, significantly enhancing the utility and flexibility of the models.

## Features

- **Multi-Model Support**: Integrates a variety of the latest large language models.
- **Real-Time Interaction**: Supports real-time query streams for instantaneous model responses.
- **Functionality Expansion**: Easily extends model capabilities, such as conducting web searches or content summarization, through built-in function call support.
- **Simplified Interface**: Provides a clean API interface, making it easy to call and manage models.

## Quick Start

Here is a guide on how to quickly integrate and use modelmerge in your Python project.

### Installation

First, you need to install modelmerge. It can be installed directly via pip:

```bash
pip install modelmerge
```

### Usage Example

Here is a simple example demonstrating how to use modelmerge to request the GPT-4 model and handle the returned streaming data:

```python
from modelmerge.models import chatgpt

# Initialize the model, set API key and chosen model
bot = chatgpt(api_key="{YOUR_API_KEY}", engine="gpt-4-turbo-2024-04-09")

# Send a request and get responses in real time
for text in bot.ask_stream("python list use"):
    print(text, end="")
```

### Configuration

You can adjust different model parameters, including API keys and model selection, by modifying the configuration file.

### Supported Models

- GPT-4
- GPT-3.5
- Claude3
- Claude2
- Gemini1.5 Pro
- DALL-E 3
- Groq

### License

This project is licensed under the MIT License.

### Contributions

Contributions are welcome! Please submit issues or pull requests on GitHub to improve the project.

### Contact

If you have any questions or need assistance, please contact us at Your Email.