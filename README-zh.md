# modelmerge

[英文](./README.md) | [中文](./README-zh.md)

modelmerge 是一个强大的库，旨在简化和统一不同大型语言模型的使用，包括 GPT-3.5/4/4 Turbo/4o/4o mini、DALL-E 3、Claude2/3/3.5、Gemini1.5 Pro/Flash、Vertex AI(Claude, Gemini) 、DuckDuckGo 和 Groq。该库支持 GPT 格式的函数调用，并内置了 Google 搜索和 URL 总结功能，极大地增强了模型的实用性和灵活性。

## 特点

- **多模型支持**：集成多种最新的大语言模型。
- **实时交互**：支持实时查询流，实时获取模型响应。
- **功能扩展**：通过内置的函数调用支持，可以轻松扩展模型的功能，例如进行网络搜索或内容摘要。
- **简易接口**：提供简洁的 API 接口，使得调用和管理模型变得轻松。

## 快速上手

以下是如何在您的 Python 项目中快速集成和使用 modelmerge 的指南。

### 安装

首先，您需要安装 modelmerge。可以通过 pip 直接安装：

```bash
pip install modelmerge
```

### 使用示例

以下是一个简单的示例，展示如何使用 modelmerge 来请求 GPT-4 模型并处理返回的流式数据：

```python
from ModelMerge import chatgpt

# 初始化模型，设置 API 密钥和所选模型
bot = chatgpt(api_key="{YOUR_API_KEY}", engine="gpt-4o")

# 获取回答
result = bot.ask("python list use")

# 发送请求并实时获取流式响应
for text in bot.ask_stream("python list use"):
    print(text, end="")
```

## 配置

您可以通过修改配置文件来调整不同模型的参数，包括 API 密钥、模型选择等。

## 支持的模型

- GPT-3.5/4/4 Turbo/4o/4o mini
- DALL-E 3
- Claude2/3/3.5
- Gemini1.5 Pro/Flash
- Vertex AI(Claude, Gemini)
- Groq
- DuckDuckGo(gpt-4o-mini, claude-3-haiku, Meta-Llama-3.1-70B, Mixtral-8x7B)

## 许可证

本项目采用 MIT 许可证授权。

## 贡献

欢迎通过 GitHub 提交问题或拉取请求来贡献改进。

## 联系方式

如有任何疑问或需要帮助，请通过 [yym68686@outlook.com](mailto:yym68686@outlook.com) 联系我们。