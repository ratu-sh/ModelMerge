# ModelMerge

## 🔌 Plugins

Our plugin system has been successfully developed and is now fully operational. We welcome everyone to contribute their code to enrich our plugin library. The following plugins are currently supported:

- **Web Search**: By default, DuckDuckGo search is provided. Google search is automatically activated when the `GOOGLE_CSE_ID` and `GOOGLE_API_KEY` environment variables are set.
- **Time Retrieval**: Retrieves the current time, date, and day of the week in the GMT+8 time zone.
- **URL Summary**: Automatically extracts URLs from queries and responds based on the content of the URLs.
- **Version Information**: Displays the current version of the bot, commit hash, update time, and developer name.

To develop plugins, please follow the steps outlined below:

- Initially, you need to add the environment variable for the plugin in the `config.PLUGINS` dictionary located in the `config.py` file. The value can be customized to be either enabled or disabled by default. It is advisable to use uppercase letters for the entire environment variable.
- Subsequently, append the function's name and description in the `tools/function_call.py` file.
- Then, enhance the `ask_stream` function in the `models/` file with the function's processing logic. You can refer to the existing examples within the `ask_stream` method for guidance on how to write it.
- Following that, write the function, as mentioned in the `tools/function_call.py` file, in the `plugins/` file.
- Lastly, don't forget to add the plugin's description in the plugins section of the README.

Please note that the above steps are a general guide and may need to be adjusted based on the specific requirements of your plugin.