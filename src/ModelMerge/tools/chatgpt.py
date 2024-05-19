function_call_list = \
{
    "base": {
        "functions": [],
        "function_call": "auto"
    },
    "current_weather": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": [
                        "celsius",
                        "fahrenheit"
                    ]
                }
            },
            "required": [
                "location"
            ]
        }
    },
    "SEARCH": {
        "name": "get_search_results",
        "description": "Search Google to enhance knowledge.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt to search."
                }
            },
            "required": [
                "prompt"
            ]
        }
    },
    "URL": {
        "name": "get_url_content",
        "description": "Get the webpage content of a URL",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "the URL to request"
                }
            },
            "required": [
                "url"
            ]
        }
    },
    "DATE": {
        "name": "get_date_time_weekday",
        "description": "Get the current time, date, and day of the week",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    "VERSION": {
        "name": "get_version_info",
        "description": "Get version information",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    "TARVEL": {
        "name": "get_city_tarvel_info",
        "description": "Get the city's travel plan by city name.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "the city to search"
                }
            },
            "required": [
                "city"
            ]
        }
    },
    "IMAGE": {
        "name": "generate_image",
        "description": "Generate images based on user descriptions.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "the prompt to generate image"
                }
            },
            "required": [
                "prompt"
            ]
        }
    },
    "CODE": {
        "name": "run_python_script",
        "description": "Convert the string to a Python script and return the Python execution result. Assign the result to the variable result. Directly output the code, without using quotation marks or other symbols to enclose the code.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "the code to run"
                }
            },
            "required": [
                "prompt"
            ]
        }
    },
}
