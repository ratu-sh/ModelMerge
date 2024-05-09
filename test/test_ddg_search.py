from duckduckgo_search import DDGS

def getddgsearchurl(query, max_results=4):
    try:
        webresult = DDGS().text(query, max_results=max_results)
        if webresult == None:
            return []
        urls = [result['href'] for result in webresult]
    except Exception as e:
        print('\033[31m')
        print("duckduckgo error", e)
        print('\033[0m')
        urls = []
    # print("ddg urls", urls)
    return urls

# 搜索关键词
query = "OpenAI"
print(getddgsearchurl(query))