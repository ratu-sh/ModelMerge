import os
PLUGINS = {
    "SEARCH" : (os.environ.get('SEARCH', "True") == "False") == False,
    "URL"    : (os.environ.get('URL', "True") == "False") == False,
    "CODE"   : (os.environ.get('CODE', "True") == "False") == False,
    "IMAGE"  : (os.environ.get('IMAGE', "False") == "False") == False,
    "DATE"   : (os.environ.get('DATE', "False") == "False") == False,
    "VERSION": (os.environ.get('VERSION', "False") == "False") == False,
    "TARVEL" : (os.environ.get('TARVEL', "False") == "False") == False,
}