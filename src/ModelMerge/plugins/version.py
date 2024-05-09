import subprocess
import os

# Plugins 使用函数
def get_version_info():
    current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(['git', '-C', current_directory, 'log', '-1'], stdout=subprocess.PIPE)
    output = result.stdout.decode()
    return output