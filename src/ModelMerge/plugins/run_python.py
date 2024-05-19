def run_python_script(script):
    # 创建一个字典来存储脚本执行的本地变量
    local_vars = {}

    try:
        # 执行脚本字符串
        exec(script, {}, local_vars)
        return local_vars['result']
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    # 示例用法
    script = """
def add(a, b):
    return a + b

result = add(5, 3)
    """

    output = run_python_script(script)
    print(output)