import requests

from ..utils.scripts import Document_extract
def download_read_arxiv_pdf(arxiv_id, save_path = "paper.pdf"):
    # 构造下载PDF的URL
    url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'

    # 发送HTTP GET请求
    response = requests.get(url)

    # 检查是否成功获取内容
    if response.status_code == 200:
        # 将PDF内容写入文件
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f'PDF下载成功，保存路径: {save_path}')
        return Document_extract(None, save_path)
    else:
        print(f'下载失败，状态码: {response.status_code}')
        return "文件下载失败"

if __name__ == '__main__':
    # 示例使用
    arxiv_id = '2305.12345'  # 替换为实际的arXiv ID
    save_path = 'paper.pdf'  # 替换为你想保存的路径和文件名
    print(download_read_arxiv_pdf(arxiv_id, save_path))