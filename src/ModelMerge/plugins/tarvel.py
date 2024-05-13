import re
import json
import execjs
import hashlib
import requests
from bs4 import BeautifulSoup
from requests.utils import add_dict_to_cookiejar
from urllib3.exceptions import InsecureRequestWarning

# 关闭ssl验证提示
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def get_cookie(response, session, header, url):
    # 提取js代码
    js_clearance = re.findall('cookie=(.*?);location', response.text)[0]
    # 执行后获得cookie参数js_clearance
    result = execjs.eval(js_clearance).split(';')[0].split('=')[1]
    # 添加cookie
    add_dict_to_cookiejar(session.cookies, {'__jsl_clearance_s': result})
    # 第二次请求
    response = session.get(url, headers=header, verify=False)
    # 提取参数并转字典
    parameter = json.loads(re.findall(r';go\((.*?)\)', response.text)[0])
    # print(parameter)
    for i in range(len(parameter['chars'])):
        for j in range(len(parameter['chars'])):
            values = parameter['bts'][0] + parameter['chars'][i] + parameter['chars'][j] + parameter['bts'][1]
            if parameter['ha'] == 'md5':
                ha = hashlib.md5(values.encode()).hexdigest()
            elif parameter['ha'] == 'sha1':
                ha = hashlib.sha1(values.encode()).hexdigest()
            elif parameter['ha'] == 'sha256':
                ha = hashlib.sha256(values.encode()).hexdigest()
            if ha == parameter['ct']:
                __jsl_clearance_s = values

    return __jsl_clearance_s

def get_mafengwo(url):
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0',
    }
    # 使用session保持会话
    session = requests.session()
    # 第一次请求
    response = session.get(url, headers=header, verify=False)
    # 获取参数及加密方式 获取cookie
    if "search" in url:
        soup = BeautifulSoup(response.text, "html.parser")
        soup_lxml = BeautifulSoup(response.text, "lxml")
        return soup, soup_lxml

    __jsl_clearance_s = get_cookie(response, session, header, url)
    # print(__jsl_clearance_s)
    # 修改cookie
    add_dict_to_cookiejar(session.cookies, {'__jsl_clearance_s': __jsl_clearance_s})
    # 第三次请求
    html = session.get(url, headers=header, verify=False)
    #print(html.cookies)
    #print(html.content.decode())
    soup = BeautifulSoup(html.text, "html.parser")
    soup_lxml = BeautifulSoup(html.text, "lxml")
    return soup, soup_lxml

def get_mafengwo_routers(soup):
    routes = soup.find_all('a', href=True, string=True)
    urls = []
    for route in routes:
        if route['href'].startswith('/mdd/') and "_" in route['href']:  # 只处理包含'mdd'的链接
            urls.append("https://www.mafengwo.cn" + route['href'])
    return urls

def get_mafengwo_urls(soup):
    routes = soup.find_all('a', href=True, string=True)
    urls = []
    for route in routes:
        urls.append(route['href'])
    return urls

def get_mafengwo_all_text(soup):
    # print(soup)
    all_text = "<travel_plan>\n\n"
    title = soup.find('h1').text.strip()
    all_text += "旅游方案：{title}\n\n".format(title=title)
    days = soup.find_all('div', class_='day-item')
    all_text += "以下是路线概览：\n\n"

    # 循环处理每一天的行程信息
    for day in days:
        # 获取天数，如 D1, D2 等
        day_number = day.find('span', class_='day-num').text.strip()
        # print(day_number)
        all_text += day_number.replace("D", "Day")  # 打印天数
        place = day.find('span', class_='place').text.strip()
        all_text += f" {place}"  # 打印景点和停留时间
        all_text += "\n\n"  # 打印景点和停留时间

        # 获取当天的详细行程描述
        itinerary_description = day.find('div', class_='poi-txt').text.strip()
        all_text += itinerary_description  # 打印行程描述
        all_text += "\n\n"  # 打印行程描述

        # 获取当天的具体景点和推荐停留时间
        # print(day)
        place_name = [item.text.strip() for item in day.find_all('a', class_='p-link')]
        stay_time = [item.text.strip() for item in day.find_all('span', class_='time')]
        place_info = [item.text.strip() for item in day.find_all('dd')]
        place_list = []
        for place, time, info in zip(place_name, stay_time, place_info):
            place_list.append(f"地点: {place}\n停留时间: {time}\n介绍: {info}\n")
        text = "路线：\n\n" + "\n".join(place_list)
        all_text += text  # 打印景点和停留时间

        # # 打印住宿攻略（如果有）
        # hotel_tips = day.find('div', class_='J_hotelpois')
        # if hotel_tips:
        #     hotel_tips_text = hotel_tips.find('div', class_='day-hd mt30').text.strip()
        #     all_text += hotel_tips_text  # 打印住宿攻略
        all_text += "\n\n\n"
    all_text += "</travel_plan>"
    all_text += "\n\n\n"
    return all_text

def get_mafengwo_all_travel_plan(routes):
    all_travel_plan = "每个旅游方案都被包裹在<travel_plan></travel_plan>标记里面，下面是所有旅游方案：\n\n"
    for url in routes:
        soup, soup_lxml = get_mafengwo(url)
        text = get_mafengwo_all_text(soup_lxml)
        all_travel_plan += text
    return all_travel_plan

def get_city_tarvel_info(city):
    url = 'https://www.mafengwo.cn/search/q.php?q={}'.format(city)
    soup, soup_lxml = get_mafengwo(url)
    urls = get_mafengwo_urls(soup)
    for item in urls:
        if "mafengwo.cn/mdd/route/" in item:
            url = item.replace("m.", "www.")
    print("search url:", url)
    soup, soup_lxml = get_mafengwo(url)
    routes = get_mafengwo_routers(soup)
    all_travel_plan = get_mafengwo_all_travel_plan(routes)
    return all_travel_plan


if __name__ == '__main__':
    # url = 'https://www.mafengwo.cn/mdd/route/10088.html'
    all_travel_plan = get_city_tarvel_info("上海")
    print(all_travel_plan)


    # # 假设你的 HTML 内容存储在一个名为 'example.txt' 的文件中
    # with open('1.txt', 'r', encoding='utf-8') as file:
    #     html_content = file.read()

    # # 使用 BeautifulSoup 解析 HTML
    # soup = BeautifulSoup(html_content, 'lxml')
    # soup = get_mafengwo_all_text(soup)
    # print(soup)