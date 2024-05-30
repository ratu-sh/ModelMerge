import time
import json
import random
import hashlib
import logging
import requests
import itertools
from datetime import datetime

city = {"AAT":"阿勒泰","ACX":"兴义","AEB":"百色","AKU":"阿克苏","AOG":"鞍山","AQG":"安庆","AVA":"安顺","AXF":"阿拉善左旗","BAV":"包头","BFJ":"毕节","BHY":"北海"
                ,"BJS":"北京","BPE":"秦皇岛","BPL":"博乐","BPX":"昌都","BSD":"保山","CAN":"广州","CDE":"承德","CGD":"常德","CGO":"郑州","CGQ":"长春","CHG":"朝阳","CIF":"赤峰"
                ,"CIH":"长治","CKG":"重庆","CSX":"长沙","CTU":"成都","CWJ":"沧源","CYI":"嘉义","CZX":"常州","DAT":"大同","DAX":"达县","DBC":"白城","DCY":"稻城","DDG":"丹东"
                ,"DIG":"香格里拉(迪庆)","DLC":"大连","DLU":"大理","DNH":"敦煌","DOY":"东营","DQA":"大庆","DSN":"鄂尔多斯","DYG":"张家界","EJN":"额济纳旗","ENH":"恩施"
                ,"ENY":"延安","ERL":"二连浩特","FOC":"福州","FUG":"阜阳","FUO":"佛山","FYJ":"抚远","GOQ":"格尔木","GYS":"广元","GYU":"固原","HAK":"海口","HDG":"邯郸"
                ,"HEK":"黑河","HET":"呼和浩特","HFE":"合肥","HGH":"杭州","HIA":"淮安","HJJ":"怀化","HKG":"香港","HLD":"海拉尔","HLH":"乌兰浩特","HMI":"哈密","HPG":"神农架"
                ,"HRB":"哈尔滨","HSN":"舟山","HTN":"和田","HUZ":"惠州","HYN":"台州","HZG":"汉中","HZH":"黎平","INC":"银川","IQM":"且末","IQN":"庆阳","JDZ":"景德镇"
                ,"JGD":"加格达奇","JGN":"嘉峪关","JGS":"井冈山","JHG":"西双版纳","JIC":"金昌","JIQ":"黔江","JIU":"九江","JJN":"晋江","JMJ":"澜沧","JMU":"佳木斯","JNG":"济宁"
                ,"JNZ":"锦州","JSJ":"建三江","JUH":"池州","JUZ":"衢州","JXA":"鸡西","JZH":"九寨沟","KCA":"库车","KGT":"康定","KHG":"喀什","KHN":"南昌","KJH":"凯里","KMG":"昆明"
                ,"KNH":"金门","KOW":"赣州","KRL":"库尔勒","KRY":"克拉玛依","KWE":"贵阳","KWL":"桂林","LCX":"龙岩","LDS":"伊春","LFQ":"临汾","LHW":"兰州","LJG":"丽江","LLB":"荔波"
                ,"LLF":"永州","LLV":"吕梁","LNJ":"临沧","LPF":"六盘水","LUM":"芒市","LXA":"拉萨","LYA":"洛阳","LYG":"连云港","LYI":"临沂","LZH":"柳州","LZO":"泸州"
                ,"LZY":"林芝","MDG":"牡丹江","MFK":"马祖","MFM":"澳门","MIG":"绵阳","MXZ":"梅州","NAO":"南充","NBS":"白山","NDG":"齐齐哈尔","NGB":"宁波","NGQ":"阿里"
                ,"NKG":"南京","NLH":"宁蒗","NNG":"南宁","NNY":"南阳","NTG":"南通","NZH":"满洲里","OHE":"漠河","PZI":"攀枝花","RHT":"阿拉善右旗","RIZ":"日照","RKZ":"日喀则"
                ,"RLK":"巴彦淖尔","SHA":"上海","SHE":"沈阳","SIA":"西安","SJW":"石家庄","SWA":"揭阳","SYM":"普洱","SYX":"三亚","SZX":"深圳","TAO":"青岛","TCG":"塔城","TCZ":"腾冲"
                ,"TEN":"铜仁","TGO":"通辽","THQ":"天水","TLQ":"吐鲁番","TNA":"济南","TSN":"天津","TVS":"唐山","TXN":"黄山","TYN":"太原","URC":"乌鲁木齐","UYN":"榆林","WEF":"潍坊"
                ,"WEH":"威海","WMT":"遵义(茅台)","WNH":"文山","WNZ":"温州","WUA":"乌海","WUH":"武汉","WUS":"武夷山","WUX":"无锡","WUZ":"梧州","WXN":"万州","XFN":"襄阳","XIC":"西昌"
                ,"XIL":"锡林浩特","XMN":"厦门","XNN":"西宁","XUZ":"徐州","YBP":"宜宾","YCU":"运城","YIC":"宜春","YIE":"阿尔山","YIH":"宜昌","YIN":"伊宁","YIW":"义乌","YNJ":"延吉"
                ,"YNT":"烟台","YNZ":"盐城","YTY":"扬州","YUS":"玉树","YZY":"张掖","ZAT":"昭通","ZHA":"湛江","ZHY":"中卫","ZQZ":"张家口","ZUH":"珠海","ZYI":"遵义(新舟)","KJI":"喀纳斯"}

city_value = list(city.items())
def city_name():
        city_name = []
        for key in city:
                city_name.append(city[key])
        return city_name

def name_code(name=None,code=None):
        if name == None and code != None:
                return city[code]
        elif name != None and code == None:
                return list(city.keys())[list(city.values()).index(name)]

def depart_arrival():
        list_info = list(itertools.permutations(city_name(), 2))
        return list_info

# 参考文章：
#   - 机场列表 - 维基百科
#     https://zh.wikipedia.org/wiki/%E4%B8%AD%E5%8D%8E%E4%BA%BA%E6%B0%91%E5%85%B1%E5%92%8C%E5%9B%BD%E6%9C%BA%E5%9C%BA%E5%88%97%E8%A1%A8
#   - 携程国际机票sign破解 https://blog.csdn.net/weixin_38927522/article/details/108214323


def get_cookie_bfa():
    random_str = "abcdefghijklmnopqrstuvwxyz1234567890"
    random_id = ""
    for _ in range(6):
        random_id += random.choice(random_str)
    t = str(int(round(time.time() * 1000)))

    bfa_list = ["1", t, random_id, "1", t, t, "1", "1"]
    bfa = "_bfa={}".format(".".join(bfa_list))
    # e.g. _bfa=1.1639722810158.u3jal2.1.1639722810158.1639722810158.1.1
    return bfa

# 获取调用携程 API 查询航班接口 Header 中所需的参数 sign
def get_sign(transaction_id, departure_city_code, arrival_city_code, departure_date):
    sign_value = transaction_id + departure_city_code + arrival_city_code + departure_date
    _sign = hashlib.md5()
    _sign.update(sign_value.encode('utf-8'))
    return _sign.hexdigest()

# 获取 transactionID 及航线数据
def get_transaction_id(departure_city_code, arrival_city_code, departure_date, cabin):
    flight_list_url = "https://flights.ctrip.com/international/search/api/flightlist" \
                      "/oneway-{}-{}?_=1&depdate={}&cabin={}&containstax=1" \
        .format(departure_city_code, arrival_city_code, departure_date, cabin)
    flight_list_req = requests.get(url=flight_list_url)
    if flight_list_req.status_code != 200:
        logging.error("get transaction id failed, status code {}".format(flight_list_req.status_code))
        return "", None

    try:
        flight_list_data = flight_list_req.json()["data"]
        transaction_id = flight_list_data["transactionID"]
    except Exception as e:
        logging.error("get transaction id failed, {}".format(e))
        return "", None

    return transaction_id, flight_list_data

# 获取航线具体信息与航班数据
def get_flight_info(departure_city_code, arrival_city_code, departure_date='2023-06-01', cabin='Y'):
    # 获取 transactionID 及航线数据
    transaction_id, flight_list_data = get_transaction_id(departure_city_code, arrival_city_code, departure_date, cabin)
    if transaction_id == "" or flight_list_data is None:
        return False, None

    # 获取调用携程 API 查询航班接口 Header 中所需的参数 sign
    sign = get_sign(transaction_id, departure_city_code, arrival_city_code, departure_date)

    # cookie 中的 bfa
    bfa = get_cookie_bfa()

    # 构造请求，查询数据
    search_url = "https://flights.ctrip.com/international/search/api/search/batchSearch"
    search_headers = {
        "transactionid": transaction_id,
        "sign": sign,
        "scope": flight_list_data["scope"],
        "origin": "https://flights.ctrip.com",
        "referer": "https://flights.ctrip.com/online/list/oneway-{}-{}"
                   "?_=1&depdate={}&cabin={}&containstax=1".format(departure_city_code, arrival_city_code,
                                                                   departure_date, cabin),
        "content-type": "application/json;charset=UTF-8",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "cookie": bfa,
    }
    r = requests.post(url=search_url, headers=search_headers, data=json.dumps(flight_list_data))

    if r.status_code != 200:
        logging.error("get flight info failed, status code {}".format(r.status_code))
        return False, None

    try:
        result_json = r.json()
        if result_json["data"]["context"]["flag"] != 0:
            logging.error("get flight info failed, {}".format(result_json))
            return False, None
    except Exception as e:
        logging.error("get flight info failed, {}".format(e))
        return False, None

    if "flightItineraryList" not in result_json["data"]:
        result_data = []
    else:
        result_data = result_json["data"]["flightItineraryList"]
    return result_data

def get_calendar_detail(departure_city_code, arrival_city_code, departure_date='2023-06-01', cabin='Y'):
    transaction_id, flight_list_data = get_transaction_id(departure_city_code, arrival_city_code, departure_date, cabin)
    bfa = get_cookie_bfa()
    detail_url = 'https://flights.ctrip.com/international/search/api/lowprice/calendar/getOwCalendarPrices'
    detail_headers = {
        "transactionid": transaction_id,
        "scope": flight_list_data["scope"],
        "origin": "https://flights.ctrip.com",
        "referer": "https://flights.ctrip.com/online/list/oneway-{}-{}"
                   "?_=1&depdate={}&cabin={}&containstax=1".format(departure_city_code, arrival_city_code,
                                                                   departure_date, cabin),
        "content-type": "application/json;charset=UTF-8",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "cookie": bfa
    }
    json_data = {
        "departCityCode": departure_city_code,
        "arrivalCityCode": arrival_city_code,
        "cabin": cabin
    }

    r = requests.get(url=detail_url, timeout=10, headers=detail_headers, params=json_data).json()
    return r

def get_flight(departcity, arrivalcity):
    depart_city_code = name_code(name=departcity)
    arrival_city_code = name_code(name=arrivalcity)
    resultstr = f"出发城市:{departcity}, 到达城市:{arrivalcity}\n起飞日期,机票价格\n"
    try:
        print(f'爬取{departcity}飞往{arrivalcity}的数据')
        delay = random.uniform(0.1, 1.5)
        res = get_calendar_detail(depart_city_code, arrival_city_code)['data']
        print('爬取完毕')
        if res == {}:
            pass
        else:
            res0 = list(res.items())
            res1 = [(departcity, arrivalcity) for _ in range(len(res0))]
            result = [(loc[0], loc[1], date, value) for (date, value), loc in zip(res0, res1)]
            sorted_result = sorted(result, key=lambda x: datetime.strptime(x[2], "%Y-%m-%d"))
            for i in sorted_result:
                resultstr = resultstr + i[2] + "," + str(i[3]) + "\n"
            return resultstr
    except requests.RequestException as e:
        print(f'{departcity}飞往{arrivalcity}爬取失败，失败原因为：{e}')

def get_Round_trip_flight_price(departcity, arrivalcity):
     result = get_flight(departcity, arrivalcity)
     result1 = result + "\n\n" + get_flight(arrivalcity, departcity)
     return result1
if __name__ == '__main__':
    print(get_Round_trip_flight_price("上海", "重庆"))
    # print(get_flight_info("SHA", "CKG", departure_date='2024-05-31'))