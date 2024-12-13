import requests

# 指定要请求的网页 URL
url = "https://www.qweather.com/weather/beijing-101010100.html"  # 替换为目标 URL

try:
    # 发起 GET 请求
    response = requests.get(url)

    # 检查请求是否成功
    if response.status_code == 200:
        # 设置响应编码
        response.encoding = response.apparent_encoding  # 自动检测编码
        print("响应状态编码：", response.encoding)
        # 获取网页内容
        content = response.text

        print("网页内容：")
        print(content)  # 打印获取的数据
        print("响应长度：", len(content))
    else:
        print(f"请求失败，状态码：{response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"请求发生错误：{e}")

