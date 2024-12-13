import xml.etree.ElementTree as ET

# 读取XML文件
tree = ET.parse(r'D:\Users\ddd\Desktop\Multi-Drone-Multi-Object-Detection-and-Tracking\new_xml\1\26-1.xml')
root = tree.getroot()

# 创建一个列表来存储每个track的数据
data_list = []

# 遍历每个track和box
for track in root.iter('track'):
    track_id = track.attrib['id']
    for box in track.iter('box'):
        frame = box.attrib['frame']
        xbr = box.attrib['xbr']
        xtl = box.attrib['xtl']
        ybr = box.attrib['ybr']
        ytl = box.attrib['ytl']
        data = (frame, track_id, xbr, xtl, ybr, ytl)
        data_list.append(data)

# 将数据写入新的txt文件
with open('output.txt', 'w') as f:
    for data in data_list:
        line = f"Frame {data[0]}: Track ID: {data[1]}, xbr: {data[2]}, xtl: {data[3]}, ybr: {data[4]}, ytl: {data[5]}"
        f.write(line + '\n')

