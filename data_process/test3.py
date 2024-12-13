data_dict = {}

# 从文本文件中读取数据
with open('output2.txt', 'r') as file:
    for line in file:
        line = line.strip()
        frame = int(line.split(':')[0].split()[-1])
        parts = line.split('|')
        for part in parts:
            if part.strip():  # 确保部分不为空
                track_id = int(part.split('Track ID: ')[1].split(',')[0])
                xbr = int(part.split('xbr: ')[1].split(',')[0])
                xtl = int(part.split('xtl: ')[1].split(',')[0])
                ybr = int(part.split('ybr: ')[1].split(',')[0])
                ytl = int(part.split('ytl: ')[1].strip())
                if frame in data_dict:
                    data_dict[frame].append((track_id, xbr, xtl, ybr, ytl))
                else:
                    data_dict[frame] = [(track_id, xbr, xtl, ybr, ytl)]

# 将数据写入新的txt文件
with open('output3.txt', 'w') as f:
    for frame in sorted(data_dict.keys()):
        for data in data_dict[frame]:
            line = f"Frame: {frame} Track ID: {data[0]}, xbr: {data[1]}, xtl: {data[2]}, ybr: {data[3]}, ytl: {data[4]}"
            f.write(line + '\n')
