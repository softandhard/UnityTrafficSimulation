data_dict = {}

# 从文本文件中读取数据
with open('output.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line.startswith('Frame'):
            frame = int(line.split(':')[0].split()[-1])
            parts = line.split(',')
            track_id = int(parts[0].split()[-1])
            xbr = int(parts[1].split()[-1])
            xtl = int(parts[2].split()[-1])
            ybr = int(parts[3].split()[-1])
            ytl = int(parts[4].split()[-1])
            if frame in data_dict:
                data_dict[frame].append((track_id, xbr, xtl, ybr, ytl))
            else:
                data_dict[frame] = [(track_id, xbr, xtl, ybr, ytl)]

# 将数据写入新的txt文件
with open('output2.txt', 'w') as f:
    for frame in sorted(data_dict.keys()):
        line = ""
        for data in data_dict[frame]:
            line += f"Frame {frame}: Track ID: {data[0]}, xbr: {data[1]}, xtl: {data[2]}, ybr: {data[3]}, ytl: {data[4]} | "
        f.write(line + '\n')

