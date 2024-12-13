
with open('output3.txt', 'r') as file:
    with open('output4.txt', 'w') as outfile:
        for line in file:
            line = line.strip()
            try:
                frame = int(line.split(':')[1].split()[0])
                track_id = int(line.split('Track ID: ')[1].split(',')[0])
                xbr = int(line.split('xbr: ')[1].split(',')[0])
                xtl = int(line.split('xtl: ')[1].split(',')[0])
                ybr = int(line.split('ybr: ')[1].split(',')[0])
                ytl = int(line.split('ytl: ')[1])
                xmin = xtl
                ymin = ytl
                width = xbr - xtl
                height = ybr - ytl
                outfile.write(f"{1} {track_id} {frame} {xmin} {ymin} {width} {height}\n")
            except (IndexError, ValueError):
                print(f"Issue with line: {line}")
