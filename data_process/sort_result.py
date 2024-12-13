from collections import defaultdict

# 读取文件内容并按照每行第一个数字进行分组
data_dict = defaultdict(list)
with open('/online_MTMC/outputs/Veri-e6e/mtmc_resnet50_ibn_a_gap.txt', 'r') as file:
    for line in file:
        first_num = int(line.split()[0])  # 获取每行的第一个数字
        data_dict[first_num].append(line)  # 将行数据添加到对应的键值列表中

# 按照数字顺序输出数据
with open('C:/Users/cjh/Desktop/MTMCT/data_process/fixed_result.txt', 'w') as output_file:
    for key in sorted(data_dict.keys()):
        for line in data_dict[key]:
            output_file.write(line)
