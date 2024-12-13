# 读取第一个文本文件的内容
with open('output5.txt', 'r') as file1:
    data1 = file1.read()

# 读取第二个文本文件的内容
with open('file2.txt', 'r') as file2:
    data2 = file2.read()

# 将两个文件的内容合并
merged_data = data1 + '\n' + data2

# 将合并后的内容写入新的文本文件
with open('output6.txt', 'w') as merged_file:
    merged_file.write(merged_data)
