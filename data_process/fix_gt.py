# 指定要减去的值
specified_value = 96

# 打开原始文本文件和新的文本文件
with open('C:/Users/cjh/Desktop/MTMCT/eval/AIC19_S02.txt', 'r') as file_in, open('/eval/gt/ground_truth_fixed.txt', 'w') as file_out:
    # 逐行读取原始文本文件
    for line in file_in:
        # 将每行数据拆分为单个数字
        numbers = [int(num) for num in line.split()]

        # 减去指定值并更新第二个数字
        numbers[1] -= specified_value

        # 将处理后的数据写入新的文本文件
        file_out.write(' '.join(map(str, numbers)) + '\n')
