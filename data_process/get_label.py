# 打开原始txt文件和新的txt文件
with open('C:/Users/cjh/Desktop/MTMCT/eval/ground_truth_validation.txt', 'r') as file_in, \
        open('C:/Users/cjh/Desktop/MTMCT/eval/ground_truth_test.txt', 'w') as file_out:
    # 读取原始txt文件的所有行
    lines = file_in.readlines()

    # 指定要提取的数据范围
    start_line = 0  # 第1行  行数减一
    end_line = 20955  # 第5081行  行数减一

    # 提取指定范围的数据并写入新的txt文件
    for line in lines[start_line:end_line + 1]:
        file_out.write(line)

