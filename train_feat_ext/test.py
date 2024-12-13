import os

# 文件夹路径
folder_path = r'C:\迅雷下载\VehicleID\data\2016-CVPR-VehicleReId-dataset\5B'


def get_prefix_num(filename):
    """
    从文件名中提取前缀数字。
    """
    prefix_num = filename.split('_', 1)[0]
    return int(prefix_num)

def rename_images(folder_path, start_value):
    """
    将第一个数字相同的图片名修改为指定的数值，之后的不同数值的图片名第一个数字依次递增。
    :param folder_path: 图片所在的文件夹路径
    :param start_value: 修改后的起始数值
    """
    # 获取所有文件名并根据第一个数字排序
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort(key=get_prefix_num)

    if not files:
        print("No files found in the directory.")
        return

    # 初始化
    prev_num = get_prefix_num(files[0])
    current_value = start_value
    # 存储已处理的数字，以防止重复递增
    processed_nums = {}

    for file_name in files:
        prefix_num = get_prefix_num(file_name)

        # 如果当前的数字已经处理过，则使用相同的修改后的数字
        if prefix_num in processed_nums:
            new_num = processed_nums[prefix_num]
        else:
            # 如果是新的数字，更新current_value并记录处理过的数字
            if prefix_num != prev_num:
                current_value += 1
                prev_num = prefix_num
            new_num = current_value
            processed_nums[prefix_num] = new_num

        # 构建新的文件名并重命名
        new_name = f"{new_num}_" + "_new-".join(file_name.split('_')[1:])
        os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_name))
        print(f'Renamed "{file_name}" to "{new_name}"')

# 162, 301, 432, 564, 780
# Example usage
start_value = 565  # 从这个数值开始，根据你的需求修改这个值
rename_images(folder_path, start_value)