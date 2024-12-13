import os

# 指定图片文件所在的文件夹路径
folder_path = 'D:/Users/ddd/VeRi/AIC22/test/S02/c009/frame'

# 获取文件夹中所有文件的列表
files = os.listdir(folder_path)

# 设置计数器
count = 1

# 遍历文件夹中的文件
for file in files:
    if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):  # 只处理图片文件
        # 构建新的文件名
        new_name = f'{count:07d}.jpg'
        # 重命名文件
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_name))
        count += 1

