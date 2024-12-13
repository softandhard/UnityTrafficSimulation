# Patch size
img_h = 384
img_w = 384

# Configurations (model)
model_name = 'resnet50_ibn_a'
pretrained = './nets/%s.pth' % model_name
avg_type = 'gap'
num_ide_class = 776

# Configurations (train)
num_samples_per_id = 10
k_num = 2
p_num = 2
batch_size = k_num * p_num
seed = 10000
num_epoch = 120
milestones = [40, 90]
init_lr = 0.00035

# Path
tr_data_dir = 'C:/Users/cjh/Desktop/Fast_Online_MTMCT-main/dataset/VeRi/'
save_path = './outputs/%s_%s/' % (model_name, avg_type)
log_path = save_path + 'log.txt'
