import os
import shutil
import yaml
from sklearn.model_selection import train_test_split


# ==== 修正后的数据集划分 ====
def split_dataset(dataset_path, seed=2025, yaml_name='dataset_config'):
    dataset_path = os.path.abspath(dataset_path)

    # 创建所有目录 (一次性)
    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dataset_path, subset, 'images'),
                    exist_ok=True)
        os.makedirs(os.path.join(dataset_path, subset, 'labels'),
                    exist_ok=True)

    # 获取图片列表
    image_dir = os.path.join(dataset_path, "images")
    all_images = sorted(
        [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

    # 8:1:1比例划分
    train_val, test_files = train_test_split(all_images,
                                             test_size=0.1,
                                             random_state=seed)
    train_files, val_files = train_test_split(train_val,
                                              test_size=1 / 9,
                                              random_state=seed)  # 修正比例

    # 复制文件而非移动
    def copy_files(file_list, subset):
        for img_file in file_list:
            # 复制图片
            img_src = os.path.join(image_dir, img_file)
            img_dst = os.path.join(dataset_path, subset, 'images', img_file)
            shutil.copy2(img_src, img_dst)  # 改用复制

            # 复制标签
            base_name = os.path.splitext(img_file)[0]
            label_src = os.path.join(dataset_path, "labels",
                                     f"{base_name}.txt")
            label_dst = os.path.join(dataset_path, subset, 'labels',
                                     f"{base_name}.txt")
            if os.path.exists(label_src):
                shutil.copy2(label_src, label_dst)  # 改用复制

    # 执行复制
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    # 创建YAML配置
    yaml_path = os.path.join(dataset_path, f"{yaml_name}.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(
            {
                # 'path': dataset_path,
                'train': os.path.join(dataset_path, '../train'),
                'val': os.path.join(dataset_path, 'val'),
                'test': os.path.join(dataset_path, '../test'),
                'names': ['0', '1', '2', '3'],
                'nc': 4
            },
            f)

    print("配置路径:", yaml_path)
    return yaml_path


# 数据集配置
split_seed = 2025
train_name = '250711_all'
dataset_path = os.path.join("E:/master files/yellow_flower_detection/data/daylily2025/", train_name)

# 划分数据集
yaml_path = split_dataset(
    dataset_path,
    seed=split_seed,
    yaml_name=train_name  # 确保YAML文件名匹配
)
