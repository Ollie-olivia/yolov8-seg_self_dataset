from ultralytics import YOLO
import os

#
# # 加载训练好的模型
# model = YOLO('/media/ubuntu/0f083fd5-b631-4342-9812-7e262eaff979/YCQ/yellow_flower_detection/code/yoloseg_for_daylily/weights/yolo11s-seg/250711_all_train/weights/best.pt') # yolo11n-seg预训练的权重
# #PATH = 'weights/0260am_best.pt'
# # 测试单张图片
# #results = model('test_images/image1.jpg', save=True)  # 结果会保存在 `runs/detect/predict/`
#
# # 测试整个文件夹
# #results = model(PATH, save=True)  # 测试所有图片
#
# # 如果有标签，可以计算指标
# results = model.val(data='/media/ubuntu/0f083fd5-b631-4342-9812-7e262eaff979/YCQ/yellow_flower_detection/data/daylily2025/250711_all/250711_all.yaml')  # 需要提供数据集配置文件
#

# 加载训练好的模型
model_path = '/media/ubuntu/0f083fd5-b631-4342-9812-7e262eaff979/YCQ/yellow_flower_detection/code/yoloseg_for_daylily/weights/yolov8s-seg/250711_all_train/weights/best.pt'
model = YOLO(model_path)

# 从模型路径中提取模型名称信息
preweight_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(model_path))))  # 获取 "yolo11s-seg"


# 构建自定义的保存路径
base_weights_dir = '/media/ubuntu/0f083fd5-b631-4342-9812-7e262eaff979/YCQ/yellow_flower_detection/code/yoloseg_for_daylily/test'
save_dir = os.path.join(base_weights_dir, f"{preweight_name}_val")

print(f"模型目录: {model_path}")
print(f"结果将保存到: {save_dir}")

# 进行验证测试，并指定自定义的保存目录
results = model.val(
    data='/media/ubuntu/0f083fd5-b631-4342-9812-7e262eaff979/YCQ/yellow_flower_detection/data/daylily2025/250711_all/250711_all.yaml',
    project=save_dir,  # 指定保存目录
    name="evaluation"  # 子目录名称
)

print(f"测试完成，结果保存在: {save_dir}")