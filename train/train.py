from ultralytics import YOLO
import torch
import os

if __name__ == '__main__':

    # 数据集配置
    split_seed = 2025
    #dataset_path = "E:/master files/yellow_flower_detection/data/daylily2025/"
    dataset_path = "/media/ubuntu/0f083fd5-b631-4342-9812-7e262eaff979/YCQ/yellow_flower_detection/data/daylily2025/"
    train_name = '250711_all'

    # 检查yaml文件
    yaml_path = os.path.join(dataset_path, train_name, train_name + '.yaml')
    if not os.path.exists(yaml_path):
        print(f'YAML文件不存在: {yaml_path}')
        exit(1)
    else:
        print("配置路径:", yaml_path)

    # 存储路径保护
    #save_path = os.path.join(dataset_path, 'models')
    save_path ="/media/ubuntu/0f083fd5-b631-4342-9812-7e262eaff979/YCQ/yellow_flower_detection/code/yoloseg_for_daylily/weights"
    os.makedirs(save_path, exist_ok=True)
    print("保存路径:", save_path)

    # GPU检查
    # device = 'cuda:0'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # model = YOLO('yolov8n-seg.pt')  # 实例分割模型
    pretrained_model_path = '/media/ubuntu/0f083fd5-b631-4342-9812-7e262eaff979/YCQ/yellow_flower_detection/code/yoloseg_for_daylily/weights/original_weigths/yolov9c-seg.pt'  #绝对路径
    #pretrained_model_path = 'yolo10n-seg.pt'   #
    # 检查模型文件是否存在，如果不存在让Ultralytics自动下载
    if not os.path.exists(pretrained_model_path):
        print(f"模型文件不存在，将尝试自动下载: {pretrained_model_path}")

    try:
        model = YOLO(pretrained_model_path)
        print(f"成功加载模型: {pretrained_model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
    model = YOLO(pretrained_model_path)

    # 从预训练模型路径中提取模型名称（不含扩展名）
    pretrained_model_name = os.path.splitext(os.path.basename(pretrained_model_path))[0]
    model_save_path = os.path.join(save_path, pretrained_model_name)
    os.makedirs(model_save_path, exist_ok=True)
    print(f"模型将保存在: {model_save_path}")

    # 训练参数
    results = model.train(
        data=yaml_path,
        epochs=300,
        imgsz=640,
        batch=32,
        device=device,
        seed=split_seed,
        deterministic=True,
        optimizer='AdamW',
        close_mosaic=10,
        mixup=0.1,
        amp=False,#训练v8时禁用amp  训练yolo11s-seg时不禁用也会报错
        name=f'{train_name}_train',
        project=model_save_path  # 直接指定项目保存路径
    )

    # 模型自动保存在project目录中
    print(f"训练完成，模型已保存至: {model_save_path}")
