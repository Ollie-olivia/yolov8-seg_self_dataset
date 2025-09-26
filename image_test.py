import cv2

# 加载原始图像
image_path = "E:/读研文件/黄花菜检测/yolov8-seg_for_yellow/yellow_train_001.jpg"
image = cv2.imread(image_path)

# 打印原始图像的分辨率

print(image.shape)