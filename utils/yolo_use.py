from ultralytics import YOLO
import cv2
import numpy as np
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from scipy.ndimage import convolve
from collections import deque


def skeletonize_mask(mask):
    try:
        # 将彩色掩码转化为灰度图：
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) #gray:np[480,640]  0...0...212...212...0...0...0
        # 二值化：将灰度图转为黑白二值图像：
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        # 形态学开运算：去除小噪点，保留主体轮廓：
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        #骨架化：提取二值图像的骨架（单像素宽的线条）（使用scikit-image的骨架算法）：
        skeleton = skeletonize(binary > 0)
        #数据类型转换：将布尔型骨架转为uint8（0-255）格式：
        return img_as_ubyte(skeleton)  #numpy
    except Exception as e:
        print(f"骨架化处理错误: {e}")
        return np.zeros_like(mask[:,:,0], dtype=np.uint8)

####################################推理######################################
model = YOLO('train/exp/weights/best.pt')

results_ori=model('yellow_train_001.jpg')
results = results_ori[0]   #实验显示，单张照片的results其实等同于results_ori
color_image = results.plot()
mask = results.masks[0].data.cpu().numpy().squeeze() #masks.data.shape：tensor[3, 480, 640]。因此[0]表示取第一个检测到的实例
mask_rgb_cv = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#canvas = np.zeros_like(color_image)#根据color_image的shape初始化全黑的一个画布
canvas=np.zeros_like(mask_rgb_cv)
pink=[203, 192, 255]
canvas[mask > 0.5] = pink  #目标物体部分为粉色像素，其余为黑色像素
skeleton = skeletonize_mask(canvas) #骨架化掩码


##################################### 查看中间结果属性##############################
#print('results_ori:',results_ori)
#print('results结果:',results)
#print('masks:',results.masks) #object
print('masks.data.shape',results.masks.data.shape)
#print('mask结果',mask)
#print('mask.shape',mask.shape)  #yellow_test:(640, 480)  yellow_train_001:(480,640)
print('canvas.shape',canvas.shape)#(1706,1279,3)        (1536,2048,3)

'''
注意：使用yolov8n.pt会导致masks为None,因为yolov8n.pt为目标检测，而非实例分割模型（如 yolov8n-seg.pt）。
目标检测模型不输出掩码，因此 masks 为 None
'''

####################################绘制#######################################
# 缩放color_image图像以防原始图像太大显示不出来完整的图
h,w= color_image.shape[:2]
screen_width, screen_height = 1920, 1080  # 例如 1080p 屏幕
scale = min(screen_width / w, screen_height / h) * 0.6  # 90% 屏幕空间
new_size = (int(w * scale), int(h * scale))
resized_image = cv2.resize(color_image, new_size, interpolation=cv2.INTER_AREA)

cv2.imshow("color_image缩放后:", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("mask:", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("赋值颜色后的canvas:",canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("skeleton:",skeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
使用best.pt检测猫图像，因为图像中不含有best.pt能检测的黄花，因此masks肯定为None
results:
ultralytics.engine.results.Results object with attributes:

boxes: ultralytics.engine.results.Boxes object
keypoints: None
masks: None
names: {0: 'yellow'}
obb: None
orig_img: array([[[168, 171, 174],
        [171, 174, 177],
        [169, 173, 178],
        ...,
        [ 45,  69, 101],
        [ 44,  71, 104],
        [ 47,  74, 107]],

       [[167, 172, 175],
        [169, 174, 177],
        [169, 173, 178],
        ...,
        [ 45,  69, 101],
        [ 46,  70, 102],
        [ 48,  72, 104]],

       [[168, 173, 176],
        [170, 175, 178],
        [168, 176, 178],
        ...,
        [ 45,  69, 101],
        [ 46,  70, 102],
        [ 48,  72, 104]],

       ...,

       [[ 13,  23,  30],
        [ 14,  24,  31],
        [ 14,  24,  31],
        ...,
        [ 51,  66,  80],
        [ 52,  66,  83],
        [ 52,  66,  83]],

       [[ 12,  22,  29],
        [ 13,  23,  30],
        [ 13,  23,  30],
        ...,
        [ 51,  65,  82],
        [ 52,  66,  83],
        [ 51,  65,  82]],

       [[ 12,  22,  29],
        [ 13,  23,  30],
        [ 13,  23,  30],
        ...,
        [ 51,  65,  82],
        [ 52,  66,  83],
        [ 52,  66,  83]]], dtype=uint8)
orig_shape: (480, 640)
path: 'image0.jpg'
probs: None
save_dir: 'runs\\segment\\predict'
speed: {'preprocess': 2.169500003219582, 'inference': 555.292000004556, 'postprocess': 0.7374999986495823}
'''

###
# | 属性名                  | 类型              | 说明                           |
# | -------------------- | --------------- | ---------------------------- |
# | `results.boxes`      | `Boxes`对象       | 边界框信息（坐标、置信度、类别）             |
# | `results.masks`      | `Masks`对象       | 分割掩码（像素级掩码，需解码）              |
# | `results.keypoints`  | `Keypoints`对象   | 关键点检测（如姿态估计）                 |
# | `results.probs`      | `Probs`对象       | 分类概率（分类模型用）                  |
# | `results.orig_img`   | `numpy.ndarray` | 原始输入图像（BGR格式）                |
# | `results.orig_shape` | `tuple`         | 原始图像尺寸 `(高, 宽)`              |
# | `results.names`      | `dict`          | 类别ID到名称的映射（如`{0: 'person'}`） |
'''
若使用yolov8n-seg.pt的模型参数来检测猫图像，masks属性不为None：
results如下:
results: ultralytics.engine.results.Results object with attributes:

boxes: ultralytics.engine.results.Boxes object
keypoints: None
masks: ultralytics.engine.results.Masks object
names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
obb: None
orig_img: array([[[167, 170, 174],
        [169, 172, 176],
        [172, 175, 179],
        ...,
        [121, 121, 127],
        [116, 116, 122],
        [110, 110, 116]],

       [[170, 173, 177],
        [171, 174, 178],
        [172, 175, 179],
        ...,
        [110, 110, 116],
        [108, 108, 114],
        [106, 106, 112]],

       [[173, 176, 180],
        [172, 175, 179],
        [171, 174, 178],
        ...,
        [ 96,  96, 102],
        [ 99,  99, 105],
        [102, 102, 108]],

       ...,

       [[101, 102,  93],
        [ 95,  96,  87],
        [ 94,  95,  86],
        ...,
        [207, 199, 200],
        [209, 201, 202],
        [209, 201, 202]],

       [[101, 102,  93],
        [100, 101,  92],
        [100, 101,  92],
        ...,
        [210, 202, 203],
        [220, 212, 213],
        [227, 219, 220]],

       [[ 97,  98,  89],
        [101, 102,  93],
        [102, 103,  94],
        ...,
        [197, 189, 190],
        [211, 203, 204],
        [220, 212, 213]]], dtype=uint8)
orig_shape: (480, 640)
path: 'E:\\读研文件\\代码\\黄花菜检测\\cat.jpg'
probs: None
save_dir: 'runs\\segment\\predict'
speed: {'preprocess': 3.2873999880393967, 'inference': 154.0065000008326, 'postprocess': 225.1689999975497}
mask [[          0           0           0 ...           0           0           0]
 [          0           0           0 ...           0           0           0]
 [          0           0           0 ...           0           0           0]
 ...
 [          0           0           0 ...           0           0           0]
 [          0           0           0 ...           0           0           0]
 [          0           0           0 ...           0           0           0]]
 
而masks: ultralytics.engine.results.Masks object with attributes:
data: tensor([[[0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         ...,
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.]],

        [[0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         ...,
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.]]])
orig_shape: (480, 640)
shape: torch.Size([2, 480, 640])
xy: [array([[        213,         102],
       [        213,         105],
       [        211,         107],
       ...,
       [        250,         107],
       [        248,         105],
       [        248,         102]], dtype=float32), array([[        221,         103],
       [        220,         104],
       [        216,         104],
       ...,
       [        244,         104],
       [        242,         104],
       [        241,         103]], dtype=float32)]
xyn: [array([[    0.33281,      0.2125],
       [    0.33281,     0.21875],
       [    0.32969,     0.22292],
       ...,
       [    0.39062,     0.22292],
       [     0.3875,     0.21875],
       [     0.3875,      0.2125]], dtype=float32), array([[    0.34531,     0.21458],
       [    0.34375,     0.21667],
       [     0.3375,     0.21667],
       ...,
       [    0.38125,     0.21667],
       [    0.37813,     0.21667],
       [    0.37656,     0.21458]], dtype=float32)]
mask [[          0           0           0 ...           0           0           0]
 [          0           0           0 ...           0           0           0]
 [          0           0           0 ...           0           0           0]
 ...
 [          0           0           0 ...           0           0           0]
 [          0           0           0 ...           0           0           0]
 [          0           0           0 ...           0           0           0]]

'''