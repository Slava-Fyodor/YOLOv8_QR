'''
import cv2
from ultralytics import YOLO
import torch
from torch.serialization import safe_globals
from ultralytics.nn.tasks import DetectionModel

# 加入可信模型结构
safe_globals().add(DetectionModel)

# 加载模型并推理
model = YOLO('runs/detect/train2/weights/best.pt')
results = model('test.jpg', save=True, conf=0.25)

# 获取保存后的图像路径
# 默认保存到 runs/detect/predict/ 目录
save_path = results[0].save_dir / results[0].path.name

# 使用 OpenCV 加载并显示图像
img = cv2.imread(str(save_path))
cv2.imshow('Detection Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/train2/weights/best.pt')
results = model('main/test1.jpg', conf=0.25)

annotated = results[0].plot()   # 直接得到画好框的图(numpy)
cv2.imshow('Detection Result', annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()