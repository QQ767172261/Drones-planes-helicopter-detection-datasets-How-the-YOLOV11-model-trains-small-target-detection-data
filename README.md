无人机、飞机、直升机检测，yolo和voc两种目标检测标注方式
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bbe6111f0abd4c47896adc7d90c5f937.png)

标签数：11998
框数: 9975![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6125bb0ed90a4caf90a57c00f3b19165.png)


含3个类别以及框数：
AirPlane (2535), Drone (4812), Helicopter (2628)![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/74a75e26638e432880ac419d05ba99b2.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/76207657378045ef9c50ad4d38a2a391.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ed04a99101464e40b3b5e60ff844149e.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ad181dc7d93f49149431f3dec0f88dd3.png)
# 无人机、飞机、直升机目标检测数据集

## 数据集信息表格

| 项目 | 详情 |
| --- | --- |
| 数据集名称 | 空中飞行器目标检测数据集 |
| 总图像数量 | 11998张 |
| 标注总框数 | 9975个 |
| 标注格式 | YOLO、VOC |
| 类别总数 | 3类 |
| 类别分布 | AirPlane(飞机):2535框   Drone(无人机):4812框   Helicopter(直升机):2628框 |

## 标签对照表

| 类别ID | 英文标签 | 中文标签 |
| --- | --- | --- |
| 0 | AirPlane | 飞机 |
| 1 | Drone | 无人机 |
| 2 | Helicopter | 直升机 |

## 应用场景

1. 低空安防、机场周边空域监控，识别闯入的无人机、直升机、民航飞机
2. 边境、厂区、重要基础设施的视频空中目标预警，区分飞行器种类
3. 遥感图像、摄像头视频流空中目标检测，辅助空域态势感知
4. 反无人机系统图像识别模块，自动区分无人机与正常飞行器，降低误报

## YOLOv11训练代码

### 1、数据集配置文件 aircraft.yaml

```
path: ./aircraft_dataset
train: images/train
val: images/val

names:
  0: AirPlane
  1: Drone
  2: Helicopter
```

### 2、训练脚本 train_aircraft.py

```
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="aircraft.yaml",
        epochs=80,
        imgsz=640,
        batch=12,
        device=0,
        workers=4,
        project="aircraft_detect",
        name="yolo11n_aircraft",
        amp=True,
        mosaic=0.8,
        conf=0.25
    )
```

### 3、推理脚本 predict_aircraft.py

```
from ultralytics import YOLO
import cv2

model = YOLO("./aircraft_detect/yolo11n_aircraft/weights/best.pt")

def detect_aircraft(img_path):
    res = model(img_path, conf=0.3)
    for r in res:
        img = r.plot()
        cv2.imwrite("aircraft_result.jpg", img)
    return img

if __name__ == '__main__':
    detect_aircraft("test_air.jpg")
```

## 环境安装

```
pip install ultralytics opencv-python
```

