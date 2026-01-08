# Face Anti-Spoofing
Several face anti-spoofing models from github.
There are eight models to detect spoofing. For real-life projects **combination of m6 and m8** gives a high accuracy (more than 99%).

1. M1: https://github.com/zeusees/HyperFAS
2. M2: https://github.com/emadeldeen24/face-anti-spoofing
3. M3: https://github.com/Davidzhangyuanhan/CelebA-Spoof
4. M4: https://github.com/JinghuiZhou/awesome_face_antispoofing
5. M5: Simple CNN model. I have trained this model from scratch.
6. M6: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
7. M7: https://github.com/Saiyam26/Face-Anti-Spoofing-using-DeePixBiS
8. M8: Another CNN model trained using [darknet](https://github.com/AlexeyAB/darknet) or ONNX(mobilenet_v3_small). We have trained this model with a private dataset. This model tries to detect mobile or printed photo in the input image. You can load onnx model by passing `load_onnx` to class `M8FaceAntiSpoofing(load_onnx=True)`.
9. M9: https://github.com/johnraivenolazo/face-antispoof-onnx

Requirements:

`python ==> 3.6`

`pip3 install pip --upgrade`

`pip3 install -r requirements.txt`

Usage:

`python3 test.py`

## 📊 Performance Results

| Model | Accuracy | Average Error in Confidence | Inference Time (per face) | Speed (FPS) on CPU |
|-------|----------|-----------------------------|---------------------------|--------------------|
| M6    | 95.32%   | 6.58%                       | 113.87 ms                | 8.78               |
| M8    | 90.97%   | 16.61%                      | 35.30 ms                 | 28.33              |
| M9    | 72.24%   | 29.02%                      | 21.23 ms                 | 47.11              |

*Tested on 299 samples of these repository benchmark. FPS calculated as: 1000 ms / inference_time_in_ms*