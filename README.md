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

Requirements:

`python ==> 3.6`

`pip3 install pip --upgrade`

`pip3 install -r requirements.txt`

Usage:

`python3 test.py`
