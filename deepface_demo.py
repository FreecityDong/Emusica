from deepface import DeepFace

# 加载并分析图像
img_path = "face1.jpeg"
result = DeepFace.analyze(img_path, actions=['emotion'])

# 输出结果
print("Detected emotion:", result[0]['dominant_emotion'])