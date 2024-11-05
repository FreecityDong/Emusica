import cv2
import pandas as pd
from deepface import DeepFace
import numpy as np

# 视频文件路径
video_path = 'expressions.mp4'  # 请替换为你的视频文件路径
output_csv = 'emotion_analysis_results.csv'  # 输出的原始情绪CSV文件路径

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频的帧率（FPS）
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")

# 每隔0.1秒分析一次，计算需要跳过的帧数
frame_interval = int(fps * 0.1)  # 每0.1秒分析一次

# 用于保存情绪分析结果
emotion_results = []

# 初始化 DeepFace 人脸检测
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 逐帧读取视频并进行情绪分析
frame_count = 0
frame_emotions = []  # 用来存储当前0.5秒内的情绪数据

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 如果没有读取到帧，则跳出循环

    # 每隔frame_interval帧分析一次
    if frame_count % frame_interval == 0:
        try:
            # 转为灰度图进行人脸检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:  # 检测到人脸时才进行分析
                # 使用 DeepFace 进行表情分析
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

                # 获取识别到的主要情绪
                dominant_emotion = result[0]['dominant_emotion']

                # 计算当前帧的时间（秒）
                timestamp = frame_count / fps

                # 保存情绪分析结果，包括时间戳
                emotion_results.append({'timestamp': timestamp, 'frame': frame_count, 'emotion': dominant_emotion})

                # 将当前帧的情绪加入到frame_emotions中
                frame_emotions.append(dominant_emotion)

                print(f"Frame {frame_count} (Time: {timestamp:.2f}s): {dominant_emotion}")
            else:
                print(f"No face detected in frame {frame_count}")
        except Exception as e:
            print(f"Error analyzing frame {frame_count}: {e}")

    # 每0.5秒进行一次表情稳定性判断（基于0.1秒的分析数据）
    if frame_count % (fps * 0.5) == 0 and len(frame_emotions) > 0:
        # 统计frame_emotions中每个情绪出现的次数
        emotion_counts = {emotion: frame_emotions.count(emotion) for emotion in set(frame_emotions)}
        total_count = len(frame_emotions)

        for emotion, count in emotion_counts.items():
            # 如果某个表情在当前0.5秒内的重复出现率大于等于80%，记录该表情
            if count / total_count >= 0.8:
                print(f"Stable emotion detected: {emotion} with {count}/{total_count} occurrences.")
                # 将情绪结果添加到最终输出中
                timestamp = frame_count / fps
                emotion_results.append({'timestamp': timestamp, 'frame': frame_count, 'emotion': emotion})

        # 清空frame_emotions以便处理下一段时间的情绪
        frame_emotions = []

    frame_count += 1

# 释放视频资源
cap.release()

# 将结果保存到原始情绪CSV文件
df = pd.DataFrame(emotion_results)
df.to_csv(output_csv, index=False)
print(f"Emotion analysis results saved to '{output_csv}'")
