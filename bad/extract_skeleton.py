import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

# ====== 你要改的三行 ======
VIDEO_PATH = r"D:\GNN_BAD\huizi.mp4"
CSV_PATH   = r"D:\GNN_BAD\impact.csv"  # 第一列是帧号
OUTPUT_NPY = r"D:\GNN_BAD\skeleton_ld_lcw.npy"
WINDOW = 16    # t0-16 ~ t0，一共 17 帧，稍后截到 16
# =========================

# 1) 读 impact.csv
df = pd.read_csv(CSV_PATH, encoding="utf-8", engine="python")
df.columns = df.columns.str.strip().str.replace("\ufeff", "")
impact_frames = df.iloc[:, 0].astype(int).tolist()
print("击球帧列表:", impact_frames)

# 2) 打开视频
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ 无法打开视频，请检查 VIDEO_PATH 是否正确（路径、中文名等）。")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"总帧数: {total_frames}, FPS: {fps}, 分辨率: {w}x{h}")

# 3) 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# 我们要生成的数据: (N, T, V, C)
samples = []
valid_t0 = []

for idx, t0 in enumerate(impact_frames):
    if t0 < 0 or t0 >= total_frames:
        print(f"⚠ 跳过非法帧号 {t0}")
        continue

    start = max(0, t0 - WINDOW)
    end = t0
    frames_skel = []

    print(f"\n=== 处理第 {idx} 个击球: t0={t0}, 区间 {start}~{end} ===")

    for f in range(start, end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠ 帧 {f} 读取失败，中止该样本")
            frames_skel = []
            break

        # BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if not result.pose_landmarks:
            # 没检测到人，填 0
            joints = np.zeros((33, 2), dtype=np.float32)
        else:
            lm = result.pose_landmarks.landmark
            joints = np.array([[p.x, p.y] for p in lm], dtype=np.float32)  # 33x2，归一化到 [0,1]

        frames_skel.append(joints)

    if not frames_skel:
        continue

    seq = np.stack(frames_skel, axis=0)  # (T', 33, 2)

    # 如果长度不是 16，可以统一到 16（比如删掉第一帧）
    if seq.shape[0] > 16:
        seq = seq[-16:, :, :]
    elif seq.shape[0] < 16:
        # 不足就重复最后一帧补齐
        last = seq[-1:, :, :]
        pad = np.repeat(last, 16 - seq.shape[0], axis=0)
        seq = np.concatenate([seq, pad], axis=0)

    samples.append(seq)
    valid_t0.append(t0)
    print(f"✅ 样本长度: {seq.shape[0]} 帧")

pose.close()
cap.release()

if not samples:
    raise RuntimeError("❌ 没有成功生成任何 skeleton 样本，请检查击球帧和视频。")

X = np.stack(samples, axis=0)  # (N, 16, 33, 2)
print("最终数组形状:", X.shape)

np.save(OUTPUT_NPY, X)
print("🎉 已保存到:", OUTPUT_NPY)
print("对应的击球帧列表:", valid_t0)
