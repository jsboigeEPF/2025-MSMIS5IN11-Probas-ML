import cv2
import pandas as pd
import os

VIDEO_PATH = "London2012.mp4"      # 你的视频
CSV_PATH   = r"D:\GNN_BAD\impact.csv" # 上面那个 CSV
OUTPUT_DIR = "clips"            # 输出 clip 目录
WINDOW = 16                     # 取 t0-16 → t0，总 17 帧

os.makedirs(OUTPUT_DIR, exist_ok=True)

# load impact frames
df = pd.read_csv(CSV_PATH)
impact_frames = df['frame'].tolist()

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
print("Video FPS:", fps)

# process each impact frame
for i, t0 in enumerate(impact_frames):
    start = max(0, t0 - WINDOW)
    end   = t0

    output_path = os.path.join(OUTPUT_DIR, f"clip_{i}_{start}_{end}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ))

    # extract frames
    for f in range(start, end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    writer.release()
    print("Saved:", output_path)

cap.release()
