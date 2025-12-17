import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 骨架连接规则
# ==========================================
SKELETON_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24), # 躯干
    (11, 13), (13, 15), # 左臂
    (12, 14), (14, 16), # 右臂
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), # 左腿
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32), # 右腿
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10) # 脸
]

def play_skeleton():
    try:
        # 1. 读取数据
        data = np.load('skeleton_ld_lcw.npy')
        
        # 取最后一组数据 (16帧, 33点, 2坐标)
        frames_data = data[-1] 
        num_frames = len(frames_data)
        print(f"🎬 准备播放: 共 {num_frames} 帧")

        # 2. 设置画布
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # --- 关键：锁定坐标轴范围 ---
        # 如果不锁定，每一帧坐标轴都会变，画面会抖动
        all_x = frames_data[:, :, 0]
        all_y = frames_data[:, :, 1]
        margin = 0.05
        ax.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
        ax.set_ylim(np.max(all_y) + margin, np.min(all_y) - margin) # Y轴翻转
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title("Skeleton Animation (Loading...)")

        # 3. 初始化绘图元素 (一开始是空的)
        lines = []
        # 为每一根骨头创建一条线对象
        for _ in SKELETON_CONNECTIONS:
            line, = ax.plot([], [], 'b-', lw=2)
            lines.append(line)
        
        # 创建关节散点对象
        points, = ax.plot([], [], 'ro', ms=4)
        # 创建一个文字对象显示帧数
        frame_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='blue')

        # 4. 动画更新函数 (每一帧都会调用这个)
        def update(frame_idx):
            current_frame = frames_data[frame_idx]
            
            # 更新每一根骨头的位置
            for line, (start, end) in zip(lines, SKELETON_CONNECTIONS):
                x_start, y_start = current_frame[start]
                x_end, y_end = current_frame[end]
                line.set_data([x_start, x_end], [y_start, y_end])
            
            # 更新所有关节的位置
            points.set_data(current_frame[:, 0], current_frame[:, 1])
            
            # 更新标题
            ax.set_title(f"Skeleton Animation - Frame {frame_idx + 1}/{num_frames}")
            
            return lines + [points]

        # 5. 创建动画
        # interval=100 表示每帧间隔 100ms (即一秒10帧)
        ani = animation.FuncAnimation(
            fig, update, frames=num_frames, interval=100, blit=False, repeat=True
        )

        plt.show()

    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    play_skeleton()