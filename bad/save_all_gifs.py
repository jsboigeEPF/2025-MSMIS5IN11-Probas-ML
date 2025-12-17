import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import os

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

def batch_generate_gifs():
    try:
        # 1. 读取数据
        filename = 'skeleton_ld_lcw.npy'
        if not os.path.exists(filename):
            print(f"❌ 找不到文件: {filename}")
            return
            
        data = np.load(filename)
        # 形状应该是 (7, 16, 33, 2) -> (组数, 帧数, 关节数, 坐标)
        num_groups = data.shape[0]
        num_frames = data.shape[1]
        
        print(f"📦 发现 {num_groups} 组数据，每组 {num_frames} 帧。开始批量生成...")

        # 2. 循环处理每一组
        for group_idx in range(num_groups):
            print(f"   正在处理第 {group_idx} 组 ({group_idx + 1}/{num_groups})...")
            
            frames_data = data[group_idx]
            
            # 创建画布
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # 锁定这一组的坐标范围
            all_x = frames_data[:, :, 0]
            all_y = frames_data[:, :, 1]
            margin = 0.05
            ax.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
            ax.set_ylim(np.max(all_y) + margin, np.min(all_y) - margin) # Y轴翻转
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Group {group_idx}")

            # 初始化绘图元素
            lines = []
            for _ in SKELETON_CONNECTIONS:
                line, = ax.plot([], [], 'b-', lw=2)
                lines.append(line)
            points, = ax.plot([], [], 'ro', ms=4)

            # 更新函数
            def update(frame_idx):
                current_frame = frames_data[frame_idx]
                for line, (start, end) in zip(lines, SKELETON_CONNECTIONS):
                    line.set_data(
                        [current_frame[start, 0], current_frame[end, 0]], 
                        [current_frame[start, 1], current_frame[end, 1]]
                    )
                points.set_data(current_frame[:, 0], current_frame[:, 1])
                return lines + [points]

            # 生成动画
            ani = animation.FuncAnimation(
                fig, update, frames=num_frames, interval=150, blit=True
            )
            
            # 保存文件
            output_name = f'skeleton_group_{group_idx}.gif'
            ani.save(output_name, writer='pillow', fps=8)
            plt.close(fig) # 关闭画布，释放内存

        print("✅ 所有 GIF 已生成完毕！请在左侧文件列表查看 skeleton_group_*.gif")

    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    batch_generate_gifs()