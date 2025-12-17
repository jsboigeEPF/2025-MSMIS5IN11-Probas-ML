import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义骨架连接规则 (MediaPipe 标准定义)
# ==========================================
# 这就是“说明书”，告诉电脑哪个点连着哪个点
SKELETON_CONNECTIONS = [
    # 躯干
    (11, 12), (11, 23), (12, 24), (23, 24),
    # 左臂
    (11, 13), (13, 15),
    # 右臂
    (12, 14), (14, 16),
    # 左腿 (连到脚跟、脚尖)
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    # 右腿
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    # 脸部 (眼睛鼻子耳朵，可选)
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)
]

def draw_stick_figure():
    try:
        # 2. 读取文件
        file_name = 'skeleton_ld_lcw.npy'
        data = np.load(file_name)
        print(f"📦 数据形状: {data.shape}")
        
        # 3. 取最后一组(最后一个人)，第1帧
        # 使用 -1 代表取最后一个索引
        person = data[-1, 0, :, :] # (33, 2)
        
        plt.figure(figsize=(8, 8))
        
        # --- A. 画骨头 (线) ---
        for connection in SKELETON_CONNECTIONS:
            start_idx, end_idx = connection
            
            # 获取两个点的坐标
            x_start, y_start = person[start_idx][0], person[start_idx][1]
            x_end, y_end = person[end_idx][0], person[end_idx][1]
            
            # 画线 (颜色用蓝色)
            plt.plot([x_start, x_end], [y_start, y_end], c='blue', linewidth=2)

        # --- B. 画关节 (点) ---
        # 头部用红色，身体用绿色，区分一下
        plt.scatter(person[:, 0], person[:, 1], c='red', s=30, zorder=10)

        # 标出鼻子(0号点)作为方向参考
        plt.text(person[0,0], person[0,1], " Head", fontsize=10, color='red', fontweight='bold')

        plt.title("Skeleton Visualization (Connected)")
        plt.gca().invert_yaxis() # 这一步最关键！一定要翻转Y轴
        plt.axis('equal') # 保持比例，不然人会变扁
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"❌ 出错了: {e}")

if __name__ == "__main__":
    draw_stick_figure()