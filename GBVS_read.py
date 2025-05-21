from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# === 1. 使用 PIL 加载图像（支持中文路径） ===
image_path = 'D:/work/2025/地面坦克目标特性/Program/Python/gbvs-master/results/saliency/1_gbvs.png'

try:
    image = Image.open(image_path).convert('L')  # 转灰度图
except Exception as e:
    raise FileNotFoundError(f"无法读取图像，请检查路径或文件有效性。\n{e}")

img_np = np.array(image).astype(np.float32) / 255.0  # 归一化
print(f"✅ 图像读取成功，尺寸为：{img_np.shape}")

# === 2. 定义交互函数 ===
ref_coords = []

def onselect(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    roi = img_np[ymin:ymax, xmin:xmax]
    if roi.size == 0:
        print("⚠️ 选区为空，请重新选择。")
        return
    mean_val = np.mean(roi)
    print(f"👉 框选区域归一化灰度均值：{mean_val:.4f}")

# === 3. 可视化图像 + 框选交互 ===
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_np, cmap='gray')
ax.set_title("请鼠标框选区域，松开后显示均值")

toggle_selector = RectangleSelector(
    ax, onselect,
    useblit=True,
    button=[1],
    minspanx=5, minspany=5,
    interactive=True
)

plt.show()