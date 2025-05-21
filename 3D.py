import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取图片并转换为灰度
image_path = 'D:/work/2025/地面坦克目标特性/Program/Python/gbvs-master/results/saliency/1_gbvs.png'

image_orig = Image.open(image_path).convert('L')
orig_w, orig_h = image_orig.size

# === 缩放图像以加快绘制 ===
resize_w, resize_h = 100, 100
image = image_orig.resize((resize_w, resize_h))
z = np.array(image).astype(np.float32)

# === 归一化图像灰度值 ===
z = (z - z.min()) / (z.max() - z.min())

# === 构建原始像素坐标并翻转Y ===
x = np.linspace(0, orig_w - 1, resize_w)
y = np.linspace(0, orig_h - 1, resize_h)
X, Y = np.meshgrid(x, y[::-1])  # 翻转Y轴，(0,0)左下
Z = z[::-1, :]                  # 同步翻转图像数据

# === 绘制 3D 表面图 & 2D 热图 ===
fig = plt.figure(figsize=(14, 6))

# --- 子图 1：交换 X、Y 的 3D 热图 ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(Y, X, Z, cmap='jet',  # 注意这里交换顺序
                        rstride=1, cstride=1, linewidth=0, antialiased=False)
fig.colorbar(surf, ax=ax1, shrink=0.6, aspect=10, label='Normalized Intensity')
ax1.set_title('3D Heatmap (X and Y Swapped)')
ax1.set_xlabel('Y Pixel')   # 标签对调
ax1.set_ylabel('X Pixel')
ax1.set_zlabel('Height')
ax1.view_init(elev=30, azim=40)


# --- 子图 2：二维热图 ---
ax2 = fig.add_subplot(1, 2, 2)
im = ax2.imshow(Z, cmap='jet', origin='lower',
                extent=[x.min(), x.max(), y.min(), y.max()])
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label='Normalized Intensity')
ax2.set_title('2D Heatmap')
ax2.set_xlabel('X Pixel')
ax2.set_ylabel('Y Pixel')

plt.tight_layout()
plt.show()