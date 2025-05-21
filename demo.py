import os
from saliency_models import gbvs, ittikochneibur
import cv2, time, numpy as np
import matplotlib.pyplot as plt

from saliency_models.gbvs import (
    compute_channel_maps,
    visualize_edges_and_lines,
    visualize_density_map,
    plot_length_vs_score,
    compute_texture_feature,
    detectFASTCorners,
    fastCornerDensity,
    compute_local_texture_saliency,
)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    texture_stats = []
    T = 10.0
    out_root = './results'
    subdirs = {
        'channel': os.path.join(out_root, 'channel_maps'),
        'saliency': os.path.join(out_root, 'saliency'),
        'edges': os.path.join(out_root, 'edges_lines'),
        'density': os.path.join(out_root, 'corner_density'),
        'texture': os.path.join(out_root, 'local_texture'),
    }
    for d in subdirs.values():
        ensure_dir(d)

    for i in range(1, 9):
        imname = f"./images/{i}.jpg"
        print("Processing", imname)

        img_color = cv2.imread(imname, cv2.IMREAD_COLOR)
        img_gray  = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # 1. 通道激活图
        ch_maps = compute_channel_maps(img_color)
        fig = plt.figure(figsize=(15,3))
        titles = ['Intensity','Grad Mag','Orientation','Contrast','Corner Density']
        for ch in range(5):
            ax = fig.add_subplot(1,5,ch+1)
            ax.imshow(ch_maps[ch], cmap='gray')
            ax.set_title(titles[ch]); ax.axis('off')
            cv2.imwrite(f"{subdirs['channel']}/{i}_ch{ch}.png", (ch_maps[ch]*255).astype(np.uint8))
        plt.tight_layout()
        plt.savefig(f"{subdirs['channel']}/{i}_all_channels.png", dpi=150)
        plt.close()

        # 2. GBVS / IKN 显著图
        sal_gbvs = gbvs.compute_saliency(img_color)
        sal_ikn  = ittikochneibur.compute_saliency(img_color)
        cv2.imwrite(f"{subdirs['saliency']}/{i}_gbvs.png", sal_gbvs)
        cv2.imwrite(f"{subdirs['saliency']}/{i}_ikn.png", sal_ikn)
        fig = plt.figure(figsize=(10,3))
        for idx, (m,t) in enumerate(zip(
            [cv2.cvtColor(img_color,cv2.COLOR_BGR2RGB), sal_gbvs, sal_ikn],
            ["Original","GBVS","IKN"] )):
            ax = fig.add_subplot(1,3,idx+1)
            ax.imshow(m, cmap='gray' if idx>0 else None)
            ax.set_title(t); ax.axis('off')
        plt.savefig(f"{subdirs['saliency']}/{i}_compare.png", dpi=150)
        plt.close()

        # 3.1 边缘+直线图
        roi = img_gray
        n, avg_len, score = compute_texture_feature(roi, T)
        texture_stats.append((n, avg_len, score))
        vis_el = visualize_edges_and_lines(roi)
        cv2.imwrite(f"{subdirs['edges']}/{i}_edges_lines.png", vis_el)

        # 3.2 角点密度图（伪彩色）
        kps = detectFASTCorners(roi.astype(np.uint8),20,True)
        density = fastCornerDensity(roi, kps, sigma=3, density_win=7)
        cmap = plt.get_cmap('jet')
        heat = (cmap(density)[:,:,:3]*255).astype(np.uint8)
        cv2.imwrite(f"{subdirs['density']}/{i}_density.png", cv2.cvtColor(heat, cv2.COLOR_RGB2BGR))

        # 3.3 局部纹理显著性
        sal_tex = compute_local_texture_saliency(
            img_gray,
            win_size=64, step=16,
            length_thresh=T,
            canny_thresh1=50, canny_thresh2=150,
            hough_threshold=20,
            min_line_length=15, max_line_gap=10
        )
        cmap2 = plt.get_cmap('jet')
        texmap = (cmap2(sal_tex, bytes=True)[:,:,:3])
        cv2.imwrite(f"{subdirs['texture']}/{i}_texture.png", cv2.cvtColor(texmap, cv2.COLOR_RGB2BGR))

    # 4. 保存散点图
    ensure_dir(out_root)
    plt.figure(figsize=(5,4))
    plot_length_vs_score(texture_stats, T)
    plt.savefig(f"{out_root}/length_vs_score_T{T}.png", dpi=150)
    plt.close()

    print("✅ 所有结果已保存至 ./results/")
