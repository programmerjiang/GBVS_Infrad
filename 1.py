import matplotlib.pyplot as plt

# 数据
labels = ['披挂坦克', '未披挂坦克', '黑体', '披挂方舱车', '未披挂方舱车']
values = [0.2967, 0.2758, 0.5105, 0.2731, 0.0462]

# 绘图
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values, color='skyblue', edgecolor='black')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
             f"{height:.2f}", ha='center', va='bottom')

# 轴标签与标题
plt.ylim(0, 1)
plt.ylabel("Camouflage effectiveness")
plt.title("各类目标的伪装效果对比")
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()
