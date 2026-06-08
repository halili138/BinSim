import matplotlib.pyplot as plt
import numpy as np

OPTIONS = {
    # 字体设置
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 14,                    # 基础字体大小
    
    # 数学字体设置
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.default": "regular",      # 确保公式字体不是默认的斜体
    
    # 坐标轴设置
    "axes.labelsize": 18,               # 坐标轴标签字号
    "axes.titlesize": 20,               # 标题字号
    "axes.labelweight": "bold",         # 坐标轴标签加粗
    
    # 刻度设置
    "xtick.labelsize": 16,              # x轴刻度标签字号
    "ytick.labelsize": 16,              # y轴刻度标签字号
    "xtick.direction": "in",            # 刻度向内
    "ytick.direction": "in",            # 刻度向内
    
    # 图例设置
    "legend.fontsize": 12,              # 图例字号 (稍微调小一点，避免遮挡曲线)
    "legend.framealpha": 0.95,          # 图例透明度
    
    # 图形设置
    "figure.dpi": 300,                  # 高清输出
    "figure.autolayout": False,         # 使用tight_layout替代
    
    # 线条设置
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
    
    # 网格设置
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}


MOL_LATEX = {
    "C2H4": r"C_2H_4",
    "CO2": r"CO_2",
    "HCN": r"HCN",
    "H2CO": r"H_2CO",
    "N2": r"N_2",
    "C2": r"C_2",
    "O2": r"O_2",
    "H2O": r"H_2O",
    "C2H6": r"C_2H_6",
    "HCL": r"HCl",    # 修正 Cl 小写
    "CH4": r"CH_4",
    "SIH4": r"SiH_4", # 修正 Si 小写
    "BEH2": r"BeH_2", # 修正 Be 小写
    "HF": r"HF",
    "NH3": r"NH_3",
    "CO": r"CO",
    "CH3OH": r"CH_3OH",
    "HCOOH": r"HCOOH"
}


def plot_speedup_ratio(save_filename="speedup_chart.png"):
    """
    绘制以 a100 为基准的速度倍率柱状图
    """
    # 如果遇到中文显示为方块的问题，请取消以下两行的注释并根据你的系统选择合适的字体
    # plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows用黑体
    # plt.rcParams['axes.unicode_minus'] = False 
    plt.rcdefaults()
    plt.rcParams.update(OPTIONS)

    # 1. 准备原始时间数据
    systems = ["N2", "O2", "HCN"]
    
    # 使用 numpy array 方便直接进行向量化除法运算
    data_9a14_96t  = np.array([3.08, 6.00, 118.81])
    data_9a14_192t = np.array([2.94, 5.53, 95.61])
    data_a100      = np.array([1.7643312, 3.4837646, 62.9066409])
    data_h100      = np.array([0.8999517, 1.7418493, 32.9227259])

    # 2. 计算速度倍率 (以 a100 为基准: 基准时间 / 当前配置时间)
    # 倍率越大，说明速度越快
    ratio_a100      = data_9a14_96t / data_a100
    ratio_h100      = data_9a14_96t / data_h100
    ratio_9a14_96t  = data_9a14_96t / data_9a14_96t
    ratio_9a14_192t = data_9a14_96t / data_9a14_192t

    # 3. 设置柱状图的 X 轴位置和柱子宽度
    x = np.arange(len(systems))
    width = 0.2

    # 4. 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制每一组柱子
    rects1 = ax.bar(x - 1.5*width, ratio_9a14_96t, width, label='EPYC-9A14-96t', color='#2ca02c')
    rects2 = ax.bar(x - 0.5*width, ratio_9a14_192t, width, label='EPYC-9A14-192t', color='#d62728')
    rects3 = ax.bar(x + 0.5*width, ratio_a100, width, label='A100-80G', color='#1f77b4')
    rects4 = ax.bar(x + 1.5*width, ratio_h100, width, label='H100-80G', color='#ff7f0e')

    # 5. 设置图表标签和标题
    ax.set_xlabel('Name', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    # ax.set_title('不同体系与配置下的计算速度倍率 (基准: a100)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=12)
    
    # 动态调高 Y 轴上限，防止最高柱子的文本标签顶出画布被截断
    ax.set_ylim(0, max(ratio_h100) * 1.15)
    ax.legend(title='', loc='upper left')

    # 6. 在柱子上方添加具体的倍率数值标签
    def autolabel(rects):
        """为每个柱状图附加一个文本标签，显示其倍率（保留2位小数）"""
        for rect in rects:
            height = rect.get_height()
            # 格式化为两位小数并加上 'x'
            ax.annotate(f'{height:.2f}x',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 垂直向上偏移 4 个像素
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    # 自动调整布局
    fig.tight_layout()

    # 保存并关闭
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"速度倍率图已成功保存为: {save_filename}")
    plt.close(fig)


if __name__ == "__main__":
    plot_speedup_ratio("system_speedup_ratio_chart.png")