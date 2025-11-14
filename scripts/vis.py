import matplotlib.pyplot as plt
import numpy as np

# Languages
languages = ['CN', 'FR', 'JP', 'KR', 'RU', 'US']

# Models
models = [
    "qwen-3-235b-a22b-thinking-2507",
    "qwen-3-235b-a22b-instruct-2507",
    "qwen-3-32b",
    "llama-4-maverick-17b-128e-instruct"
]

# --- Data ---
# qwen-3-235b-a22b-thinking-2507
macro_f1_exact_thinking = [19.33, 19.94, 2.27, 67.64, 30.36, 33.33]
macro_f1_pairs_thinking = [48.12, 42.66, 24.28, 97.5, 48.99, 33.33]
micro_f1_thinking       = [63.81, 44.93, 32.07, 95.52, 57.47, 23.86]

# qwen-3-235b-a22b-instruct-2507
macro_f1_exact_instruct = [22.78, 19.28, 7.46, 52.27, 65.29, 10.97]
macro_f1_pairs_instruct = [71.61, 75.19, 50.19, 60.74, 76.86, 38.31]
micro_f1_instruct       = [68.86, 76.24, 63.17, 54.17, 85.97, 47.33]

# qwen-3-32b
macro_f1_exact_32b = [4.1, 17.42, 4.55, 30.92, 0.0, 8.26]
macro_f1_pairs_32b = [19.85, 48.66, 11.09, 58.93, 17.49, 32.8]
micro_f1_32b       = [21.28, 66.76, 4.69, 85.58, 23.24, 29.5]

# llama-4-maverick-17b-128e-instruct
macro_f1_exact_llama = [10.0, 14.32, 3.31, 33.33, 61.35, 5.56]
macro_f1_pairs_llama = [10.0, 32.28, 17.28, 33.33, 82.35, 19.42]
micro_f1_llama       = [6.9, 28.57, 8.26, 12.61, 83.94, 9.29]

# Organize data for plotting
macro_f1_exact = [
    macro_f1_exact_thinking,
    macro_f1_exact_instruct,
    macro_f1_exact_32b,
    macro_f1_exact_llama
]
macro_f1_pairs = [
    macro_f1_pairs_thinking,
    macro_f1_pairs_instruct,
    macro_f1_pairs_32b,
    macro_f1_pairs_llama
]
micro_f1 = [
    micro_f1_thinking,
    micro_f1_instruct,
    micro_f1_32b,
    micro_f1_llama
]

# Setup
x = np.arange(len(languages))
width = 0.18
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']  # 4 distinguishable colors

fig, axs = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
metric_names = ['Glass Composition F1 (Exact)', 'Glass Composition F1 (Pairs)', 'Individual Component F1 (Pairs)']
data = [macro_f1_exact, macro_f1_pairs, micro_f1]

plt.style.use("seaborn-v0_8-whitegrid")

for i, ax in enumerate(axs):
    for j, model in enumerate(models):
        bars = ax.bar(x + width*(j-1.5), data[i][j], width, 
                      label=model if i == 0 else "", 
                      color=colors[j])
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(languages, fontsize=12)
    ax.set_title(metric_names[i], fontsize=14, weight='bold')
    ax.set_ylabel('Score (%)' if i == 0 else "", fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

axs[0].legend(fontsize=11, frameon=True, loc="upper left")
fig.suptitle('Model Comparison by Language and Metric', fontsize=18, weight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('model_comparison_by_language.png', dpi=300)
plt.show()
