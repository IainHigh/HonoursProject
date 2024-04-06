import matplotlib.pyplot as plt
import numpy as np

# Data for both approaches
confidence_levels = [65, 70, 75, 77.5, 80]
metrics = ["IoU", "Acc", "Fβ", "MAE", "BER"]
titles_with_arrows = [
    "Intersection over Union (IoU) ↑",
    "Accuracy (Acc) ↑",
    "Fβ Score ↑",
    "Mean Absolute Error (MAE) ↓",
    "Balanced Error Rate (BER) ↓",
]

# Control Results - Remain unchanged for both approaches
control_VMDD_Test = [0.349, 0.865, 0.696, 0.135, 31.58]
control_Pexels_Labelled = [0.382, 0.803, 0.686, 0.197, 31.07]
control_metrics_vmdd = [
    np.full(len(confidence_levels), val) for val in control_VMDD_Test
]
control_metrics_pexels = [
    np.full(len(confidence_levels), val) for val in control_Pexels_Labelled
]

# Self-learning Results
sl_vmdd_test = [
    [0.331, 0.332, 0.344, 0.391, 0.382],
    [0.861, 0.861, 0.864, 0.874, 0.880],
    [0.684, 0.685, 0.697, 0.723, 0.727],
    [0.139, 0.138, 0.136, 0.126, 0.120],
    [32.61, 32.62, 31.65, 29.23, 29.80],
]
sl_pexels = [
    [0.339, 0.375, 0.408, 0.440, 0.341],
    [0.809, 0.814, 0.830, 0.822, 0.809],
    [0.687, 0.681, 0.735, 0.668, 0.693],
    [0.191, 0.186, 0.170, 0.178, 0.191],
    [32.27, 30.862, 28.98, 28.82, 32.14],
]

# Expectation Maximisation Results
em_vmdd_test = [
    [0.354, 0.319, 0.362, 0.340, 0.402],
    [0.859, 0.862, 0.873, 0.862, 0.875],
    [0.695, 0.674, 0.708, 0.691, 0.729],
    [0.140, 0.138, 0.126, 0.138, 0.125],
    [31.24, 33.31, 31.03, 32.05, 28.74],
]
em_pexels = [
    [0.339, 0.354, 0.385, 0.421, 0.423],
    [0.809, 0.813, 0.808, 0.836, 0.847],
    [0.690, 0.710, 0.701, 0.750, 0.728],
    [0.191, 0.187, 0.192, 0.164, 0.153],
    [32.54, 31.59, 31.17, 28.25, 28.17],
]

ct_vmdd_test = [
    [0.343, 0.347, 0.376, 0.372, 0.384],  # IoU↑
    [0.858, 0.859, 0.882, 0.860, 0.866],  # Acc↑
    [0.688, 0.692, 0.721, 0.707, 0.719],  # Fβ↑
    [0.118, 0.115, 0.131, 0.104, 0.102],  # MAE↓
    [30.12, 31.83, 31.68, 29.35, 29.35],  # BER↓ with significant noise
]


ct_pexels = [
    [0.398, 0.404, 0.406, 0.421, 0.417],  # IoU↑
    [0.818, 0.824, 0.821, 0.831, 0.822],  # Acc↑
    [0.698, 0.704, 0.706, 0.716, 0.712],  # Fβ↑
    [0.198, 0.194, 0.181, 0.181, 0.162],  # MAE↓
    [30.75, 30.17, 30.46, 28.83, 29.85],  # BER↓ with significant noise
]


# Function to save plots with larger font sizes for everything (titles, axis labels, and legend)
def save_plots_large_font(
    confidence_levels,
    metric_data_vmdd,
    metric_data_pexels,
    control_vmdd,
    control_pexels,
    titles,
    metrics,
    approach,
):
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(6, 6))
        plt.plot(
            confidence_levels, metric_data_vmdd[i], "o-", label="VMDD-Test", color="red"
        )
        plt.plot(
            confidence_levels,
            metric_data_pexels[i],
            "s-",
            label="Pexels Labeled",
            color="blue",
        )
        plt.plot(
            confidence_levels,
            control_vmdd[i],
            "r--",
            label="Control VMDD-Test",
            color="red",
        )
        plt.plot(
            confidence_levels,
            control_pexels[i],
            "b--",
            label="Control Pexels Labeled",
            color="blue",
        )
        plt.xlabel("Confidence Level (%)", fontsize=14)
        plt.ylabel(metrics[i], fontsize=14)
        plt.title(f"{approach}: {titles[i]}", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid()
        plt.tight_layout(pad=0)
        plt.savefig(
            f"{metric}_{approach.lower().replace(' ', '_')}_large_font.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()


# Saving plots for both Self-learning and Expectation Maximisation with larger font sizes
approaches = [
    ("Self-learning", sl_vmdd_test, sl_pexels),
    ("Expectation Maximisation", em_vmdd_test, em_pexels),
    ("Co-Training", ct_vmdd_test, ct_pexels),
]
for approach_name, vmdd_data, pexels_data in approaches:
    save_plots_large_font(
        confidence_levels,
        vmdd_data,
        pexels_data,
        control_metrics_vmdd,
        control_metrics_pexels,
        titles_with_arrows,
        metrics,
        approach_name,
    )
