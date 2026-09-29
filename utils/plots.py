import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREY = "#1f4e79", "#9aa5b1"


def bar_counts(counts, title, xlabel="Category", horizontal=True):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    s = counts.sort_values()
    ax.barh(s.index, s.values, color=BLUE)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    ax.set_xlabel("Number of articles"); ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout(); return fig


def metric_bars(table):
    metrics = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3), sharey=True)
    short = {"Multinomial Naive Bayes": "NB", "Logistic Regression": "LR", "Linear SVM": "SVM"}
    for ax, m in zip(axes, metrics):
        cols = [BLUE if s == "Selected" else GREY for s in table["Status"]]
        ax.bar([short.get(n, n) for n in table["Model Name"]], table[m] * 100, color=cols)
        for i, v in enumerate(table[m] * 100):
            ax.text(i, v + 0.5, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_title(m, fontsize=9); ax.set_ylim(0, 105)
    axes[0].set_ylabel("%"); fig.tight_layout(); return fig


def confusion_heatmap(cm, classes, normalize=False, title=""):
    import numpy as np
    data = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(data, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 8}, linewidths=.4, linecolor="#e5e8ec")
    ax.set_xlabel("Predicted category"); ax.set_ylabel("Actual category"); ax.set_title(title, loc="left", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right"); fig.tight_layout(); return fig


def class_f1(per_class):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    d = per_class.sort_values("F1")
    ax.barh(d["Category"], d["F1"] * 100, color=BLUE)
    for i, v in enumerate(d["F1"] * 100):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
    ax.set_xlim(0, 108); ax.set_xlabel("F1 score (%)"); fig.tight_layout(); return fig


def split_bars(train_counts, test_counts):
    import pandas as pd
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    pd.DataFrame({"Train": train_counts, "Test": test_counts}).plot.bar(ax=ax, color=[BLUE, GREY], width=.8)
    ax.set_ylabel("Articles"); ax.set_xlabel(""); plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(); return fig
