import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """
    Display a 2×2 confusion matrix as a heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix, shape (2,2).

    title : str
        Plot title.
    """
    
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
def plot_precision(m: Dict[str, List[float]]) -> None:
    """
    Plot precision values across thresholds for both classes.

    Parameters
    ----------
    m : dict
        Dictionary containing:
            - "thresholds"
            - "precision_0"
            - "precision_1"
    """
    plt.figure(figsize=(8, 5))
    plt.plot(m["thresholds"], m["precision_0"], marker="o", label="Precision: class 0")
    plt.plot(m["thresholds"], m["precision_1"], marker="o", label="Precision: class 1")
    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.title("Precision vs Threshold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.show()

def plot_recall(m: Dict[str, List[float]]) -> None:
    """
    Plot recall values across thresholds for both classes.

    Parameters
    ----------
    m : dict
        Dictionary containing:
            - "thresholds"
            - "recall_0"
            - "recall_1"
    """
    plt.figure(figsize=(8, 5))
    plt.plot(m["thresholds"], m["recall_0"], marker="o", label="Recall: class 0")
    plt.plot(m["thresholds"], m["recall_1"], marker="o", label="Recall: class 1")
    plt.xlabel("Threshold")
    plt.ylabel("Recall")
    plt.title("Recall vs Threshold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.show()


def plot_f1(m: Dict[str, List[float]]) -> None:
    """
    Plot F1-score values across thresholds for both classes.

    Parameters
    ----------
    m : dict
        Dictionary containing:
            - "thresholds"
            - "f1_0"
            - "f1_1"
    """
    plt.figure(figsize=(8, 5))
    plt.plot(m["thresholds"], m["f1_0"], marker="o", label="F1: class 0")
    plt.plot(m["thresholds"], m["f1_1"], marker="o", label="F1: class 1")
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.title("F1-score vs Threshold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.show()

def plot_accuracy(m: Dict[str, List[float]]) -> None:
    """
    Plot overall accuracy across thresholds.

    Parameters
    ----------
    m : dict
        Dictionary containing:
            - "thresholds"
            - "accuracy"
    """
    plt.figure(figsize=(8, 5))
    plt.plot(m["thresholds"], m["accuracy"], marker="o", label="Accuracy")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Threshold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.show()

def plot_precision_recall_both_classes(metrics: Dict[str, List[float]]) -> None:
    """
    Plot both precision and recall curves for both classes on a single graph.
    
    Colors:
        Precision class 0 → dark green (#006400)
        Precision class 1 → light green (#90EE90)
        Recall class 0    → dark red (#8B0000)
        Recall class 1    → light red (#FFA07A)

    Parameters
    ----------
    metrics : dict
        Must contain keys:
            - "thresholds"
            - "precision_0", "precision_1"
            - "recall_0", "recall_1"
    """
    thresholds = metrics["thresholds"]

    plt.figure(figsize=(10, 6))

    plt.plot(
        thresholds,
        metrics["precision_0"],
        marker="o",
        color="#006400",
        label="Precision (class 0)"
    )
    plt.plot(
        thresholds,
        metrics["precision_1"],
        marker="o",
        color="#90EE90",
        label="Precision (class 1)"
    )

    plt.plot(
        thresholds,
        metrics["recall_0"],
        marker="s",
        color="#8B0000",
        label="Recall (class 0)"
    )
    plt.plot(
        thresholds,
        metrics["recall_1"],
        marker="s",
        color="#FFA07A",
        label="Recall (class 1)"
    )

    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Precision & Recall vs Threshold (both classes)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()