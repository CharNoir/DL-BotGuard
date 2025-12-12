import matplotlib.pyplot as plt
import seaborn as sns

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