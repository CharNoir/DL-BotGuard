import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()