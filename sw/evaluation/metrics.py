from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Print standard classification report and return the confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.

    y_pred : np.ndarray
        Model predictions.

    Returns
    -------
    cm : np.ndarray
        2×2 confusion matrix.
    """
    print(classification_report(y_true, y_pred))
    return confusion_matrix(y_true, y_pred)