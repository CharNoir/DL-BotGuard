import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_with_thresholds(
    model,
    X_test,
    y_test,
    thresholds=None,
    batch_size=1024,
    device=None,
    print_reports=True,
):
    """
    Evaluate a binary classifier across multiple probability thresholds.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model which outputs either logits or raw scores.
        Must produce shape (batch, 2) for softmax classification.

    X_test : np.ndarray
        Test dataset of shape (N, T, F) or (N, F).

    y_test : np.ndarray
        Ground-truth binary labels of shape (N,).

    thresholds : np.ndarray, optional
        Array of thresholds to evaluate, e.g. np.linspace(0.1, 0.9, 9).

    batch_size : int
        Batch size for forward inference.

    device : torch.device, optional
        Device to use for inference ("cpu" or "cuda").
        If None → auto-select CUDA if available.

    print_reports : bool
        Whether to print classification reports for each threshold.

    Returns
    -------
    probs : np.ndarray
        Model predicted probabilities for the positive class. Shape (N,).

    results : dict
        A dictionary indexed by threshold, each containing:
            "preds" : model predictions at that threshold
            "metrics" : sklearn classification_report(dict form)
            "confusion_matrix" : confusion matrix for that threshold
    """
    # Default thresholds
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)

    # Prepare model
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Collect probabilities
    all_probs = []
    N = X_test.shape[0]

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = start + batch_size
            xb = torch.tensor(X_test[start:end], dtype=torch.float32).to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)

    # Collect results for each threshold
    results = {}
    for thr in thresholds:
        preds = (probs >= thr).astype(int)

        metrics_text = classification_report(y_test, preds, digits=3, output_dict=False)
        metrics_dict = classification_report(y_test, preds, digits=3, output_dict=True)
        cm = confusion_matrix(y_test, preds)

        results[thr] = {
            "preds": preds,
            "metrics": metrics_dict,
            "confusion_matrix": cm,
        }

        if print_reports:
            print(f"\n===== Threshold = {thr:.2f} =====")
            print(metrics_text)

    return probs, results
