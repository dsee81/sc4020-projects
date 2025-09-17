import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import List, Optional, Union


def evaluate_classification(num_classes: int, y_true: Union[List, np.ndarray], 
                          y_pred: Union[List, np.ndarray], class_names: Optional[List[str]] = None,
                          print_report: bool = True, plot_cm: bool = True, 
                          save_path: Optional[str] = None) -> dict:
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    metrics = {
        #we probs dn all these metrics but just in case
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_per_class': precision_score(y_true, y_pred, average=None, zero_division=0),
        'recall_per_class': recall_score(y_true, y_pred, average=None, zero_division=0),
        'f1_per_class': f1_score(y_true, y_pred, average=None, zero_division=0)
    }
    
    if print_report:
        print("CLASSIFICATION METRICS")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
        print(f"Weighted Precision: {metrics['precision_weighted']:.4f}")
        print(f"Weighted Recall: {metrics['recall_weighted']:.4f}")
        print(f"Macro F1: {metrics['f1_macro']:.4f}")
        
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        for i, row in enumerate(cm):
            print(f"{class_names[i]}: {row}")
    
    if plot_cm:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    return metrics