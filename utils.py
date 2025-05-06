
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def one_hot(y, num_classes):
    y = np.asarray(y).flatten()
    N = y.shape[0]
    OH = np.zeros((N, num_classes), dtype=np.float32)
    OH[np.arange(N), y] = 1.0
    return OH

def print_classification_report(y_true, y_pred, class_names):
    """
    打印分类报告和混淆矩阵
    y_true: 形状 (N,) 的真实标签
    y_pred: 形状 (N,) 的预测标签
    class_names: 长度为 num_classes 的类别名称列表
    """
    # 分类报告：每个类别的 precision, recall, f1-score，以及总体平均
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("=== 分类报告 ===")
    print(report)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print("=== 混淆矩阵 ===")
    print(cm)
