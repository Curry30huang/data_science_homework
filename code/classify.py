import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy.interpolate import interp1d
import numpy as np

# 读取标注数据
data_path = '../data/labeled_data.csv'
df = pd.read_csv(data_path)

# 提取特征和标签
X = df.loc[:, 'pca_feature1':'pca_feature100']
y = df['label']

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 使用One-vs-Rest策略的逻辑回归
lr = OneVsRestClassifier(LogisticRegression())
lr.fit(X_train, y_train)
lr_y_score = lr.predict_proba(X_test)

# 使用One-vs-Rest策略的支持向量机(SVM)
svm = OneVsRestClassifier(SVC(probability=True))
svm.fit(X_train, y_train)
svm_y_score = svm.predict_proba(X_test)

# 绘制逻辑回归的ROC曲线
plt.figure(figsize=(10, 6))
for i in range(lr_y_score.shape[1]):
    fpr_lr, tpr_lr, _ = roc_curve(label_binarize(y_test, classes=list(set(y)))[:, i], lr_y_score[:, i])
    roc_auc_lr = roc_auc_score(label_binarize(y_test, classes=list(set(y)))[:, i], lr_y_score[:, i])

    # 使用插值法使曲线更平滑
    interp_fpr_lr = interp1d(fpr_lr, tpr_lr, kind='linear')
    smooth_fpr_lr = np.linspace(0, 1, 100)
    smooth_tpr_lr = interp_fpr_lr(smooth_fpr_lr)

    plt.plot(smooth_fpr_lr, smooth_tpr_lr, label=f'LR Class {i} (AUC = {roc_auc_lr:.2f}')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression - Smoothed Multi-Class ROC Curve')
plt.legend(loc="lower right")
plt.xlim([0, 0.2])  # 自定义X轴范围
plt.ylim([0.8, 1.0])  # 自定义Y轴范围
plt.show()

# 绘制SVM的ROC曲线
plt.figure(figsize=(10, 6))
for i in range(svm_y_score.shape[1]):
    fpr_svm, tpr_svm, _ = roc_curve(label_binarize(y_test, classes=list(set(y)))[:, i], svm_y_score[:, i],
                                    drop_intermediate=False)
    roc_auc_svm = roc_auc_score(label_binarize(y_test, classes=list(set(y)))[:, i], svm_y_score[:, i])

    # 使用插值法使曲线更平滑
    interp_fpr_svm = interp1d(fpr_svm, tpr_svm, kind='linear')
    smooth_fpr_svm = np.linspace(0, 1, 100)
    smooth_tpr_svm = interp_fpr_svm(smooth_fpr_svm)

    plt.plot(smooth_fpr_svm, smooth_tpr_svm, label=f'SVM Class {i} (AUC = {roc_auc_svm:.2f})', linestyle='-')

plt.plot([0, 1], [0, 1], color='navy', linestyle='-')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM - Smoothed Multi-Class ROC Curve')
plt.legend(loc="lower right")
plt.xlim([0, 0.2])  # 自定义X轴范围
plt.ylim([0.8, 1.0])  # 自定义Y轴范围
plt.show()

# 输出整体准确率
lr_y_pred = lr.predict(X_test)
svm_y_pred = svm.predict(X_test)

print(f'逻辑回归准确率: {accuracy_score(y_test, lr_y_pred):.2f}')
print(f'SVM准确率: {accuracy_score(y_test, svm_y_pred):.2f}')
