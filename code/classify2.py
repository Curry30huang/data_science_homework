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
# 读取数据
data_path = r'E:\pythonProject\9topics\labeled_data.csv'
df = pd.read_csv(data_path)

# 提取特征和标签
X = df.loc[:, 'pca_feature1':'pca_feature500']
y = df['label']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# One-vs-Rest Logistic Regression
lr = OneVsRestClassifier(LogisticRegression())
lr.fit(X_train, y_train)
lr_y_score = lr.predict_proba(X_test)

# One-vs-Rest SVM
svm = OneVsRestClassifier(SVC(probability=True))
svm.fit(X_train, y_train)
svm_y_score = svm.predict_proba(X_test)

# Plot ROC curve for Logistic Regression
plt.figure(figsize=(10, 6))
for i in range(len(set(y))):
    fpr_lr, tpr_lr, _ = roc_curve(label_binarize(y_test, classes=list(set(y)))[:, i], lr_y_score[:, i],
                                  drop_intermediate=False)
    roc_auc_lr = roc_auc_score(label_binarize(y_test, classes=list(set(y)))[:, i], lr_y_score[:, i])

    # Interpolate the points to create smoother curves
    interp_fpr_lr = interp1d(fpr_lr, tpr_lr, kind='linear')
    smooth_fpr_lr = np.linspace(0, 1, 100)
    smooth_tpr_lr = interp_fpr_lr(smooth_fpr_lr)

    plt.plot(smooth_fpr_lr, smooth_tpr_lr, label=f'Logistic Regression Class {i} (AUC = {roc_auc_lr:.2f}')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression - Smoothed Multi-Class ROC Curve')
plt.legend(loc="lower right")
plt.xlim([0, 0.2])  # Customize x-axis limit
plt.ylim([0.8, 1.0])  # Customize y-axis limit
plt.show()

# Plot ROC curve for SVM
plt.figure(figsize=(10, 6))
for i in range(len(set(y))):
    fpr_svm, tpr_svm, _ = roc_curve(label_binarize(y_test, classes=list(set(y)))[:, i], svm_y_score[:, i],
                                    drop_intermediate=False)
    roc_auc_svm = roc_auc_score(label_binarize(y_test, classes=list(set(y)))[:, i], svm_y_score[:, i])

    # Interpolate the points to create smoother curves
    interp_fpr_svm = interp1d(fpr_svm, tpr_svm, kind='linear')
    smooth_fpr_svm = np.linspace(0, 1, 100)
    smooth_tpr_svm = interp_fpr_svm(smooth_fpr_svm)

    plt.plot(smooth_fpr_svm, smooth_tpr_svm, label=f'SVM Class {i} (AUC = {roc_auc_svm:.2f})', linestyle='-')

plt.plot([0, 1], [0, 1], color='navy', linestyle='-')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM - Smoothed Multi-Class ROC Curve')
plt.legend(loc="lower right")
plt.xlim([0, 0.2])  # Customize x-axis limit
plt.ylim([0.8, 1.0])  # Customize y-axis limit
plt.show()

# Print overall accuracy
lr_y_pred = lr.predict(X_test)
svm_y_pred = svm.predict(X_test)

print(f'Logistic Regression Accuracy: {accuracy_score(y_test, lr_y_pred):.2f}')
print(f'SVM Accuracy: {accuracy_score(y_test, svm_y_pred):.2f}')
