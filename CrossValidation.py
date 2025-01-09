import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns

# 載入模型
model_path = "vgg16_cats_vs_dogs.h5"
model = load_model(model_path)

# 資料路徑
data_dir = "/Users/yangjames/Documents/python/VGGNet_CatvsDog/dogs-vs-cats/train/validation"

# 設定參數
input_size = (150, 150)
batch_size = 32
num_folds = 5

# 資料生成器
datagen = ImageDataGenerator(rescale=1./255)
data_generator = datagen.flow_from_directory(
    data_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary',  # 二元分類
    shuffle=False  # 交叉驗證時固定順序
)

# 取得資料與標籤
X, y = [], []
for i in range(len(data_generator)):
    batch_x, batch_y = data_generator[i]
    X.extend(batch_x)
    y.extend(batch_y)
    if len(X) >= data_generator.samples:
        break

X = np.array(X)
y = np.array(y)

# 使用 KFold 進行交叉驗證
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_no = 1
metrics_list = []

for train_index, val_index in kf.split(X):
    print(f"Training fold {fold_no}...")

    # 分割訓練集與驗證集
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 驗證集預測
    val_predictions_prob = model.predict(X_val)
    val_predictions = (val_predictions_prob > 0.5).astype(int).flatten()

    # 計算評估指標
    acc = accuracy_score(y_val, val_predictions)
    precision = precision_score(y_val, val_predictions)
    recall = recall_score(y_val, val_predictions)
    f1 = f1_score(y_val, val_predictions)
    cm = confusion_matrix(y_val, val_predictions)

    # 計算 ROC 曲線和 AUC
    fpr, tpr, _ = roc_curve(y_val, val_predictions_prob)
    roc_auc = auc(fpr, tpr)

    metrics_list.append({
        'fold': fold_no,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': roc_auc
    })

    # 繪製混淆矩陣
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.title(f'Confusion Matrix - Fold {fold_no}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # 繪製 ROC 曲線
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title(f'ROC Curve - Fold {fold_no}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

    fold_no += 1

# 總結指標
for metrics in metrics_list:
    print(f"Fold {metrics['fold']} - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1_score']:.4f}, AUC: {metrics['auc']:.4f}")

average_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0] if key != 'fold'}
print("\nAverage Metrics:")
for metric, value in average_metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")
