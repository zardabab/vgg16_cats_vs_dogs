import itertools
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 資料路徑
train_dir = "/Users/yangjames/Documents/python/VGGNet_CatvsDog/dogs-vs-cats/train/train"
validation_dir = "/Users/yangjames/Documents/python/VGGNet_CatvsDog/dogs-vs-cats/train/validation"

# 資料預處理與增強
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

#將圖像的像素值從 [0, 255] 範圍縮放到 [0, 1] 範圍，這樣可以使模型更快地收斂，並且有助於提高模型的性能。
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# 加載圖片
#此段程式碼將圖片從磁盤加載到內存中，並將其轉換為大小為 150x150 的張量。
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

# 載入預訓練的 VGG16 模型，不包含頂層全連接層
vgg_base = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))

# 凍結預訓練層
vgg_base.trainable = True
for layer in vgg_base.layers[:-4]:  # 僅訓練最後4層
    layer.trainable = False

# 添加分類層(頂層全連接層)
model = Sequential([
    vgg_base,
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # 二分類
])

# 編譯模型
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=10,         
    restore_best_weights=True
)

# # 訓練模型
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     epochs=100,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // validation_generator.batch_size,
    
#     callbacks=[early_stopping]
# )

# # 繪製訓練和驗證的準確率與損失
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(len(acc))

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(epochs, acc, 'bo-', label='Training accuracy')
# plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(epochs, loss, 'bo-', label='Training loss')
# plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()

# 保存模型
# model.save("vgg16_cats_vs_dogs.h5")

# Cross-validation simulation for metrics
num_folds = 5
cv_results = {
    "fold": [],
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1-score": [],
    "auc": []
}

#交叉驗證
for fold in range(1, num_folds + 1):
    # # Simulating validation predictions and true labels
    # #隨機生成100個樣本的真實標籤和預測分數
    # true_labels = np.random.randint(0, 2, 100)
    # #生成100個隨機數作為預測分數
    # pred_scores = np.random.rand(100)
    # #將預測分數轉換為預測標籤
    # pred_labels = (pred_scores > 0.5).astype(int)

    # 訓練模型
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        # shffle=True,  # 打亂數據順序
        callbacks=[early_stopping]
    )

    # 繪製訓練和驗證的準確率與損失
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    plot_confusion_matrix(history.history['true_labels'], history.history['pred_labels'], classes=["Class 0", "Class 1"])

    #畫出訓練和驗證的準確率和損失圖
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    # Metrics
    # accuracy = np.mean(pred_labels == true_labels)
    accuracy =val_acc
    # report = classification_report(true_labels, pred_labels, output_dict=True)
    report = classification_report(history.history['true_labels'], history.history['pred_labels'], output_dict=True)
    
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]

    # ROC and AUC
    fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    roc_auc = auc(fpr, tpr)

    # Store results
    cv_results["fold"].append(fold)
    cv_results["accuracy"].append(accuracy)
    cv_results["precision"].append(precision)
    cv_results["recall"].append(recall)
    cv_results["f1-score"].append(f1)
    cv_results["auc"].append(roc_auc)

    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Show Confusion Matrix for the fold
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["Class 0", "Class 1"])
    plt.yticks([0, 1], ["Class 0", "Class 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
    plt.show()

    # Show ROC Curve for the fold
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Fold {fold} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Fold {fold}")
    plt.legend()
    plt.show()

# Calculate mean and variance for cross-validation results
cv_summary = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"],
    "Mean": [np.mean(cv_results[metric]) for metric in ["accuracy", "precision", "recall", "f1-score", "auc"]],
    "Variance": [np.var(cv_results[metric]) for metric in ["accuracy", "precision", "recall", "f1-score", "auc"]],
}

# Convert summary to a DataFrame
cv_summary_df = pd.DataFrame(cv_summary)

# Display the summary table
print(cv_summary_df)
