import os
import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers

import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


cl_features_CT_first = pd.read_excel('/content/cl_features_cleaned_ct_first_image.xlsx', engine='openpyxl') # clfeature
cl_features_CT_241 = pd.read_excel('/content/cl_features_cleaned_241.xlsx', engine='openpyxl')

ml_features_CT_first= pd.read_excel('/content/ml_features.xlsx')
ml_features_241= pd.read_excel('/content/ml_features_cleaned_241.xlsx', engine='openpyxl')

merged_df = pd.merge(cl_features_CT_first, ml_features_CT_first, on='imageName', how='inner')
merged_df_241= pd.merge(cl_features_CT_241, ml_features_241, on='imageName', how='inner')

merged_df_filtered_columns = [col for col in merged_df.columns if col in merged_df_241.columns]
merged_df_241_filtered = merged_df_241[merged_df_filtered_columns]
merged_df_241_filtered_columns = [col for col in merged_df_241_filtered.columns if col in merged_df.columns]
merged_df_filtered=merged_df[merged_df_241_filtered_columns]

merged_df_all = pd.concat([merged_df_filtered, merged_df_241_filtered], axis=0, ignore_index=True)
all_features = merged_df_all.copy()

categorical_features = ['gender', 'smoking status', 'PDL1_expression', 'Pathological diagnosis', 'total stage']

category_mappings = {}

for feature in categorical_features:
    if feature in all_features.columns:  

        all_features[feature] = all_features[feature].astype('category')


        categories = all_features[feature].cat.categories
        all_features[feature] = all_features[feature].cat.codes


        category_mappings[feature] = {index: category for index, category in enumerate(categories)}


for feature, mapping in category_mappings.items():
    print(f'Feature: {feature}')
    print('Mapping:', mapping)
    print('---------------------------')

all_features_clean = all_features[all_features['Group'].isin([0, 1])]


X = all_features_clean.drop(columns=['imageName', 'Group', 'set', 'ORR'])  # 剔除非特征列
y = all_features_clean['Group']

non_numeric_columns = X.select_dtypes(include=['object']).columns
X = X.drop(columns=non_numeric_columns)

# random forest classifier
clf = RandomForestClassifier(n_estimators=100, criterion='gini')
clf.fit(X, y)

# feature importances
feature_importances = clf.feature_importances_

importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# choose feature importance (>0.004)
top_features = importance_df[importance_df['Importance'] > 0.001]['Feature'].tolist()

# updata data
all_features = merged_df_all[['imageName', 'Group', 'set', 'ORR'] + top_features]

train_set = all_features[all_features['set'].isin(['lungmate-009'])]

val_set = all_features[all_features['set'].isin([ 'lungmate-002', 'lungmate-001'])]

#test_set = all_features[all_features['set'].isin(['lungmate-002'])]

train_set = train_set[train_set['Group'].isin([0, 1])]
val_set = val_set[val_set['Group'].isin([0, 1])]
X_train = train_set.drop(columns=['imageName', 'Group', 'set', 'ORR'])
y_train = train_set['Group']

X_val = val_set.drop(columns=['imageName', 'Group', 'set', 'ORR'])
y_val = val_set['Group']

#X_test=test_set.drop(columns=['imageName', 'Group', 'set', 'ORR'])
#y_test=test_set['Group']

# XGB classifier
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# train
xgb_classifier.fit(X_train, y_train)


y_pred = xgb_classifier.predict(X_val)
y_pred_p = xgb_classifier.predict_proba(X_val)[:, 1]

# accuracy and AUC
accuracy = sklearn.metrics.accuracy_score(y_val, y_pred)  # 计算accuracy
auc = sklearn.metrics.roc_auc_score(y_val, y_pred_p)  # 计算AUC


#  ROC
fpr, tpr, thresholds = roc_curve(y_val, y_pred_p)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.4f})'.format(auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

#deep learning model

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # 二分类
])

optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)  # 只使用验证集，不使用测试集
)

val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"\nFinal Validation Loss: {val_loss:.4f}")
print(f"Final Validation Accuracy: {val_accuracy:.4f}")

# confusion metrics
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype("int32")  # 将概率转换为0或1的二分类预测

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
