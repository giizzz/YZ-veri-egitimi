import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


import warnings 
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split




# Veri yükleme
df_train = pd.read_csv('C:\\Users\\gizem\\OneDrive\\Masaüstü\\YZ\\.venv\\train.csv')
original = pd.read_csv('C:\\Users\\gizem\\OneDrive\\Masaüstü\\YZ\\.venv\\ObesityDataSet.xls')
df_test = pd.read_csv('C:\\Users\\gizem\\OneDrive\\Masaüstü\\YZ\\.venv\\test.csv')

# Veri setlerinin şekli
print(f'The Train dataset has {df_train.shape[0]} rows and {df_train.shape[1]} columns')
print(f'The Test dataset has {df_test.shape[0]} rows and {df_test.shape[1]} columns')


# İlk 5 satır kontrol
print('Train:', df_train.head())
print('Test:', df_test.head())

# NObeyesdad sütununa göre gruplama
if 'Age' in df_train.columns:
    grouped = df_train.groupby('NObeyesdad')['Age'].describe().reset_index()
    print(grouped)
else:
    print("'Age' sütunu bulunamadı.")



# Eğitim ve test verisi hazırlama
train = pd.concat([df_train, original]).drop(['id'], axis=1, errors='ignore').drop_duplicates()
test = df_test.drop(['id'], axis=1, errors='ignore')


# Kategorik sütunları kategorik olarak işaretleme
categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
for col in categorical_columns:
    if col in train.columns:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')


# Model için giriş ve çıkış verisi
X = train.drop(['NObeyesdad'], axis=1, errors='ignore')
y = train['NObeyesdad']

# Eğitim ve doğrulama verilerini ayırma
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# LGBMClassifier için hiperparametreler

best_params = {
    "objective": "multiclass",          # Objective function for the model
    "metric": "multi_logloss",          # Evaluation metric
    "verbosity": -1,                    # Verbosity level (-1 for silent)
    "boosting_type": "gbdt",            # Gradient boosting type
    "random_state": 42,       # Random state for reproducibility
    "num_class": 7,                     # Number of classes in the dataset
    'learning_rate': 0.01197852738297134,  # Learning rate for gradient boosting
    'n_estimators': 509,                # Number of boosting iterations
    'lambda_l1': 0.009715116714365275,  # L1 regularization term
    'lambda_l2': 0.03853395161282091,   # L2 regularization term
    'max_depth': 11,                    # Maximum depth of the trees
    'colsample_bytree': 0.7364306508830604,  # Fraction of features to consider for each tree
    'subsample': 0.9529973839959326,    # Fraction of samples to consider for each boosting iteration
    'min_child_samples': 17             # Minimum number of data needed in a leaf
}


# Modeli eğitme
lgbm_classifier = LGBMClassifier(**best_params)
lgbm_classifier.fit(X_train, y_train)

# Tahmin yapma
y_pred = lgbm_classifier.predict(X_valid)

# Performans değerlendirmesi

accuracy = accuracy_score(y_valid, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")  # Yüzdelik olarak bastırma
print(classification_report(y_valid, y_pred))

# print("Accuracy:", accuracy_score(y_valid, y_pred))
# print(classification_report(y_valid, y_pred))

# Confusion matrix görselleştirme
plt.figure(figsize=(15, 6))
conf_matrix = confusion_matrix(y_valid, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_valid), yticklabels=np.unique(y_valid))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# Öznitelik önemini görselleştirme
feature_importance = lgbm_classifier.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
sns.despine(left=True, bottom=True)
plt.show()


# Test verisiyle tahmin yapma
predictions = lgbm_classifier.predict(test)

# Sonuçları kaydetme
submission = pd.read_csv("C:\\Users\\gizem\\OneDrive\\Masaüstü\\YZ\\.venv\\sample_submission.csv")
submission["NObeyesdad"] = predictions
submission.to_csv("submission1.csv", index=False)
print(submission.head())
