# ==========================
# 1. Libraries
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)



# ==========================
# 2. Load Dataset
# ==========================
df = pd.read_csv("IBM HR Employee Attrition Data.csv")
df.head()

# ==========================
# 3. Check Missing Values
# ==========================
print("Missing values per column:\n", df.isnull().sum())
print("Percentage of missing values:\n", (df.isnull().sum() / len(df)) * 100)

if df.isnull().sum().sum() == 0:
    print("No missing values found in dataset")
else:
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

# ==========================
# 4. Exploratory Data Analysis (EDA)
# ==========================
# Attrition distribution
sns.countplot(x='Attrition', data=df)
plt.title("Attrition Distribution")
plt.show()

# Department-wise Attrition
sns.countplot(x='Department', hue='Attrition', data=df)
plt.xticks(rotation=45)
plt.title("Department-wise Attrition")
plt.show()

# Salary vs Attrition
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.title("Salary vs Attrition")
plt.show()

# Work-Life Balance vs Attrition
sns.violinplot(x='Attrition', y='WorkLifeBalance', data=df)
plt.title("Work-Life Balance vs Attrition")
plt.show()

# Years at Company vs Attrition
sns.histplot(data=df, x='YearsAtCompany', hue='Attrition', bins=20)
plt.title("Years at Company vs Attrition")
plt.show()

# ==========================
# 5. Preprocessing
# ==========================
# Encode categorical target
df['Attrition'] = LabelEncoder().fit_transform(df['Attrition'])

# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==========================
# 6. Model Training
# ==========================
# Logistic Regression (L2)
lr_l2 = LogisticRegression(penalty='l2')
lr_l2.fit(X_train, y_train)
y_pred_lr = lr_l2.predict(X_test)

# Logistic Regression (L1)
lr_l1 = LogisticRegression(penalty='l1', solver='liblinear')
lr_l1.fit(X_train, y_train)
y_pred_l1 = lr_l1.predict(X_test)

coef = pd.Series(lr_l1.coef_[0], index=X.columns)
selected_features = coef[coef != 0].index
print("Selected Features by L1:", selected_features)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ==========================
# 7. Model Evaluation
# ==========================
def evaluate_model(name, y_test, y_pred):
    print(f"\n{name}")
    print(classification_report(y_test, y_pred))

evaluate_model("Logistic Regression (L2)", y_test, y_pred_lr)
evaluate_model("Logistic Regression (L1)", y_test, y_pred_l1)
evaluate_model("Random Forest", y_test, y_pred_rf)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# ROC Curve and AUC
y_prob_rf = rf.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.show()
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob_rf))

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob_rf)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Random Forest")
plt.show()

# Feature Importance
importances = rf.feature_importances_
features = X.columns
feat_imp = pd.Series(importances, index=features)
feat_imp.nlargest(10).plot(kind='barh')
plt.title("Top Features Affecting Attrition")
plt.show()

# Model Comparison (Accuracy)
accuracy = [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf)]
models = ['Logistic Regression', 'Random Forest']
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# SHAP Values (Advanced Explanation)
explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)