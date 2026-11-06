import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.makedirs("images", exist_ok=True)

# 1. Load data
df = sns.load_dataset("titanic")
print(df.shape)
print(df.isnull().sum())

# 2. Clean data
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df = df.drop(columns=["deck", "embark_town", "alive", "class", "who", "adult_male", "alone"])

# 3. Exploratory data analysis
sns.countplot(x="survived", data=df)
plt.title("Survival Count")
plt.savefig("images/survival_count.png")
plt.clf()

sns.barplot(x="sex", y="survived", data=df)
plt.title("Survival Rate by Gender")
plt.savefig("images/survival_by_gender.png")
plt.clf()

sns.barplot(x="pclass", y="survived", data=df)
plt.title("Survival Rate by Passenger Class")
plt.savefig("images/survival_by_class.png")
plt.clf()

sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png")
plt.clf()

# 4. Encode categorical columns
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["embarked"], drop_first=True)

# 5. Train / test split
X = df.drop("survived", axis=1)
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_test, preds), 3))
    print(classification_report(y_test, preds))

# 7. Confusion matrix for the Random Forest
rf_preds = models["Random Forest"].predict(X_test)
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("images/confusion_matrix.png")
