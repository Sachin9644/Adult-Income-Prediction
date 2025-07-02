# Adult Income Classification - Major Project 

#  Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#  Load and Inspect Dataset
file_path = "75045e1e-f939-41d0-9340-a78c43738106.csv"
df = pd.read_csv(file_path)

print("\nDataset Information:")
print(df.info())
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nChecking for missing values:")
print(df.isnull().sum())

#  Data Cleaning and Preprocessing

df.columns = df.columns.str.strip()

s
df = df.replace('?', np.nan)
df = df.dropna()

print("\nShape after dropping null values:", df.shape)

# Encode categorical columns using LabelEncoder
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nColumns after encoding:")
print(df.columns.tolist())

#  Feature Engineering
# Separate independent variables (X) and target variable (y)
X = df.drop('income', axis=1)
y = df['income']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Splitting the Data
# Split into training and testing sets with 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

#  Data Visualization
# Visualization 1: Income Distribution
plt.figure(figsize=(8, 4))
sns.countplot(x='income', data=df)
plt.title('Distribution of Income Classes')
plt.xlabel('Income Category')
plt.ylabel('Count')
plt.show()

# Visualization 2: Age Histogram
sns.histplot(df['age'], bins=30, kde=True, color='green')
plt.title('Age Distribution of Individuals')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Visualization 3: Boxplot of Hours-per-Week by Income
plt.figure(figsize=(8, 4))
sns.boxplot(x='income', y='hours-per-week', data=df)
plt.title('Work Hours per Week by Income Category')
plt.show()

# Visualization 4: Education vs Hours-per-Week
plt.figure(figsize=(10, 5))
sns.barplot(x='education', y='hours-per-week', data=df)
plt.xticks(rotation=90)
plt.title('Average Weekly Work Hours by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Hours per Week')
plt.show()

# Visualization 5: Heatmap of Feature Correlation
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Features")
plt.show()

# Visualization 6: Capital Gain Histogram (Plotly)
px.histogram(df, x='capital-gain', color='income', title='Capital Gain by Income Category').show()

# Visualization 7: Age vs Hours-per-Week Scatter (Plotly)
px.scatter(df, x='age', y='hours-per-week', color='income', title='Age vs Hours-per-Week by Income').show()

# Visualization 8: Education Number by Income Boxplot (Plotly)
px.box(df, x='income', y='education-num', title='Education Number by Income').show()

#  Model Building and Evaluation
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

for name, model in models.items():
    print(f"\n==== {name} ====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


