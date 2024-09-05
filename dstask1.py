import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = 'C:/Users/Atulya Kumar/Downloads/Titanic-Dataset.csv'  
titanic_df = pd.read_csv(file_path)

titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())

titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])

titanic_df['HasCabin'] = titanic_df['Cabin'].notna().astype(int)

titanic_df.drop(columns=['Cabin'], inplace=True)

titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})

titanic_df = pd.get_dummies(titanic_df, columns=['Embarked'], drop_first=True)

titanic_df.drop(columns=['Name', 'Ticket'], inplace=True)

X = titanic_df.drop(columns=['Survived', 'PassengerId'])
y = titanic_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)
