# Employees dataset with false positives

### Import libraries
```python
!pip install pandas scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
```

### Load the dataset
```python
df = pd.read_csv('/Users/layanalghamdi/Desktop/employees_dataset_with_false_positives.csv')

#Create the "False_Positive" label
df['False_Positive'] = ((df['Performance_Score'] == 'High') & (df['Needs_Training'] == 'Yes')).astype(int)

#Prepare the features and labels
features = ['Department', 'Job_Title', 'Performance_Score']
X = df[features]
y = df['False_Positive']
```
### Encode categorical features
```python
encoder = LabelEncoder()
for col in X.columns:
    X[col] = encoder.fit_transform(X[col])
```
### train & test & split the data
```python
#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Predict on test data
y_pred = model.predict(X_test)
```
### Evaluate the model
```python
print("="*50)
print("Classification Report")
print("="*50)
print(classification_report(y_test, y_pred, target_names=["Not False Positive", "False Positive"]))

print("="*50)
print("Confusion Matrix")
print("="*50)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not False Positive", "False Positive"])
disp.plot(cmap=plt.cm.Blues)
plt.show()
```
<img width="470" alt="Screenshot 1446-10-30 at 1 45 20 PM" src="https://github.com/user-attachments/assets/73eeea1d-835f-4180-9f4d-371d117b5c85" />
