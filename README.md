# Employees dataset with false positives

### Import libraries
```python
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
### Split, train and test the dataset
```python
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
<img width="517" alt="Screenshot 1446-10-30 at 1 56 14 PM" src="https://github.com/user-attachments/assets/37aac0c1-4043-4509-97ad-a72ab05624a8" />
<img width="757" alt="Screenshot 1446-10-30 at 1 56 47 PM" src="https://github.com/user-attachments/assets/940d9fb9-7efd-4a89-8faf-03ee89a625c6" />


### List of False Positives in the Full Dataset
<img width="568" alt="Screenshot 1446-10-30 at 2 02 00 PM" src="https://github.com/user-attachments/assets/b171ba3a-80bc-42af-aa3d-d435fe9a5786" />
<img width="568" alt="Screenshot 1446-10-30 at 2 02 18 PM" src="https://github.com/user-attachments/assets/2b3afece-c740-4a64-a966-2046caa6acce" />

### Highlight and display False Positives in the full dataset
```python
false_positives = df[df['False_Positive'] == 1]

print("="*50)
print("List of False Positives in the Full Dataset")
print("="*50)
display(false_positives[['Employee_ID', 'Name', 'Department', 'Job_Title', 'Performance_Score', 'Needs_Training']])

# Step 8: Highlight all false positives nicely in the full dataset
def highlight_false_positive(row):
    return ['background-color: lightcoral' if row.False_Positive == 1 else '' for _ in row]

styled_df = df.style.apply(highlight_false_positive, axis=1)
styled_df
```

<img width="653" alt="Screenshot 1446-10-30 at 2 03 04 PM" src="https://github.com/user-attachments/assets/63441f5b-1505-445a-971a-8fa20b3800c7" />



