# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit
import pandas as pd

df = pd.read_csv('parkinsons.csv')
df.head()
df = df.dropna()
import seaborn as sns
sns.pairplot(df, hue='status')
selected_features = ['PPE', 'spread1']
target = 'status'

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_for_analysis = scaler.fit_transform(df[selected_features])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_for_analysis, df[target], test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5) 
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
import joblib

joblib.dump(model, 'new_model.joblib')
