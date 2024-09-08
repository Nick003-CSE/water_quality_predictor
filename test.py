import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from termcolor import colored
from sklearn import preprocessing
from pickle import dump
from pickle import load
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.metrics import confusion_matrix

data = pd.read_csv('/input/water-potability/water_potability.csv')
data.head()

data.isnull().sum()

data.fillna(data.median(), inplace=True)
data.isnull().sum()

data.info()

data.shape

data.dtypes

plt.figure(figsize=(6,6))

data['Potability'].value_counts().plot.pie(explode=[0.1,0.1],
                    autopct='%1.1f%%', shadow=True,
                    textprops={'fontsize':16}).set_title("Target distribution");


new_directory = 'Models'

os.makedirs(new_directory, exist_ok=True)

data_numeric = data.drop(columns = ['Potability'])
normalizer = preprocessing.MinMaxScaler()
model_normalizer = normalizer.fit(data_numeric)
dump(model_normalizer, open('Models/normalizer_water.pkl', 'wb'))

data_numeric_normalizer = model_normalizer.fit_transform(data_numeric)
data_numeric_normalizer = pd.DataFrame(data = data_numeric_normalizer, columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])
data_numeric_normalizer.head()

X = data_numeric_normalizer
y = data['Potability']

print('Frequencies of the y before balanced')
y_count = Counter(y)
print(y_count)

# Construct object SMOTE
resampler = SMOTE()

# Perform Balance
data_x_b, data_y_b = resampler.fit_resample(X, y)

# Print the frequencies of the classes after balanced
print('Frequencies of the y after balanced')
y_count = Counter(data_y_b)
print(y_count)

# Balanced data join
data_final = data_x_b.join(data_y_b, how = 'left')
data_final

#Convert the y balanced in DataFrames
data_y_b = pd.DataFrame(data_y_b)
data_y_b

#Training that divides the base into 70 for training and 30 for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# A dictionary to define parameters to test in algorithm
parameters = {
    'n_estimators' : [300], 
    'max_depth' : [15], 
    'max_features' : ['sqrt'], 
    'random_state' : [42], 
    'min_samples_leaf' : [1], 
    'min_samples_split' : [4]
}

rf = RandomForestClassifier()
rf_cv = GridSearchCV(estimator=rf, param_grid=parameters, cv=20).fit(X_train, y_train)
print(colored('Tuned hyper parameters :\n{}'.format(rf_cv.best_params_), 'blue'))

rf = RandomForestClassifier(**rf_cv.best_params_).fit(X_train, y_train)

# Use of pred_proba for metrics.
y_pred_proba = rf.predict_proba(X_test)

# Predict the classes in the test set
y_pred = rf.classes_[y_pred_proba.argmax(axis=1)]
#y_pred = rf.predic}t(atributos_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Use of cross validate for metrics
scores_cross = cross_validate(rf, data_x_b, data_y_b.values.ravel(), scoring=['precision_macro', 'recall_macro'],
               cv=10)

cross_val_score(rf, data_x_b, data_y_b.values.ravel(), cv=10).mean()

dump(rf, open('Models/water_model_cross.pkl', 'wb'))

print("Precision Macro:", scores_cross['test_precision_macro'].mean())
print("Recall Macro:", scores_cross['test_recall_macro'].mean())


#confusion matrix
cm = confusion_matrix(y_test, y_pred, labels = rf.classes_)
print(cm)

graphic = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
graphic.plot()

dump(rf, open('Models/water_cm.pkl', 'wb'))

# Randomly select an index
idx_random = np.random.randint(0, len(data))

# Select the new data by row
new_data = data.iloc[[idx_random]]

# Save the true label (Potability) before dropping the column
true_label = new_data['Potability'].values[0]

# Remove the 'Potability' column if it exists
if 'Potability' in new_data.columns:
    new_data = new_data.drop(columns=['Potability'])

# Load the normalizer
normalizer = load(open('Models/normalizer_water.pkl', 'rb'))

# Normalize the new data
new_data_normalizer = normalizer.transform(new_data)
new_data_normalizer_df = pd.DataFrame(new_data_normalizer, columns=new_data.columns)

# Output the normalized instance
#print(f'{new_data_normalizer_df.to_string(index=False)}\n')

# Load the classifier
water_classifier = load(open('Models/water_model_cross.pkl', 'rb'))

# Classifier
result = water_classifier.predict(new_data_normalizer_df)
dist_proba = water_classifier.predict_proba(new_data_normalizer_df)

# Find the index of highest probability
idx = np.argmax(dist_proba[0])
y_pred = water_classifier.classes_[idx]
score = dist_proba[0][idx]

# Compare with the true label
true_y = water_classifier.classes_[true_label]

print("Target: ", true_y)
print("Classified as: ", y_pred, "\nScore: ", str(score))
print("0 = Not Potable  \n1 = Potable")