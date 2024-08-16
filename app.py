import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load dataset
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('cancer patient data sets.csv')
    return df

df = load_data()

# Preprocess the data
def preprocess_data(df):
    df.drop('index', axis=1, inplace=True, errors='ignore')

    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    Q1 = df[numerical_columns].quantile(0.25)
    Q3 = df[numerical_columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_threshold = Q1 - 1.5 * IQR
    upper_threshold = Q3 + 1.5 * IQR
    for col in numerical_columns:
        df[col] = np.where((df[col] < lower_threshold[col]) | (df[col] > upper_threshold[col]),
                           np.where(df[col] < lower_threshold[col], lower_threshold[col], upper_threshold[col]),
                           df[col])

    label_encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    return df

df = preprocess_data(df)

# Define features and target
features = ['Age', 'Gender','chronic Lung Disease', 'Obesity', 'Smoking', 'Snoring']
target = 'Level'

# Split the data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn) * 100
precision_knn = precision_score(y_test, y_pred_knn, average='weighted') * 100
recall_knn = recall_score(y_test, y_pred_knn, average='weighted') * 100
f1_knn = f1_score(y_test, y_pred_knn, average='weighted') * 100

# Streamlit app
st.title("Cancer Patient Data Analysis and Prediction")

st.write("### Dataset")
st.write(df.head())

st.write("### Model Evaluation")
st.write(f"**Accuracy:** {accuracy_knn:.1f}%")
st.write(f"**Precision:** {precision_knn:.1f}%")
st.write(f"**Recall:** {recall_knn:.1f}%")
st.write(f"**F1 Score:** {f1_knn:.1f}%")

st.write("### Make a Prediction")
age = st.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
chronic_lung_disease = st.slider("Chronic Lung Disease (1-10)", 1, 10, 5)
obesity = st.slider("Obesity (1-10)", 1, 10, 5)
smoking = st.slider("Smoking (1-10)", 1, 10, 5)
snoring = st.slider("Snoring (1-10)", 1, 10, 5)

user_data = np.array([age, gender, chronic_lung_disease, obesity, smoking, snoring]).reshape(1, -1)
user_data_scaled = scaler.transform(user_data)
# prediction = knn_model.predict(user_data_scaled)

# st.write(f"**Predicted Level:** {prediction[0]}")

if st.button("Predict"):
    if(age<25 or age>80):
        st.write(f"age is invalid")
    else:
        prediction = knn_model.predict(user_data_scaled)
        st.write(f"Predicted result for the input: {prediction[0]}")

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import LabelEncoder

# @st.cache_data(allow_output_mutation=True)
# def load_data():
#     df = pd.read_csv('cancer_patient_data.csv')
#     df.drop('index', axis=1, inplace=True)
#     numerical_columns = df.select_dtypes(include=['int64']).columns
#     Q1 = df[numerical_columns].quantile(0.25)
#     Q3 = df[numerical_columns].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_threshold = Q1 - 1.5 * IQR
#     upper_threshold = Q3 + 1.5 * IQR
#     for col in numerical_columns:
#         df[col] = np.where((df[col] < lower_threshold[col]) | (df[col] > upper_threshold[col]),
#                            np.where(df[col] < lower_threshold[col], lower_threshold[col], upper_threshold[col]),
#                            df[col])
#     categorical_columns = df.select_dtypes(include=['object']).columns
#     label_encoder = LabelEncoder()
#     for col in categorical_columns:
#         df[col] = label_encoder.fit_transform(df[col])
#     return df

# df = load_data()

# features = ['Age', 'Gender', 'chronic Lung Disease', 'Obesity', 'Smoking', 'Snoring']
# target = 'Level'

# X = df[features]
# y = df[target]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# knn_model = KNeighborsClassifier(n_neighbors=5)
# knn_model.fit(X_train, y_train)

# y_pred_knn = knn_model.predict(X_test)
# accuracy_knn = accuracy_score(y_test, y_pred_knn) * 100
# precision_knn = precision_score(y_test, y_pred_knn, average='weighted') * 100
# recall_knn = recall_score(y_test, y_pred_knn, average='weighted') * 100
# f1_knn = f1_score(y_test, y_pred_knn, average='weighted') * 100

# st.write("KNN Classifier Model:")
# st.write(f"Accuracy: {accuracy_knn:.1f}%")
# st.write(f"Precision: {precision_knn:.1f}%")
# st.write(f"Recall: {recall_knn:.1f}%")
# st.write(f"F1 Score: {f1_knn:.1f}%")

# def get_user_input():
#     user_data = []
#     user_data.append(int(st.text_input("Age: ")))
#     user_data.append(int(st.selectbox("Gender (Male: 0, Female: 1): ", [0, 1])))
#     user_data.append(int(st.text_input("Chronic Lung Disease (1-10): ")))
#     user_data.append(int(st.text_input("Obesity (1-10): ")))
#     user_data.append(int(st.text_input("Smoking (1-10): ")))
#     user_data.append(int(st.text_input("Snoring (1-10): ")))
#     return np.array(user_data).reshape(1, -1)

# user_input = get_user_input()
# user_input_scaled = scaler.transform(user_input)

# if st.button("Predict"):
#     prediction = knn_model.predict(user_input_scaled)
#     st.write(f"Predicted result for the input: {prediction[0]}")
