import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles, make_blobs, make_moons, make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import MinMaxScaler

# Set page config
st.set_page_config(page_title="Boundary Surfaces Visualization", page_icon=":bar_chart:", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        /* Page background */
        body {
            background-color: #f5f7fa;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #2a3f5f;
            color: #ffffff;
            padding: 20px;
            border-radius: 15px;
        }

        /* Title styling */
        .css-1v3fvcr h1 {
            color: #0F52BA;
            text-align: center;
            font-size: 40px;
            font-family: 'Arial Black', sans-serif;
        }

        /* Plot background */
        .stApp .plot-container {
            background-color: #ffffff;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 10px;
            margin-top: 20px;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
        }

        /* Adjust font size of sidebar elements */
        .stSelectbox, .stSlider {
            font-size: 16px;
        }

        /* Customize buttons */
        .stButton button {
            background-color: #50C878;
            color: white;
            border-radius: 10px;
            border: none;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #45B567;
            transition: 0.3s;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the app
st.title('Boundary Surfaces Visualization')

# Sidebar for data selection
st.sidebar.header('Select Dataset and Classifier')

# Create a synthetic dataset
data = st.sidebar.selectbox('Choose the type of data', ('Classification', 'Circles', 'Blobs', 'Moons', 'Gaussian Quantiles'))

if data == 'Classification':
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
elif data == 'Circles':
    X, y = make_circles(n_samples=100, factor=0.5, noise=0.05)
elif data == 'Blobs':
    X, y = make_blobs(n_samples=250, centers=2, n_features=2, cluster_std=1.0, random_state=42)
elif data == 'Moons':
    X, y = make_moons(n_samples=250, noise=0.1, random_state=42)
elif data == 'Gaussian Quantiles':
    X, y = make_gaussian_quantiles(n_samples=100, n_features=2, n_classes=3, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to plot decision surfaces
def plot_decision_surface(X, y, model, title):
    plt.figure(figsize=(8, 6))  # Explicitly create a figure
    plot_decision_regions(X, y, clf=model, colors='#50C878,#E0115F,#0F52BA')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Check if there are any labels for the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend(loc='best')
    
    st.pyplot(plt)  # Ensure plt is passed to st.pyplot

# Dropdown menu to select the classifier
classifier_name = st.sidebar.selectbox('Select Classifier', ('KNN', 'Naive Bayes', 'Logistic Regression', 'DecisionTreeClassifier'))

# Classifier settings and visualization
if classifier_name == 'KNN':
    n_neighbors = st.sidebar.slider('Number of Neighbors (k)', 1, 15, 3)
    weights = st.sidebar.selectbox('Weight Function', ('uniform', 'distance'))
    algorithm = st.sidebar.selectbox('Algorithm', ('auto', 'ball_tree', 'kd_tree', 'brute'))

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    knn.fit(X_train, y_train)
    plot_decision_surface(X, y, knn, 'KNeighborsClassifier')
    st.pyplot(plt, clear_figure=True)

elif classifier_name == 'Naive Bayes':
    nb_type = st.sidebar.selectbox('Select Naive Bayes Type', ('Gaussian', 'Multinomial', 'Bernoulli'))

    if nb_type == 'Gaussian':
        nb = GaussianNB()
        nb.fit(X_train, y_train)
    elif nb_type == 'Multinomial':
        alpha = st.sidebar.slider('Alpha (Smoothing)', 0.0, 1.0, 1.0)
        scaler = MinMaxScaler()  # Scale data to non-negative values
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        nb = MultinomialNB(alpha=alpha)
        nb.fit(X_train_scaled, y_train)
    elif nb_type == 'Bernoulli':
        alpha = st.sidebar.slider('Alpha (Smoothing)', 0.0, 1.0, 1.0)
        binarize = st.sidebar.slider('Binarize Threshold', 0.0, 1.0, 0.0)
        nb = BernoulliNB(alpha=alpha, binarize=binarize)
        nb.fit(X_train, y_train)

    if nb_type == 'Multinomial':
        plot_decision_surface(X_test_scaled, y_test, nb, f'{nb_type} Naive Bayes Decision Surface')
    else:
        plot_decision_surface(X, y, nb, f'{nb_type} Naive Bayes Decision Surface')
    
    st.pyplot(plt, clear_figure=True)

elif classifier_name == 'DecisionTreeClassifier':
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    plot_decision_surface(X, y, dt, 'DecisionTree Classifier Decision Surface')
    st.pyplot(plt, clear_figure=True)

else:
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    plot_decision_surface(X, y, lr, 'Logistic Regression Decision Surface')
    st.pyplot(plt, clear_figure=True)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <p style="font-size: 16px; color: #888;">Developed by Your Name - © 2024</p>
    </div>
""", unsafe_allow_html=True)
