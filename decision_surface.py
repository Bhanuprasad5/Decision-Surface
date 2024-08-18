import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification,make_circles,make_blobs,make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions

#st.image("inno_image.webp",width=200)
# Streamlit application
st.title('Boundary Surfaces Visualization')

# Create a synthetic dataset
data = st.sidebar.selectbox('type of data ', ('classification','circles','blobs', 'moons'))

if data == 'classification':
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
elif data == 'circles':
    X, y = make_circles(n_samples=100, factor=0.5, noise=0.05)
    
elif data == 'blobs':
    X,y = make_blobs(n_samples=250, centers=2, n_features=2, cluster_std=1.0, random_state=42)
    
elif data == 'moons':
    X,y = make_moons(n_samples=250,noise=0.1 ,random_state=42)
    


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def plot_decision_surface(X, y, model,title):
    plot_decision_regions(X,y,clf = model,colors="#7f7f7f,#bcbd22,#17becf")
    plt.show()
    plt.title(title)

# Dropdown menu to select the classifier
classifier_name = st.sidebar.selectbox('Select Classifier', ('KNN', 'Naive Bayes', 'Logistic Regression','DecisionTreeClassifier'))


# Add input for KNN parameters
if classifier_name == 'KNN':
    n_neighbors = st.sidebar.slider('Number of Neighbors (k)', 1, 15, 3)
    weights = st.sidebar.selectbox('Weight Function', ('uniform', 'distance'))
    algorithm = st.sidebar.selectbox('Algorithm', ('auto', 'ball_tree', 'kd_tree', 'brute'))
    n_jobs = st.sidebar.number_input('Number of Parallel Jobs (n_jobs)', -1, 10, 1)
    
    
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, n_jobs=n_jobs)
    knn.fit(X_train, y_train)
    plot_decision_surface(X, y, knn,'KNeighborsClassifier')
    st.pyplot(plt,clear_figure=True)

elif classifier_name == 'Naive Bayes':
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    plot_decision_surface(X, y, nb, 'Naive Bayes Decision Surface')
    st.pyplot(plt,clear_figure=True)

elif classifier_name =='DecisionTreeClassifier':
    dt=DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    plot_decision_surface(X, y, dt, 'DecisionTree Classifier Decision Surface')
    st.pyplot(plt,clear_figure=True)

else:
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    plot_decision_surface(X, y, lr, 'Logistic Regression Decision Surface')
    st.pyplot(plt,clear_figure=True)
