
---

# 🛍️ Customer Segmentation using K-Means | Streamlit Deployment

## 📌 Project Overview

Customer segmentation is a crucial business strategy used to divide customers into meaningful groups based on their behavior and characteristics.
In this project, I built a **machine learning–based customer segmentation system** using **K-Means clustering** and deployed it as an **interactive Streamlit web application**.

The application allows users to input customer details and instantly predicts the **customer segment**, helping businesses make data-driven marketing decisions.

Live Demo [https://customersegmentationml-htvawf38s7rcaatjapachm.streamlit.app/]

---

## 🎯 Problem Statement

Businesses often treat all customers the same, which leads to inefficient marketing and low customer retention.
The goal of this project is to:

* Identify distinct customer groups
* Understand customer behavior
* Enable personalized marketing strategies

---

## 🧠 Solution Approach

I applied an **unsupervised machine learning approach** using **K-Means clustering** to segment customers based on their attributes.

---

## 🔄 Project Workflow

```
1. Load Dataset
2. Data Cleaning
3. Preprocessing
   - Feature Engineering
   - Feature Selection
   - Feature Scaling (StandardScaler)
4. Elbow Method (Optimal K selection)
5. PCA (for visualization)
6. K-Means Model Training
7. Model Evaluation
8. Cluster Labeling
9. Streamlit Deployment
```

---

## 🛠️ Technologies & Tools Used

* **Python**
* **Pandas & NumPy** – Data manipulation
* **Scikit-learn** – Machine learning models
* **Matplotlib / Seaborn** – Visualization
* **Joblib** – Model persistence
* **Streamlit** – Web app deployment
* **GitHub** – Version control

---

## 📊 Machine Learning Model

* **Algorithm:** K-Means Clustering
* **Why K-Means?**

  * Fast and scalable
  * Widely used for customer segmentation
  * Easy to interpret for business use cases

---

## 📐 Feature Scaling

Feature scaling was performed using **StandardScaler** because K-Means is a distance-based algorithm.

> Feature scaling, PCA, and K-Means were combined into a single **scikit-learn Pipeline**, which was trained and saved using `joblib` to ensure consistent preprocessing during deployment.

---

## 🧩 PCA (Principal Component Analysis)

* Used only for **visualization**
* Reduced high-dimensional data into 2D
* Helped in visual interpretation of customer clusters

---

## 🏷️ Customer Segments

After model training and evaluation, clusters were labeled into meaningful business segments such as:

* High-level Customers
* Mid-level Customers
* Low-leve/At-Risk Customers
  

*(Exact labels may vary based on dataset)*

---

## 🚀 Deployment

The trained model pipeline was deployed using **Streamlit Community Cloud**.

### Deployment Features:

* User-friendly input form
* Real-time customer segmentation
* Business-readable segment output

---

## 📁 Repository Structure  

```
customer-segmentation-streamlit/
│
├── app.py              # Streamlit application
├── model.joblib        # Trained ML pipeline (Scaler + PCA + K-Means)
├── Kmeans.ipynb        # Model development & analysis notebook
├── requirements.txt    # Required libraries
└── README.md           # Project documentation
```

---

## ▶️ How to Run the Project Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---


