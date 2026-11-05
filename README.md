# Iris Species Classification – Streamlit ML Dashboard

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Decision%20Tree-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

An interactive Machine Learning web application built with Streamlit and Scikit-Learn that allows users to perform Exploratory Data Analysis (EDA), tune Decision Tree hyperparameters dynamically, view evaluation metrics, and test live predictions on custom flower measurements.

## Features

- **Interactive Hyperparameter Tuning:** Adjust test data split ratio and Decision Tree depth live from the sidebar.
- **Exploratory Data Analysis (EDA):** Interactive dataset previews, summary statistics, and species counts.
- **Data Visualization:** Embedded Matplotlib and Seaborn plots for feature relationships and confusion matrix heatmaps.
- **Real-Time Inference:** Instant species classification widget for custom user inputs.

## Project Structure

```text
Iris-Species-Classification/
│
├── IrisClassifier.py
├── iris.csv
├── requirements.txt
├── .gitignore
├── README.md
├── LICENSE
└── screenshots/