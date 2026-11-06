# 🌺 Iris Species Classification - Interactive ML Dashboard

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Decision_Tree-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas&logoColor=white)
![Visualization](https://img.shields.io/badge/Data-Visualization-blueviolet)
![License](https://img.shields.io/badge/License-MIT-yellow?logo=opensourceinitiative&logoColor=white)

**Can a machine learning model identify a flower just by looking at its measurements? 🌸**

Turns out, it can!

**Iris Species Classification** is an interactive Machine Learning web application built using Python, Streamlit, and Scikit-Learn. It uses a **Decision Tree Classifier** to classify Iris flowers into three different species based on their physical measurements.

But this isn't just another model that spits out predictions.

The application provides an interactive dashboard where users can explore the dataset, understand feature relationships, experiment with model hyperparameters, evaluate classification performance, and even test their own flower measurements.

From exploratory data analysis to model evaluation and interactive prediction, the entire machine learning workflow is brought together in one place.

---

## 🚀 Key Features

### 1. 📊 Dataset Exploration & EDA

Explore the Iris dataset and understand what goes into training the model.

- Preview the dataset directly through the dashboard.
- View the total number of records and columns.
- Examine the distribution of different Iris species.
- Generate statistical summaries using Pandas.
- Understand the dataset before training the classifier.

### 2. 📈 Interactive Data Visualization

Visualize relationships between flower measurements.

- Scatter plot of Petal Length vs Petal Width.
- Different visual representations for each Iris species.
- Color-coded species distribution.
- Grid and legend for easier interpretation.

The visualization helps understand how petal measurements differ across the three flower species.

### 3. ⚙️ Dynamic Model Configuration

Experiment with Decision Tree hyperparameters directly from the sidebar.

- Adjust the training and testing split ratio.
- Configure the maximum depth of the Decision Tree.
- Retrain the classifier using the selected parameters.
- Observe how model configuration affects classification performance.

**Available configurations:**

| Parameter | Minimum | Maximum | Default |
|---|---:|---:|---:|
| Test Split Size | 10% | 90% | 50% |
| Maximum Tree Depth | 1 | 10 | 5 |

### 4. 🎯 Model Evaluation & Performance

Understand how well the trained classifier performs.

- Overall classification accuracy.
- Confusion matrix heatmap.
- Precision, Recall, and F1-score.
- Detailed classification report.
- Comparison between actual and predicted species.

### 5. 🔮 Interactive Flower Prediction

Test the trained model using custom flower measurements.

Users can adjust:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

Click the **Classify Species** button to generate a prediction.

The application displays the predicted Iris species instantly.

---

## 🧠 Machine Learning Approach

This project implements a **Supervised Machine Learning Classification Model** using the Decision Tree algorithm.

### 1. Dataset Preparation

The application uses the Iris dataset containing 150 observations and 5 columns.

The dataset consists of four numerical input features and one target variable.

**Input Features:**

| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal in centimeters |
| Sepal Width | Width of the sepal in centimeters |
| Petal Length | Length of the petal in centimeters |
| Petal Width | Width of the petal in centimeters |

**Target Variable:**

The model predicts one of the following three Iris species:

- Setosa
- Versicolor
- Virginica

Each species contains 50 records in the supplied dataset.

### 2. Train-Test Split

The dataset is divided into training and testing subsets using Scikit-Learn's `train_test_split()`.

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=test_size,
    random_state=42
)
```

- Training data is used to train the classifier.
- Testing data is used to evaluate its performance.
- The test split ratio can be adjusted through the Streamlit sidebar.

### 3. Decision Tree Classification

The project uses Scikit-Learn's `DecisionTreeClassifier`.

```python
model = DecisionTreeClassifier(max_depth=max_depth)

model.fit(X_train, Y_train)
```

A Decision Tree learns patterns from the input features by creating a series of decision rules.

For example, it can learn how petal length and petal width help distinguish different Iris species.

The maximum depth parameter controls how deep the decision tree can grow.

### 4. Model Prediction

After training, the model predicts the species of flowers in the testing dataset.

```python
Y_pred = model.predict(X_test)
```

The predictions are compared against the actual labels to evaluate the classifier.

### 5. Model Evaluation

The application calculates three main evaluation outputs:

**Accuracy Score**

Measures the proportion of correctly classified test samples.

**Confusion Matrix**

Shows the number of correctly and incorrectly classified observations for each species.

**Classification Report**

Provides:
- Precision
- Recall
- F1-score
- Support

These metrics help evaluate the classifier beyond accuracy alone.

---

## 🎮 Application Dashboard

The application is divided into four interactive tabs.

| Dashboard Tab | Functionality |
|---|---|
| 📊 Dataset & EDA | Dataset preview, species distribution and statistical summary |
| 📈 Visualization | Petal length vs petal width scatter plot |
| 🎯 Model Metrics | Accuracy, confusion matrix and classification report |
| 🔮 Real-time Prediction | Predict species using custom flower measurements |

### How to Use

1. Launch the Streamlit application.
2. Adjust the test split size and maximum tree depth from the sidebar.
3. Explore the dataset using the Dataset & EDA tab.
4. Understand feature relationships through the Visualization tab.
5. Evaluate model performance in the Model Metrics tab.
6. Enter custom flower measurements in the Real-time Prediction tab.
7. Click **Classify Species** to view the predicted result.

---

## 🧰 Technologies & Libraries Used

| Technology / Library | Purpose |
|---|---|
| Python 3 | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data processing and manipulation |
| Matplotlib | Data visualization |
| Seaborn | Confusion matrix visualization |
| Scikit-Learn | Machine Learning implementation |
| DecisionTreeClassifier | Classification algorithm |
| train_test_split | Training and testing data separation |
| accuracy_score | Model accuracy evaluation |
| confusion_matrix | Classification performance analysis |
| classification_report | Precision, Recall and F1-score |

---

## 📂 Project Structure

```text
Iris-Species-Classification/
│
├── IrisClassifier.py       # Main Streamlit application
│
├── iris.csv                # Iris flower dataset
│
├── requirements.txt        # Project dependencies
│
├── .gitignore              # Git ignored files
│
├── README.md               # Project documentation
│
└── LICENSE                 # MIT License
```

---

## 💻 Installation & Setup

Want to run the Iris Classifier locally? Follow these steps.

### Prerequisites

- Python 3.x installed on your system.
- pip package manager.
- Basic knowledge of Python and command-line operations.

### Step 1: Clone the Repository

```bash
git clone https://github.com/bemo-codes/Iris-Species-Classification
```

Navigate to the project directory:

```bash
cd Iris-Species-Classification
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt
```

The project uses the following dependencies:

```text
streamlit
pandas
matplotlib
seaborn
scikit-learn
```

### Step 4: Run the Application

Execute the following command:

```bash
streamlit run IrisClassifier.py
```

The application will start locally.

Open the following URL in your browser:

```text
http://localhost:8501
```

You're ready to explore the Iris dataset and experiment with Machine Learning! 🌺

---

## ⚡ Implementation Highlights

Some important implementation details:

- **Streamlit Caching:** Uses `@st.cache_data` to avoid repeatedly loading the CSV dataset.
- **Dynamic Training:** The classifier is trained using the hyperparameters selected through the sidebar.
- **Reproducible Data Split:** Uses `random_state=42` in the train-test split.
- **Interactive Visualizations:** Uses Matplotlib and Seaborn to present feature relationships and evaluation results.
- **User-Driven Inference:** Allows users to provide custom flower measurements and obtain predictions from the trained model.

The model is trained again when the Streamlit application reruns, using the currently selected configuration.

---

## ⚠️ Current Limitations

- The project currently uses only the Iris dataset.
- Classification is limited to three Iris species.
- The model uses a single train-test split rather than cross-validation.
- Model performance depends on the selected test split and tree depth.
- The Decision Tree model is not currently saved for reuse between application sessions.
- Feature importance and Decision Tree visualization are not currently implemented.
- The application does not currently support uploading custom datasets.

---

## 🔮 Future Enhancements

There is plenty of room to expand this project!

Some possible improvements include:

- **Decision Tree Visualization:** Display the trained tree structure graphically.
- **Feature Importance Analysis:** Identify which flower measurements contribute most to classification.
- **Additional ML Algorithms:** Compare Decision Tree with Random Forest, Logistic Regression, and KNN.
- **Cross-Validation:** Evaluate model performance across multiple data splits.
- **Model Comparison Dashboard:** Compare multiple classifiers using different evaluation metrics.
- **Dataset Upload:** Allow users to upload their own CSV datasets.
- **Prediction Confidence:** Display prediction probabilities alongside predicted species.
- **Model Persistence:** Save trained models using Joblib or Pickle.
- **Enhanced Visualizations:** Introduce additional plots such as pair plots and feature distributions.

---

## 👨‍💻 Author

**Shivam Ramesh Kurlekar**

GitHub: [@bemo-codes](https://github.com/bemo-codes)

---

## 📄 License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for more information.

---

<p align="center">
  Built with Python 🐍 | Powered by Machine Learning 🤖 | Inspired by Nature 🌺
</p>

<p align="center">
  ⭐ If you find this project interesting, consider giving the repository a star!
</p>
