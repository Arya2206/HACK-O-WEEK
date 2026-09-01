# 🎓 ScoreSense — Student Performance Prediction System

ScoreSense is a simple and interactive machine learning project developed for Hackoweek. It uses regression and classification algorithms to analyze student academic performance, predict final scores, and identify whether a student is likely to pass or need additional academic support.

## 🚀 Features

* Predicts a student's final score
* Classifies students as **Pass** or **Needs Support**
* Compares multiple machine learning models
* Interactive Streamlit dashboard
* Visual model comparison
* Confusion matrix for classification
* Easy-to-use student prediction interface

## 🤖 Machine Learning Models

### Regression

The project compares four regression algorithms:

* Linear Regression
* Polynomial Regression
* Ridge Regression
* Lasso Regression

Regression performance is evaluated using:

* MAE
* RMSE
* R² Score

### Classification

The project uses two classification algorithms:

* Logistic Regression
* K-Nearest Neighbors (KNN)

Classification performance is evaluated using:

* Accuracy
* Confusion Matrix

## 📊 Input Features

The models use the following student information:

| Feature               | Description                     |
| --------------------- | ------------------------------- |
| Study Hours           | Average daily study hours       |
| Attendance            | Attendance percentage           |
| Previous Score        | Previous academic score         |
| Assignments Completed | Number of completed assignments |
| Sleep Hours           | Average daily sleep             |

## 🖥️ Project Dashboard

The project includes an interactive Streamlit dashboard where users can:

1. View model performance
2. Compare regression models
3. Compare classification models
4. View the confusion matrix
5. Enter student details
6. Generate a predicted final score
7. View the predicted student status

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Streamlit

## ▶️ How to Run

Clone the repository and open the project folder in VS Code.

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn streamlit
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 📁 Project Structure

```text
ScoreSense/
│
├── data/
│   └── student_data.csv
│
├── main.py
├── app.py
└── README.md
```

## 🎯 Objective

The objective of ScoreSense is to demonstrate how different machine learning algorithms can be applied to the same student-performance problem and compared based on their performance.

The project also demonstrates the difference between **regression**, which predicts a numerical score, and **classification**, which predicts a student category.

## 🔮 Future Improvements

Possible future improvements include:

* Adding more student features
* Using a larger real-world dataset
* Adding more machine learning models
* Improving prediction accuracy
* Deploying the Streamlit application online
* Adding personalized study recommendations

## 👨‍💻 Project

Developed as part of **Hackoweek** to demonstrate practical applications of Machine Learning.
