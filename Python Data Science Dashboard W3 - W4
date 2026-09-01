import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# OOP - DATASET MANAGER
# ============================================================

class DatasetManager:

    def __init__(self):

        # Main student dataset
        self.students = pd.DataFrame({
            "ID": [1, 2, 3, 4, 5, 6, 7, 8],
            "Name": [
                "Anushka",
                "Arpeet",
                "Rahul",
                "Priya",
                "Rohan",
                "Sneha",
                "Parth",
                "Arnav"
            ],
            "Department": [
                "CSE",
                "CSE",
                "ECE",
                "CSE",
                "ECE",
                "IT",
                "IT",
                "CSE"
            ],
            "Math": [
                85, 78, 92, np.nan,
                67, 88, 74, 95
            ],
            "Python": [
                90, 82, 89, 76,
                np.nan, 91, 70, 94
            ],
            "DataScience": [
                88, 79, 95, 80,
                72, 93, 75, np.nan
            ]
        })

        # Second dataset for MERGING
        self.attendance = pd.DataFrame({
            "ID": [1, 2, 3, 4, 5, 6, 7, 8],
            "Attendance": [
                92, 85, 96, 78,
                81, 94, 76, 98
            ]
        })


    # ========================================================
    # FUNCTION - CLEAN DATA
    # ========================================================

    def clean_data(self):

        cleaned = self.students.copy()

        # Fill missing marks with column mean
        subjects = [
            "Math",
            "Python",
            "DataScience"
        ]

        for subject in subjects:

            cleaned[subject] = cleaned[subject].fillna(
                cleaned[subject].mean()
            )

        return cleaned


    # ========================================================
    # FUNCTION - CALCULATE AVERAGE
    # ========================================================

    def calculate_average(self, row):

        return (
            row["Math"]
            + row["Python"]
            + row["DataScience"]
        ) / 3


    # ========================================================
    # LIST COMPREHENSION
    # ========================================================

    def list_comprehension_demo(self):

        marks = [85, 78, 92, 67, 88, 95]

        # Students scoring >= 80
        high_scores = [
            mark
            for mark in marks
            if mark >= 80
        ]

        return high_scores


    # ========================================================
    # DICTIONARY COMPREHENSION
    # ========================================================

    def dictionary_comprehension_demo(self):

        subjects = {
            "Math": 85,
            "Python": 90,
            "Data Science": 88
        }

        # Add 5 bonus marks
        updated_marks = {
            subject: mark + 5
            for subject, mark in subjects.items()
        }

        return updated_marks


    # ========================================================
    # NUMPY ARRAY
    # ========================================================

    def numpy_demo(self):

        marks = np.array([
            [85, 90, 88],
            [78, 82, 79],
            [92, 89, 95]
        ])

        return marks


    # ========================================================
    # NUMPY BROADCASTING
    # ========================================================

    def broadcasting_demo(self):

        marks = np.array([
            [85, 90, 88],
            [78, 82, 79],
            [92, 89, 95]
        ])

        # Broadcasting 5 marks to every value
        result = marks + 5

        return result


    # ========================================================
    # VECTORIZED OPERATIONS
    # ========================================================

    def vectorized_demo(self):

        marks = np.array([
            80,
            85,
            90,
            75,
            95
        ])

        # No loop required
        percentage = marks * 1.0

        return percentage


    # ========================================================
    # PANDAS GROUPBY
    # ========================================================

    def groupby_analysis(self):

        cleaned = self.clean_data()

        cleaned["Average"] = cleaned.apply(
            self.calculate_average,
            axis=1
        )

        result = cleaned.groupby(
            "Department"
        )["Average"].mean()

        return result


    # ========================================================
    # MERGING DATAFRAMES
    # ========================================================

    def merge_data(self):

        merged = pd.merge(
            self.students,
            self.attendance,
            on="ID"
        )

        return merged


# ============================================================
# CREATE DATASET OBJECT
# ============================================================

data_manager = DatasetManager()


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "Python Data Science Dashboard"
)

root.geometry("1100x700")

root.configure(
    bg="#f4f6f8"
)


# ============================================================
# FUNCTIONS FOR GUI
# ============================================================

def clear_content():

    for widget in content_frame.winfo_children():

        widget.destroy()


def title(text, description):

    tk.Label(
        content_frame,
        text=text,
        font=("Arial", 24, "bold"),
        bg="#f4f6f8",
        fg="#111827"
    ).pack(pady=(20, 5))

    tk.Label(
        content_frame,
        text=description,
        font=("Arial", 11),
        bg="#f4f6f8",
        fg="#4b5563"
    ).pack(pady=(0, 20))


# ============================================================
# HOME
# ============================================================

def home():

    clear_content()

    title(
        "🐍 Python Data Science Dashboard",
        "Learn Python, NumPy, Pandas and Data Visualization"
    )

    text = """
This project demonstrates the basic tools used in
Python Data Science.

Python:
    • Functions
    • Object-Oriented Programming
    • List Comprehension
    • Dictionary Comprehension

NumPy:
    • Arrays
    • Broadcasting
    • Vectorized Operations

Pandas:
    • DataFrames
    • Data Cleaning
    • Merging
    • GroupBy

Visualization:
    • Matplotlib
    • Seaborn
"""

    tk.Label(
        content_frame,
        text=text,
        font=("Arial", 14),
        bg="white",
        justify="left",
        padx=40,
        pady=30
    ).pack(pady=20)


# ============================================================
# PYTHON SECTION
# ============================================================

def python_section():

    clear_content()

    title(
        "🐍 Python Essentials",
        "Functions, OOP, List Comprehension and Dictionary Comprehension"
    )

    # Function demonstration

    high_scores = data_manager.list_comprehension_demo()

    dictionary_result = (
        data_manager.dictionary_comprehension_demo()
    )

    output = f"""
LIST COMPREHENSION
------------------

Original marks:
[85, 78, 92, 67, 88, 95]

Marks >= 80:

{high_scores}


DICTIONARY COMPREHENSION
------------------------

Original dictionary:

Math = 85
Python = 90
Data Science = 88

After adding 5 bonus marks:

{dictionary_result}


OOP
---

DatasetManager is a Python class.

An object of DatasetManager is used to
load and process the student dataset.
"""

    tk.Label(
        content_frame,
        text=output,
        font=("Courier New", 12),
        bg="white",
        justify="left",
        padx=30,
        pady=30
    ).pack(pady=20)


# ============================================================
# NUMPY SECTION
# ============================================================

def numpy_section():

    clear_content()

    title(
        "🔢 NumPy Operations",
        "Arrays, broadcasting and vectorized operations"
    )

    array = data_manager.numpy_demo()

    broadcast = data_manager.broadcasting_demo()

    vectorized = data_manager.vectorized_demo()

    output = f"""
NUMPY ARRAY
-----------

{array}


BROADCASTING
------------

Original Array:

{array}

Adding 5 using broadcasting:

{broadcast}


VECTORIZED OPERATION
--------------------

Original:

{vectorized}

Each value can be processed without
writing an explicit Python loop.
"""

    tk.Label(
        content_frame,
        text=output,
        font=("Courier New", 12),
        bg="white",
        justify="left",
        padx=30,
        pady=25
    ).pack(pady=20)


# ============================================================
# PANDAS SECTION
# ============================================================

def pandas_section():

    clear_content()

    title(
        "🐼 Pandas Data Analysis",
        "DataFrames, cleaning, merging and GroupBy"
    )

    cleaned = data_manager.clean_data()

    grouped = data_manager.groupby_analysis()

    merged = data_manager.merge_data()

    text = f"""
DATA CLEANING
-------------

Missing values were found in the marks.

They were replaced using the mean
of the corresponding subject.


GROUPBY ANALYSIS
----------------

Average marks by department:

{grouped.to_string()}


MERGING
-------

Student dataset was merged with
the attendance dataset using ID.

First rows:

{merged.head().to_string(index=False)}
"""

    tk.Label(
        content_frame,
        text=text,
        font=("Courier New", 10),
        bg="white",
        justify="left",
        padx=25,
        pady=25
    ).pack(pady=10)


# ============================================================
# DATASET TABLE
# ============================================================

def dataset_section():

    clear_content()

    title(
        "📊 Student Dataset",
        "Pandas DataFrame"
    )

    df = data_manager.students

    columns = list(df.columns)

    tree = ttk.Treeview(
        content_frame,
        columns=columns,
        show="headings",
        height=12
    )

    for column in columns:

        tree.heading(
            column,
            text=column
        )

        tree.column(
            column,
            width=100
        )

    for _, row in df.iterrows():

        values = list(row)

        tree.insert(
            "",
            "end",
            values=values
        )

    tree.pack(
        padx=20,
        pady=20,
        fill="both",
        expand=True
    )


# ============================================================
# MATPLOTLIB BAR CHART
# ============================================================

def matplotlib_chart():

    cleaned = data_manager.clean_data()

    cleaned["Average"] = cleaned.apply(
        data_manager.calculate_average,
        axis=1
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        cleaned["Name"],
        cleaned["Average"]
    )

    plt.title(
        "Student Average Marks"
    )

    plt.xlabel(
        "Students"
    )

    plt.ylabel(
        "Average Marks"
    )

    plt.xticks(
        rotation=30
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# SEABORN CHART
# ============================================================

def seaborn_chart():

    cleaned = data_manager.clean_data()

    cleaned["Average"] = cleaned.apply(
        data_manager.calculate_average,
        axis=1
    )

    plt.figure(
        figsize=(9, 5)
    )

    sns.barplot(
        data=cleaned,
        x="Department",
        y="Average"
    )

    plt.title(
        "Average Marks by Department"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# VISUALIZATION SECTION
# ============================================================

def visualization_section():

    clear_content()

    title(
        "📈 Data Visualization",
        "Create charts using Matplotlib and Seaborn"
    )

    tk.Button(
        content_frame,
        text="📊 Matplotlib Chart",
        command=matplotlib_chart,
        font=("Arial", 13, "bold"),
        bg="#2563eb",
        fg="white",
        padx=25,
        pady=10
    ).pack(pady=20)

    tk.Button(
        content_frame,
        text="📈 Seaborn Chart",
        command=seaborn_chart,
        font=("Arial", 13, "bold"),
        bg="#16a34a",
        fg="white",
        padx=25,
        pady=10
    ).pack(pady=10)

    tk.Label(
        content_frame,
        text="""
Matplotlib:
Used for creating general-purpose graphs.

Seaborn:
Built on Matplotlib and provides
more attractive statistical visualizations.
""",
        font=("Arial", 13),
        bg="white",
        justify="left",
        padx=30,
        pady=25
    ).pack(pady=20)


# ============================================================
# SIDEBAR
# ============================================================

sidebar = tk.Frame(
    root,
    width=230,
    bg="#111827"
)

sidebar.pack(
    side="left",
    fill="y"
)


tk.Label(
    sidebar,
    text="🐍\nPython Data\nScience",
    font=("Arial", 20, "bold"),
    bg="#111827",
    fg="white",
    pady=30
).pack()


buttons = [

    ("🏠  Home", home),

    ("🐍  Python", python_section),

    ("🔢  NumPy", numpy_section),

    ("🐼  Pandas", pandas_section),

    ("📋  Dataset", dataset_section),

    ("📊  Visualization", visualization_section)
]


for text, command in buttons:

    tk.Button(
        sidebar,
        text=text,
        command=command,
        font=("Arial", 11, "bold"),
        bg="#1f2937",
        fg="white",
        activebackground="#374151",
        activeforeground="white",
        relief="flat",
        anchor="w",
        padx=20,
        pady=12
    ).pack(
        fill="x",
        padx=10,
        pady=4
    )


# ============================================================
# CONTENT FRAME
# ============================================================

content_frame = tk.Frame(
    root,
    bg="#f4f6f8"
)

content_frame.pack(
    side="right",
    fill="both",
    expand=True
)


# Start application

home()


root.mainloop()
