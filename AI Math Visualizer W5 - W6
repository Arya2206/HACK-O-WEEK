import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sympy as sp


# =========================================================
# MAIN WINDOW
# =========================================================

root = tk.Tk()
root.title("AI Mathematics Visualizer")
root.geometry("1000x700")
root.configure(bg="#f4f6f8")


# =========================================================
# GENERAL FUNCTIONS
# =========================================================

def clear_frame():
    for widget in content_frame.winfo_children():
        widget.destroy()


def create_title(title, description):
    tk.Label(
        content_frame,
        text=title,
        font=("Arial", 24, "bold"),
        bg="#f4f6f8",
        fg="#1f2937"
    ).pack(pady=(20, 5))

    tk.Label(
        content_frame,
        text=description,
        font=("Arial", 11),
        bg="#f4f6f8",
        fg="#4b5563"
    ).pack(pady=(0, 20))


# =========================================================
# VECTOR SECTION
# =========================================================

def vector_section():

    clear_frame()

    create_title(
        "📐 Vector Operations",
        "Perform vector addition, magnitude and dot product"
    )

    frame = tk.Frame(content_frame, bg="#f4f6f8")
    frame.pack(pady=10)

    tk.Label(
        frame,
        text="Vector A",
        font=("Arial", 14, "bold"),
        bg="#f4f6f8"
    ).grid(row=0, column=0, padx=30)

    tk.Label(
        frame,
        text="Vector B",
        font=("Arial", 14, "bold"),
        bg="#f4f6f8"
    ).grid(row=0, column=2, padx=30)

    ax = tk.Entry(frame, width=8, font=("Arial", 13))
    ay = tk.Entry(frame, width=8, font=("Arial", 13))

    bx = tk.Entry(frame, width=8, font=("Arial", 13))
    by = tk.Entry(frame, width=8, font=("Arial", 13))

    ax.insert(0, "3")
    ay.insert(0, "4")

    bx.insert(0, "2")
    by.insert(0, "1")

    ax.grid(row=1, column=0, padx=5, pady=5)
    ay.grid(row=2, column=0, padx=5, pady=5)

    bx.grid(row=1, column=2, padx=5, pady=5)
    by.grid(row=2, column=2, padx=5, pady=5)

    def calculate():

        try:
            A = np.array([
                float(ax.get()),
                float(ay.get())
            ])

            B = np.array([
                float(bx.get()),
                float(by.get())
            ])

            addition = A + B
            dot = np.dot(A, B)

            magnitude_A = np.linalg.norm(A)
            magnitude_B = np.linalg.norm(B)

            result.config(
                text=f"""
Vector A + Vector B
= {addition}

Dot Product
= {dot:.2f}

Magnitude of A
= {magnitude_A:.2f}

Magnitude of B
= {magnitude_B:.2f}
"""
            )

        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Please enter valid numbers."
            )

    tk.Button(
        content_frame,
        text="Calculate",
        command=calculate,
        font=("Arial", 12, "bold"),
        bg="#2563eb",
        fg="white",
        padx=25,
        pady=8
    ).pack(pady=20)

    result = tk.Label(
        content_frame,
        text="Enter vector values and click Calculate",
        font=("Arial", 13),
        bg="white",
        justify="left",
        padx=30,
        pady=20
    )

    result.pack(pady=20)


# =========================================================
# MATRIX SECTION
# =========================================================

def matrix_section():

    clear_frame()

    create_title(
        "🔢 Matrix Operations",
        "Multiply two 2 × 2 matrices"
    )

    main = tk.Frame(content_frame, bg="#f4f6f8")
    main.pack(pady=10)

    # Matrix A

    tk.Label(
        main,
        text="Matrix A",
        font=("Arial", 14, "bold"),
        bg="#f4f6f8"
    ).grid(row=0, column=0, columnspan=2)

    A_entries = []

    for i in range(2):
        row = []

        for j in range(2):

            entry = tk.Entry(
                main,
                width=6,
                font=("Arial", 14)
            )

            entry.insert(0, str(i + j + 1))
            entry.grid(
                row=i + 1,
                column=j,
                padx=5,
                pady=5
            )

            row.append(entry)

        A_entries.append(row)

    # Matrix B

    tk.Label(
        main,
        text="Matrix B",
        font=("Arial", 14, "bold"),
        bg="#f4f6f8"
    ).grid(row=0, column=3, columnspan=2)

    B_entries = []

    for i in range(2):
        row = []

        for j in range(2):

            entry = tk.Entry(
                main,
                width=6,
                font=("Arial", 14)
            )

            entry.insert(0, str(i + j + 5))
            entry.grid(
                row=i + 1,
                column=j + 3,
                padx=5,
                pady=5
            )

            row.append(entry)

        B_entries.append(row)

    def multiply():

        try:

            A = np.array([
                [float(A_entries[i][j].get()) for j in range(2)]
                for i in range(2)
            ])

            B = np.array([
                [float(B_entries[i][j].get()) for j in range(2)]
                for i in range(2)
            ])

            C = np.matmul(A, B)

            result.config(
                text=f"""
Matrix A:

{A}

Matrix B:

{B}

A × B:

{C}
"""
            )

        except ValueError:

            messagebox.showerror(
                "Invalid Input",
                "Please enter valid numbers."
            )

    tk.Button(
        content_frame,
        text="Multiply Matrices",
        command=multiply,
        font=("Arial", 12, "bold"),
        bg="#2563eb",
        fg="white",
        padx=20,
        pady=8
    ).pack(pady=25)

    result = tk.Label(
        content_frame,
        text="Matrix result will appear here",
        font=("Courier New", 12),
        bg="white",
        justify="left",
        padx=30,
        pady=20
    )

    result.pack()


# =========================================================
# EIGENVALUE SECTION
# =========================================================

def eigen_section():

    clear_frame()

    create_title(
        "🔄 Eigenvalues & Eigenvectors",
        "Find eigenvalues and eigenvectors of a matrix"
    )

    tk.Label(
        content_frame,
        text="Enter a 2 × 2 matrix",
        font=("Arial", 13),
        bg="#f4f6f8"
    ).pack()

    entries = []

    matrix_frame = tk.Frame(
        content_frame,
        bg="#f4f6f8"
    )

    matrix_frame.pack(pady=20)

    for i in range(2):

        row = []

        for j in range(2):

            entry = tk.Entry(
                matrix_frame,
                width=7,
                font=("Arial", 14)
            )

            entry.insert(
                0,
                "2" if i == j else "0"
            )

            entry.grid(
                row=i,
                column=j,
                padx=5,
                pady=5
            )

            row.append(entry)

        entries.append(row)

    def calculate():

        try:

            matrix = np.array([
                [
                    float(entries[i][j].get())
                    for j in range(2)
                ]
                for i in range(2)
            ])

            eigenvalues, eigenvectors = np.linalg.eig(matrix)

            result.config(
                text=f"""
Matrix:

{matrix}

Eigenvalues:

{eigenvalues}

Eigenvectors:

{eigenvectors}

Intuition:
Eigenvectors are special directions that
remain in the same direction after a matrix
transformation.

Eigenvalues tell us how much those directions
are stretched or compressed.
"""
            )

        except ValueError:

            messagebox.showerror(
                "Invalid Input",
                "Please enter valid numbers."
            )

    tk.Button(
        content_frame,
        text="Calculate Eigenvalues",
        command=calculate,
        font=("Arial", 12, "bold"),
        bg="#2563eb",
        fg="white",
        padx=20,
        pady=8
    ).pack(pady=15)

    result = tk.Label(
        content_frame,
        text="Results will appear here",
        font=("Courier New", 11),
        bg="white",
        justify="left",
        padx=25,
        pady=20
    )

    result.pack(pady=10)


# =========================================================
# DERIVATIVE SECTION
# =========================================================

def derivative_section():

    clear_frame()

    create_title(
        "📈 Derivative Calculator",
        "Calculate the derivative of a mathematical function"
    )

    tk.Label(
        content_frame,
        text="Enter function in x:",
        font=("Arial", 13),
        bg="#f4f6f8"
    ).pack()

    function_entry = tk.Entry(
        content_frame,
        width=30,
        font=("Arial", 15)
    )

    function_entry.insert(0, "x**2 + 3*x")
    function_entry.pack(pady=15)

    def calculate():

        try:

            x = sp.symbols("x")

            expression = sp.sympify(
                function_entry.get()
            )

            derivative = sp.diff(
                expression,
                x
            )

            result.config(
                text=f"""
Function:

f(x) = {expression}

Derivative:

f'(x) = {derivative}

The derivative tells us how quickly
the function changes with respect to x.

Machine learning uses derivatives to
determine how model parameters should change.
"""
            )

        except Exception:

            messagebox.showerror(
                "Invalid Function",
                "Example: x**2 + 3*x"
            )

    tk.Button(
        content_frame,
        text="Calculate Derivative",
        command=calculate,
        font=("Arial", 12, "bold"),
        bg="#2563eb",
        fg="white",
        padx=20,
        pady=8
    ).pack(pady=15)

    result = tk.Label(
        content_frame,
        text="Derivative result will appear here",
        font=("Arial", 12),
        bg="white",
        justify="left",
        padx=25,
        pady=20
    )

    result.pack()


# =========================================================
# GRADIENT SECTION
# =========================================================

def gradient_section():

    clear_frame()

    create_title(
        "🎯 Gradient Calculator",
        "Calculate the gradient of a multivariable function"
    )

    tk.Label(
        content_frame,
        text="Function f(x, y):",
        font=("Arial", 13),
        bg="#f4f6f8"
    ).pack()

    function_entry = tk.Entry(
        content_frame,
        width=30,
        font=("Arial", 15)
    )

    function_entry.insert(
        0,
        "x**2 + y**2"
    )

    function_entry.pack(pady=15)

    def calculate():

        try:

            x, y = sp.symbols("x y")

            expression = sp.sympify(
                function_entry.get()
            )

            dx = sp.diff(
                expression,
                x
            )

            dy = sp.diff(
                expression,
                y
            )

            result.config(
                text=f"""
Function:

f(x,y) = {expression}

∂f/∂x = {dx}

∂f/∂y = {dy}


Gradient:

∇f = [{dx}, {dy}]


The gradient points in the direction
of the steepest increase.

In machine learning, gradient descent
moves in the opposite direction to
minimize the loss function.
"""
            )

        except Exception:

            messagebox.showerror(
                "Invalid Function",
                "Example: x**2 + y**2"
            )

    tk.Button(
        content_frame,
        text="Calculate Gradient",
        command=calculate,
        font=("Arial", 12, "bold"),
        bg="#2563eb",
        fg="white",
        padx=20,
        pady=8
    ).pack(pady=15)

    result = tk.Label(
        content_frame,
        text="Gradient result will appear here",
        font=("Arial", 12),
        bg="white",
        justify="left",
        padx=25,
        pady=20
    )

    result.pack()


# =========================================================
# CHAIN RULE / BACKPROPAGATION
# =========================================================

def chain_rule_section():

    clear_frame()

    create_title(
        "⛓️ Chain Rule & Backpropagation",
        "Understand how gradients flow through functions"
    )

    explanation = """
Example:

z = x²
y = 3z

Using the chain rule:

dy/dx = (dy/dz) × (dz/dx)

dy/dz = 3
dz/dx = 2x

Therefore:

dy/dx = 3 × 2x
      = 6x

This idea is fundamental to
neural-network backpropagation.
"""

    tk.Label(
        content_frame,
        text=explanation,
        font=("Arial", 14),
        bg="white",
        justify="left",
        padx=30,
        pady=25
    ).pack(pady=20)

    x = tk.Entry(
        content_frame,
        width=10,
        font=("Arial", 14)
    )

    x.insert(0, "2")
    x.pack(pady=10)

    def calculate():

        try:

            value = float(x.get())

            dz_dx = 2 * value

            dy_dz = 3

            dy_dx = dy_dz * dz_dx

            result.config(
                text=f"""
x = {value}

dz/dx = {dz_dx}

dy/dz = {dy_dz}

dy/dx = (dy/dz)(dz/dx)

dy/dx = {dy_dx}

This demonstrates how the chain rule
allows gradients to flow backward through
multiple layers of calculations.
"""
            )

        except ValueError:

            messagebox.showerror(
                "Invalid Input",
                "Enter a number."
            )

    tk.Button(
        content_frame,
        text="Calculate",
        command=calculate,
        font=("Arial", 12, "bold"),
        bg="#2563eb",
        fg="white",
        padx=25,
        pady=8
    ).pack(pady=15)

    result = tk.Label(
        content_frame,
        text="Result will appear here",
        font=("Arial", 12),
        bg="white",
        justify="left",
        padx=30,
        pady=20
    )

    result.pack()


# =========================================================
# ABOUT SECTION
# =========================================================

def about_section():

    clear_frame()

    create_title(
        "🧠 About This Project",
        "Mathematical foundations of Artificial Intelligence"
    )

    text = """
AI and Machine Learning rely heavily on mathematics.

Linear Algebra
    → Vectors and matrices represent data and model parameters.

Eigenvalues
    → Help understand matrix transformations and dimensionality reduction.

Calculus
    → Derivatives tell us how functions change.

Gradients
    → Tell us the direction of maximum increase.

Chain Rule
    → Allows gradients to flow through multiple operations.

Backpropagation
    → Uses the chain rule to calculate gradients in neural networks.

This application combines all these concepts
into one interactive Python-based visualizer.
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


# =========================================================
# SIDEBAR
# =========================================================

sidebar = tk.Frame(
    root,
    width=220,
    bg="#111827"
)

sidebar.pack(
    side="left",
    fill="y"
)

tk.Label(
    sidebar,
    text="🧠\nAI Math\nVisualizer",
    font=("Arial", 20, "bold"),
    bg="#111827",
    fg="white",
    pady=30
).pack()


buttons = [
    ("📐  Vectors", vector_section),
    ("🔢  Matrices", matrix_section),
    ("🔄  Eigenvalues", eigen_section),
    ("📈  Derivatives", derivative_section),
    ("🎯  Gradients", gradient_section),
    ("⛓️  Chain Rule", chain_rule_section),
    ("ℹ️  About", about_section)
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


# =========================================================
# CONTENT AREA
# =========================================================

content_frame = tk.Frame(
    root,
    bg="#f4f6f8"
)

content_frame.pack(
    side="right",
    fill="both",
    expand=True
)


# Start with About page

about_section()


# =========================================================
# RUN APPLICATION
# =========================================================

root.mainloop()
