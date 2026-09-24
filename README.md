# 🚀 Python & Data Science Bootcamp

This repository contains notebooks, assignments, and datasets from the hands-on Python and Data Science Bootcamp sessions.

---

## 📂 Repository Structure

| File / Folder | Description |
| :--- | :--- |
| **`Day_1_Assignment_1.ipynb`** | Day 1 fundamentals: Python basics, string manipulations, lists, loops, and random number operations. |
| **`Day_2.ipynb`** | Day 2 concepts: NumPy arrays, statistics (mean, std), Matplotlib wave plotting, 2D array generation, and Pandas DataFrames. |
| **`Day_3.ipynb`** | Day 3 Machine Learning: Decision Tree Classification using Scikit-Learn, data visualization of student patterns, train-test splitting, and model training. |
| **`Assignment_2.ipynb`** | Assignment 2 covering: <br>1. Generating experiment datasets to CSV (`khubh_mela_data.csv`)<br>2. 2D random data generation with NumPy (`sample_2d_data.csv`)<br>3. Data cleaning in Pandas using Regular Expressions (RegEx)<br>4. Plotting a Sine Wave with Matplotlib |
| **`student_performance.csv`** | Student performance dataset containing academic factors, study time, attendance, assignments, and outcomes for classification modeling. |
| **`khubh_mela_data.csv`** | Tabular visitor dataset used for data cleaning and preprocessing experiments. |
| **`sample_2d_data.csv`** | 2D random numerical data exported via NumPy. |
| **`data.csv`** | Random generated numerical dataset. |
| **`221666051/`** | Resourcesat-2 LISS-III Remote Sensing & Machine Learning Masterclass materials. |

---

## 🛠️ Technologies & Libraries

- **Language:** Python 3
- **Numerical Computing:** [NumPy](https://numpy.org/)
- **Data Manipulation & Cleaning:** [Pandas](https://pandas.pydata.org/) (including RegEx string methods)
- **Data Visualization:** [Matplotlib](https://matplotlib.org/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (`DecisionTreeClassifier`, `train_test_split`, `metrics`, `plot_tree`)
- **Environment:** Jupyter Notebooks (`.ipynb`)

---

## ⚙️ Setup & Installation

To run these notebooks locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tanmay4tyagi/Bootcamp.git
   cd Bootcamp
   ```

2. **Install required dependencies:**
   ```bash
   pip install numpy pandas matplotlib scikit-learn jupyter
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   *Or open the folder directly in VS Code with the Jupyter extension installed.*

---

## 📝 Key Topics Covered

- **NumPy Operations:** Creating 1D/2D arrays, calculating mean/standard deviation, generating random integer grids with `np.random.randint`, saving arrays to CSV with `np.savetxt`.
- **Pandas Data Cleaning:** Handling missing values (`NaN`), text preprocessing using RegEx (`str.replace`), whitespace stripping, category standardization, and groupby aggregation.
- **Data Visualization:** Line plots, sine wave generation with `np.linspace` and `np.sin`, axis labeling, grid styling, and student performance scatter plots.
- **Machine Learning (Day 3):**
  - Supervised learning with **Decision Trees** (`DecisionTreeClassifier`).
  - Feature selection (`study_hours`, `attendance`, `assignments`) and target definition (`result`).
  - Stratified Train-Test splitting with `train_test_split`.
  - Model fitting and performance evaluation.
