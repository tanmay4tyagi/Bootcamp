# 🚀 Python, Data Science & AI Bootcamp

This repository contains notebooks, assignments, datasets, and the **Final Capstone Project** from the hands-on Python, Data Science, Machine Learning, and Artificial Intelligence Bootcamp.

---

## 🌟 Featured Project: OmniVision AI (Capstone Submission)
**File:** [`Final_Project_Capstone.ipynb`](Final_Project_Capstone.ipynb)  
An integrated multimodal AI system combining all core competencies taught throughout the bootcamp into a unified, laptop-friendly pipeline:
- **Module 1: Machine Learning & Predictive Analytics** — Customer purchase prediction with Decision Tree Classification, EDA scatter/distribution plots, train-test splitting, confusion matrix, and visual tree logic graph.
- **Module 2: Deep Learning Image Classification (CNN)** — Convolutional Neural Network built with TensorFlow/Keras using Conv2D, MaxPooling2D, Flatten, Dense, and Dropout, with full loss/accuracy epoch diagnostic curves.
- **Module 3: Real-Time Object Detection & Live Counting (YOLOv8)** — Live webcam object detection and visitor counting powered by Ultralytics YOLOv8 Nano (`yolov8n.pt`) with Windows DirectShow backend, auto camera index detection, and overlay counter.
- **Module 4: Natural Language Processing & Sentiment Word Cloud** — Text corpus analysis with NLTK tokenization, stopword removal, frequency distribution analysis via `Counter`, and high-resolution Word Cloud generation.

---

## 📂 Repository Structure

| File / Folder | Description |
| :--- | :--- |
| **`Final_Project_Capstone.ipynb`** | **⭐ Capstone Project:** Unified Multimodal AI system (Decision Tree + CNN + YOLOv8 + Word Cloud). |
| **`Day_1_Assignment_1.ipynb`** | Day 1 fundamentals: Python basics, string manipulations, lists, loops, and random number operations. |
| **`Day_2.ipynb`** | Day 2 concepts: NumPy arrays, statistics (mean, std), Matplotlib wave plotting, 2D array generation, and Pandas DataFrames. |
| **`Day_3.ipynb`** | Day 3 Machine Learning: Decision Tree Classification using Scikit-Learn, data visualization of student patterns, train-test splitting, and model training. |
| **`Day_4.ipynb`** | Day 4 Deep Learning & Computer Vision: OpenCV transformations, CNN image classification, YOLOv8 live object counting, and NLTK Word Cloud visualization. |
| **`Assignment_2.ipynb`** | Assignment 2 covering: <br>1. Generating experiment datasets to CSV (`khubh_mela_data.csv`)<br>2. 2D random data generation with NumPy (`sample_2d_data.csv`)<br>3. Data cleaning in Pandas using Regular Expressions (RegEx)<br>4. Plotting a Sine Wave with Matplotlib |
| **`Assignment_3.ipynb`** | Assignment 3: Linear Regression (power prediction) and K-Means Clustering on the Marvel Cinematic Universe dataset. |
| **`customer_analytics_data.csv`** | Generated tabular dataset used for the Capstone Machine Learning classification module. |
| **`student_performance.csv`** | Student performance dataset containing academic factors, study time, attendance, assignments, and outcomes for classification modeling. |
| **`khubh_mela_data.csv`** | Tabular visitor dataset used for data cleaning and preprocessing experiments. |
| **`sample_2d_data.csv`** | 2D random numerical data exported via NumPy. |
| **`data.csv`** | Random generated numerical dataset. |
| **`mcu.csv`** | Marvel Cinematic Universe dataset for data analysis. |
| **`yolov8n.pt`** | Pre-trained YOLOv8 Nano PyTorch weights (~6.5MB) for real-time object detection. |
| **`ISRO Datasets and image processing/`** | Remote sensing and satellite image processing masterclass resources. |

---

## 🛠️ Technologies & Libraries

- **Language:** Python 3
- **Numerical Computing:** [NumPy](https://numpy.org/)
- **Data Manipulation & Cleaning:** [Pandas](https://pandas.pydata.org/) (including RegEx string methods)
- **Data Visualization:** [Matplotlib](https://matplotlib.org/), [WordCloud](https://github.com/amueller/word_cloud)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (`DecisionTreeClassifier`, `LinearRegression`, `KMeans`, `train_test_split`, `metrics`, `plot_tree`)
- **Deep Learning & Computer Vision:** [OpenCV](https://opencv.org/) (`cv2`), [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/), [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Natural Language Processing:** [NLTK](https://www.nltk.org/) (`word_tokenize`, `stopwords`)
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
   pip install numpy pandas matplotlib scikit-learn opencv-python tensorflow keras ultralytics nltk wordcloud pillow
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   *Or open the folder directly in VS Code with the Jupyter extension installed.*

---

## 📝 Key Topics Covered

### 🔹 Day 1 & 2: Foundations, NumPy & Pandas
- **NumPy Operations:** 1D/2D arrays, mean, standard deviation, random integer generation with `np.random.randint`, saving arrays to CSV with `np.savetxt`.
- **Pandas Data Cleaning:** Handling `NaN` values, RegEx text cleaning (`str.replace`), category standardization, and groupby aggregation.
- **Matplotlib Visualization:** Sine waves (`np.linspace`, `np.sin`), scatter plots, and custom styling.

### 🔹 Day 3: Machine Learning
- **Decision Trees:** Supervised learning using `DecisionTreeClassifier` on student performance metrics.
- **Pipeline:** Feature selection, train-test splitting, model fitting, and evaluation metrics.

### 🔹 Day 4: Deep Learning, Computer Vision & NLP
- **OpenCV Image Processing:** Image loading (`cv2.imread`), dimensions checking, resizing (`cv2.resize`), flipping (`cv2.flip`), and cropping.
- **Convolutional Neural Networks (CNN):**
  - Image data sanitization and preprocessing with PIL.
  - Multi-layer CNN architecture with `Rescaling`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dropout`, and `Dense` layers using Keras.
- **Real-Time Object Detection & Counting (YOLOv8):**
  - Pre-trained lightweight YOLOv8 Nano model (`yolov8n.pt`).
  - Windows webcam connection with DirectShow (`cv2.CAP_DSHOW`) and device fallback.
  - Real-time frame annotation, object counting overlay, and safe hardware cleanup.
- **Word Cloud Generation (NLP):**
  - Tokenization using NLTK `word_tokenize`.
  - Stopwords removal and punctuation filtering.
  - Word frequency distribution with `collections.Counter`.
  - Inline Word Cloud rendering using `wordcloud` and `matplotlib.pyplot`.
