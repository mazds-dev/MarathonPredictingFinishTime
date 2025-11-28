# Marathon Predicting Finish Time (Linear Regression)  

## 🚀 Live Demo
Access here: 👉 **https://marathonpredictingfinishtime.streamlit.app/**  

### Student Details
- **Name:** Marvin Adorian Zanchi dos Santos  
- **Student Number:** C00288302  
- **Course:** BSc Software Development  
- **Module:** Data Science & Machine Learning 1
- **Lecturer:** Ben OShaughnessy 
- **Submission Date:** 4 November 2025  

---

# Project Overview

This project applies a **Linear Regression** model to predict marathon finish times using runners’ demographic and event-related data.

The dataset, sourced from **Kaggle (2023 Marathon Results)**, contains over **420,000** entries from **600+ marathon events** across the United States.

Each record includes:

* Age
* Gender
* Race name
* Finish time (seconds)

The aim is to analyze how these variables influence marathon performance and develop a model that estimates a runner’s expected finish time.

---

# Project Objectives

1. Clean and prepare a large, real-world marathon dataset for machine learning.
2. Explore relationships between age, gender, race, and finish time.
3. Train and evaluate a **Linear Regression** model.
4. Interpret the model coefficients to understand what drives performance.
5. Summarize insights and propose future directions for improvement.

---

# Dataset

**Source:**
[Kaggle – 2023 Marathon Results](https://www.kaggle.com/datasets/runningwithrock/2023-marathon-results?resource=download&select=Results.csv)

**Dataset Summary**

* ~429,000 marathon results
* Columns: Name, Race, Year, Gender, Age, Finish, Age Bracket
* Age = -1 indicates unknown (removed during cleaning)
* All events occurred in **2023**

**Note:**
Due to its size, the dataset is **not included** in this repository.
Download it from Kaggle and place it inside a folder named `/data`.

---

# Tools & Technologies

* **Language:** Python
* **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
* **Environment:** Jupyter Notebook / Visual Studio Code
* **Version Control:** GitHub

---

# Methodology

### **1. Data Cleaning**

* Removed invalid ages (<16 or >90, or age = -1)
* Kept only Male/Female categories
* Encoded Gender numerically (Male = 1, Female = 0)
* Removed columns not required for modelling: Name, Age Bracket, Year
* Removed extreme finish times (> 20,000 seconds) to avoid ultra-marathon outliers

### **2. Exploratory Data Analysis (EDA)**

* Visualized distribution of finish times (in hours)
* Analysed age distribution
* Explored how Age, Gender, and Race relate to finish time
* Created a correlation heatmap

### **3. Feature Engineering**

* Selected features: **Age**, **Gender**, **Race**
* Applied one-hot encoding to Race
* Split data into training and testing (80/20)

### **4. Model Training**

* Trained a Linear Regression model using scikit-learn
* Predicted `Finish` (seconds)

### **5. Evaluation**

- **R²:** 0.186  
- **MAE:** 32.29 minutes  
- **RMSE:** 39.16 minutes  

### **6. Interpretation**

* **Age effect:** +34.7 seconds/year (~0.58 minutes per year)
* **Gender effect:** Male runners finish ~19 minutes faster on average
* **Race:** Significant impact due to course difficulty, terrain, and climate

---

# Key Visualisations

* Distribution of finish times
* Distribution of runner ages
* Age vs Finish scatterplot
* Gender vs Finish time boxplot
* Actual vs Predicted scatterplot with perfect-fit line
* Feature coefficient ranking

---

# Results Summary

The linear regression model explains around **19%** of the variance in marathon finish times.
This is expected given that marathon performance depends on many additional factors:

* training load
* pacing strategy
* athlete experience
* physiological variables (VO₂ max)
* course elevation
* weather conditions

Even so, the model provides **clear and interpretable insights** into how age, gender, and race influence performance.

---

# How to Run the Notebook

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Place the dataset file here:

   ```
   /data/Results.csv
   ```
3. Open and run the notebook:

   ```
   notebook/marathon_predicting_finish_time.ipynb
   ```

---

# References

* Dataset: Kaggle – 2023 Marathon Results
* Python Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
* Course Content: Data Science & Machine Learning 1
* YouTube Tutorials: Exploratory data analysis & regression modelling
* ChatGPT (OpenAI, 2025): Assisted with documentation, structure, and formatting

---

# Author

Developed by Marvin Adorian Zanchi Santos
BSc Software Development — Year 4
South East Technological University — Carlow

---
