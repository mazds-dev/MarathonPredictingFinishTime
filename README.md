# Marathon Predicting Finish Time 

### Student Details
- **Name:** Marvin Adorian Zanchi dos Santos  
- **Student Number:** C00288302  
- **Course:** BSc Software Development  
- **Module:** Data Science & Machine Learning 1
- **Lecturer:** Ben OShaughnessy 
- **Submission Date:** 4 November 2025  

---

## Project Overview
This project, titled **"Marathon Predicting Finish Time"**, applies a **Linear Regression** model to predict marathon completion times based on runners’ demographic and event data.  

The dataset, sourced from **Kaggle (2023 Marathon Results)**, contains over 420,000 entries from 600+ races worldwide.  
Each record includes the runner’s **age**, **gender**, **race name**, and **finish time** (in seconds).  

The objective is to analyze how these factors influence marathon performance and to build a predictive model that estimates a runner’s expected finish time.

---

## Project Objectives
1. Clean and prepare the marathon dataset for machine learning.  
2. Explore relationships between age, gender, race, and finish times using data visualization.  
3. Train and evaluate a **Linear Regression model** to predict finish time.  
4. Interpret the model’s coefficients to identify which variables have the most influence on performance.  
5. Present results, insights, and recommendations for future work.

---

## Dataset
**Source:** [Kaggle – 2023 Marathon Results](https://www.kaggle.com/datasets/runningwithrock/2023-marathon-results?resource=download&select=Results.csv)  
**Description:**  
- ~429,000 marathon results  
- Columns: Name, Race, Year, Gender, Age, Finish (seconds), Age Bracket  
- Year = 2023 (constant)  
- Age -1 indicates unknown (removed during cleaning)

**Note:** The dataset is too large to attach to this submission.  
You can download it directly from the Kaggle link above.

---

## Tools and Technologies
- **Language:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn  
- **Environment:** Jupyter Notebook / VS Code  
- **Version Control:** GitHub  

---

## Methodology
1. **Data Cleaning:**  
   - Removed rows with missing or invalid values (Age = -1, unknown Gender).  
   - Encoded categorical features (Gender → numeric, Race → one-hot encoding).  
2. **EDA (Exploratory Data Analysis):**  
   - Visualized distributions (Age, Finish time).  
   - Analyzed correlations (Age vs Finish, Gender vs Finish).  
3. **Model Training:**  
   - Split dataset 80/20 for training and testing.  
   - Applied `LinearRegression` model from `scikit-learn`.  
4. **Evaluation:**  
   - R² = 0.3588  
   - MAE = 2,673.62 seconds (~44.6 minutes)  
   - RMSE = 3,391.50 seconds (~56.5 minutes)  
5. **Interpretation:**  
   - Older runners generally have longer finish times.  
   - Male runners finish slightly faster on average.  
   - Certain races (e.g., trail marathons) show longer finish times due to difficulty.

---

## Key Visualizations
- Distribution of marathon finish times  
- Age vs Finish Time (with regression line)  
- Actual vs Predicted Finish Time (with perfect-fit red line)  
- Coefficient analysis showing the most influential races and demographics  

---

## Results Summary
The model explains about **36% of the variance in marathon finish times**, 
indicating that age, gender, and race are significant but not exhaustive predictors.  
Further improvements could involve additional variables such as pace, training load, or environmental conditions.

---

## How to Run This Notebook
1. Install dependencies:
2. Place the dataset file (`Results.csv`) inside a folder called `/data`  
inside the same directory as the notebook.
3. Open and run the notebook file:  
`notebooks/marathon_predicting_finish_time.ipynb`

---

## References
- Dataset: [Kaggle – 2023 Marathon Results](https://www.kaggle.com/datasets/runningwithrock/2023-marathon-results?resource=download&select=Results.csv)  
- Python Libraries: [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/)  
- Course Material: Data Science & Machine Learning 1 (College Module)  
- YouTube Tutorials: Various videos on data cleaning, exploratory data analysis, and linear regression in Python (used for learning best practices).  
- ChatGPT (OpenAI, 2025): Used to assist with project structure, code explanations, and documentation formatting.

---

## Author
Developed by Marvin Adorian Zanchi Santos  
BSc Software Development — Year 4  
South East Technological University - Carlow

