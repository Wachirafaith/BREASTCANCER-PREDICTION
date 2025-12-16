
# BREAST CANCER DIAGNOSTIC SYSTEM

## Project Overview

This project is an end-to-end machine learning powered breast cancer prediction/detection system designed to assist laboratory professionals, healthcare workers and patients in understanding breast cancer diagnostic results.

The system analyzes diagnostic measurements from breast cell nuclei and classifies tumors as **Benign (Non-Cancerous)** or **Malignant (Cancerous)**. In addition to prediction, the application provides **confidence indicators**, **personalized recommendations** and an **AI-powered health assistant** through an interactive Streamlit user interface.

This project demonstrates:
- Exploratory Data Analysis (EDA)
- Model building and evaluation
- Model selection
- Model deployment
- A streamlit user-interface(UI)

---

## Dataset

- **Dataset Name:** Breast Cancer Wisconsin Diagnostic Dataset (WDBC)
- **Source:** UCI Machine Learning Repository
- **Number of Samples:** 569
- **Number of Features:** 30 numerical diagnostic features
- **Target Variable:** Diagnosis (Benign / Malignant)

The dataset contains measurements computed from digitized images of **fine needle aspirate (FNA)** of breast masses. Each feature represents characteristics of the cell nucleus such as radius, texture, perimeter, area, smoothness and concavity.

---

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the structure, quality and relationships within the data before model training.

### Data Cleaning
- Checked for missing values and removed a column containing null values to ensure data quality
- Dropped the ID column since it had no predictive value
- Checked for duplicate records to ensure data consistency

### Data Visualization
The following visualization techniques were used:
