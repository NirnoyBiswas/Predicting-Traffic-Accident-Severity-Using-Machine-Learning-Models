# Predicting Traffic Accident Severity Using Machine Learning Models
This repository contains the implementation accompanying our research paper on predicting traffic accident severity across four ordinal levels using the large-scale **US Accidents dataset (2016–2023)**. This study compares multiple machine learning algorithms and introduces a highly efficient class imbalance resolution pipeline to yield deployment-ready road safety models.

## 📌 Project Overview
 * **Dataset:** US Accidents Dataset (2016-2023) comprising over **7.7 million records**.
 * **Core Challenge:** Severe class imbalance (Severity Level 2 accounts for ~83% of data, while Severity Levels 1 and 4 combine for less than 2%).
 * **Methodology:** Comparative evaluation of 5 machine learning models alongside two sampling variations (Pure SMOTE vs. a Custom Hybrid Sampling Strategy).
 * **Key Results:** **XGBoost** achieved the highest overall accuracy (**85.2%**) and AUC-ROC (**0.903**). Our hybrid sampling approach boosted critical **Severity-4 Recall** from **39.2% to 68.4%**.
## 🛠️ Data Preprocessing Pipeline
To scale down the sheer volume of 7.7 million instances under computational constraints, a **stratified random sample of approximately 500,000 records** was used.
The data was cleaned and transformed using the following pipeline:
 1. **Feature Elimination:** Removed irrelevant identifiers and text fields (ID, Source, Description, Street).
 2. **Categorical Encoding:** One-hot encoded high-cardinality features like State and Weather_Condition, and mapped binary features (Sunrise_Sunset) to {0, 1}.
 3. **Imputation:** Missing numerical values were imputed using column medians, and missing categorical items were assigned an Unknown sentinel tag.
 4. **Feature Engineering:** Extracted Hour, Weekday, and Month temporal metrics from Start_Time.
*The resulting engineered training dataframe yielded a total of **290 features**.*
## ⚖️ Handling Class Imbalance
We evaluated two separate class balance strategies applied strictly after an 80/20 train-test split to avoid data leakage:
 * **Pure SMOTE:** Oversampled minority classes to match the majority class (Severity 2), creating roughly ~332,000 training instances per class.
 * **Hybrid Sampling:** Implemented Random Under-Sampling (RUS) to cap the dominant Severity 2 class at 100,000 instances, followed by SMOTE to equalize the rest. This decreased overall training time by over 60% while dramatically improving rare-event metric detection.
## 🤖 Machine Learning Models
Five classifiers were constructed, tuned, and analyzed:
 * **XGBoost:** Top-performer, utilizing the histogram method for fast split finding and GPU-accelerated training.
 * **Random Forest:** Ensemble of 50 randomized decision trees utilizing Mean Decrease in Impurity for ranking feature dynamics.
 * **Logistic Regression:** Multinomial baseline configured with L2 regularization (max\_iter=1000).
 * **Decision Tree:** CART-based approach using Gini impurity.
 * **Gaussian Naive Bayes:** Generative probabilistic baseline model.
## 📈 Key Experimental Results
### 1. Performance Summary (Pure SMOTE)
| Model | Accuracy | Macro F1 | AUC-ROC | Sev-4 Recall |
|---|---|---|---|---|
| **XGBoost** | **0.852** | **0.621** | **0.903** | **0.463** |
| Random Forest | 0.831 | 0.584 | 0.882 | 0.392 |
| Logistic Regression | 0.831 | 0.524 | 0.882 | 0.392 |
| Decision Tree | 0.792 | 0.501 | 0.841 | 0.341 |
| Gaussian Naive Bayes | 0.641 | 0.382 | 0.791 | 0.298 |
### 2. Pure SMOTE vs. Hybrid Sampling Trade-Off
When using **Random Forest**, the hybrid pipeline optimizes minor-class recall significantly—making it highly relevant for safety-critical configurations where missing a catastrophic incident is far costlier than a false alarm:
 * **Accuracy:** Dropped slightly from 83.1% to 81.0%.
 * **Macro F1-Score:** Increased from 0.584 to **0.670**.
 * **Severity-4 Recall:** Substantially spiked from 39.2% to **68.4%**.
## 🌡️ Feature Importance Analysis
According to the Random Forest Mean Decrease in Impurity analysis, meteorological metrics play a massive role in severity indexing, with the top three variables accounting for more than 45% of total predictive significance:
 1. **Temperature (°F)**
 2. **Humidity (%)**
 3. **Pressure (in)**
 4. Wind Speed (mph)
 5. Visibility (mi)
 6. Distance (mi)
 7. Start Latitude
 8. Start Longitude
 9. Sunrise/Sunset
 10. Wind Chill (°F)
## 🚀 Repository Structure
```text
├── data/                  # Metadata links / Sampling scripts
├── notebooks/             # EDA and Preprocessing steps
├── src/
│   ├── preprocess.py      # Cleans data, handles imputation, 290 feature outputs
│   ├── balancing.py       # Pure SMOTE & Hybrid pipeline functions
│   └── train.py           # Model training execution scripts
├── requirements.txt       # Necessary Python libraries (xgboost, imblearn, etc.)
└── README.md

```
