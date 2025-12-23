# 🧑‍💼 Employee Retention Prediction  
**Capstone Project – Data Science & Artificial Intelligence**

---

## 📌 Project Overview

Employee attrition is a critical challenge for organizations, leading to increased recruitment costs and loss of skilled talent.  
This project aims to **predict employee attrition risk** using machine learning, enabling HR teams to take **proactive retention actions**.

The solution includes:
- A **LightGBM classification model**
- Feature preprocessing with **Label Encoding and Standard Scaling**
- An interactive **Streamlit web application** for real-time prediction

---

## 🎯 Objectives

- Analyze employee-related data to identify attrition patterns  
- Build a predictive ML model with high performance  
- Deploy the model as a user-friendly Streamlit app  
- Provide actionable insights for HR decision-making  

---

## 🗂️ Project Structure

```
Employee-Retention-Prediction/
│
├── Employee Retention Prediction.ipynb   # Training & analysis notebook
├── app.py                                # Streamlit application
├── model.pkl                             # Trained LightGBM model
├── scaler.pkl                            # StandardScaler used during training
├── README.md                             # Project documentation
└── requirements.txt                     # Required Python libraries
```

---

## 📊 Dataset Description

The dataset contains employee demographic, education, experience, and company-related features.

### Key Features:
- city  
- city_development_index  
- gender  
- relevent_experience  
- enrolled_university  
- education_level  
- major_discipline  
- experience  
- company_size  
- company_type  
- last_new_job  
- training_hours  

### Target Variable:
- **target**  
  - 0 → Employee stays  
  - 1 → Employee leaves  

---

## 🧠 Machine Learning Approach

### Model Used
- **LightGBM Classifier**

### Preprocessing Steps
- Label Encoding for categorical variables  
- Standard Scaling applied to features  
- Removal of non-predictive columns (`enrollee_id`, `target`) before model training  

> ⚠️ Note:  
> The scaler was fitted on the full dataset during training, so the Streamlit app reconstructs the same feature structure during inference.

---

## 📈 Model Evaluation

The model was evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC–AUC Curve  

LightGBM was chosen due to:
- Strong performance on tabular data  
- Ability to handle feature interactions  
- Fast training and inference  

---

## 🚀 Streamlit Application

The Streamlit app allows HR users to:
- Input employee details via a form  
- Predict attrition probability instantly  
- Categorize risk as:
  - 🟢 Low Risk  
  - 🟡 Medium Risk  
  - 🔴 High Risk  

### Run the app:
```bash
streamlit run app.py
```

---

## ⚙️ Installation & Setup

### 1️⃣ Create Virtual Environment (optional)
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App
```bash
streamlit run app.py
```

---

## 🧪 Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- LightGBM  
- Streamlit  
- Joblib  

---

## 🎓 Viva / Interview Explanation (Quick)

> *“This project predicts employee attrition using a LightGBM classifier.  
Preprocessing includes label encoding and standard scaling.  
The trained model is deployed using Streamlit to allow HR teams to assess attrition risk and take preventive actions.”*

---

## 🔮 Future Enhancements

- Convert preprocessing + model into a single sklearn Pipeline  
- Add SHAP explainability for feature importance  
- Integrate database support for real-time HR data  
- Improve UI with dashboards and trends  

---

## 👩‍💻 Author

**Sangeetha A**  
Capstone Project – Data Science & Artificial Intelligence  
