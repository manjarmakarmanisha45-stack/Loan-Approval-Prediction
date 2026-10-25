
import joblib
import pandas as pd
import gradio as gr

model = joblib.load("loan_approval_model.pkl")
columns = joblib.load("loan_approval_columns.pkl")

def predict_loan(gender, married, dependents, education,
                 self_employed, income, co_income,
                 loan_amount, term, credit_history, property_area):

    applicant = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": income,
        "CoapplicantIncome": co_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": term,
        "Credit_History": float(credit_history),
        "Property_Area": property_area
    }])

    applicant = pd.get_dummies(applicant)
    applicant = applicant.reindex(columns=columns, fill_value=0)

    prediction = model.predict(applicant)[0]

    if prediction == 1:
        return "Loan Prediction: APPROVED"
    return "Loan Prediction: REJECTED"

demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Radio(["Yes", "No"], label="Married"),
        gr.Dropdown(["0", "1", "2", "3+"], label="Dependents"),
        gr.Radio(["Graduate", "Not Graduate"], label="Education"),
        gr.Radio(["Yes", "No"], label="Self Employed"),
        gr.Number(label="Applicant Income", value=5000),
        gr.Number(label="Coapplicant Income", value=1500),
        gr.Number(label="Loan Amount (thousands)", value=150),
        gr.Dropdown([360, 180, 120, 60, 36, 12],
                    label="Loan Term (months)", value=360),
        gr.Radio(["1.0", "0.0"], label="Credit History"),
        gr.Radio(["Urban", "Semiurban", "Rural"],
                 label="Property Area")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Loan Approval Prediction System"
)

demo.launch()
