from fastapi import FastAPI
import os
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import requests
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

app = FastAPI()
templates = Jinja2Templates(directory='templates')

scaler_model1 = pickle.load(open('models/scaler_model1.sav', 'rb'))
scaler_model2 = pickle.load(open('models/scaler_model2.sav', 'rb'))

model_1 = joblib.load('models/pipeline_GBC.pkl')
model_2 = joblib.load('models/pipeline2_GBC.pkl')

@app.get('/')
def main(request: Request):
    return templates.TemplateResponse('index.html', {'request':request})


@app.get('/predict_earlystage')
def predict_earlystage(request: Request,
            patientAge: float,
            vital_bps: float,
            vital_bpd: float,
            Creatinine: float,
            BUN: float,
            Albumin: float,
            WBC: float,
            HGB: float,
            Fasting_glucose: float,
            Potassium: float,
            Sodium: float,
            RBC: float,
            Hypertension: float,
            Diabetes: float,
            Anemia: float):

    df1 = [[
        patientAge,
        vital_bps,
        vital_bpd,
        Creatinine,
        BUN,
        Albumin,
        WBC,
        HGB,
        Fasting_glucose,
        Potassium,
        Sodium,
        RBC,
        Hypertension,
        Diabetes,
        Anemia
    ]]

    df1 = pd.DataFrame(df1, columns=[
        'patientAge',
        'vital_bps',
        'vital_bpd',
        'Creatinine',
        'BUN',
        'Albumin',
        'WBC',
        'HGB',
        'Fasting_glucose',
        'Potassium',
        'Sodium',
        'RBC',
        'Hypertension',
        'Diabetes',
        'Anemia'
    ])

    print(df1)

    x1_norm = scaler_model1.transform(df1)
    y_predict_1 = model_1.predict_proba(x1_norm)

    result = y_predict_1[0][1]

    return templates.TemplateResponse('index.html', context={'request':request, 'prediction_text':result})


@app.get('/predict_endstage')
def predict_endstage(request: Request,
    e_patientAge: float,
    e_patientSexName: float,
    e_vital_bps: float,
    e_vital_bpd: float,
    e_vital_height:float,
    e_vital_weight: float,
    e_smoking: float,
    e_Creatinine: float,
    e_BUN: float,
    e_ALT: float,
    e_AST: float,
    e_ALP: float,
    e_Total_protein: float,
    e_Albumin: float,
    e_Calcium: float,
    e_Phosphorous: float,
    e_CaxP: float,
    e_WBC: float,
    e_HGB: float,
    e_PLT: float,
    e_Triglyceride: float,
    e_HDL_c: float,
    e_LDL_c: float,
    e_Fasting_glucose: float,
    e_Potassium: float,
    e_Sodium: float,
    e_Chlorine: float,
    e_Bicarbonate: float,
    e_Hypertension: float,
    e_Diabetes: float,
    e_Anemia: float):

    if e_vital_height != 0:
        e_vital_height = e_vital_height/100
        e_vital_bmi = e_vital_weight / (e_vital_height*e_vital_height)
    else:
        pass

    df = [[
        e_patientAge,
        e_patientSexName,
        e_vital_bps,
        e_vital_bpd,
        e_vital_bmi,
        e_smoking,
        e_Creatinine,
        e_BUN,
        e_ALT,
        e_AST,
        e_ALP,
        e_Total_protein,
        e_Albumin,
        e_Calcium,
        e_Phosphorous,
        e_CaxP,
        e_WBC,
        e_HGB,
        e_PLT,
        e_Triglyceride,
        e_HDL_c,
        e_LDL_c,
        e_Fasting_glucose,
        e_Potassium,
        e_Sodium,
        e_Chlorine,
        e_Bicarbonate,
        e_Hypertension,
        e_Diabetes,
        e_Anemia
    ]]

    df = pd.DataFrame(df, columns=[
        'patientAge', 'patientSexName', 'vital_bps', 'vital_bpd', 'vital_bmi',
        'smoking', 'Creatinine', 'BUN', 'ALT', 'AST', 'ALP', 'Total_protein',
        'Albumin', 'Calcium', 'Phosphorous', 'Ca × P', 'WBC', 'HGB', 'PLT',
        'Triglyceride', 'HDL_c', 'LDL_c', 'Fasting_glucose', 'Potassium',
        'Sodium', 'Chlorine', 'Bicarbonate', 'Hypertension', 'Diabetes',
        'Anemia'
    ])

    x1_norm = scaler_model2.transform(df)
    y_predict_2 = model_2.predict_proba(x1_norm)

    result2 = y_predict_2[0][1]

    return templates.TemplateResponse('index.html', context={'request': request, 'prediction_text2': result2})


if __name__ == '__main__':
    uvicorn.run(app)