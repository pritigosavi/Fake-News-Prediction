import uvicorn
from fastapi import FastAPI, Depends, status,Form,File,UploadFile, Response, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse,RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
from starlette.templating import Jinja2Templates

# templates = Jinja2Templates(directory="templates")



app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/', status_code = status.HTTP_201_CREATED)
def index(request: Request): 
    response = templates.TemplateResponse("index.html", {"request":request,"conversation" : "indexpp"})
    return response

@app.get("/alzeimer", response_class=HTMLResponse)
async def read_alzeimer(request: Request):
    return templates.TemplateResponse("alzeimers.html", {"request": request})


@app.get("/diabetes", response_class=HTMLResponse)
async def read_diabetes(request: Request):
    return templates.TemplateResponse("diabetes.html", {"request": request})


@app.get("/heart", response_class=HTMLResponse)
async def read_heart(request: Request):
    return templates.TemplateResponse("heart.html", {"request": request})


@app.get("/kidney", response_class=HTMLResponse)
async def read_kidney(request: Request):
    return templates.TemplateResponse("kidney.html", {"request": request})

@app.get("/covid19", response_class=HTMLResponse)
async def read_covid19(request: Request):
    return templates.TemplateResponse("covid19.html", {"request": request})

@app.get("/liver", response_class=HTMLResponse)
async def read_liver(request: Request):
    return templates.TemplateResponse("liver.html", {"request": request})


@app.get("/result")
def result(request: Request):
    val = request.query_params.get("pred_txt")
    response = templates.TemplateResponse("result.html", {"request":request})
    return response

@app.post('/kidney_health_prediction_result', response_class=HTMLResponse)
async def kidney_health_prediction(
    request: Request,
    Age: int = Form(...),
    BloodPressure: int = Form(...),
    SpecificGravity: float = Form(...),
    Albumin: int = Form(...),
    Sugar: int = Form(...),
    RedBloodCells: int = Form(...),
    PusCell: int = Form(...),
    PusCellClumps: int = Form(...),
    Bacteria: int = Form(...),
    BloodGlucoseRandom: int = Form(...),
    BloodUrea: int = Form(...),
    SerumCreatinine: float = Form(...),
    Sodium: int = Form(...),
    Potassium: float = Form(...),
    Hemoglobin: float = Form(...),
    PackedCellVolume: int = Form(...),
    WhiteBloodCellCount: int = Form(...),
    RedCellCount: int = Form(...),
    Hypertension: int = Form(...),
    DiabetesMellitus: int = Form(...),
    CoronaryArteryDisease: int = Form(...),
    Appetite: int = Form(...),
    PedalEdema: int = Form(...),
    Anemia: int = Form(...),
):
    print(111111111111111111111)
    disease_name = "Kidney Health Prediction"
    
    try:
        input_data = [Age, BloodPressure, SpecificGravity, Albumin, Sugar,
                      RedBloodCells, PusCell, PusCellClumps, Bacteria,
                      BloodGlucoseRandom, BloodUrea, SerumCreatinine, Sodium,
                      Potassium, Hemoglobin, PackedCellVolume, WhiteBloodCellCount,
                      RedCellCount, Hypertension, DiabetesMellitus,
                      CoronaryArteryDisease, Appetite, PedalEdema, Anemia]
        print(222222222222222222222222222222)
        columns = ['Age', 'BloodPressure', 'SpecificGravity', 'Albumin', 'Sugar',
                   'RedBloodCells', 'PusCell', 'PusCellClumps', 'Bacteria',
                   'BloodGlucoseRandom', 'BloodUrea', 'SerumCreatinine', 'Sodium',
                   'Potassium', 'Hemoglobin', 'PackedCellVolume', 'WhiteBloodCellCount',
                   'RedCellCount', 'Hypertension', 'DiabetesMellitus',
                   'CoronaryArteryDisease', 'Appetite', 'PedalEdema', 'Anemia']
        print(333333333333333333333333)
        df = pd.DataFrame([input_data], columns=columns)
        print(4444444444444444444444444444)
        model = joblib.load("./models/kidney_model.pkl")
        print(555555555555555555)
        prediction = model.predict(df)
        print(7777777777777777)
        
        if prediction[0] == 1:
            pred_text = "Based on the analysis, it is predicted that there are signs of kidney disease. It is recommended to consult with a healthcare professional for a detailed evaluation and necessary guidance."
        else:
            pred_text = "Based on the analysis, it is predicted that there are no significant signs of kidney disease. However, for personalized advice, it is recommended to consult with a healthcare professional."

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing: {str(e)}")

    return templates.TemplateResponse("result.html", {"request": request, "disease_name": disease_name, "pred_text": pred_text})


@app.post('/covid_predicton_result', response_class=HTMLResponse)
async def diag(
    request: Request,
    Breathing_Problem: bool = Form(False),
    Fever: bool = Form(False),
    Dry_Cough: bool = Form(False),
    Sore_Throat: bool = Form(False),
    Hyper_Tension: bool = Form(False),
    Fatigue: bool = Form(False),
    Abroad_Travel: bool = Form(False),
    Contact_with_COVID_Patient: bool = Form(False),
    Attended_Large_Gathering: bool = Form(False),
    Visited_Public_Exposed_Places: bool = Form(False),
    Family_Working_in_Public_Exposed_Places: bool = Form(False),
    ):

    disease_name = "COVID-19 Prediction"
    import pickle
    model = pickle.load(open('models/covid_model.pkl', 'rb'))

    try:
        input_data = [Breathing_Problem, Fever, Dry_Cough, Sore_Throat, Hyper_Tension, Fatigue, Abroad_Travel, Contact_with_COVID_Patient, Attended_Large_Gathering, Visited_Public_Exposed_Places, Family_Working_in_Public_Exposed_Places]
                   
        print(input_data)
        prediction = model.predict([input_data])

        if prediction[0] == 0:
            pred_text = "Based on the analysis, it is predicted that you do not have COVID-19. However, it is essential to stay vigilant and follow health guidelines."
        elif prediction[0] == 1:
            pred_text = "Based on the analysis, it is predicted that you are at a high risk of having COVID-19. It is advisable to consult with a healthcare professional for further evaluation and guidance."

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing: {str(e)}")

    return templates.TemplateResponse("result.html", {"request": request, "disease_name": disease_name, "pred_text": pred_text})





@app.post('/kidney_disease_predicton_result', response_class=HTMLResponse)
async def diag(
    request: Request,
    gravity: float = Form(...),
    ph: float = Form(...),
    osmo: int = Form(...),
    cond: float = Form(...),
    urea: int = Form(...),
    calc: float = Form(...),
    ):

    disease_name = "Kidney Disease Prediction"
    model = joblib.load("./models/kidney_model.pkl")

    try:
        input_data = [gravity, ph, osmo, cond, urea, calc]
        print(input_data)
        prediction = model.predict([input_data])
        
        if prediction[0] == 1:
            pred_text = "The analysis suggests that you may have a risk of kidney stone."
        else:
            pred_text = "The analysis indicates that you do not have a significant risk of kidney stone. "
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing: {str(e)}")
        print(e)

    return templates.TemplateResponse("result.html", {"request": request, "disease_name" : disease_name, "pred_text": pred_text})










@app.post('/heart_failure_predicton_result', response_class=HTMLResponse)

async def diag(
    request: Request,
    Age: float = Form(...),
    gender: int = Form(...),
    chestPainType: int = Form(...),
    RestingBP: int = Form(...),
    Cholesterol: int = Form(...),
    FastingBS: int = Form(...),
    RestingECG: int = Form(...),
    MaxHR: int = Form(...),
    ExerciseAngina: int = Form(...),
    stDepression: float = Form(...),
    stSegmentSlope: int = Form(...),
    majorVesselsColored: int = Form(...),
    thalassemia: int = Form(...),
    ):
    
    disease_name = "Heart Failure Prediction"
    import joblib
    model = joblib.load("./models/heart_model.pkl")
 
    try:
        # input_data = [Age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        input_data = [Age, gender, chestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR,ExerciseAngina, stDepression, stSegmentSlope,majorVesselsColored,thalassemia]
        print(type(input_data[0]))
        # df = pd.DataFrame([input_data])
        prediction = model.predict([input_data])

        if prediction[0] == 1:
            pred_text = "Based on the analysis, it is predicted that you are not at risk of having a heart attack. However, for personalized advice, it is recommended to consult with a healthcare professional."
        else:
            pred_text = "Based on the analysis, it is predicted that you are at high risk of having a heart attack. It is strongly advised to consult with a healthcare professional promptly for a detailed evaluation and necessary guidance."
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing: {str(e)}")
  
    return templates.TemplateResponse("result.html", {"request": request, "disease_name": disease_name, "pred_text": pred_text})

@app.get("/result")
def result(request: Request):
    
    val = request.query_params.get("pred_txt")
    response = templates.TemplateResponse("result.html", {"request":request, "conversation" : val})
 
    return response


@app.post('/diabetes_predicton_result', response_class=HTMLResponse)
async def diag(
    request: Request,
    Pregnancies: int = Form(...),
    Glucose: int = Form(...),
    BloodPressure: int = Form(...),
    SkinThickness: int = Form(...),
    Insulin: int = Form(...),
    BMI: float = Form(...),
    DiabetesPedigreeFunction: int = Form(...),
    Age: int = Form(...),
    ):
    print(1111111111111111111111111111111111111)
    disease_name = "Diabetes Prediction"
    print(2222222222222222222222222222)
    model = joblib.load("./models/diabetes_model.pkl")
    # model = joblib.load("./models/heart_model.pkl")
    print(3333333333333333333333333333333)

    try:
        print(444444444444444444444444444444444)
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        prediction = model.predict([input_data])
        print(prediction)
        if prediction[0] == 1:
            pred_text = "The analysis suggests that you may have a risk of diabetes. It is advisable to consult with a healthcare professional for a more detailed evaluation and guidance."
        else:
            pred_text = "The analysis indicates that you do not have a significant risk of diabetes. However, for personalized advice, it is recommended to consult with a healthcare professional."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing: {str(e)}")
    print(3333333333333333333333333333333333333333333)
    return templates.TemplateResponse("result.html", {"request": request, "disease_name" : disease_name, "pred_text": pred_text})




@app.post("/alzheimer's_disease_prediction_result", response_class=HTMLResponse)
async def diag(request: Request, image: UploadFile = File(...)):

    disease_name = "Alzheimer's Disease Prediction"
    from tensorflow.keras.models import load_model
    import numpy as np
    import cv2
    from fastapi import HTTPException

    model = load_model("./models/Alzheimer_model.h5")

    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (176, 176))
        img = np.expand_dims(img, axis=0)
        output = (model.predict(img))[0]
        class_num = np.sort(['Alzheimer_s disease', 'Cognitively normal', 'Early mild Cognitive Impairment', 'Late mild Cognitive Impairment'])
        output = class_num[np.argmax(output)]
        if output == 'Cognitively normal':
            pred_text = "The analysis suggests that you are at a low risk of Alzheimer's Disease, and you are considered cognitively normal. No cause for concern, you are safe."
        elif output == 'Alzheimer_s disease':
            pred_text = "The analysis indicates a potential risk of Alzheimer's Disease. It is strongly recommended to consult with a healthcare professional immediately for a thorough evaluation and guidance."
        else:
            pred_text = f"The analysis suggests that you may have a risk of {output}. It is advisable to consult with a healthcare professional for a more detailed evaluation and guidance."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    return templates.TemplateResponse("result.html", {"request": request, "disease_name" : disease_name, "pred_text": pred_text})

@app.post('/covid_predicton_result', response_class=HTMLResponse)
async def diag(
    request: Request,
    Breathing_Problem: bool = Form(False),
    Fever: bool = Form(False),
    Dry_Cough: bool = Form(False),
    Sore_Throat: bool = Form(False),
    Running_Nose: bool = Form(False),
    Asthma: bool = Form(False),
    Chronic_Lung_Disease: bool = Form(False),
    Headache: bool = Form(False),
    Heart_Disease: bool = Form(False),
    Diabetes: bool = Form(False),
    Hyper_Tension: bool = Form(False),
    Fatigue: bool = Form(False),
    Gastrointestinal: bool = Form(False),
    Abroad_Travel: bool = Form(False),
    Contact_with_COVID_Patient: bool = Form(False),
    Attended_Large_Gathering: bool = Form(False),
    Visited_Public_Exposed_Places: bool = Form(False),
    Family_Working_in_Public_Exposed_Places: bool = Form(False),
    Wearing_Masks: bool = Form(False),
    Sanitization_from_Market: bool = Form(False),
    ):

    disease_name = "COVID-19 Prediction"
    import pickle
    model = pickle.load(open('models/covid_model.pkl', 'rb'))

    try:
        input_data = [Breathing_Problem, Fever, Dry_Cough, Sore_Throat, Running_Nose, Asthma, Chronic_Lung_Disease, Headache, Heart_Disease, Diabetes, Hyper_Tension, Fatigue, Gastrointestinal, Abroad_Travel, Contact_with_COVID_Patient, Attended_Large_Gathering, Visited_Public_Exposed_Places, Family_Working_in_Public_Exposed_Places, Wearing_Masks, Sanitization_from_Market]
        column_names = ['Breathing_Problem', 'Fever', 'Dry_Cough', 'Sore_Throat', 'Running_Nose', 'Asthma', 'Chronic_Lung_Disease', 'Headache', 'Heart_Disease', 'Diabetes', 'Hyper_Tension', 'Fatigue', 'Gastrointestinal', 'Abroad_Travel', 'Contact_with_COVID_Patient', 'Attended_Large_Gathering', 'Visited_Public_Exposed_Places', 'Family_Working_in_Public_Exposed_Places', 'Wearing_Masks', 'Sanitization_from_Market']        
        df = pd.DataFrame([[int(i) for i in input_data]], columns = column_names)

        prediction = model.predict(df)

        if prediction[0] == 0:
            pred_text = "Based on the analysis, it is predicted that you do not have COVID-19. However, it is essential to stay vigilant and follow recommended health guidelines."
        elif prediction[0] == 1:
            pred_text = "Based on the analysis, it is predicted that you are at a high risk of having COVID-19. It is advisable to consult with a healthcare professional for further evaluation and guidance."

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing: {str(e)}")

    return templates.TemplateResponse("result.html", {"request": request, "disease_name": disease_name, "pred_text": pred_text})







if __name__ == '__main__':
    uvicorn.run(app, host = "127.0.0.1", port = 8080) # running server
