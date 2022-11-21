#uvicorn apps.server-diagnosis.main:app --reload

# Import API libraries 
from fastapi import FastAPI
from fastapi import Body, FastAPI
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware

# Import Utility libraries
import pickle
from typing import Union
import numpy as np

import pandas as pd
import numpy as np
import pickle
from experta import *

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df=pd.read_csv("cvd_risk_train.csv",sep=',')
df=df.drop(['id'],axis=1)
x=df.drop(['cardio'],axis=1)
y=df['cardio']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.3,random_state=1)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10)
rfc.fit(xtrain,ytrain)

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(xtrain,ytrain)

from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier()
knc.fit(xtrain,ytrain)

from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier()
sgd.fit(xtrain,ytrain)


models = [rfc, dtc, knc, sgd]


class CVDExpertSystem(KnowledgeEngine):
    def diagnose(self, form_dictionary):
        _overweight = ((form_dictionary["weight"]/((form_dictionary["height"]/100)**2))>25)
        _active = form_dictionary["active"]==1
        _above_50 = form_dictionary["age"]>50
        _hypotension = (form_dictionary["ap_hi"]<90)|(form_dictionary["ap_lo"]<60)
        _hypertension = (form_dictionary["ap_hi"]>140)|(form_dictionary["ap_lo"]>90)
        _smoking = form_dictionary["smoke"]==1
        _high_cholesterol = form_dictionary["cholesterol"]>1
        _diabetes = form_dictionary["gluc"]>1
        _racially_african = form_dictionary["racial_identity"]==1
        self.declare(Fact(overweight = _overweight))
        self.declare(Fact(active = _active))
        self.declare(Fact(above_50 = _above_50))
        self.declare(Fact(hypotension = _hypotension))
        self.declare(Fact(hypertension = _hypertension))
        self.declare(Fact(smoking = _smoking))
        self.declare(Fact(high_cholesterol = _high_cholesterol))
        self.declare(Fact(diabetes = _diabetes))
        self.declare(Fact(racially_african = _racially_african))
        self.declare(Fact(blood_clotting_disorder = form_dictionary["blood_clotting_disorder"]==1))
        self.declare(Fact(kidney_disease = form_dictionary["kidney_disease"]==1))
        self.declare(Fact(chest_pains = form_dictionary["chest_pains"]))
        self.declare(Fact(breathlessness = form_dictionary["breathlessness"]==1))
        self.declare(Fact(nausea = form_dictionary["nausea"]==1))
        self.declare(Fact(faintings = form_dictionary["faintings"]==1))
        self.declare(Fact(fatigue = form_dictionary["fatigue"]==1))
        self.declare(Fact(swollen_ankles = form_dictionary["swollen_ankles"]==1))
        self.declare(Fact(drastic_weight_changes = form_dictionary["drastic_weight_changes"]==1))
        self.declare(Fact(bloating = form_dictionary["bloating"]==1))
        self.declare(Fact(loss_of_appetite = form_dictionary["loss_of_appetite"]==1))
        self.declare(Fact(dizziness_confusion = form_dictionary["dizziness_confusion"]==1))
        self.declare(Fact(palpitations = form_dictionary["palpitations"]==1))
        self.declare(Fact(atrial_fibrillation = form_dictionary["atrial_fibrillation"]==1))
        self.declare(Fact(sudden_headache = form_dictionary["sudden_headache"]==1))
        self.declare(Fact(sudden_lossofvision = form_dictionary["sudden_lossofvision"]==1))
        self.declare(Fact(face_dropping = form_dictionary["face_dropping"]==1))
        self.declare(Fact(numbness = form_dictionary["numbness"]==1))
        self.declare(Fact(speech_problem = form_dictionary["speech_problem"]==1))
        self.declare(Fact(leg_pain_cramping = form_dictionary["leg_pain_cramping"]==1))
        self.declare(Fact(burning_aching_toes = form_dictionary["burning_aching_toes"]==1))
        self.declare(Fact(cool_feet_skin = form_dictionary["cool_feet_skin"]==1))
        self.declare(Fact(redness_colorchanges_skin = form_dictionary["redness_colorchanges_skin"]==1))
        self.declare(Fact(back_pain = form_dictionary["back_pain"]==1))
        self.declare(Fact(coughing = form_dictionary["coughing"]==1))
        self.declare(Fact(hoarseness = form_dictionary["hoarseness"]==1))
        self.declare(Fact(tenderness_pain_chest = form_dictionary["tenderness_pain_chest"]==1))
        self.declare(Fact(sharp_sudden_pain_upperback = form_dictionary["sharp_sudden_pain_upperback"]==1))
        self.declare(Fact(pain_chest_jaw_neck_arms = form_dictionary["pain_chest_jaw_neck_arms"]==1))
        self.declare(Fact(loss_of_consciousness = form_dictionary["loss_of_consciousness"]==1))
        self.declare(Fact(difficulty_breathing = form_dictionary["difficulty_breathing"]==1))
        self.declare(Fact(trouble_swallowing_food = form_dictionary["trouble_swallowing_food"]==1))
        self.declare(Fact(deep_constant_belly_pain = form_dictionary["deep_constant_belly_pain"]==1))
        self.declare(Fact(family_history_coronaryheartdisease = form_dictionary["family_history_coronaryheartdisease"]==1))
        self.declare(Fact(personal_family_history_blood_bloodvesseldisease = form_dictionary["personal_family_history_blood_bloodvesseldisease"]==1))
        self.run()
        return self.symptoms_probability_dictionary
    
    symptoms_probability_dictionary = {"coronary_heart_disease":0, "stroke":0, "peripheral_arterial_disease":0, "thoracic_aortic_aneurysms":0, "abdominal_aortic_aneurysm":0}

    #coronary heart disease 
    #possible 
    @Rule(Fact(smoking = True)| Fact(hypertension = True) | Fact(high_cholesterol = True) | Fact(active = False) | Fact(diabetes = True) | Fact(overweight = True) | Fact(family_history_coronaryheartdisease = True), salience = 11)
    def possible_coronary_heart_disease(self):
        self.symptoms_probability_dictionary["coronary_heart_disease"] = 0.1
        
    #probable
    @Rule(AND(Fact(chest_pains = True), Fact(breathlessness=True), Fact(nausea = True), Fact(faintings = True)), salience = 10)
    def probable_coronary_heart_disease(self):
        self.facts 
        self.symptoms_probability_dictionary["coronary_heart_disease"] = 0.5
        
    #confirmed 
    @Rule(Fact(breathlessness=True), Fact(fatigue = True), Fact(swollen_ankles = True), Fact(coughing = True), Fact(palpitations = True), salience = 9)
    def confirmed_coronary_heart_disease(self):
        self.symptoms_probability_dictionary["coronary_heart_disease"] = 0.9

    #stroke
    #possible
    @Rule(Fact(overweight = True) | Fact(smoking = True) | Fact(high_cholesterol = True) | Fact(hypertension = True) | Fact(diabetes = True) | Fact(atrial_fibrillation = True), salience = 8)
    def possible_stroke(self):
        self.symptoms_probability_dictionary["stroke"] = 0.1
    
    #probable 
    @Rule(Fact(sudden_headache = True), Fact(dizziness_confusion = True), Fact(sudden_lossofvision = True), Fact(numbness = True), salience = 7)
    def probable_stroke(self):
        self.symptoms_probability_dictionary["stroke"] = 0.5

    #confirmed 
    @Rule(Fact(face_dropping = True), Fact(numbness = True), Fact(speech_problem = True), salience = 6)
    def confirmed_stroke(self):
        self.symptoms_probability_dictionary["stroke"] = 0.9
        
    #peripheral arterial disease
    #possible 
    @Rule(Fact(smoking = True) | Fact(diabetes = True) | Fact(above_50 = True) | Fact(racially_african = True) | Fact(personal_family_history_blood_bloodvesseldisease = True) | Fact(blood_clotting_disorder = True) | Fact(kidney_disease = True), salience = 5)
    def possible_peripheral_arterial_disease(self):
        self.symptoms_probability_dictionary["peripheral_arterial_disease"] = 0.1

    #probable 
    @Rule(Fact(leg_pain_cramping = True), Fact(burning_aching_toes = True), Fact(cool_feet_skin = True), Fact(redness_colorchanges_skin = True), salience = 4)
    def probable_peripheral_arterial_disease(self):
        self.symptoms_probability_dictionary["peripheral_arterial_disease"] = 0.5
        
    #Thoracic aortic aneurysms
    #possible 
    @Rule(Fact(back_pain = True) | Fact(coughing = True) | Fact(hoarseness = True) | Fact(difficulty_breathing = True) | Fact(tenderness_pain_chest = True), salience = 3)
    def possible_thoracic_aortic_aneurysms(self):
        self.symptoms_probability_dictionary["thoracic_aortic_aneurysms"] = 0.1

    #confirmed 
    @Rule(Fact(sharp_sudden_pain_upperback = True), Fact(pain_chest_jaw_neck_arms = True), Fact(difficulty_breathing = True), Fact(hypertension = True), Fact(loss_of_consciousness = True), Fact(breathlessness=True), Fact(trouble_swallowing_food = True), salience = 2)
    def probable_thoracic_aortic_aneurysms(self):
        self.symptoms_probability_dictionary["thoracic_aortic_aneurysms"] = 0.9

    #abdominal aortic aneurysm
    #confirmed
    @Rule(Fact(deep_constant_belly_pain = True)& Fact(back_pain = True), salience = 1)
    def confirmed_abdominal_aortic_aneurysm(self):
        self.symptoms_probability_dictionary["abdominal_aortic_aneurysm"] = 0.9


expert_system = CVDExpertSystem()

@app.get("/")
def hello():
    return {"message":"test"}

@app.post("/diagnose/")
async def diagnose(form_dictionary:dict):
    for i in form_dictionary.keys():
        if i != "racial_identity":
            if type(form_dictionary[i]) != int:
                form_dictionary[i] = int(form_dictionary[i])
    prediction = predict_risk(form_dictionary)
    diagnosis = diagnose_disease(form_dictionary)
    return {"risk": prediction, "diagnosis": diagnosis}
#age:int,gender:int,height:float,weight:float,ap_hi:int,ap_lo:int,cholesterol:int,gluc:int,smoke:int,alco:int,active:int

def predict_risk(form_dictionary:dict):
    positive = 0
    number_of_model = len(models)
    for model in models:
        prediction = model.predict(np.array([form_dictionary["age"], form_dictionary["gender"], form_dictionary["height"], form_dictionary["weight"], form_dictionary["ap_hi"], form_dictionary["ap_lo"], form_dictionary["cholesterol"], form_dictionary["gluc"], form_dictionary["smoke"], form_dictionary["alco"], form_dictionary["active"]]).reshape(1,-1))
        if prediction == [1]:
            positive += 1
    return {"number_of_positive": positive, "number_of_model":number_of_model}

def diagnose_disease(form_dictionary:dict):
    return expert_system.diagnose(form_dictionary)
