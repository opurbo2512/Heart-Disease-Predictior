import streamlit as st
import torch
from torch import nn

st.set_page_config(page_title="Heart disease prediction",page_icon="🫀")

st.title("Heart disease predictor🫀")

with st.expander("About this app"):
    st.markdown("**What can this app do?**")
    st.info("This app use a deep learning model.Using this model and based on some data it will predict that have you heart disease or not.")

with st.container():
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.subheader("Input requeirment for prediction")

with st.container():
    c4,c5 = st.columns(2)

    with st.form(key="a"):
        with c4:
            age = st.number_input("Enter your age:",0,120)
            sex = st.radio("Sex",["Male","Female"])
            cpain = st.slider("Chest pain type",1,4)
            resting_bp = st.number_input("Resting Bloos Pressure:",0,200)
            chol = st.slider("cholesterol",0,1000)
            f_sugar = st.radio("fasting blood sugar?",["Yes","No"])

        with c5:
            r_ecg = st.number_input("resting ecg",0,3)
            h_rate = st.slider("max heart rate",50,220)
            e_angeina = st.radio("exercise angina",["Yes","No"])
            oldpeak = st.number_input("oldpeak",-10,10)
            ST = st.slider("ST slope",1,4)
    
    if sex == "Male":
        sex = 1
    else:
        sex = 0

    if f_sugar == "Yes":
        f_sugar = 1
    else:
        f_sugar = 0

    if e_angeina == "Yes":
        e_angeina = 1
    else:
        e_angeina = 0

    input_list = [age,sex,cpain,resting_bp,chol,f_sugar,r_ecg,h_rate,e_angeina,oldpeak,ST]
    tensor = torch.tensor(input_list).type(torch.float)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(11,16),
                nn.ReLU(),
                nn.Linear(16,32),
                nn.ReLU(),
                nn.Linear(32,16),
                nn.ReLU(),
                nn.Linear(16,1)
                )

        def forward(self,x):
            return self.model(x)


    model = Model()
    model.load_state_dict(torch.load("model/model.pth"))

    model.eval()
    with torch.inference_mode():
        logs = model(tensor).squeeze()
        pred = torch.round(torch.sigmoid(logs)).item()

    if st.button("Predict"):
        
        if pred == 0:
            st.success("You have not any heart disease.")
        else:
            st.warning("Ohhhh hohhh, you have heart desease.")


        
        
        
    
    

