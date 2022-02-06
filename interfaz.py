import streamlit as st
import pandas as pd
from pickle import load
import joblib
import pickle
import numpy as np
import math
from PIL import Image
import os
#from config.definitions import ROOT_DIR

#Indication to run interfaz in a localhost
#1 open the terminal cmd and change root direcory to file directory
#2 write this: streamlit run interfaz_streamlit.py --server.port=9876


PROJECT_ROOT_DIR = "."
OutModel_PATH = os.path.join(PROJECT_ROOT_DIR, "model_output")

#Recovering regression model
model_file = os.path.join(OutModel_PATH, "final_model_PCbeams.pkl")
with open(model_file, 'rb') as f:
    loaded_model_reg = pickle.load(f)

#Recovering classifcation model
model_file = os.path.join(OutModel_PATH, "final_model_PCbeams_class.pkl")
with open(model_file, 'rb') as f:
    loaded_model_cla = pickle.load(f)


st.title('Shear strength and Failure mode of Prestressed Concrete Beams Predicted by ML Methods')
st.subheader('Dimensional Parameters')
st.sidebar.header('User Input Parameters')

PROJECT_ROOT_DIR_Fig = "."
OutModel_PATH2 = os.path.join(PROJECT_ROOT_DIR_Fig, "figures_interfaz")

image = Image.open(os.path.join(OutModel_PATH2,'viga.png'))
st.image(image)


def user_input_features():
    bw = st.sidebar.slider('bw (mm)', min_value=25, max_value=375, step=10)
    D = st.sidebar.slider('D (mm)', min_value=150, max_value=1600, step=50)
    a_deff = st.sidebar.slider('a/Deff', min_value=0.0, max_value=8.0, step=0.5)
    rho_l = st.sidebar.slider('rho_l', min_value=0.0, max_value=1.0, step=0.1)
    rho_lp = st.sidebar.slider('rho_lp', min_value=0.01, max_value=0.1, step=0.01) 
    rho_t = st.sidebar.slider('rho_t', min_value=0.0, max_value=0.1, step=0.01) 
    fc = st.sidebar.slider('fc (MPa)', min_value=10, max_value=120, step=10)
    fy = st.sidebar.slider('fy (MPa)', min_value=0, max_value=900, step=50)
    fyt = st.sidebar.slider('fyt (MPa)', min_value=0, max_value=900, step=50) 
    fpy = st.sidebar.slider('fpy (MPa)', min_value=600, max_value=4400, step=20) 
    fpu = st.sidebar.slider('fpu (MPa)', min_value=900, max_value=5200, step=50) 
    fpo = st.sidebar.slider('fpo (MPa)', min_value=20, max_value=1760, step=20) 
    Fpo = st.sidebar.slider('Fpo (N)', min_value=3700, max_value=10140000, step=3000) 

    data = {'bw (mm)': bw,
            'D (mm)': D,
            'a/Deff': a_deff,
            'rho_l': rho_l,
            'rho_lp': rho_lp,
            'rho_t': rho_t,
            'fc (MPa)': fc,  
            'fy (MPa)': fy,   
            'fyt (MPa)': fyt,
            'fpy (MPa)': fpy,
            'fpu (MPa)': fpu, 
            'fpo (MPa)': fpo,
            'Fpo (N)': Fpo}                       
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

bw1=df['bw (mm)'].values.item()
D1=df['D (mm)'].values.item()
a_deff1=df['a/Deff'].values.item()
rho_l1=df['rho_l'].values.item()
rho_lp1=df['rho_lp'].values.item()
rho_t1=df['rho_t'].values.item()
fc1=df['fc (MPa)'].values.item()
fy1=df['fy (MPa)'].values.item()
fyt1=df['fyt (MPa)'].values.item()
fpy1=df['fpy (MPa)'].values.item()
fpu1=df['fpu (MPa)'].values.item()
fpo1=df['fpo (MPa)'].values.item()
Fpo1=df['Fpo (N)'].values.item()

bw_D=bw1*D1
fc2=fc1
fpo_fpu=fpo1/fpu1
a_Deff=a_deff1
rhol_fy_fc = rho_l1*fy1/fc1
rholp_fpu_fc=rho_lp1*fpu1/fc1
rhot_fyt_fc=rho_t1*fyt1/fc1
eta_p=Fpo1/(bw1*D1*fc1)
sq_fc=np.sqrt(fc1)
lamb=(rho_lp1*fpy1)/(rho_lp1*fpy1+rho_l1*fy1)

user_input={'bw (mm)': "{:.0f}".format(bw1),
            'D (mm)': "{:.0f}".format(D1),
            'a/Deff': "{:.0f}".format(a_deff1),
            'rho_l': "{:.2f}".format(rho_l1),
            'rho_lp': "{:.2f}".format(rho_lp1),
            'rho_t': "{:.2f}".format(rho_t1),
            'fc (MPa)': "{:.0f}".format(fc1),
            'fy (MPa)': "{:.0f}".format(fy1),
            'fyt (MPa)': "{:.0f}".format(fyt1),
            'fpy (MPa)': "{:.0f}".format(fpy1),
            'fpu (MPa)': "{:.0f}".format(fpu1),
            'fpo (MPa)': "{:.0f}".format(fpo1),
			'Fpo (N)': "{:.0f}".format(Fpo1)}

user_input_df=pd.DataFrame(user_input, index=[0])
st.subheader('User Input Parameters')
#st.dataframe(user_input_df, 900, 1500)
st.table(user_input_df)
#st.write(user_input_df)
#
#Parameters for regression
calculated_param={'bw_D (mm2)': "{:.2f}".format(bw_D),
                  'fc (MPa)': "{:.2f}".format(fc2),
                  'fpo/fpu': "{:.2f}".format(fpo_fpu),
                  'a/Deff': "{:.2f}".format(a_deff1),
                  'rhot_fyt_fc': "{:.2f}".format(rhot_fyt_fc),
                  'rhol_fy_fc': "{:.2f}".format(rhol_fy_fc),
                  'rholp_fpu_fc': "{:.2f}".format(rholp_fpu_fc),
                  'eta_p': "{:.2f}".format(eta_p)}
calculated_param_df=pd.DataFrame(calculated_param, index=[0])
st.subheader('Model Input Parameters for Shear Strength')
st.table(calculated_param_df)
#
#Parameters for regression
calculated_param_cla={'bw_D (mm2)': "{:.2f}".format(bw_D),
                  'sq_fc (MPa)': "{:.2f}".format(sq_fc),
                  'fpo/fpu': "{:.2f}".format(fpo_fpu),
                  'a/Deff': "{:.2f}".format(a_deff1),
                  'eta_p': "{:.2f}".format(eta_p),
                  'lambda': "{:.2f}".format(lamb),
                  'rhot_fyt_fc': "{:.2f}".format(rhot_fyt_fc)}
calculated_param_df_cla=pd.DataFrame(calculated_param_cla, index=[0])
st.subheader('Model Input Parameters for Failure Mode')
st.table(calculated_param_df_cla)
#
#Transform some input to log space
bw_D_log=np.log(bw_D)
fc2_log=np.log(fc2)
a_deff1_log=np.log(a_deff1)
rholp_fpu_fc_log=np.log(rholp_fpu_fc)
eta_p_log=np.log(eta_p)

#Definning input to model predictions
reg=np.array([[bw_D_log,fc2_log,fpo_fpu,a_deff1_log,rhot_fyt_fc,rhol_fy_fc,rholp_fpu_fc_log,eta_p_log]])

cla=np.array([[bw_D,sq_fc,fpo_fpu,a_deff1,eta_p,lamb,rhot_fyt_fc]])

#Predictions
##Regression
Load_pred_reg=loaded_model_reg.predict(reg).item()
V_test=np.exp(Load_pred_reg)*np.exp(reg[0,1])*np.exp(reg[0,0])/1000

##Classification
Load_pred_cla=loaded_model_cla.predict(cla).item()
resultado=Load_pred_cla
res=str()
if resultado==0:
   res="Diagonal tension failure"
if resultado==1:
   res="Flexure-shear failure"
if resultado==2:
   res="Horizontal shear failure"
if resultado==3:
   res="Shear-compression failure"
if resultado==4:
   res="Stirrup rupture"
if resultado==5:
   res="Web crushing failure"
if resultado==6:
   res="Web-shear failure"

st.subheader('XGBoost Model Predictions')
w_cr_results={'Shear Strength (kN)':"{:.2f}".format(V_test),
               'Failure Mode':format(res)}
w_cr_results_df=pd.DataFrame(w_cr_results, index=[0])
st.table(w_cr_results_df)

image_flex = Image.open(os.path.join(OutModel_PATH2,'flexural_shear.png'))
image_web = Image.open(os.path.join(OutModel_PATH2,'web_shear.png'))
image_diag = Image.open(os.path.join(OutModel_PATH2,'diagonal_tension_failure.png'))
image_hor = Image.open(os.path.join(OutModel_PATH2,'horizontal_shear_failure.png'))
image_com = Image.open(os.path.join(OutModel_PATH2,'shear_compression_failure.png'))
image_sti = Image.open(os.path.join(OutModel_PATH2,'strirrup_rupture.png'))
image_cru = Image.open(os.path.join(OutModel_PATH2,'web_crushing_failure.png'))

if resultado==0:
   st.image(image_diag)
if resultado==1:
   st.image(image_flex)
if resultado==2:
   st.image(image_hor)
if resultado==3:
   st.image(image_com)
if resultado==4:
   st.image(image_sti)
if resultado==5:
   st.image(image_cru)
if resultado==6:
   st.image(image_web)