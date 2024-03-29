import streamlit as st
import pandas as pd
from pickle import load
#import joblib
import pickle
import numpy as np
import math
#import matplotlip
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler
#from config.definitions import ROOT_DIR

#Indication to run interfaz in a localhost
#1 open the terminal cmd and change root direcory to file directory
#2 write this: streamlit run interfaz.py --server.port=9876
PROJECT_ROOT_DIR = "."
OutModel_PATH = os.path.join(PROJECT_ROOT_DIR, "model_output")

#Recovering regression model
model_file = os.path.join(OutModel_PATH, "final_model_PCbeams-041023-actualized.pkl")
with open(model_file, 'rb') as f:
    loaded_model_reg = pickle.load(f)

#Recovering classifcation model
model_file = os.path.join(OutModel_PATH, "final_model_PCbeams_class_041023-actualized.pkl")
with open(model_file, 'rb') as f:
    loaded_model_cla = pickle.load(f)


st.title('PCbeam_Bot: ML model for the shear design of Prestressed concrete beams')
st.subheader('Dimensional Parameters')
st.sidebar.header('User Input Parameters')

PROJECT_ROOT_DIR_Fig = "."
OutModel_PATH2 = os.path.join(PROJECT_ROOT_DIR_Fig, "figures_interfaz")

image = Image.open(os.path.join(OutModel_PATH2,'viga.png'))
st.image(image)


def user_input_features():
    #prueba1=st.number_input('Insert a number',min_value=100.00,max_value=1000.00,value=200.00,step=50.00)
    bw = st.sidebar.slider('bw (mm)', min_value=25, max_value=375, step=10)
    D = st.sidebar.slider('D (mm)', min_value=150, max_value=1600, step=50)
    Ac = st.sidebar.slider('Ac (mm2)', min_value=10000, max_value=718000, step=1000)
    a_deff = st.sidebar.slider('a/Deff', min_value=0.40, max_value=8.00, step=0.01)
    rho_l = st.sidebar.slider('rho_l', min_value=0.000, max_value=0.200, step=0.001)
    rho_lp = st.sidebar.slider('rho_lp', min_value=0.001, max_value=0.060, step=0.001) 
    rho_t = st.sidebar.slider('rho_t', min_value=0.00, max_value=0.05, step=0.01) 
    fc = st.sidebar.slider('fc (MPa)', min_value=10, max_value=120, step=5)
    fy = st.sidebar.slider('fy (MPa)', min_value=205, max_value=900, step=10)
    fyt = st.sidebar.slider('fyt (MPa)', min_value=85, max_value=900, step=10) 
    fpy = st.sidebar.slider('fpy (MPa)', min_value=600, max_value=4400, step=20) 
    fpu = st.sidebar.slider('fpu (MPa)', min_value=900, max_value=5200, step=20) 
    fpo = st.sidebar.slider('fpo (MPa)', min_value=20, max_value=1760, step=20)  

    data = {'bw (mm)': bw,
            'D (mm)': D,
            'Ac (mm2)':Ac,
            'a/Deff': a_deff,
            'rho_l': rho_l,
            'rho_lp': rho_lp,
            'rho_t': rho_t,
            'fc (MPa)': fc,  
            'fy (MPa)': fy,   
            'fyt (MPa)': fyt,
            'fpy (MPa)': fpy,
            'fpu (MPa)': fpu, 
            'fpo (MPa)': fpo}                      
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

bw1=df['bw (mm)'].values.item()
D1=df['D (mm)'].values.item()
Ac1=df['Ac (mm2)'].values.item()
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

bw_D=bw1*D1
bw_Ac=bw1/Ac1
fc2=fc1
fpo_fpu=fpo1/fpu1
a_Deff=a_deff1
rhol_fy_fc = rho_l1*fy1/fc1
rholp_fpu_fc=rho_lp1*fpu1/fc1
rhot_fyt_fc=rho_t1*fyt1/fc1
Ap=rho_lp1*bw1*D1
Fpo=Ap*fpo1
eta_p=Fpo/(bw1*D1*fc1)
sq_fc=np.sqrt(fc1)
lamb=(rho_lp1*fpy1)/(rho_lp1*fpy1+rho_l1*fy1)

user_input={'bw (mm)': "{:.0f}".format(bw1),
            'D (mm)': "{:.0f}".format(D1),
            'Ac (mm2)':"{:.0f}".format(Ac1),
            'a/Deff': "{:.2f}".format(a_deff1),
            'rho_l': "{:.2f}".format(rho_l1),
            'rho_lp': "{:.2f}".format(rho_lp1),
            'rho_t': "{:.2f}".format(rho_t1),
            'fc (MPa)': "{:.0f}".format(fc1),
            'fy (MPa)': "{:.0f}".format(fy1),
            'fyt (MPa)': "{:.0f}".format(fyt1),
            'fpy (MPa)': "{:.0f}".format(fpy1),
            'fpu (MPa)': "{:.0f}".format(fpu1),
            'fpo (MPa)': "{:.0f}".format(fpo1)}

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
                  'rhot_fyt/fc': "{:.2f}".format(rhot_fyt_fc),
                  'rhol_fy/fc': "{:.2f}".format(rhol_fy_fc),
                  'rholp_fpu/fc': "{:.2f}".format(rholp_fpu_fc),
                  'eta_p': "{:.2f}".format(eta_p)}
calculated_param_df=pd.DataFrame(calculated_param, index=[0])
st.subheader('Model Input Parameters for Shear Strength')
st.table(calculated_param_df)
#
#Parameters for classification
calculated_param_cla={'bw_D (mm2)': "{:.2f}".format(bw_D),
                  '\u221Afc (\u221AMPa)': "{:.2f}".format(sq_fc),
                  'fpo/fpu': "{:.2f}".format(fpo_fpu),
                  'a/Deff': "{:.2f}".format(a_deff1),
                  'eta_p': "{:.2f}".format(eta_p),
                  'lambda': "{:.2f}".format(lamb),
                  'rhot_fyt/fc': "{:.2f}".format(rhot_fyt_fc),
                  'bw/Ac x 10\u00AF\u00B3 (1/mm)': "{:.2f}".format(bw_Ac*1000)}
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

var_names_reg = ['bw_D', 'fc', 'fpo/fpu', "a/Deff", 'rhot_fyt/fc', 'rhol_fy/fc', 'rholp_fpu/fc','eta_p']
var_names_cla = ['bw_D', 'sqrt_fc', 'fpo/fpu', "a/Deff",'eta_p', 'lambda', 'rhot_fyt/fc','bw_Ac']
#Definning input to model predictions
reg=np.array([[bw_D_log,fc2_log,fpo_fpu,a_deff1_log,rhot_fyt_fc,rhol_fy_fc,rholp_fpu_fc_log,eta_p_log]])
cla=np.array([[bw_D,sq_fc,fpo_fpu,a_deff1,eta_p,lamb,rhot_fyt_fc,bw_Ac]])
# Escalando los inputs (forma correcta para los inputs)
s_reg = np.load('std_scale_reg_041023-actualized.npy')
m_reg = np.load('mean_scale_reg_041023-actualized.npy')

s_cla = np.load('std_scale_cla_041023-actualized.npy')
m_cla = np.load('mean_scale_cla_041023-actualized.npy')

reg_sca=pd.DataFrame((reg-m_reg)/s_reg,index=[0])
cla_sca=pd.DataFrame((cla-m_cla)/s_cla,index=[0])

#Factor reduction phi
phi=0.3/(1+np.exp(-150*(rholp_fpu_fc-0.18)))+0.6

##Regression
Load_pred_reg=loaded_model_reg.predict(reg_sca.values).item()
V_test=np.exp(Load_pred_reg)*np.exp(reg[0,1])*np.exp(reg[0,0])/1000

##Classification  
Load_pred_cla=loaded_model_cla.predict(cla_sca.values).item()
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
st.write('$\\phi$ = Resistance Factor Design')  
w_cr_results={'Shear Strength (kN)':"{:.2f}".format(V_test),
               '𝜙Shear Strength (kN)':"{:.2f}".format(V_test*phi),
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

st.subheader('Nomenclature')
st.write('*fpo/fpu* is the ultimate effective prestress ratio; *rhot_fyt/fc* is the transverse reinforcement index; *rhol_fy/fc* is the passive longitudinal reinforcement index; \
   *rholp_fpu/fc* is the prestress index; *eta_p* is the total axial force ratio; *lambda* is the reinforcement moment ratio; *bw* is the web width; *D* is the total depth; *Ac* is the section area; \
   *a/Deff* is the shear span-to-depth ratio; *rho_l* is the longitudinal reinforcement ratio; *rho_lp* is the prestressing reinforcement ratio; *rho_t* is the transverse reinforcement ratio; \
   *fc* is the concrete strength, *fy* is the yield strength of longitudinal reinforcement; *fyt* is the yield strength of transverse reinforcement; *fpy* is the yield strength of prestressing tendons; \
   *fpu* is the tensile strength of the tendons; *fpo* is the actual prestress after all losses.')

st.subheader('Reference')
st.write('When usingn this app in your own work, please use the following citation:')
st.write('Bedriñana, L. A., J. Sucasaca, J. Tovar, and H. Burton. 2023. “Design-Oriented Machine-Learning Models for Predicting the Shear Strength of Prestressed Concrete Beams.” J. Bridg. Eng., 28 (4): 04023009. https://doi.org/10.1061/JBENF2.BEENG-6013.')
st.write('Universidad de Ingenieria y Tecnologia - UTEC')
st.write('Department of Civil Engeneering')