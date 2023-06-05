import streamlit as sl
import pickle
import numpy as np
import pandas as pd
import requests as re
from streamlit_lottie import st_lottie
import json 
import warnings
warnings.filterwarnings("ignore")



# import the model
RFG = pickle.load(open('D:\SEM 6\DE\RFG_LRS.pkl','rb'))
df = pickle.load(open('D:\SEM 6\DE\df_LRS.pkl','rb'))

def lottieurl_load(url: str):
    r= re.get(url)
    if r.status_code !=200:
        return None
    return r.json()
lottie_img = lottieurl_load("https://assets2.lottiefiles.com/packages/lf20_nApXIX.json")
col1, col2, col3 = sl.columns(3)
with col1:
    sl.write(' ')

with col2:
    st_lottie(lottie_img,speed=1,reverse=False,loop=True,quality="medium",height=200,width=200,key=None)

with col3:
    sl.write(' ')
sl.title("Laptop Recommender System")

# brand
company = sl.selectbox('Brand',df['Company'].unique())

# type of laptop
type = sl.selectbox('Type',df['TypeName'].unique())

# Ram
ram = sl.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = sl.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = sl.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = sl.selectbox('IPS',['No','Yes'])

# screen size
screen_size = sl.number_input('Screen Size (Ex: 15.6, 17.2')

# resolution
resolution = sl.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = sl.selectbox('CPU',df['Cpu brand'].unique())

hdd = sl.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = sl.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = sl.selectbox('GPU',df['Gpu brand'].unique())

os = sl.selectbox('OS',df['os'].unique())


if sl.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
       
        query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
      
    query = query.reshape(1,12)
    sl.title("Price of the laptop according to selected configuration comes out to be: " + str(int(np.exp(RFG.predict(query)[0]))))


predicted_price=float(np.exp(RFG.predict(query)[0]))

start_price = 9270.00
end_price = 324954.72
difference = 10000

price_ranges = []
current_price = start_price

while current_price <= end_price:
    price_ranges.append((current_price, current_price + difference))
    current_price += difference



sl.title("Recommended laptops are from:")
df2 = df[['Company', 'Price']].copy()

matched_laptops = set()
#matched_laptops = []
#prices =set()
for index, row in df2.iterrows():
    for price_range in price_ranges:
            if price_range[0] <= predicted_price < price_range[1] and price_range[0] <= row['Price'] < price_range[1]:
                matched_laptops.add(row['Company'])
                #prices.add(row["Price"])
         

# Print the matched laptop names
for laptop_name in matched_laptops:
    sl.write(f"Brand: {laptop_name}")


sl.title("Reach us: ")

# LinkedIn profile link
linkedin_link1="https://www.linkedin.com/in/vishv-patel-b03b13229"
linkedin_link2="https://www.linkedin.com/in/dev-patel-45b96521b"
linkedin_link3="https://www.linkedin.com/in/kirtan-tank-7a958921b"
linkedin_link4="https://www.linkedin.com/in/anup-tailor-58b121220"




sl.write(f'Contact 1: {linkedin_link1}')
sl.write(f'Contact 2: {linkedin_link2}')
sl.write(f'Contact 3: {linkedin_link3}')
sl.write(f'Contact 4: {linkedin_link4}')
