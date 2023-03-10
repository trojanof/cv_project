import streamlit as st
import matplotlib.pyplot as plt
import torch
import torchvision
st.set_page_config(
    page_title='Проект по Computer Vision',
    page_icon= ":computer:",
    layout='wide'
)
st.sidebar.header("Home page")
c1, c2 = st.columns(2)        
c2.image('data/og-computer-vision-facebook.jpg')
c1.markdown("""
# Проект по Computer Vision
Cостоит из 3 частей:
## 1.Генерация заданной цифры с помощью **Conditional GAN**
## 2.Детекция объектов с помощью **YOLO v5**
## 3.Очищение документов от шумов с помощью **Автоэнкодера**
""")
