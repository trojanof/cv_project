import streamlit as st
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
import numpy as np
import torch.nn as nn

import os

#for key in st.session_state.keys():
#    del st.session_state[key]

st.set_page_config(
    page_title='Детекция объектов',
    page_icon= ":ladder:",
    layout='wide'
)
st.sidebar.header("Detection")
c1, c2 = st.columns(2)        
c2.image('data/27490275.jpg')
c1.markdown("""
## Детекция объектов на строительной площадке
Мы использовали [датасет 'Средства индивидуальной защиты'](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model) на сайте [Roboflow](https://roboflow.com/)
Для детекции объектов мы использовали YOLO v5
Подробнее об этой нейросети можно узнать [здесь](https://habr.com/ru/post/576738/)
## Благодаря нашему сервису Вы можете отследить нарушения техники безопасности на стройплощадке, ведь наша модель фиксирует наличие и отсутствие:
* ### Защитных очков
* ### Перчаток
* ### Каски
* ### Маски
* ### Защитного жилета
Также фиксирует 
* ### Разметочные  конусы 
* ### Людей
#### Чтобы загрузить изображение воспользуйтесь боковой панелью слева 
""")
            
image_file = st.file_uploader(
    "Можно перетащить (драг-н-дроп)", 
    type=["jpg", "jpeg", "png"], key='helmet'
    )


new_names = {0: 'Падение',
 1: 'Перчатки',
 2: 'Очки',
 3: 'Каска',
 4: 'Лестница',
 5: 'Маска',
 6: 'НЕТ перчаток',
 7: 'НЕТ очков',
 8: 'НЕТ каски',
 9: 'НЕТ маски',
 10: 'НЕТ защитного жилета',
 11: 'Человек',
 12: 'Разметочный конус',
 13: 'Защитный жилет'}


if image_file is not None:
    img = Image.open(image_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption='Uploaded Image', use_column_width='always')
        
        imgpath = image_file.name
        
        outputpath = os.path.basename(imgpath)
        
        with open(imgpath, mode="wb") as f:
            f.write(image_file.getbuffer())
        #call Model prediction--
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='data/best170.pt', force_reload=True, device='cpu') 
        model.names = new_names
        model.conf = 0.4
        
        model.cpu()
        pred = model(imgpath)
        pred.render()  # render bbox in image
        for im in pred.ims:
            im_base64 = Image.fromarray(im)
            im_base64.save(outputpath)
            #--Display predicton
            
        img_ = Image.open(outputpath)
        with col2:
            st.image(img_, caption='Model Prediction(s)', use_column_width='always')