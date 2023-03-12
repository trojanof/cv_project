import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

for key in st.session_state.keys():
    del st.session_state[key]

st.set_page_config(
    page_title='Очистка текста',
    page_icon= ":scroll:",
    layout='wide'
)
st.sidebar.header("Clear paper")
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder 
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2),
            nn.BatchNorm2d(64),
            nn.SELU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.SELU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.SELU()
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=2),
            nn.BatchNorm2d(8),
            nn.SELU()
            )
        
        self.pool = nn.MaxPool2d(2, 2, return_indices=True) 
        
        self.unpool = nn.MaxUnpool2d(2, 2)
        
        self.conv1_t = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.SELU()
            )
        self.conv2_t = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.SELU()
            )
        self.conv3_t = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=2),
            nn.BatchNorm2d(64),
            nn.SELU()
            )
        self.conv4_t = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=2),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
            )     
    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x, indicies = self.pool(x) # ⟸ bottleneck
        return x, indicies
    def decode(self, x, indicies):
        x = self.unpool(x, indicies)
        x = self.conv1_t(x)
        x = self.conv2_t(x)
        x = self.conv3_t(x)
        x = self.conv4_t(x)
        return x
    def forward(self, x):
        latent, indicies = self.encode(x)
        out = self.decode(latent, indicies)      
        return out

@st.cache_data
def load_model():
    model = ConvAutoencoder()
    model.load_state_dict(torch.load('data/weights200epoch.pt'))    
    return model

model = load_model()

model.eval()
c1, c2 = st.columns(2) 
c2.image('data/denoise.gif')
c1.write('''
# Очистка картинок с текстом

## Пролили чай на документы?
## Остался след от чашки кофе?
## Ребенок смял газету?
# Наш сервис решит Вашу проблему!
''')
st.write('''         
# Просто загрузите изображение документа при помощи боковой панели, которая находится слева, и наш автоэнкодер очистит изображение!
''')
         
         
st.sidebar.header('Загрузите картинку с текстом')
text_file = st.sidebar.file_uploader(
    "Можно перетащить (драг-н-дроп)", 
    type=["jpg", "jpeg", "png"]
    )

if text_file is not None:
    
    # подготовка картинок для вывода
    img_arr = np.array(Image.open(text_file).convert('L')) # L - переводим в ч/б
    img_tensor = torch.Tensor(img_arr).unsqueeze(0)
    img_tensor = (img_tensor.float()/255).unsqueeze(0)
    res_img = model(img_tensor)
    
    # вывод в две колонки
    col1, col2 = st.columns(2)
    with col1:
        st.write('''
            Исходная картинка:
            ''')
        fig, ax = plt.subplots(1,1)
        ax.imshow(img_arr, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)
    with col2:
        st.write('''
            Результат очистки:
            ''')
        fig, ax = plt.subplots(1,1)
        ax.imshow(torch.permute(res_img.squeeze(0).detach(), (1, 2, 0)), cmap='gray')
        ax.axis('off')
        st.pyplot(fig)