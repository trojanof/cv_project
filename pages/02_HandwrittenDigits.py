import streamlit as st
import matplotlib.pyplot as plt
import torch
import torchvision
st.set_page_config(
    page_title='Генерация цифр',
    page_icon= ":phone:"
)
st.sidebar.header("Генерация цифр")
st.title('Генерация')
c1, c2 = st.columns(2) 
c1.write('''
## При помощи **Conditional GAN**      Вы можете сгенерировать цифры, и они будут похожи на рукописные
''')
st.write('''
         ### Вы даже можете увидеть свой номер телефона сгенерированный нейросетью. 
## P. S. Мы не сохраняем личные данные
''') 
c2.image('data/movie2.gif')
import torch
import torch.nn as nn
import streamlit as st #pip install streamlit
import torchvision # pip install torchvision
from torch.autograd import Variable 
from torchvision.utils import make_grid # pip install torchutils
import matplotlib.pyplot as plt
import numpy as np
import os
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)
    
def load_generator():
    generator = Generator()  
    generator.load_state_dict(torch.load('data/CondGan_mnist_2_generator.pth', map_location='cpu'))
    generator.eval()
    return generator
def generate_mob_num(generator, input_number, theme):
  res = [int(x) for x in str(input_number)]
  z = Variable(torch.randn(len(str(input_number)), 100)).cpu()
  labels = torch.LongTensor(res).cpu()
  images = generator(z, labels).unsqueeze(1)
  if theme == 'light':
    images = abs(1 - images)
    grid = make_grid(images, nrow=len(str(input_number)), normalize=True, padding=0).cpu()
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(grid.permute(1, 2, 0).data, cmap='gray')
    ax.axis('off')
  else:
    grid = make_grid(images, nrow=len(str(input_number)), normalize=True, padding=10).cpu()
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(grid.permute(1, 2, 0).data, cmap='gray')
    ax.axis('off')
    ax.set_xmargin(0)
  st.pyplot(plt)
generator = load_generator()
st.write('Пожалуйста введите номер телефона')
option = st.number_input('Введите цифры от 0 до 9',value=11)
check = st.checkbox('Dark output')
col1, col2= st.columns(2)
with col1:
   gen_button = st.button("Generate")
with col2:
   if check:
      theme = 'dark'
   else:
      theme = 'light'
if gen_button:
   generate_mob_num(generator, option, theme)