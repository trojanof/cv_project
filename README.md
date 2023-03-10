[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/streamlit/demo-face-gan/)

# Streamlit Project: 

This project has collected three solutions for solving various problems using computer vision and machine learning models.

## How to run this demo

The demo requires Python 3.6 or 3.7 (The version of TensorFlow we use is not supported in Python 3.8+). 
**We suggest creating a new virtual environment**, then running:

```
git clone https://github.com/kostiks/elb_cv_project
create new Virtual environment using requirements.txt
run streamlit run main.py
```

## Model CGAN (generating handwritten numbers) 

Using the Conditional Generative Adversarial Network model based on the MNIST dataset, training is performed with the subsequent generation of handwritten digits based on a user request.

## Model AutoEncoder (Denoising Dirty Documents)

Using the Convolutional autoencoder model, analyzing and training the model for the subsequent removal of noise and defects in images containing text.

## Model YOLOv5 (recognition of construction equipment attributes)

Using the YoloV5 computer vision model, training takes place on a labeled sample from the roboflow.com website containing the attributes of construction equipment, and the subsequent detection of objects using the provided photo or real-time video from the construction site

## Questions? Comments?

[Link](Telegram)