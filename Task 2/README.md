Trained models --> https://drive.google.com/drive/folders/1dAmD4Kz_HmMJLq4JLWD3yAJ9HdzeBeQS?usp=sharing

This directory contains a deep learning project that predict the
following outputs:-
a. Color of product
b. Type of product (T-shirt, shoes, etc.)
c. Preferable season to use this product
d. Gender (Men, Women, Unisex) on fashion product images

Task 2 EDA.ipynb - A comprehensive Exploratory Data Analysis (EDA) was conducted to understand the dataset, including

Task 2 Model build.ipynb - The model architecture is designed to predict multiple outputs from a single input image. The model is built using the ResNet50 architecture and predicts:
                            The color of the product
                            The type of product (e.g., T-shirt, shoes)
                            The preferable season for the product
                            The gender category (Men, Women, Unisex)
                            (The models performance is the result of limited epochs as the gpu compute available.)

streamlit_app.py - A Streamlit-based UI for easy interaction with the model. This UI allows users to upload images and see the model's predictions in real-time.

1.png, 2.jpg, 3.jpg, 4.jpg, 5.jpg - Sample screenshots of fashion products were taken from the Amazon website. These samples were used to test the model's prediction capabilities and validate its performance.

Output.png - The model's predictions for sample fashion product images from Amazon

