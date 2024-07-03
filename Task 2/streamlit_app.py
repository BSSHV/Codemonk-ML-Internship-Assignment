import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions, ResNet50V2
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Classes
gender_class = ['Boys', 'Girls', 'Men', 'Unisex', 'Women']
type_class = ['Accessory Gift Set', 'Baby Dolls', 'Backpacks', 'Bangle', 'Basketballs', 'Bath Robe', 
              'Beauty Accessory', 'Belts', 'Blazers', 'Body Lotion', 'Body Wash and Scrub', 'Booties', 
              'Boxers', 'Bra', 'Bracelet', 'Briefs', 'Camisoles', 'Capris', 'Caps', 'Casual Shoes', 
              'Churidar', 'Clothing Set', 'Clutches', 'Compact', 'Concealer', 'Cufflinks', 'Cushion Covers', 
              'Deodorant', 'Dresses', 'Duffel Bag', 'Dupatta', 'Earrings', 'Eye Cream', 'Eyeshadow', 'Face Moisturisers', 
              'Face Scrub and Exfoliator', 'Face Serum and Gel', 'Face Wash and Cleanser', 'Flats', 'Flip Flops', 
              'Footballs', 'Formal Shoes', 'Foundation and Primer', 'Fragrance Gift Set', 'Free Gifts', 'Gloves', 
              'Hair Accessory', 'Hair Colour', 'Handbags', 'Hat', 'Headband', 'Heels', 'Highlighter and Blush', 
              'Innerwear Vests', 'Ipad', 'Jackets', 'Jeans', 'Jeggings', 'Jewellery Set', 'Jumpsuit', 'Kajal and Eyeliner', 
              'Key chain', 'Kurta Sets', 'Kurtas', 'Kurtis', 'Laptop Bag', 'Leggings', 'Lehenga Choli', 'Lip Care', 
              'Lip Gloss', 'Lip Liner', 'Lip Plumper', 'Lipstick', 'Lounge Pants', 'Lounge Shorts', 'Lounge Tshirts', 
              'Makeup Remover', 'Mascara', 'Mask and Peel', 'Mens Grooming Kit', 'Messenger Bag', 'Mobile Pouch', 'Mufflers', 
              'Nail Essentials', 'Nail Polish', 'Necklace and Chains', 'Nehru Jackets', 'Night suits', 'Nightdress', 
              'Patiala', 'Pendant', 'Perfume and Body Mist', 'Rain Jacket', 'Rain Trousers', 'Ring', 'Robe', 'Rompers', 
              'Rucksacks', 'Salwar', 'Salwar and Dupatta', 'Sandals', 'Sarees', 'Scarves', 'Shapewear', 'Shirts', 'Shoe Accessories', 
              'Shoe Laces', 'Shorts', 'Shrug', 'Skirts', 'Socks', 'Sports Sandals', 'Sports Shoes', 'Stockings', 'Stoles', 
              'Sunglasses', 'Sunscreen', 'Suspenders', 'Sweaters', 'Sweatshirts', 'Swimwear', 'Tablet Sleeve', 'Ties', 'Ties and Cufflinks', 
              'Tights', 'Toner', 'Tops', 'Track Pants', 'Tracksuits', 'Travel Accessory', 'Trolley Bag', 'Trousers', 'Trunk', 
              'Tshirts', 'Tunics', 'Umbrellas', 'Waist Pouch', 'Waistcoat', 'Wallets', 'Watches', 'Water Bottle', 'Wristbands']
season_class = ['Fall', 'Spring', 'Summer', 'Unknown', 'Winter']
color_class = ['Beige', 'Black', 'Blue', 'Bronze', 'Brown', 'Burgundy', 'Charcoal', 'Coffee Brown', 'Copper', 'Cream', 'Fluorescent Green', 
               'Gold', 'Green', 'Grey', 'Grey Melange', 'Khaki', 'Lavender', 'Lime Green', 'Magenta', 'Maroon', 'Mauve', 'Metallic', 'Multi',
               'Mushroom Brown', 'Mustard', 'Navy Blue', 'Nude', 'Off White', 'Olive', 'Orange', 'Peach', 'Pink', 'Purple', 'Red', 'Rose', 
               'Rust', 'Sea Green', 'Silver', 'Skin', 'Steel', 'Tan', 'Taupe', 'Teal', 'Turquoise Blue', 'Unknown', 'White', 'Yellow']

# Load the trained prediction models
def load_model_weights(model, weights_path):
    model.load_weights(weights_path)
    return model

def build_model(num_classes, weights_path):
    res = ResNet50V2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    for layer in res.layers:
        layer.trainable = False
    x = Flatten()(res.output)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=res.input, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return load_model_weights(model, weights_path)

gender_model = build_model(5, 'models/model_gender.h5')
type_model = build_model(142, 'models/model.h5')
season_model = build_model(5, 'models/model_season.h5')
colour_model = build_model(47, 'models/model_colour.h5')
#Make predictions
def load_and_prep_image(img, img_shape=224):
    img = img.resize((img_shape, img_shape))
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

def pred_and_plot(model, image, class_names):
    img = load_and_prep_image(image)
    pred = model.predict(np.expand_dims(img, axis=0))
    if len(pred[0]) > 1:
        pred_class = class_names[np.argmax(pred)]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]
    return pred_class

# GUI
st.title("Fashion Attributes Prediction ")
st.write("Upload an image ")

uploaded_image = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Fashion Attributes"):
        gender = pred_and_plot(gender_model, image, gender_class)
        fashion_type = pred_and_plot(type_model, image, type_class)
        season = pred_and_plot(season_model, image, season_class)
        color = pred_and_plot(colour_model, image, color_class)
        st.success(f"Predicted Gender: {gender}")
        st.success(f"Predicted Type: {fashion_type}")
        st.success(f"Predicted Season: {season}")
        st.success(f"Predicted color: {color}")
