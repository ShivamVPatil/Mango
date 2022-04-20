import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model

model = load_model('./Model/BC.h5',compile=False)
lab = {0: 'HealthyBadami', 1: 'HealthyTotapuri', 2: 'PartiallyRipeBadami', 3: 'PartiallyRipeTotapuri', 4: 'RawBadami', 5: 'RawTotapuri', 6: 'RottenBadami', 7: 'RottenNeelam', 8: 'RottenTotapuri', 9: 'UnripeBadami', 10: 'UnripeTotapuri'}

def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(res)
    return res

def run():
    img1 = Image.open('./meta/logo1.png')
    img1 = img1.resize((350,350))
    st.image(img1,use_column_width=False)
    st.title("Mango Varient Detection")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based "270 Bird Species also see 70 Sports Dataset"</h4>''',
                unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image of Bird", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = processed_img(save_image_path)
            st.success("Predicted Bird is: "+result)
run()
