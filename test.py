# =====================================================
# PLANTVILLAGE DATASET - TÜRKÇE SINIF ADLARI
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------------------------------
# YOLLAR
# -----------------------------------------------------
MODEL_PATH = r"D:\plant_resnet50_v2.h5"
IMAGE_PATH = r"D:\elmacuruk.jpg"
TRAIN_DIR  = r"D:\small_dataset"   # bu klasörün içi ekran görüntüsündeki gibi
IMG_SIZE = 224

# -----------------------------------------------------
# MODELİ YÜKLE
# -----------------------------------------------------
model = load_model(MODEL_PATH)
print("✔ Model yüklendi")

# -----------------------------------------------------
# SINIF ADLARINI OTOMATİK AL (KLASÖRLERDEN)
# -----------------------------------------------------
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

tmp_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# index -> İngilizce sınıf adı
class_names = {v: k for k, v in tmp_gen.class_indices.items()}

# -----------------------------------------------------
# İNGİLİZCE → TÜRKÇE SINIF ADLARI
# -----------------------------------------------------
turkce = {
    "Apple___Apple_scab": "Elma – Elma Karalekesi",
    "Apple___Black_rot": "Elma – Siyah Çürüklük",
    "Apple___Cedar_apple_rust": "Elma – Sedir Elma Pası",
    "Apple___healthy": "Elma – Sağlıklı",

    "Blueberry___healthy": "Yaban Mersini – Sağlıklı",

    "Cherry_(including_sour)___healthy": "Kiraz – Sağlıklı",
    "Cherry_(including_sour)___Powdery_mildew": "Kiraz – Külleme",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Mısır – Yaprak Lekesi (Cercospora)",
    "Corn_(maize)___Common_rust": "Mısır – Yaygın Pas",
    "Corn_(maize)___healthy": "Mısır – Sağlıklı",
    "Corn_(maize)___Northern_Leaf_Blight": "Mısır – Kuzey Yaprak Yanıklığı",

    "Grape___Black_rot": "Üzüm – Siyah Çürüklük",
    "Grape___Esca_(Black_Measles)": "Üzüm – Esca (Siyah Benek)",
    "Grape___healthy": "Üzüm – Sağlıklı",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Üzüm – Yaprak Yanıklığı",

    "Orange___Haunglongbing_(Citrus_greening)": "Portakal – Yeşillenme Hastalığı",

    "Peach___Bacterial_spot": "Şeftali – Bakteriyel Leke",
    "Peach___healthy": "Şeftali – Sağlıklı",

    "Pepper,_bell___Bacterial_spot": "Biber – Bakteriyel Leke",
    "Pepper,_bell___healthy": "Biber – Sağlıklı",

    "Potato___Early_blight": "Patates – Erken Yanıklık",
    "Potato___Late_blight": "Patates – Geç Yanıklık",
    "Potato___healthy": "Patates – Sağlıklı",

    "Raspberry___healthy": "Ahududu – Sağlıklı",

    "Soybean___healthy": "Soya – Sağlıklı",

    "Squash___Powdery_mildew": "Kabak – Külleme",

    "Strawberry___healthy": "Çilek – Sağlıklı",
    "Strawberry___Leaf_scorch": "Çilek – Yaprak Yanıklığı",

    "Tomato___Bacterial_spot": "Domates – Bakteriyel Leke",
    "Tomato___Early_blight": "Domates – Erken Yanıklık",
    "Tomato___Late_blight": "Domates – Geç Yanıklık",
    "Tomato___Leaf_Mold": "Domates – Yaprak Küfü",
    "Tomato___Septoria_leaf_spot": "Domates – Septorya Yaprak Lekesi",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Domates – Kırmızı Örümcek",
    "Tomato___Target_Spot": "Domates – Hedef Leke",
    "Tomato___Tomato_mosaic_virus": "Domates – Mozaik Virüsü",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Domates – Sarı Yaprak Kıvırcıklık Virüsü",
    "Tomato___healthy": "Domates – Sağlıklı"
}

# -----------------------------------------------------
# GÖRSELİ YÜKLE
# -----------------------------------------------------
img = image.load_img(IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# -----------------------------------------------------
# TAHMİN
# -----------------------------------------------------
pred = model.predict(img_array)

idx = int(np.argmax(pred))
pred_en = class_names[idx]
pred_tr = turkce.get(pred_en, pred_en)
confidence = float(np.max(pred))

# -----------------------------------------------------
# SONUÇ
# -----------------------------------------------------
print("📌 Tahmin:", pred_tr)
print("📊 Güven:", round(confidence * 100, 2), "%")

plt.imshow(img)
plt.axis("off")
plt.title(f"{pred_tr}")
plt.show()
