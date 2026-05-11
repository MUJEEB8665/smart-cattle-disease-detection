import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Smart Cattle Disease Detection",
    page_icon="🐄",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #111827;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #4b5563;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #d1fae5;
    color: #065f46;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}

.confidence-box {
    padding: 15px;
    border-radius: 12px;
    background-color: #dbeafe;
    color: #1e3a8a;
    font-size: 20px;
    font-weight: bold;
    margin-top: 15px;
}

.precaution-box {
    padding: 25px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    margin-top: 20px;
}

.feature-box {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #

model = tf.keras.models.load_model(
    'model/cattle_disease_model.h5'
)

# ---------------- CLASS NAMES ---------------- #

class_names = [
    'Foot and Mouth Disease',
    'Healthy',
    'Lumpy Skin Disease',
    'Mastitis'
]

# ---------------- RECOMMENDATIONS ---------------- #

recommendations = {

    "Foot and Mouth Disease": """
    <ul>
    <li>Isolate infected cattle immediately</li>
    <li>Consult a veterinarian</li>
    <li>Maintain proper hygiene and sanitation</li>
    <li>Avoid movement of infected cattle</li>
    <li>Disinfect feeding and water areas</li>
    </ul>
    """,

    "Healthy": """
    <ul>
    <li>Cattle appears healthy</li>
    <li>Continue regular health monitoring</li>
    <li>Provide balanced nutrition</li>
    <li>Maintain clean environment</li>
    <li>Schedule periodic veterinary checkups</li>
    </ul>
    """,

    "Lumpy Skin Disease": """
    <ul>
    <li>Provide immediate veterinary care</li>
    <li>Isolate infected cattle</li>
    <li>Use insect and mosquito control</li>
    <li>Maintain clean surroundings</li>
    <li>Monitor skin lesions regularly</li>
    </ul>
    """,

    "Mastitis": """
    <ul>
    <li>Maintain proper milking hygiene</li>
    <li>Clean udder before and after milking</li>
    <li>Consult veterinarian for treatment</li>
    <li>Monitor milk production carefully</li>
    <li>Keep cattle resting area clean</li>
    </ul>
    """
}

# ---------------- DISEASE INFORMATION ---------------- #

disease_info = {

    "Foot and Mouth Disease": {
        "Symptoms": """
        - Fever  
        - Blisters on mouth and feet  
        - Excessive salivation  
        - Difficulty walking  
        """,

        "Causes": """
        Viral infection caused by Foot-and-Mouth Disease Virus (FMDV).
        """,

        "Prevention": """
        - Vaccination  
        - Isolation of infected cattle  
        - Farm sanitation  
        """,

        "Treatment": """
        No specific cure exists. Supportive veterinary care is recommended.
        """
    },

    "Healthy": {
        "Symptoms": """
        - Normal eating habits  
        - Good physical activity  
        - Healthy skin and coat  
        """,

        "Causes": """
        Cattle appears healthy with no visible disease symptoms.
        """,

        "Prevention": """
        - Regular monitoring  
        - Balanced nutrition  
        - Clean environment  
        """,

        "Treatment": """
        No treatment required.
        """
    },

    "Lumpy Skin Disease": {
        "Symptoms": """
        - Skin nodules  
        - Fever  
        - Swollen lymph nodes  
        - Reduced milk production  
        """,

        "Causes": """
        Viral infection spread mainly through insects and mosquitoes.
        """,

        "Prevention": """
        - Insect control  
        - Vaccination  
        - Isolation of infected cattle  
        """,

        "Treatment": """
        Veterinary care and supportive treatment recommended.
        """
    },

    "Mastitis": {
        "Symptoms": """
        - Swollen udder  
        - Abnormal milk  
        - Pain during milking  
        - Reduced milk production  
        """,

        "Causes": """
        Bacterial infection affecting udder tissues.
        """,

        "Prevention": """
        - Proper milking hygiene  
        - Clean cattle environment  
        - Regular udder cleaning  
        """,

        "Treatment": """
        Antibiotics and veterinary supervision recommended.
        """
    }
}

# ---------------- TITLE ---------------- #

st.markdown(
    '<div class="title">🐄 VetVision AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Based Smart Veterinary Disease Detection System</div>',
    unsafe_allow_html=True
)

# ---------------- LAYOUT ---------------- #

col1, col2 = st.columns([2.5, 1])

# ---------------- LEFT COLUMN ---------------- #

with col1:

    uploaded_file = st.file_uploader(
        "Upload Cattle Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Open image
        image = Image.open(uploaded_file)

        # Display image
        st.image(
            image,
            caption="Uploaded Cattle Image",
            width=500
        )

        # Resize image
        img = image.resize((224, 224))

        # Convert image to array
        img_array = np.array(img) / 255.0

        # Convert RGBA to RGB
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        # Expand dimensions
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)

        predicted_class = class_names[np.argmax(prediction)]

        confidence = np.max(prediction) * 100

        # Result box
        st.markdown(
            f'''
            <div class="result-box">
            Predicted Disease: {predicted_class}
            </div>
            ''',
            unsafe_allow_html=True
        )

        # Confidence box
        st.markdown(
            f'''
            <div class="confidence-box">
            Confidence Score: {confidence:.2f}%
            </div>
            ''',
            unsafe_allow_html=True
        )

        # Recommendations
        st.markdown(
            f'''
            <div class="precaution-box">
            <h2>Precautions & Recommendations</h2>
            {recommendations[predicted_class]}
            </div>
            ''',
            unsafe_allow_html=True
        )

        # Disease Information
        info = disease_info[predicted_class]

        st.markdown("## 🩺 Disease Information")

        st.info(f"### Symptoms\n{info['Symptoms']}")

        st.warning(f"### Causes\n{info['Causes']}")

        st.success(f"### Prevention\n{info['Prevention']}")

        st.error(f"### Treatment\n{info['Treatment']}")

# ---------------- RIGHT COLUMN ---------------- #

with col2:

    st.subheader("📌 About Project")

    st.info("""
This project uses Artificial Intelligence, Deep Learning, and Computer Vision techniques to detect cattle diseases from images.
""")

    st.success("✅ Model Accuracy: 95.7%")

    st.subheader("🚀 Technologies Used")

    st.markdown("""
- TensorFlow
- Keras
- MobileNetV2
- Streamlit
- Python
- Deep Learning
""")

    st.subheader("✨ Project Features")

    st.markdown("""
✅ AI Disease Detection  
✅ Deep Learning Model  
✅ Image Classification  
✅ Real-Time Prediction  
✅ Disease Recommendations  
✅ Smart Veterinary Assistance  
""")