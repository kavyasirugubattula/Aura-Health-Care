import base64
import re
import mysql
import mysql.connector
import requests
import streamlit as st
from PIL import Image
from openai import OpenAI
from tensorflow.keras.models import load_model
import os
import cv2
import pandas as pd
import folium
import pyttsx3
from streamlit_folium import folium_static
from geopy.distance import geodesic
import numpy as np
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

client = OpenAI(
    base_url="https://huggingface.co/api/inference-proxy/together",
    api_key="....................................................."
)

def openai_generate_text(user_input):
    try:
        messages = [
            {
                "role": "user",
                "content": user_input
            }
        ]
        completion = client.chat.completions.create(
            model="mistralai/Mistral-Small-24B-Instruct-2501",
            messages=messages,
            max_tokens=500
        )
        # Access the generated response text
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"


# Function to handle navigation
def navigate(page):
    st.session_state["current_page"] = page

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

# Navbar options
nav_options = {
    "Home": "🏠 Home",
    "Skin Disease": "🩺 Skin Disease Detector",
    "Professional Doctor": "👮‍♂ Professional Doctor",
    "Nearby Services": "🟠 Nearby Services",
    "Query": "❓ Query"
}

# Sidebar Styling
sidebar_style = """
<style>
[data-testid="stSidebar"] {
    background-color: #FFCCBC;
    color: black;
}
[data-testid="stSidebar"] button {
    font-size: 16px;
    color: rgb(237, 41, 130);
    font-weight: bold;
}
</style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)

# Sidebar for navigation
sidebar_logo_path = "C:/Users/kavya/Downloads/Aura-Healthcare-logo-bottom-removebg-preview (1).png"
try:
    st.sidebar.image(sidebar_logo_path, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("Logo image not found.")

for key, value in nav_options.items():
    if st.sidebar.button(value):
        navigate(key)

# Disease-Doctor mapping dictionary
disease_doctor_mapping = {
    "fever": {
        "doctor_name": "Dr. A Sharma",
        "specialization": "General Physician",
        "email": "a.sharma@example.com",
        "contact": "+1234567890"
    },
    "toothache": {
        "doctor_name": "Dr. R Malhotra",
        "specialization": "Dentist",
        "email": "r.malhotra@example.com",
        "contact": "+1987654312"
    },

}

def home_page():
    # Display the welcome section at the top
    col1, col2 = st.columns([1, 4])  # Adjust the column ratio as needed
    with col1:
        st.image("C:/Users/kavya/Downloads/robot.png", width=150)  # Adjust the width as necessary
    with col2:
        st.title("Welcome to Aura Health Care")

    # Text Generation Section
    with st.container():
        st.markdown("---")
        st.markdown("### Ask Aura")
        user_input = st.text_area("Enter your question or symptoms here", height=100)

        if st.button("Generate Response"):
            try:
                # Generate the response
                response = openai_generate_text(user_input)

                # Display the response in a styled container
                st.markdown("### Response:")
                st.markdown(
                    f"""
                    <div style="
                        background-color:rgb(59, 58, 58);
                        padding: 15px;
                        border-radius: 10px;
                        border: 1px solid #ddd;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                        font-family: Arial, sans-serif;
                        font-size: 16px;
                        line-height: 1.5;
                    ">
                        {response}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Error generating response: {e}")

    # About Aura Health Care Section
    with st.container():
        st.markdown("---")
        st.markdown("### About Aura Health Care")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(
                "https://miro.medium.com/v2/resize:fit:700/1*GAzVE-9hV0Ps8zYAbXbc2A.jpeg",
                caption="Your Health, Our Priority",
                use_container_width=True
            )

        with col2:
            st.markdown(
                """
                Welcome to Aura Health Care, your trusted digital companion for all things health-related.  
                - Our platform is designed to empower individuals with accurate health insights and provide easy access to healthcare services.  
                - From detecting skin diseases using AI to connecting with professional doctors and locating nearby medical services, we aim to simplify your health journey.  
                """
            )

    # About Us Section
    with st.container():
        st.markdown("---")  # Adds a horizontal line for separation
        st.markdown("### About Us", unsafe_allow_html=True)  # Section title
        st.markdown(
            """
            We are a passionate team of tech enthusiasts, healthcare innovators, and creative minds dedicated to revolutionizing digital health solutions.  
            At Aura Health Care, we believe in blending technology with care to make health accessible and personalized for everyone.  
            
            We chose anime aesthetics to represent our creativity and unique approach to problem-solving. These visuals add a fun, vibrant vibe while keeping our vision professional and futuristic.  
            
            Together, let's build a healthier future—one step at a time!
            """,
            unsafe_allow_html=True,
        )
        st.image(
            "C:/Users/kavya/Downloads/girls.webp",  # Replace with the actual path to your image
            caption="Our Creative Spirit",
            use_column_width=True
        )

#skin disease detector

def skin_disease_page():
    st.title("Skin Disease Detector")
    st.markdown("""
    Upload Image: Provide a clear photo of the affected skin area.
    AI Analysis: The system compares your image with a trained database of skin conditions.
    Results: Get a prediction of the possible skin disease along with helpful details.
    """)

    # Load the pre-trained model
    model_path = "skin_disease_model.h5"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Please ensure the model file is uploaded and the path is correct.")
        return

    model = load_model(model_path)

    # Disease descriptions and doctor details
    disease_info = {
        "actinic keratosis": {
            "description": "A rough, scaly patch on the skin caused by years of sun exposure. Can develop into skin cancer.",
            "doctor": "Dr. Rajesh Kumar (Dermatologist)",
            "contact": "+91 9876543210",
            "email": "rajesh.derma@hospital.com",
            "hospital": "Apollo Hospital, Delhi"
        },
        "basal cell carcinoma": {
            "description": "A common type of skin cancer that appears as a transparent bump on the skin. It rarely spreads but should be treated.",
            "doctor": "Dr. Priya Sharma (Oncologist)",
            "contact": "+91 9876543211",
            "email": "priya.onco@hospital.com",
            "hospital": "Fortis Cancer Institute, Mumbai"
        },
        "dermatofibroma": {
            "description": "A harmless skin growth that feels firm to the touch. Often appears on legs and can be itchy.",
            "doctor": "Dr. Aman Verma (Dermatologist)",
            "contact": "+91 9876543212",
            "email": "aman.derma@hospital.com",
            "hospital": "Medanta Hospital, Gurgaon"
        },
        "melanoma": {
            "description": "The most serious type of skin cancer, developing from pigment-producing cells. Early detection is critical.",
            "doctor": "Dr. Neha Gupta (Oncologist)",
            "contact": "+91 9876543213",
            "email": "neha.onco@hospital.com",
            "hospital": "Tata Memorial Hospital, Mumbai"
        },
        "nevus": {
            "description": "A benign mole or birthmark, usually harmless. However, changes in shape or color should be monitored.",
            "doctor": "Dr. Sandeep Joshi (Dermatologist)",
            "contact": "+91 9876543214",
            "email": "sandeep.derma@hospital.com",
            "hospital": "AIIMS, Delhi"
        },
        "seborrheic keratosis": {
            "description": "A noncancerous skin growth that looks waxy or scaly. Often appears in older adults.",
            "doctor": "Dr. Ritu Mehta (Dermatologist)",
            "contact": "+91 9876543215",
            "email": "ritu.derma@hospital.com",
            "hospital": "Max Healthcare, Bangalore"
        },
        "squamous cell carcinoma": {
            "description": "A type of skin cancer that may look like a red, scaly sore. Can spread if untreated.",
            "doctor": "Dr. Akash Rao (Oncologist)",
            "contact": "+91 9876543216",
            "email": "akash.onco@hospital.com",
            "hospital": "Apollo Cancer Centre, Chennai"
        },
        "vascular lesion": {
            "description": "Abnormal blood vessels on the skin, can be benign or indicate underlying conditions.",
            "doctor": "Dr. Kavita Nair (Vascular Surgeon)",
            "contact": "+91 9876543217",
            "email": "kavita.vascular@hospital.com",
            "hospital": "Narayana Health, Hyderabad"
        },
        "pigmented benign keratosis": {
            "description": "A dark, noncancerous growth often found in older adults.",
            "doctor": "Dr. Anil Kapoor (Dermatologist)",
            "contact": "+91 9876543218",
            "email": "anil.derma@hospital.com",
            "hospital": "Manipal Hospital, Pune"
        }
    }

    # Upload an image
    uploaded_file = st.file_uploader("Upload an image of the skin lesion (JPG, PNG, etc.)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Load and preprocess the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image_resized = cv2.resize(image_rgb, (150, 150))
        image_expanded = np.expand_dims(image_resized, axis=0)
        image_normalized = image_expanded / 255.0  # Normalize

        # Predict the disease
        predictions = model.predict(image_normalized)
        confidence_scores = predictions[0] * 100  # Convert to percentage
        predicted_index = np.argmax(predictions)
        predicted_class = list(disease_info.keys())[predicted_index]
        predicted_confidence = confidence_scores[predicted_index]

        # Display prediction results
        st.subheader("Prediction Results")
        st.write(f"Predicted Disease: {predicted_class}")
        st.write(f"Confidence: {predicted_confidence:.2f}%")
        
        # Display disease description
        disease_details = disease_info.get(predicted_class, {})
        st.write(f"Description: {disease_details.get('description', 'No description available.')}")
        
        # Display doctor details
        st.subheader("Doctor Details")
        st.write(f"👨‍⚕ Doctor Name:** {disease_details.get('doctor', 'Not Available')}")
        st.write(f"📞 Contact Number:** {disease_details.get('contact', 'Not Available')}")
        st.write(f"📧 Email:** {disease_details.get('email', 'Not Available')}")
        st.write(f"🏥 Hospital:** {disease_details.get('hospital', 'Not Available')}")

# Function to render the Professional Doctor Page
def professional_doctor_page():
    st.title("👨‍⚕ Find Professional Doctors")
    st.markdown(
        """
        ### Connect with Experts
        Get access to professional doctors for personalized consultations on various health conditions.
        Simply type the name of the disease to find a recommended doctor!
        """
    )
    search_query = st.text_input("🔍 Enter Disease Name:").strip().lower()

    if search_query:
        # Fetch the doctor info based on the disease name
        doctor_info = disease_doctor_mapping.get(search_query, None)

        if doctor_info:
            st.success("✅ Doctor Found!")
            st.write(f"🦠 Disease:** {search_query.title()}")
            st.write(f"👨‍⚕ Doctor's Name:** {doctor_info['doctor_name']}")
            st.write(f"🏥 Specialization:** {doctor_info['specialization']}")
            st.write(f"📧 Email:** {doctor_info['email']}")
            st.write(f"📞 Contact:** {doctor_info['contact']}")
        else:
            st.error("❌ No doctor found for the given disease.")

# KIET College, Kakinada (Fixed Location)
USER_LAT, USER_LON = 16.8124121, 82.2402992

def speak_text(text):
    """Speaks the given text using text-to-speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def nearby_services_page():
    st.title("Nearby Medical Services 🏥💊")
    st.markdown(
        """
        ### Locate Services
        Find nearby pharmacies, clinics, and hospitals to meet your healthcare needs.
        """
    )
    
    # Button to trigger the service locator
    if st.button("Find Nearby Services", key="find_services_button"):

        # Medical Services Data (Hospitals & Pharmacies)
        data = {
            "Name": [
                "Medicover Hospitals", "Apollo Hospitals", "Trust Multispecialty Hospitals", "Sai Sudha Multispecialty Hospital",
                "Inodaya Hospitals", "Siddhartha Hospital", "Government General Hospital", "Surya Global Hospitals",
                "Narayana Superspeciality Hospital", "Aditya Multispecialty Hospital", "Ramesh Hospitals", "Sunrise Hospital",
                "Balaji Hospital", "Manasa Hospital", "Swarna Hospital", "Care Hospital", "Lotus Hospitals for Women & Children",
                "KIMS Hospital", "Amrutha Hospital", "Bollineni Hospitals", "Sri Ramachandra Hospital", "Usha Mullapudi Cardiac Center",
                "Pragati Hospital", "Gayatri Vidya Parishad Hospital", "Queens NRI Hospital", "City Hospital",
                "Apollo Pharmacy", "MedPlus Pharmacy", "Sri Balaji Medicals", "Sri Sai Medicals", "Jan Aushadhi Kendra"
            ],
            "Latitude": [16.950000, 16.951378, 16.934271, 16.945600, 16.952100, 16.939500, 16.940315, 16.945985,
                        16.947500, 16.953200, 16.944800, 16.949000, 16.946500, 16.948700, 16.950800, 16.951900, 16.955000,
                        16.956700, 16.954300, 16.957800, 16.958900, 16.959500, 16.960200, 16.961000, 16.962500, 16.963200,
                        16.948000, 16.947500, 16.949800, 16.950300, 16.951000],
            "Longitude": [82.242500, 82.238218, 82.240719, 82.249800, 82.241700, 82.236800, 82.238976, 82.259548,
                        82.243500, 82.245700, 82.239600, 82.247900, 82.241200, 82.250300, 82.244900, 82.248600, 82.252000,
                        82.253500, 82.254700, 82.255800, 82.256900, 82.257500, 82.258200, 82.259000, 82.260500, 82.261200,
                        82.243800, 82.246000, 82.248500, 82.250000, 82.251200],
            "Type": ["Hospital"] * 26 + ["Pharmacy"] * 5
        }

        df = pd.DataFrame(data)

        # Calculate distances from user location (KIET College)
        df["Distance (km)"] = df.apply(lambda row: geodesic((USER_LAT, USER_LON), (row["Latitude"], row["Longitude"])).km, axis=1)

        # Sort by nearest places
        df = df.sort_values(by="Distance (km)")

        # Select hospital for route & distance calculation
        selected_hospital = st.selectbox("🏥 Select a Hospital:", df[df["Type"] == "Hospital"]["Name"], key="select_hospital")
        selected_hospital_data = df[df["Name"] == selected_hospital].iloc[0]
        selected_lat, selected_lon = selected_hospital_data["Latitude"], selected_hospital_data["Longitude"]
        selected_distance = selected_hospital_data["Distance (km)"]

        st.write(f"📍 {selected_hospital} is {selected_distance:.2f} km away.")

        # Speak out the selected hospital's name and distance
        speech_text = f"{selected_hospital} is {selected_distance:.2f} kilometers away."
        if st.button("🔊 Speak Location", key="speak_button"):
            speak_text(speech_text)

        # Create Map
        hospital_map = folium.Map(location=[USER_LAT, USER_LON], zoom_start=14, tiles="CartoDB positron")

        # Add User Location Marker (Always Blue)
        folium.Marker(
            [USER_LAT, USER_LON], popup="📍 KIET College", tooltip="You are here",
            icon=folium.Icon(color="blue", icon="user")
        ).add_to(hospital_map)

        # Add Hospital & Pharmacy Markers
        for _, row in df.iterrows():
            marker_color = "red" if row["Type"] == "Hospital" else "green"
            marker_icon = "plus" if row["Type"] == "Hospital" else "shopping-cart"

            # Highlight selected hospital in Pink
            if row["Name"] == selected_hospital:
                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    popup=f"<b>{row['Name']}</b><br>Distance: {row['Distance (km)']:.2f} km",
                    tooltip=row["Name"],
                    icon=folium.Icon(color="pink", icon="hospital"),
                ).add_to(hospital_map)
            else:
                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    popup=f"<b>{row['Name']}</b><br>Distance: {row['Distance (km)']:.2f} km",
                    tooltip=row["Name"],
                    icon=folium.Icon(color=marker_color, icon=marker_icon),
                ).add_to(hospital_map)

        # Draw Arrow from User to Selected Hospital
        folium.PolyLine(
            [(USER_LAT, USER_LON), (selected_lat, selected_lon)],
            color="blue", weight=5, opacity=0.8, tooltip="Route to Hospital"
        ).add_to(hospital_map)

        # Show the Map
        folium_static(hospital_map)

conn = mysql.connector.connect(
    host="......",  # e.g., "localhost"
    port=...,
    user="......",  # e.g., "root"
    password="......",  # e.g., "password123"
    database="......."  # e.g., "queries_db"
)
cursor = conn.cursor()

# Function to render the Query Page
def query_page():
    st.title("Submit Your Query ❓")
    st.markdown("Fill in the form below to submit your query to our team.")

    # Initialize session state for form fields
    if "query_form_data" not in st.session_state:
        st.session_state["query_form_data"] = {
            "name": "",
            "email": "",
            "phone": "",
            "message": "",
        }

    # Form to collect user data
    with st.form(key="query_form"):
        name = st.text_input("Name:", value=st.session_state["query_form_data"]["name"])
        email = st.text_input("Email:", value=st.session_state["query_form_data"]["email"])
        phone = st.text_input("Phone Number:", value=st.session_state["query_form_data"]["phone"])
        message = st.text_area("Message:", value=st.session_state["query_form_data"]["message"])
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        # Validate inputs
        if not name or not email or not phone or not message:
            st.error("❌ Please fill in all fields.")
        elif len(phone) != 10 or not phone.isdigit():
            st.error("❌ Phone number must be 10 digits.")
        else:
            # Insert data into the database
            try:
                cursor.execute(
                    "INSERT INTO queries (name, email, phonenumber, message) VALUES (%s, %s, %s, %s)",
                    (name, email, phone, message)
                )
                conn.commit()
                st.success("✅ Your query has been submitted successfully!")
            except mysql.connector.Error as err:
                st.error(f"❌ Error: {err}")

# Render the correct page based on session state
if st.session_state["current_page"] == "Home":
    home_page()
elif st.session_state["current_page"] == "Skin Disease":
    skin_disease_page()
elif st.session_state["current_page"] == "Professional Doctor":
    professional_doctor_page()
elif st.session_state["current_page"] == "Nearby Services":
    nearby_services_page()
elif st.session_state["current_page"] == "Query":
    query_page()

# Footer
st.sidebar.markdown(
    """
     Contact Us  
    Email: support@aurahealth.com  
    Phone: 9524026927
"""
)