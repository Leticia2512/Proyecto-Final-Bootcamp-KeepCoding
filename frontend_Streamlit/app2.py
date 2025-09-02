import streamlit as st
import requests
import json
from PIL import Image
import io

# FastAPI backend URL
FASTAPI_URL = "http://localhost:8000/prediction"

st.title("👁️ Ocular Prediction System")
st.write("Upload an eye image and provide patient information for prediction")

# Create form for user input
with st.form("prediction_form"):
    st.header("Patient Information")
    
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    gender = st.selectbox("Gender", ["male", "female"])
    
    st.header("Image Upload")
    uploaded_file = st.file_uploader("Choose an eye image", type=["jpg", "jpeg"])
    
    submitted = st.form_submit_button("Make Prediction")

if submitted:
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Eye Image", use_container_width=False)
            
            # Save the image temporarily (or you can send it directly as bytes)
            image_path = f"temp_{uploaded_file.name}"
            image.save(image_path)
            
            # Prepare the data for FastAPI
            data = {
                "age": age,
                "gender": gender,
                "image_url": image_path  # This should be the path your FastAPI can access
            }
            
            # Show loading spinner
            with st.spinner("Making prediction..."):
                # Send POST request to FastAPI
                response = requests.post(FASTAPI_URL, json=data)
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("✅ Prediction successful!")
                
                # Display results
                st.subheader("Prediction Results")
                
                # Map class numbers to human-readable labels
                class_mapping = {
                    0: "DR",
                    1: "GLAUCOMA",
                    2: "CATARACT",
                    5: "NORMAL",
                    6: "OTHER"
                }
                
                predicted_class = result["predicted_class"]
                probabilities = result["probabilities"]
                
                st.metric("Predicted Condition", class_mapping.get(predicted_class, "UNKNOWN"))
                
                # Display probabilities
                st.subheader("Probability Distribution")
                
                for i, prob in enumerate(probabilities):
                    class_name = class_mapping.get(i, f"Class {i}")
                    st.progress(prob, text=f"{class_name}: {prob:.2%}")
                
                # Show raw JSON data
                with st.expander("View Raw Response"):
                    st.json(result)
                    
            else:
                st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
    else:
        st.warning("⚠️ Please upload an image first.")
