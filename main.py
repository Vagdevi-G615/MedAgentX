import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from image_model import MedicalImageModel
from text_model import MedicalTextModel
from explainability import ExplainabilityEngine
from utils import DatabaseManager, preprocess_image, format_confidence, get_recommendation

# Initialize models and components
def load_models():
    return {
        'image_model': MedicalImageModel(),
        'text_model': MedicalTextModel(),
        'explainer': ExplainabilityEngine(),
        'db': DatabaseManager()
    }

def main():
    st.set_page_config(page_title="MedAgentX", page_icon="🏥", layout="wide")
    
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0;'>🏥 MedAgentX</h1>
        <p style='color: white; margin: 5px 0 0 0; font-size: 18px;'>AI-Powered Medical Diagnosis Support</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("⚠️ **Medical Disclaimer**: This tool is for educational purposes only. Always consult healthcare professionals for medical decisions.")
    
    # Load models
    models = load_models()
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        page = st.selectbox("Choose a page", ["🔍 Diagnosis", "📚 History"])
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("Supported Conditions", "11")
        st.metric("Image Types", "MRI & X-Ray")
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info("📸 Upload clear, high-quality medical images")
        st.info("📝 Describe symptoms in detail")
    
    if "Diagnosis" in page:
        diagnosis_page(models)
    else:
        history_page(models['db'])

def diagnosis_page(models):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Medical Image Upload")
        uploaded_file = st.file_uploader(
            "Choose a medical image", 
            type=['png', 'jpg', 'jpeg'],
            help="Supported: PNG, JPG, JPEG"
        )
        
        if uploaded_file:
            image = preprocess_image(uploaded_file)
            st.image(image, caption="✅ Uploaded Medical Image", use_column_width=True)
            st.success(f"Image uploaded: {uploaded_file.size} bytes")
    
    with col2:
        st.markdown("### 📝 Clinical Symptoms")
        symptoms = st.text_area(
            "Describe patient symptoms:",
            placeholder="e.g., persistent cough, fever, shortness of breath, chest pain, headache, vision problems...",
            height=200,
            help="Be detailed for better accuracy"
        )
        
        if symptoms:
            st.success(f"Symptoms recorded: {len(symptoms.split())} words")
    
    st.markdown("---")
    
    if st.button("🔍 Get Diagnosis", type="primary", use_container_width=True, help="Click to analyze your medical data"):
        if uploaded_file and symptoms:
            with st.spinner("Analyzing medical data..."):
                # Get predictions
                image_pred = models['image_model'].predict(image)
                text_pred = models['text_model'].predict(symptoms)
                final_pred = models['explainer'].combine_predictions(image_pred, text_pred)
                
                # Get explanations
                attention_map = models['image_model'].get_attention_map(image)
                attention_tokens = models['text_model'].get_attention_weights(symptoms)
                
                # Display results
                display_results(models, image, image_pred, text_pred, final_pred, 
                              attention_map, attention_tokens, symptoms)
                
                # Save to database
                models['db'].save_diagnosis(image_pred, text_pred, final_pred, symptoms)
        else:
            st.error("Please upload an image and enter symptoms.")

def display_results(models, image, image_pred, text_pred, final_pred, attention_map, attention_tokens, symptoms):
    st.markdown("## 🎯 Diagnosis Results")
    
    # Main prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🎯 Final Diagnosis**")
        st.success(final_pred['prediction'])
    with col2:
        st.markdown("**📊 Confidence**")
        st.info(format_confidence(final_pred['confidence']))
    with col3:
        risk_level = "High" if final_pred['confidence'] > 0.8 else "Medium" if final_pred['confidence'] > 0.6 else "Low"
        st.markdown("**⚠️ Risk Level**")
        if risk_level == "High":
            st.error(risk_level)
        elif risk_level == "Medium":
            st.warning(risk_level)
        else:
            st.success(risk_level)
    
    # Detailed breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Prediction Breakdown")
        st.write("**Image Analysis:**")
        for label, prob in image_pred['probabilities'].items():
            st.write(f"- {label}: {format_confidence(prob)}")
        
        st.write("**Symptom Analysis:**")
        for label, prob in text_pred['probabilities'].items():
            st.write(f"- {label}: {format_confidence(prob)}")
    
    with col2:
        st.subheader("🔍 Visual Explanation")
        image_type = image_pred.get('image_type', 'X-ray')
        st.write(f"**Image Type Detected:** {image_type}")
        with st.expander("🔧 Technical Details"):
            brightness = np.mean(np.array(image.convert('L')))
            contrast = np.std(np.array(image.convert('L')))
            st.metric("Brightness", f"{brightness:.1f}")
            st.metric("Contrast", f"{contrast:.1f}")
        heatmap = models['explainer'].generate_gradcam_heatmap(attention_map, image)
        st.image(heatmap, caption="Attention Heatmap", use_column_width=True)
    
    # Tabbed layout
    tab1, tab2 = st.tabs(["📋 Clinical Analysis", "💡 Recommendations"])
    
    with tab1:
        explanation = models['explainer'].generate_text_explanation(image_pred, text_pred, attention_tokens)
        st.markdown(explanation)
        
        st.markdown("**📊 Probability Breakdown:**")
        for condition, prob in sorted(final_pred['probabilities'].items(), key=lambda x: x[1], reverse=True)[:5]:
            st.progress(prob, text=f"{condition}: {format_confidence(prob)}")
    
    with tab2:
        recommendation = get_recommendation(final_pred['prediction'], final_pred['confidence'])
        st.info(recommendation)
        
        if final_pred['confidence'] < 0.6:
            st.warning("⚠️ Low confidence prediction. Seek professional medical consultation.")
        
        st.markdown("**🏥 Next Steps:**")
        st.markdown("• Consult with a healthcare professional")
        st.markdown("• Consider additional diagnostic tests")
        st.markdown("• Monitor symptoms closely")

def history_page(db):
    st.header("📚 Diagnosis History")
    
    history = db.get_history()
    
    if history:
        for record in history:
            with st.expander(f"Diagnosis from {record[1][:19]}"):
                st.write(f"**Final Prediction:** {record[4]}")
                st.write(f"**Confidence:** {format_confidence(record[5])}")
                st.write(f"**Symptoms:** {record[6]}")
    else:
        st.info("No diagnosis history available.")

if __name__ == "__main__":
    main()