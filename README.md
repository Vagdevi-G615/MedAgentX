# MedAgentX: AI-Powered Medical Diagnosis Support

🏥 A healthcare diagnosis support web application that analyzes medical images and clinical symptoms using Vision Transformers and Large Language Models.

## Features

- **Multi-modal Analysis**: Combines medical image analysis (Vision Transformer) with symptom text analysis (ClinicalBERT)
- **Explainable AI**: Provides visual attention maps and textual explanations for predictions
- **Diagnosis History**: Saves and tracks previous diagnoses locally
- **User-friendly Interface**: Clean Streamlit web interface

## Tech Stack

- **Frontend**: Streamlit
- **Image Model**: Vision Transformer (ViT)
- **Text Model**: Bio_ClinicalBERT
- **Explainability**: Attention visualization, Grad-CAM-style heatmaps
- **Database**: SQLite for local storage

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MedAgentX
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Launch the application:
```bash
streamlit run main.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload a medical image (X-ray, skin lesion, etc.)

4. Enter clinical symptoms in the text area

5. Click "Get Diagnosis" to receive AI-powered analysis


## Project Structure

```
MedAgentX/
├── main.py              # Streamlit web application
├── image_model.py       # Vision Transformer for medical images
├── text_model.py        # ClinicalBERT for symptom analysis
├── explainability.py    # XAI components (attention, heatmaps)
├── utils.py             # Database and utility functions
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Important Notes

⚠️ **Disclaimer**: This application is for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

## Future Enhancements

- [ ] PDF report generation
- [ ] Mobile responsive design
- [ ] Integration with real medical datasets
- [ ] Multi-language support
- [ ] Advanced ensemble methods
