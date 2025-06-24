import json
import sqlite3
from datetime import datetime
from PIL import Image
import io
import base64

class DatabaseManager:
    def __init__(self, db_path="diagnosis_history.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                image_prediction TEXT,
                text_prediction TEXT,
                final_prediction TEXT,
                confidence REAL,
                symptoms TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_diagnosis(self, image_pred, text_pred, final_pred, symptoms):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO diagnoses (timestamp, image_prediction, text_prediction, final_prediction, confidence, symptoms)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            json.dumps(image_pred),
            json.dumps(text_pred),
            final_pred['prediction'],
            final_pred['confidence'],
            symptoms
        ))
        conn.commit()
        conn.close()
    
    def get_history(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM diagnoses ORDER BY timestamp DESC LIMIT ?', (limit,))
        results = cursor.fetchall()
        conn.close()
        return results

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def format_confidence(confidence):
    return f"{confidence:.1%}"

def get_recommendation(prediction, confidence):
    recommendations = {
        'Normal': "No immediate medical attention required. Continue regular health monitoring.",
        'Pneumonia': "Recommend consultation with a pulmonologist.",
        'Atelectasis': "Recommend pulmonary evaluation.",
        'Cardiomegaly': "Recommend cardiology consultation.",
        'Effusion': "Recommend chest imaging follow-up.",
        'Infiltration': "Recommend pulmonary assessment."
    }
    
    base_rec = recommendations.get(prediction, "Consult healthcare professional for evaluation.")
    
    if confidence < 0.6:
        base_rec += " Note: Low confidence prediction - seek professional medical opinion."
    
    return base_rec