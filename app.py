"""
================================================================================
FLASK WEB APPLICATION - WINE QUALITY PREDICTION
================================================================================
Aplikasi web untuk prediksi kualitas wine menggunakan Neural Network
yang telah dilatih dengan algoritma Backpropagation.

Fitur:
- Form input untuk 11 parameter wine
- Real-time prediction
- Error handling
- Responsive design dengan Bootstrap 5
================================================================================
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import logging

# ============================================================================
# KONFIGURASI APLIKASI
# ============================================================================

# Inisialisasi Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# LOAD MODEL DAN SCALER
# ============================================================================

try:
    # Load model neural network
    model = load_model('model.keras')
    logger.info("✓ Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

try:
    # Load scaler untuk normalisasi data
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    logger.info("✓ Scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading scaler: {e}")
    scaler = None

# ============================================================================
# FEATURE NAMES (Nama fitur sesuai urutan di dataset)
# ============================================================================

FEATURE_NAMES = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]

# Range nilai yang valid (untuk validasi input)
FEATURE_RANGES = {
    'fixed acidity': (4.6, 15.9),
    'volatile acidity': (0.12, 1.58),
    'citric acid': (0.0, 1.0),
    'residual sugar': (0.9, 15.5),
    'chlorides': (0.012, 0.611),
    'free sulfur dioxide': (1.0, 72.0),
    'total sulfur dioxide': (6.0, 289.0),
    'density': (0.99007, 1.00369),
    'pH': (2.74, 4.01),
    'sulphates': (0.33, 2.0),
    'alcohol': (8.4, 14.9)
}

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """
    Route untuk halaman utama (home page)
    Menampilkan form input untuk prediksi wine quality
    """
    try:
        return render_template('index.html', features=FEATURE_NAMES)
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        return f"Error: {e}", 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Route untuk melakukan prediksi kualitas wine
    
    Request: POST dengan JSON data
    Response: JSON dengan hasil prediksi
    
    Format request:
    {
        "fixed_acidity": value,
        "volatile_acidity": value,
        ...
    }
    
    Format response:
    {
        "success": true/false,
        "prediction": "GOOD" / "BAD",
        "confidence": 0.0-1.0,
        "message": "string",
        "error": "string (jika ada error)"
    }
    """
    
    try:
        # Cek apakah model dan scaler sudah loaded
        if model is None or scaler is None:
            return jsonify({
                'success': False,
                'error': 'Model atau scaler tidak berhasil dimuat'
            }), 500
        
        # Get JSON data dari request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Data kosong atau format tidak valid'
            }), 400
        
        # ====================================================================
        # VALIDASI INPUT
        # ====================================================================
        
        input_values = []
        errors = []
        
        for feature_name in FEATURE_NAMES:
            # Convert feature name untuk JavaScript (underscore to space)
            js_feature_name = feature_name.replace(' ', '_')
            
            if js_feature_name not in data:
                errors.append(f"Input '{feature_name}' tidak ditemukan")
                continue
            
            try:
                # Konversi ke float
                value = float(data[js_feature_name])
                
                # Validasi range
                min_val, max_val = FEATURE_RANGES[feature_name]
                if not (min_val <= value <= max_val):
                    errors.append(
                        f"'{feature_name}' harus antara {min_val} - {max_val}, "
                        f"Anda memasukkan {value}"
                    )
                else:
                    input_values.append(value)
                    
            except ValueError:
                errors.append(f"'{feature_name}' harus berupa angka")
        
        # Jika ada error validasi
        if errors:
            return jsonify({
                'success': False,
                'error': '\n'.join(errors)
            }), 400
        
        # ====================================================================
        # PREPROCESSING DATA
        # ====================================================================
        
        # Convert ke numpy array
        input_array = np.array(input_values).reshape(1, -1)
        
        # Normalisasi menggunakan scaler
        input_scaled = scaler.transform(input_array)
        
        # ====================================================================
        # PREDIKSI MENGGUNAKAN MODEL
        # ====================================================================
        
        # Prediksi probability
        prediction_prob = model.predict(input_scaled, verbose=0)[0][0]
        
        # Konversi ke label (threshold = 0.5)
        prediction_label = 1 if prediction_prob >= 0.5 else 0
        
        # Convert ke text
        quality_text = "GOOD" if prediction_label == 1 else "BAD"
        confidence = max(prediction_prob, 1 - prediction_prob)
        
        # ====================================================================
        # RESPONSE
        # ====================================================================
        
        return jsonify({
            'success': True,
            'prediction': quality_text,
            'confidence': float(confidence),
            'probability': float(prediction_prob),
            'message': f'Wine ini diprediksi berkualitas {quality_text} '
                      f'dengan confidence {confidence*100:.2f}%'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Terjadi error: {str(e)}'
        }), 500


@app.route('/api/ranges', methods=['GET'])
def get_ranges():
    """
    API endpoint untuk mendapatkan range valid setiap fitur
    Digunakan untuk validasi di frontend
    """
    try:
        ranges = {}
        for feature_name, (min_val, max_val) in FEATURE_RANGES.items():
            js_feature_name = feature_name.replace(' ', '_')
            ranges[js_feature_name] = {
                'min': min_val,
                'max': max_val,
                'display_name': feature_name
            }
        return jsonify(ranges), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint untuk memastikan server berjalan
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    }), 200


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Page not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Verifikasi model dan scaler
    if model is None or scaler is None:
        print("\n⚠️  WARNING: Model atau scaler tidak berhasil dimuat!")
        print("Pastikan file 'model.h5' dan 'scaler.pkl' ada di direktori yang sama dengan app.py")
        print("Jalankan 'python train_model.py' terlebih dahulu untuk membuat model.\n")
    
    print("\n" + "="*80)
    print("STARTING WINE QUALITY PREDICTION WEB APP")
    print("="*80)
    print("Server berjalan di: http://localhost:5000")
    print("Tekan CTRL+C untuk menghentikan server")
    print("="*80 + "\n")
    
    # Jalankan Flask app
    app.run(
        debug=True,           # Debug mode ON
        host='localhost',     # Localhost
        port=5000             # Port 5000
    )