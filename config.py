"""
================================================================================
CONFIG.PY - KONFIGURASI PROJECT WINE QUALITY PREDICTION
================================================================================
File ini berisi semua konstanta dan konfigurasi yang digunakan di project
"""

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Path ke dataset
DATASET_PATH = '/mnt/user-data/uploads/WineQT.csv'

# Kolom fitur yang digunakan (11 features)
FEATURE_COLUMNS = [
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

# Kolom target
TARGET_COLUMN = 'quality'

# Threshold untuk binary classification
# quality >= QUALITY_THRESHOLD → GOOD (1)
# quality < QUALITY_THRESHOLD → BAD (0)
QUALITY_THRESHOLD = 6

# ============================================================================
# DATA PREPROCESSING CONFIGURATION
# ============================================================================

# Ratio untuk split train-test
TEST_SIZE = 0.2  # 20% test, 80% train

# Random state untuk reproducibility
RANDOM_STATE = 42

# ============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# ============================================================================

# Input dimension (number of features)
INPUT_DIM = 11

# Hidden layer configurations
HIDDEN_LAYERS = [
    {
        'units': 64,
        'activation': 'relu',
        'dropout': 0.2,
        'name': 'hidden_layer_1'
    },
    {
        'units': 32,
        'activation': 'relu',
        'dropout': 0.2,
        'name': 'hidden_layer_2'
    },
    {
        'units': 16,
        'activation': 'relu',
        'dropout': 0.1,
        'name': 'hidden_layer_3'
    }
]

# Output layer configuration
OUTPUT_UNITS = 1
OUTPUT_ACTIVATION = 'sigmoid'  # Binary classification

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Optimizer
OPTIMIZER = 'adam'

# Loss function (binary classification)
LOSS_FUNCTION = 'binary_crossentropy'

# Metrics
METRICS = ['accuracy']

# Training parameters
EPOCHS = 200
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2  # 20% dari training data untuk validation

# ============================================================================
# EARLY STOPPING CONFIGURATION
# ============================================================================

# Monitor validation loss
EARLY_STOPPING_MONITOR = 'val_loss'

# Berapa epoch tanpa improvement sebelum stop
EARLY_STOPPING_PATIENCE = 15

# Restore bobot terbaik
EARLY_STOPPING_RESTORE_BEST_WEIGHTS = True

# ============================================================================
# MODEL PATHS
# ============================================================================

# Path untuk menyimpan trained model
MODEL_PATH = 'model.h5'

# Path untuk menyimpan scaler
SCALER_PATH = 'scaler.pkl'

# Path untuk visualization hasil training
TRAINING_RESULTS_IMAGE = 'static/training_results.png'

# ============================================================================
# FLASK CONFIGURATION
# ============================================================================

# Flask app configuration
FLASK_DEBUG = True
FLASK_HOST = 'localhost'
FLASK_PORT = 5000

# Max content length (untuk upload, jika ada)
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# ============================================================================
# PREDICTION CONFIGURATION
# ============================================================================

# Prediction threshold (0.5 untuk binary classification)
PREDICTION_THRESHOLD = 0.5

# Label mapping
PREDICTION_LABELS = {
    0: 'BAD',
    1: 'GOOD'
}

# Feature ranges (untuk validasi input)
# Format: {feature_name: (min_value, max_value)}
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
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# DISPLAY CONFIGURATION (untuk visualization)
# ============================================================================

# Figure size untuk matplotlib
FIGURE_SIZE = (14, 10)

# DPI untuk saved images
IMAGE_DPI = 300

# ============================================================================
# DESCRIPTION & DOCUMENTATION
# ============================================================================

PROJECT_NAME = "Wine Quality Prediction System"
PROJECT_DESCRIPTION = """
Implementasi Jaringan Saraf Tiruan dengan Algoritma Backpropagation 
untuk Prediksi Kualitas Wine Berbasis Web Flask
"""

PROJECT_VERSION = "1.0.0"
PROJECT_AUTHOR = "Student"
PROJECT_DATE = "2024"