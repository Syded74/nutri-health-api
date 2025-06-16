from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Global variables for model components
model = None
scaler = None
feature_names = None
label_mapping = None

# Nutrition reference values and defaults
NUTRITION_DEFAULTS = {
    "caloric_value": {
        "default": 150,  # Average calories per 100g for processed foods
        "description": "Calories per 100g"
    },
    "fat": {
        "default": 5.0,  # Conservative estimate
        "description": "Total fat (g) per 100g"
    },
    "saturated_fat": {
        "default": 2.0,  # Usually 30-40% of total fat
        "description": "Saturated fat (g) per 100g"
    },
    "sugars": {
        "default": 8.0,  # Moderate sugar content
        "description": "Sugars (g) per 100g"
    },
    "sodium": {
        "default": 0.3,  # Conservative sodium estimate (300mg per 100g)
        "description": "Sodium (g) per 100g"
    },
    "protein": {
        "default": 6.0,  # Moderate protein content
        "description": "Protein (g) per 100g"
    },
    "vitamin_a": {
        "default": 10.0,  # Minimal vitamin A
        "description": "Vitamin A (μg) per 100g"
    },
    "vitamin_c": {
        "default": 5.0,  # Minimal vitamin C
        "description": "Vitamin C (mg) per 100g"
    },
    "iron": {
        "default": 1.5,  # Minimal iron
        "description": "Iron (mg) per 100g"
    },
    "calcium": {
        "default": 50.0,  # Minimal calcium
        "description": "Calcium (mg) per 100g"
    }
}

def load_model_components():
    """Load all model components at startup"""
    global model, scaler, feature_names, label_mapping
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'model')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at: {model_path}")
        
        model = joblib.load(os.path.join(model_path, 'nutrisafe_model.pkl'))
        scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        feature_names = joblib.load(os.path.join(model_path, 'feature_names.pkl'))
        label_mapping = joblib.load(os.path.join(model_path, 'label_mapping.pkl'))
        
        logger.info("✅ All model components loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model components: {str(e)}")
        return False

def impute_missing_values(data, method='smart'):
    """
    Handle missing nutrition values with different strategies
    
    Args:
        data: Dictionary with nutrition values (None, empty string, or numbers)
        method: 'smart', 'conservative', 'zero', or 'category_based'
    
    Returns:
        Dictionary with imputed values and metadata
    """
    imputed_data = {}
    missing_fields = []
    imputation_info = {}
    
    for field in feature_names:
        value = data.get(field)
        
        # Check if value is missing or invalid
        if value is None or value == "" or (isinstance(value, str) and value.strip() == ""):
            missing_fields.append(field)
            
            if method == 'smart':
                # Smart imputation based on food category patterns
                imputed_value = get_smart_default(field, data)
            elif method == 'conservative':
                # Use conservative estimates that won't bias toward unhealthy
                imputed_value = NUTRITION_DEFAULTS[field]['default']
            elif method == 'category_based':
                # Estimate based on food category (if provided)
                imputed_value = get_category_based_default(field, data)
            else:  # zero
                imputed_value = 0.0
            
            imputed_data[field] = imputed_value
            imputation_info[field] = {
                'original': None,
                'imputed': imputed_value,
                'method': method
            }
        else:
            # Convert to float if valid
            try:
                imputed_data[field] = float(value)
                imputation_info[field] = {
                    'original': float(value),
                    'imputed': float(value),
                    'method': 'provided'
                }
            except (ValueError, TypeError):
                missing_fields.append(field)
                imputed_value = NUTRITION_DEFAULTS[field]['default']
                imputed_data[field] = imputed_value
                imputation_info[field] = {
                    'original': value,
                    'imputed': imputed_value,
                    'method': 'invalid_converted'
                }
    
    return imputed_data, missing_fields, imputation_info

def get_smart_default(field, data):
    """
    Provide smart defaults based on relationships between nutrients
    """
    # Get any provided values to make educated guesses
    calories = safe_float(data.get('caloric_value'))
    fat = safe_float(data.get('fat'))
    protein = safe_float(data.get('protein'))
    
    smart_defaults = {
        'caloric_value': 150,  # Reasonable middle ground
        'fat': max(3.0, calories * 0.03 if calories else 5.0),  # ~3% of calories from fat
        'saturated_fat': (fat * 0.3) if fat else 1.5,  # ~30% of total fat
        'sugars': max(2.0, calories * 0.05 if calories else 6.0),  # ~5% of calories
        'sodium': 0.25,  # Conservative sodium estimate
        'protein': max(4.0, calories * 0.1 if calories else 6.0),  # ~10% of calories
        'vitamin_a': 5.0,   # Minimal values for vitamins/minerals
        'vitamin_c': 2.0,
        'iron': 1.0,
        'calcium': 30.0
    }
    
    return smart_defaults.get(field, NUTRITION_DEFAULTS[field]['default'])

def get_category_based_default(field, data):
    """
    Estimate values based on product category (if provided)
    """
    category = data.get('category', '').lower()
    
    # Category-based defaults (you can expand this)
    category_defaults = {
        'dairy': {
            'protein': 8.0, 'calcium': 120.0, 'fat': 3.5
        },
        'beverage': {
            'caloric_value': 40, 'sugars': 10.0, 'sodium': 0.01
        },
        'snack': {
            'caloric_value': 450, 'fat': 20.0, 'sodium': 0.8
        },
        'cereal': {
            'caloric_value': 350, 'sugars': 15.0, 'iron': 8.0
        }
    }
    
    if category in category_defaults and field in category_defaults[category]:
        return category_defaults[category][field]
    
    return NUTRITION_DEFAULTS[field]['default']

def safe_float(value):
    """Safely convert value to float, return None if not possible"""
    try:
        return float(value) if value is not None and value != "" else None
    except (ValueError, TypeError):
        return None

def calculate_confidence_adjustment(missing_fields, total_fields):
    """
    Adjust prediction confidence based on missing data
    """
    missing_ratio = len(missing_fields) / total_fields
    
    if missing_ratio == 0:
        return 1.0  # Full confidence
    elif missing_ratio <= 0.2:
        return 0.9  # Minor adjustment
    elif missing_ratio <= 0.4:
        return 0.75  # Moderate adjustment
    elif missing_ratio <= 0.6:
        return 0.6   # Significant adjustment
    else:
        return 0.4   # Major adjustment

def get_nutrition_advice(prediction, probabilities, missing_fields, confidence_adjustment):
    """Generate nutrition advice with missing data consideration"""
    base_advice = {
        0: {
            "category": "Healthy",
            "message": "This food appears to have a good nutritional profile.",
            "tips": [
                "Maintain balanced portions",
                "Pair with other nutritious foods",
                "Stay hydrated"
            ],
            "color": "#4CAF50"
        },
        1: {
            "category": "Moderate",
            "message": "This food is okay in moderation.",
            "tips": [
                "Limit portion sizes",
                "Balance with healthier options",
                "Consider frequency of consumption"
            ],
            "color": "#FF9800"
        },
        2: {
            "category": "Unhealthy",
            "message": "Consider healthier alternatives when possible.",
            "tips": [
                "Consume sparingly",
                "Look for lower sodium/sugar alternatives",
                "Balance with nutrient-dense foods"
            ],
            "color": "#F44336"
        }
    }
    
    advice = base_advice.get(prediction, base_advice[1])
    
    # Add warning about missing data if significant
    if len(missing_fields) > 0:
        confidence_level = "high" if confidence_adjustment > 0.8 else "moderate" if confidence_adjustment > 0.6 else "low"
        
        advice["data_quality_warning"] = {
            "missing_nutrients": missing_fields,
            "confidence_level": confidence_level,
            "message": f"Prediction based on {len(feature_names) - len(missing_fields)}/{len(feature_names)} nutrients. For more accurate results, try to obtain complete nutritional information."
        }
    
    return advice

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction with missing value handling"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Get imputation method from request (default to 'smart')
        imputation_method = data.get('imputation_method', 'smart')
        
        # Handle missing values
        imputed_data, missing_fields, imputation_info = impute_missing_values(data, imputation_method)
        
        # Prepare input array
        input_array = np.array([[imputed_data[feature] for feature in feature_names]])
        
        # Scale input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Adjust confidence based on missing data
        confidence_adjustment = calculate_confidence_adjustment(missing_fields, len(feature_names))
        adjusted_confidence = float(max(probabilities)) * confidence_adjustment * 100
        
        # Get advice with missing data consideration
        advice = get_nutrition_advice(prediction, probabilities, missing_fields, confidence_adjustment)
        
        # Prepare response
        response = {
            "prediction": {
                "risk_level": int(prediction),
                "category": label_mapping[prediction],
                "confidence": adjusted_confidence,
                "raw_confidence": float(max(probabilities)) * 100,
                "confidence_adjustment": confidence_adjustment
            },
            "probabilities": {
                "healthy": float(probabilities[0]) * 100,
                "moderate": float(probabilities[1]) * 100,
                "unhealthy": float(probabilities[2]) * 100
            },
            "data_quality": {
                "missing_fields": missing_fields,
                "missing_count": len(missing_fields),
                "total_fields": len(feature_names),
                "completeness_ratio": (len(feature_names) - len(missing_fields)) / len(feature_names),
                "imputation_method": imputation_method,
                "imputation_details": imputation_info
            },
            "advice": advice,
            "input_data": imputed_data,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Prediction: {label_mapping[prediction]} ({adjusted_confidence:.1f}%), Missing: {len(missing_fields)}/{len(feature_names)}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

@app.route('/imputation-methods', methods=['GET'])
def get_imputation_methods():
    """Get available imputation methods"""
    return jsonify({
        "methods": {
            "smart": {
                "description": "Uses relationships between nutrients to estimate missing values",
                "recommended": True
            },
            "conservative": {
                "description": "Uses moderate default values that won't bias toward unhealthy",
                "recommended": False
            },
            "category_based": {
                "description": "Estimates based on product category (requires 'category' field)",
                "recommended": False
            },
            "zero": {
                "description": "Sets missing values to zero (not recommended)",
                "recommended": False
            }
        },
        "default_method": "smart"
    })

@app.route('/nutrition-defaults', methods=['GET'])
def get_nutrition_defaults():
    """Get default values used for imputation"""
    return jsonify({
        "defaults": NUTRITION_DEFAULTS,
        "note": "These are fallback values used when nutritional information is missing"
    })

# Keep existing endpoints
@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        "message": "🔬 NutriSafe API - Enhanced Nutrition Risk Assessment",
        "version": "2.0.0",
        "status": "active",
        "features": [
            "Smart missing value imputation",
            "Confidence adjustment for incomplete data",
            "Multiple imputation strategies",
            "Data quality reporting"
        ],
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/predict": "Nutrition risk prediction with missing value handling (POST)",
            "/batch-predict": "Batch predictions (POST)",
            "/features": "Get required input features",
            "/imputation-methods": "Get available imputation methods",
            "/nutrition-defaults": "Get default values for missing nutrients"
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Get required input features with enhanced information"""
    return jsonify({
        "required_features": feature_names,
        "feature_descriptions": {k: v["description"] for k, v in NUTRITION_DEFAULTS.items()},
        "missing_value_handling": "Fields can be omitted or set to null/empty - the API will intelligently impute missing values",
        "example_input": {
            "caloric_value": 250,
            "fat": 10,
            "saturated_fat": 3,
            "sugars": 15,
            "sodium": 0.5,
            "protein": 8,
            "vitamin_a": None,  # Example of missing value
            "vitamin_c": "",    # Example of empty value
            "iron": 2,
            "calcium": 100,
            "imputation_method": "smart"  # Optional parameter
        }
    })

if __name__ == '__main__':
    if not load_model_components():
        logger.error("Failed to load model components. Exiting...")
        exit(1)
    
    logger.info("🚀 Starting Enhanced NutriSafe API...")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port) 
