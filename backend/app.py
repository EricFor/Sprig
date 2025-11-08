from flask import Flask, request, jsonify
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
import base64
import io
from PIL import Image
import os
import google.generativeai as genai
import json
import requests
from typing import Optional, Any
import cv2
import numpy as np
import random

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY", "PMXJsunsSTtzwaHuRPQX")
)

MODEL_ID = "refrigerator-food/3"

# Initialize Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAJL3YKnXY5wzfyguhG3GV3z9ZbezDNVNo")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "google/gemini-2.5-flash")

# Extract model name from "google/gemini-2.5-flash" format to "gemini-2.5-flash"
GEMINI_MODEL_NAME = GEMINI_MODEL.replace("google/", "") if "/" in GEMINI_MODEL else GEMINI_MODEL
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"

# Also configure the SDK as fallback
genai.configure(api_key=GEMINI_API_KEY)

def preprocess_image(image_pil):
    """
    Preprocess image to match dataset style with noise and cutouts
    Similar to the sample image with heavy noise and black rectangular occlusions
    """
    # Convert PIL to OpenCV format
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = image_cv.shape[:2]
    
    # Add heavy noise (similar to sample image)
    noise_intensity = 0.15
    noise = np.random.normal(0, noise_intensity * 255, image_cv.shape).astype(np.float32)
    noisy_image = image_cv.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    # Add black rectangular cutouts (occlusions)
    # 10x smaller by area means ~3.16x smaller dimensions (sqrt(10) ≈ 3.16)
    num_cutouts = random.randint(8, 15)  # Random number of cutouts
    cutout_size_range = (13, 38)  # Size range for cutouts (10x smaller area: 126/3.16 to 379/3.16)
    
    for _ in range(num_cutouts):
        # Random position
        x = random.randint(0, w - cutout_size_range[1])
        y = random.randint(0, h - cutout_size_range[1])
        
        # Random size within range
        cutout_w = random.randint(cutout_size_range[0], cutout_size_range[1])
        cutout_h = random.randint(cutout_size_range[0], cutout_size_range[1])
        
        # Ensure cutout doesn't go out of bounds
        x = min(x, w - cutout_w)
        y = min(y, h - cutout_h)
        
        # Draw black rectangle (cutout)
        noisy_image[y:y+cutout_h, x:x+cutout_w] = 0
    
    # Convert back to PIL
    preprocessed_pil = Image.fromarray(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
    
    return preprocessed_pil, noisy_image

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information"""
    return jsonify({
        "message": "EcoFridge API",
        "version": "1.0.0",
        "endpoints": {
            "/health": {
                "method": "GET",
                "description": "Health check endpoint"
            },
            "/api/analyze-fridge": {
                "method": "POST",
                "description": "Analyze fridge image and generate recipes",
                "required": ["image (file)"],
                "optional": ["preferences (JSON)"]
            }
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/analyze-fridge', methods=['POST'])
def analyze_fridge():
    """
    Analyze fridge image using Roboflow model
    Expects: multipart/form-data with 'image' file and optional 'preferences' JSON
    """
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Get preferences if provided
        preferences = {}
        if 'preferences' in request.form:
            preferences = json.loads(request.form['preferences'])
        
        # Read image file
        image_bytes = image_file.read()
        
        # Convert to PIL Image for processing
        original_image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image to match dataset style (noise + cutouts)
        print("Preprocessing image with noise and cutouts...")
        preprocessed_image_pil, preprocessed_image_cv = preprocess_image(original_image)
        
        # Encode preprocessed image as base64 for frontend display
        _, buffer = cv2.imencode('.jpg', preprocessed_image_cv, [cv2.IMWRITE_JPEG_QUALITY, 85])
        preprocessed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        preprocessed_image_base64 = f"data:image/jpeg;base64,{preprocessed_image_base64}"
        
        print("Image preprocessing complete. Using preprocessed image for inference.")
        
        # Run inference with Roboflow on preprocessed image
        result = CLIENT.infer(preprocessed_image_pil, model_id=MODEL_ID)
        
        # Process results to extract detected ingredients
        ingredients = []
        if result and 'predictions' in result:
            # Extract unique ingredient names with confidence scores
            seen_ingredients = {}
            for prediction in result['predictions']:
                class_name = prediction.get('class', '').strip()
                confidence = prediction.get('confidence', 0)
                
                if class_name:
                    # If we've seen this ingredient, keep the one with higher confidence
                    if class_name not in seen_ingredients or confidence > seen_ingredients[class_name]['confidence']:
                        seen_ingredients[class_name] = {
                            'name': class_name,
                            'confidence': confidence
                        }
            
            # Convert to list and sort by confidence
            ingredients = list(seen_ingredients.values())
            ingredients.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Generate recipes using Gemini AI based on detected ingredients
        print(f"Generating recipes for {len(ingredients)} ingredients")
        recipes = generate_recipes(ingredients, preferences)
        print(f"Generated {len(recipes)} recipes")
        
        # Extract missing ingredients from recipes
        detected_ingredient_names = [ing['name'].lower() for ing in ingredients]
        missing_ingredients = []
        for recipe in recipes:
            for missing in recipe.get('missingIngredients', []):
                if missing not in missing_ingredients:
                    missing_ingredients.append(missing)
        
        # Generate shopping suggestions (mock data for now)
        shopping_suggestions = generate_shopping_suggestions(missing_ingredients)
        
        print(f"Returning: {len(ingredients)} ingredients, {len(recipes)} recipes, {len(missing_ingredients)} missing ingredients")
        
        return jsonify({
            "ingredients": ingredients,
            "recipes": recipes,
            "missingIngredients": missing_ingredients,
            "shoppingSuggestions": shopping_suggestions,
            "preprocessedImage": preprocessed_image_base64  # For debugging
        }), 200
        
    except Exception as e:
        print(f"Error analyzing fridge: {str(e)}")
        return jsonify({"error": f"Failed to analyze image: {str(e)}"}), 500

def generate_recipes(ingredients, preferences):
    """
    Generate recipe suggestions using Google Gemini AI based on detected ingredients
    """
    try:
        # Extract ingredient names
        ingredient_names = [ing['name'] for ing in ingredients]
        
        if not ingredient_names:
            print("No ingredients provided, returning empty recipes")
            return []
        
        print(f"Starting recipe generation for ingredients: {ingredient_names}")
        
        # Build preferences string
        pref_strings = []
        if preferences.get('vegan'):
            pref_strings.append("vegan")
        if preferences.get('vegetarian'):
            pref_strings.append("vegetarian")
        if preferences.get('spicy'):
            pref_strings.append("spicy")
        if preferences.get('lowCarb'):
            pref_strings.append("low-carb")
        if preferences.get('glutenFree'):
            pref_strings.append("gluten-free")
        if preferences.get('dairyFree'):
            pref_strings.append("dairy-free")
        
        preferences_text = ", ".join(pref_strings) if pref_strings else "no specific dietary restrictions"
        
        # Create prompt for Gemini
        prompt = f"""Generate 4 recipe recommendations based on these available ingredients: {', '.join(ingredient_names)}.

User preferences: {preferences_text}

For each recipe, provide:
1. A creative and appealing recipe name
2. A brief description (1-2 sentences)
3. Prep time in minutes
4. Cook time in minutes
5. List of available ingredients used from the provided list: {', '.join(ingredient_names)}
6. List of missing ingredients needed (common pantry items like oil, salt, pepper can be assumed)
7. Relevant tags (e.g., "vegetarian", "vegan-option", "high-protein", "gluten-free", "low-carb", "spicy")

Return ONLY a valid JSON array with this exact structure (no markdown, no code blocks, just pure JSON):
[
  {{
    "name": "Recipe Name",
    "description": "Brief description",
    "prepTime": "X min",
    "cookTime": "X min",
    "availableIngredients": ["Ingredient1", "Ingredient2"],
    "missingIngredients": ["Ingredient1", "Ingredient2"],
    "tags": ["tag1", "tag2"]
  }}
]

Make sure the recipes are practical, use the available ingredients creatively, and respect the user's dietary preferences. The availableIngredients should be a subset of the provided ingredients list."""

        # Call Gemini API using direct HTTP request
        print(f"Calling Gemini API with {len(ingredient_names)} ingredients")
        print(f"Using model: {GEMINI_MODEL_NAME} via URL: {GEMINI_URL}")
        
        try:
            # Make direct HTTP request to Gemini API
            headers = {
                'Content-Type': 'application/json',
            }
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }
            
            response = requests.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            print(f"Gemini API call successful, status: {response.status_code}")
            
            # Extract text from response
            response_text = None
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                candidate = response_data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    if len(candidate['content']['parts']) > 0:
                        response_text = candidate['content']['parts'][0].get('text', '').strip()
                        print("Extracted text from API response")
                    else:
                        print("No parts in candidate content")
                else:
                    print(f"Candidate structure: {candidate.keys()}")
            else:
                print(f"Unexpected response format: {response_data}")
                return []
            
            if not response_text:
                print("No text extracted from Gemini response")
                return []
                
        except requests.exceptions.RequestException as api_error:
            print(f"Gemini API HTTP request failed: {str(api_error)}")
            # Fallback to SDK method
            print("Falling back to SDK method...")
            try:
                model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                response = model.generate_content(prompt)
                if hasattr(response, 'text'):
                    response_text = response.text.strip()
                elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        if len(candidate.content.parts) > 0:
                            response_text = candidate.content.parts[0].text.strip()
                if not response_text:
                    return []
            except Exception as sdk_error:
                print(f"SDK fallback also failed: {str(sdk_error)}")
                import traceback
                traceback.print_exc()
                return []
        except Exception as api_error:
            print(f"Gemini API call failed: {str(api_error)}")
            import traceback
            traceback.print_exc()
            return []
        
        print(f"Gemini raw response: {response_text[:200]}...")  # Log first 200 chars
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON response
        recipes = json.loads(response_text)
        
        # Ensure it's a list
        if not isinstance(recipes, list):
            recipes = [recipes]
        
        print(f"Successfully parsed {len(recipes)} recipes from Gemini")
        
        # Limit to 4 recipes
        return recipes[:4]
        
    except json.JSONDecodeError as e:
        print(f"Error parsing Gemini JSON response: {e}")
        print(f"Response was: {response_text if 'response_text' in locals() else 'N/A'}")
        import traceback
        traceback.print_exc()
        # Fallback to empty list
        return []
    except Exception as e:
        print(f"Error generating recipes with Gemini: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fallback to empty list
        return []

def generate_shopping_suggestions(missing_ingredients):
    """
    Generate shopping suggestions with eco-scores
    This is mock data - in production, integrate with store APIs
    """
    suggestions = []
    
    for ingredient in missing_ingredients:
        stores = [
            {
                "name": "Local Farmers Market",
                "distance": "0.5 mi",
                "ecoScore": 95,
                "sustainability": "Excellent",
                "rating": "Local sourcing, zero-waste packaging",
                "price": "$8.99"
            },
            {
                "name": "Whole Foods Market",
                "distance": "2.1 mi",
                "ecoScore": 82,
                "sustainability": "Very Good",
                "rating": "Organic options, sustainable packaging",
                "price": "$12.99"
            },
            {
                "name": "Trader Joe's",
                "distance": "3.5 mi",
                "ecoScore": 75,
                "sustainability": "Good",
                "rating": "Sustainable sourcing, recyclable packaging",
                "price": "$9.99"
            }
        ]
        
        suggestions.append({
            "ingredient": ingredient,
            "stores": stores
        })
    
    return suggestions

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

