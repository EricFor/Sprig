from flask import Flask, request, jsonify
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
import base64
import io
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY", "PMXJsunsSTtzwaHuRPQX")
)

MODEL_ID = "refrigerator-food/3"

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
            import json
            preferences = json.loads(request.form['preferences'])
        
        # Read image file
        image_bytes = image_file.read()
        
        # Convert to PIL Image for processing
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run inference with Roboflow
        result = CLIENT.infer(image, model_id=MODEL_ID)
        
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
        
        # Generate mock recipes based on detected ingredients
        # In a real app, you'd use a recipe API or database
        recipes = generate_recipes(ingredients, preferences)
        
        # Extract missing ingredients from recipes
        detected_ingredient_names = [ing['name'].lower() for ing in ingredients]
        missing_ingredients = []
        for recipe in recipes:
            for missing in recipe.get('missingIngredients', []):
                if missing not in missing_ingredients:
                    missing_ingredients.append(missing)
        
        # Generate shopping suggestions (mock data for now)
        shopping_suggestions = generate_shopping_suggestions(missing_ingredients)
        
        return jsonify({
            "ingredients": ingredients,
            "recipes": recipes,
            "missingIngredients": missing_ingredients,
            "shoppingSuggestions": shopping_suggestions
        }), 200
        
    except Exception as e:
        print(f"Error analyzing fridge: {str(e)}")
        return jsonify({"error": f"Failed to analyze image: {str(e)}"}), 500

def generate_recipes(ingredients, preferences):
    """
    Generate recipe suggestions based on detected ingredients
    This is a simplified version - in production, use a recipe API
    """
    ingredient_names = [ing['name'].lower() for ing in ingredients]
    
    # Recipe database (simplified)
    all_recipes = [
        {
            "name": "Chicken Stir-Fry",
            "description": "A quick and healthy stir-fry with your available ingredients",
            "prepTime": "15 min",
            "cookTime": "20 min",
            "missingIngredients": ["Olive Oil"],
            "tags": ["high-protein", "gluten-free"],
            "requiredIngredients": ["chicken", "bell pepper", "onion"]
        },
        {
            "name": "Mediterranean Quinoa Bowl",
            "description": "Fresh and vibrant bowl with quinoa, vegetables, and herbs",
            "prepTime": "10 min",
            "cookTime": "20 min",
            "missingIngredients": ["Quinoa", "Olive Oil"],
            "tags": ["vegetarian", "gluten-free", "high-protein"],
            "requiredIngredients": ["tomato", "cucumber", "onion"]
        },
        {
            "name": "Stuffed Bell Peppers",
            "description": "Hearty bell peppers stuffed with chicken and vegetables",
            "prepTime": "20 min",
            "cookTime": "45 min",
            "missingIngredients": ["Olive Oil"],
            "tags": ["high-protein", "gluten-free"],
            "requiredIngredients": ["bell pepper", "chicken", "onion"]
        },
        {
            "name": "Black Bean and Mushroom Tacos",
            "description": "Flavorful vegetarian tacos with black beans and mushrooms",
            "prepTime": "15 min",
            "cookTime": "25 min",
            "missingIngredients": ["Black Beans", "Olive Oil"],
            "tags": ["vegetarian", "vegan-option", "spicy"],
            "requiredIngredients": ["mushroom", "onion"]
        },
        {
            "name": "Caprese Salad",
            "description": "Classic Italian salad with fresh tomatoes, mozzarella, and basil",
            "prepTime": "10 min",
            "cookTime": "0 min",
            "missingIngredients": ["Mozzarella", "Basil", "Olive Oil"],
            "tags": ["vegetarian", "gluten-free", "low-carb"],
            "requiredIngredients": ["tomato"]
        },
        {
            "name": "Vegetable Stir-Fry",
            "description": "Colorful mix of fresh vegetables in a savory sauce",
            "prepTime": "15 min",
            "cookTime": "15 min",
            "missingIngredients": ["Soy Sauce", "Olive Oil"],
            "tags": ["vegetarian", "vegan-option", "gluten-free"],
            "requiredIngredients": ["bell pepper", "onion", "mushroom"]
        }
    ]
    
    # Filter recipes based on available ingredients
    matching_recipes = []
    for recipe in all_recipes:
        required = recipe.get("requiredIngredients", [])
        # Check if at least one required ingredient is available
        if any(req in ingredient_names for req in required):
            matching_recipes.append(recipe)
    
    # Apply preference filters
    filtered_recipes = matching_recipes
    if preferences.get('vegan') or preferences.get('vegetarian'):
        filtered_recipes = [r for r in filtered_recipes if 
                          'vegetarian' in r['tags'] or 'vegan-option' in r['tags']]
    if preferences.get('lowCarb'):
        filtered_recipes = [r for r in filtered_recipes if 
                          'low-carb' in r['tags']]
    if preferences.get('glutenFree'):
        filtered_recipes = [r for r in filtered_recipes if 
                          'gluten-free' in r['tags']]
    
    # Remove requiredIngredients from response (internal use only)
    for recipe in filtered_recipes:
        recipe.pop('requiredIngredients', None)
    
    return filtered_recipes[:4]  # Return top 4 matches

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

