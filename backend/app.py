from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import base64
import io
import os
import json
import time
import logging
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image
import cv2
import numpy as np
from openai import OpenAI
import re
import traceback
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from utils.dedalus_utils import run_ai_query

# Load environment variables from .env file

env_path = load_dotenv(dotenv_path = find_dotenv(), override=True)


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dedalus Labs API client 
DEDALUS_API_KEY = os.getenv("DEDALUS_API_KEY")
DEDALUS_BASE_URL = os.getenv("DEDALUS_BASE_URL", "https://api.dedaluslabs.ai/v1")

if not DEDALUS_API_KEY:
    raise ValueError("DEDALUS_API_KEY environment variable is required. Get your API key from Dedalus Labs.")

# Initialize OpenAI-compatible client with Dedalus Labs endpoint
client = OpenAI(
    api_key=DEDALUS_API_KEY,
    base_url=DEDALUS_BASE_URL
)

# Model configuration (Dedalus Labs model names)
PRIMARY_MODEL = os.getenv("DEDALUS_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o"))
FALLBACK_MODEL = os.getenv("DEDALUS_MODEL_FALLBACK", os.getenv("OPENAI_MODEL_FALLBACK", "gpt-4o-mini"))
USE_FALLBACK = os.getenv("USE_FALLBACK_MODEL", "false").lower() == "true"

# Default confidence threshold
DEFAULT_MIN_CONFIDENCE = 0.50
DEFAULT_TOP_K = None  # No limit by default

# Ingredient aliases removed to preserve ingredient differentiation

# Deny list for non-ingredient items
NON_INGREDIENT_DENY_LIST = {
    "container", "bag", "package", "packaging", "box", "bottle", "jar", "can",
    "utensil", "fork", "knife", "spoon", "plate", "bowl", "cup", "glass",
    "food", "stuff", "thing", "item", "object", "product", "label", "text",
    "shelf", "drawer", "door", "handle", "refrigerator", "fridge", "light",
    "plastic", "metal", "paper", "cardboard", "foil", "wrap", "lid", "cap",
}


def validate_image_quality(image_pil: Image.Image) -> Tuple[bool, Optional[str]]:
    """
    Validate image quality before processing.
    Returns (is_valid, error_message)
    """
    # Check resolution
    width, height = image_pil.size
    min_dimension = 320
    if width < min_dimension or height < min_dimension:
        return False, f"Image resolution too low ({width}x{height}). Please use at least {min_dimension}x{min_dimension} pixels."
    
    # Check brightness
    img_array = np.array(image_pil.convert('L'))
    mean_brightness = np.mean(img_array)
    
    if mean_brightness < 30:
        return False, "Image too dark. Please ensure good lighting when taking the photo."
    if mean_brightness > 225:
        return False, "Image too bright. Please reduce glare or adjust lighting."
    
    # Check blur (Laplacian variance)
    gray = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < 100:
        return False, "Image too blurry. Please take a clearer photo with better focus."
    
    return True, None

def optimize_image_size(image_pil: Image.Image, max_dimension: int = 1024) -> Image.Image:
    """
    Resize image to optimal size for inference while maintaining aspect ratio.
    """
    width, height = image_pil.size
    max_size = max(width, height)
    
    if max_size > max_dimension:
        # Maintain aspect ratio
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        
        return image_pil.resize((new_width, new_height), Image.LANCZOS)
    
    return image_pil

def image_to_base64(image_pil: Image.Image, format: str = "JPEG", quality: int = 90) -> str:
    """
    Convert PIL Image to base64 string for API.
    Saves as JPEG with quality 85-92% (default 90%) for optimal file size/quality balance.
    """
    buffer = io.BytesIO()
    # Save with optimized quality (85-92% range)
    quality = max(85, min(92, quality))
    image_pil.save(buffer, format=format, quality=quality, optimize=True)
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

def classify_image_with_openai(image_base64: str, model: str = PRIMARY_MODEL) -> Dict[str, Any]:
    """
    Classify fridge image using Dedalus Labs API 
    """
    system_message = """You are an expert at identifying food ingredients in refrigerator images.

Your task is to identify ONLY actual food ingredients. Do not include any non-food items.

Rules:
1. Return ONLY a valid JSON object with this exact structure:
   {
     "ingredients": [
       {"name": "ingredient_name", "confidence": "confidence_score_between_0.00_and_1.00"}
     ],
     "notes": "optional brief context"
   }

2. Exclude:
   - Brand names (e.g., "Coca-Cola", "Heinz")
   - bags, packaging, utensils
   - Vague labels like "food", "stuff", "things"
   - Any item you are less than 0.50 confident about
   - Any non-food items

3. Only include ingredients you can clearly identify with medium-high confidence (≥0.50).

4. Use common ingredient names (e.g., "tomato" not "ripe red tomato", "chicken" not "organic chicken breast").

5. Return ingredients as singular nouns when possible (e.g., "egg" not "eggs", "tomato" not "tomatoes").

6. Do not include confidence scores below 0.50 confidence score.

7. Return ONLY valid JSON, no markdown, no blank spaces, no code blocks, no explanations outside the JSON structure.

8. Return a continuous confidence score (increments of 0.01) for each ingredient from 0.00 to 1.00, no percentage signs or other symbols. Include only two decimal places.

9. For any packaged ingredients, use visible labels to identify the ingredient. Do not include the brand name in the ingredient name."""



    user_message = """Analyze this refrigerator image and identify all visible food ingredients.

Return ONLY a JSON object with the structure specified in the system message.
Include only ingredients you can identify with high confidence (≥0.50)."""

    try:
        # Note: response_format may not be supported for vision models in all cases
        # We'll try with it first, and parse JSON from text if needed
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_message},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"
                        }}
                    ]}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for consistent outputs
                max_tokens=1000
            )
        except Exception as format_error:
            # If response_format fails, try without it and parse JSON from text
            logger.warning(f"Response format not supported, trying without it: {format_error}")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_message},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"
                        }}
                    ]}
                ],
                temperature=0.1,
                max_tokens=1000
            )
        
        content = response.choices[0].message.content
        
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        result = json.loads(content)
        
        # Log token usage
        if hasattr(response, 'usage'):
            logger.info(f"Tokens used - Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")
        
        return result
    
    except Exception as e:
        logger.error(f"Dedalus Labs API error: {str(e)}")
        raise

def normalize_ingredient_name(name: str) -> str:
    """
    Normalize ingredient name: minimal processing to preserve differentiation.
    Only does basic cleaning: lowercase and strip whitespace.
    No aliasing, singularization, or adjective removal to preserve ingredient variants.
    """
    # Only do minimal normalization: lowercase and strip
    # This preserves ingredient differentiation (e.g., "cherry tomato" vs "tomato")
    normalized = name.lower().strip()
    
    # Remove special characters except spaces and hyphens
    normalized = re.sub(r'[^\w\s-]', '', normalized)
    
    return normalized.strip()

def filter_non_ingredients(ingredient_name: str) -> bool:
    """
    Check if ingredient name should be filtered out (non-ingredient items).
    Returns True if should be kept, False if should be filtered out.
    """
    normalized = ingredient_name.lower()
    
    # Check deny list
    if normalized in NON_INGREDIENT_DENY_LIST:
        return False
    
    # Check if any word in the name is in deny list
    words = normalized.split()
    for word in words:
        if word in NON_INGREDIENT_DENY_LIST:
            return False
    
    return True

def post_process_ingredients(raw_ingredients: List[Dict[str, Any]], min_confidence: float = DEFAULT_MIN_CONFIDENCE, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Post-process ingredients: filter, deduplicate, apply confidence threshold.
    """
    processed = []
    seen_normalized = {}  # Map normalized name to best ingredient
    
    for ingredient in raw_ingredients:
        name = ingredient.get("name", "").strip()
        confidence = float(ingredient.get("confidence", 0.0))
        
        # Skip if confidence too low
        if confidence < min_confidence:
            continue
        
        # Skip if name is empty
        if not name:
            continue
        
        # Normalize name
        normalized_name = normalize_ingredient_name(name)
        
        # Filter out non-ingredients
        if not filter_non_ingredients(normalized_name):
            continue
        
        # Deduplicate: keep highest confidence, but preserve original name
        if normalized_name not in seen_normalized:
            seen_normalized[normalized_name] = {
                "name": name,  # Keep original name for differentiation
                "confidence": confidence
            }
        else:
            # Update if this has higher confidence
            if confidence > seen_normalized[normalized_name]["confidence"]:
                seen_normalized[normalized_name] = {
                    "name": name,  # Keep original name for differentiation
                    "confidence": confidence
                }
    
    # Convert to list and sort by confidence
    processed = list(seen_normalized.values())
    processed.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Apply top_k limit
    if top_k and top_k > 0:
        processed = processed[:top_k]
    
    return processed

def merge_ingredient_lists(ingredient_lists: List[List[Dict[str, Any]]], min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> List[Dict[str, Any]]:
    """
    Merge multiple ingredient lists from multiple images, deduplicating by normalized name.
    """
    merged = {}
    
    for ingredient_list in ingredient_lists:
        for ingredient in ingredient_list:
            name = ingredient.get("name", "").strip()
            confidence = float(ingredient.get("confidence", 0.0))
            
            # Skip if confidence too low
            if confidence < min_confidence:
                continue
            
            # Normalize name
            normalized_name = normalize_ingredient_name(name)
            
            # Filter out non-ingredients
            if not filter_non_ingredients(normalized_name):
                continue
            
            # Deduplicate: keep highest confidence, but preserve original name
            if normalized_name not in merged:
                merged[normalized_name] = {
                    "name": name,  # Keep original name for differentiation
                    "confidence": confidence
                }
            else:
                # Update if this has higher confidence
                if confidence > merged[normalized_name]["confidence"]:
                    merged[normalized_name] = {
                        "name": name,  # Keep original name for differentiation
                        "confidence": confidence
                    }
    
    # Convert to list and sort by confidence
    result = list(merged.values())
    result.sort(key=lambda x: x["confidence"], reverse=True)
    
    return result

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/classify-fridge', methods=['POST'])
def classify_fridge():
    """
    Classify fridge image(s) using Dedalus Labs API 
    
    Accepts:
    - One or more images (files with key 'images' or 'image')
    - Optional: min_conf (float, default 0.50)
    - Optional: top_k (int, default None)
    - Optional: image_urls (list of URLs, for remote images)
    
    Returns:
    - ingredients: List of {name, confidence}
    - notes: Optional context
    - diagnostics: Processing metadata
    """
    start_time = time.time()
    
    try:
        # Get parameters
        min_conf = float(request.form.get('min_conf', DEFAULT_MIN_CONFIDENCE))
        top_k = request.form.get('top_k')
        top_k = int(top_k) if top_k else None
        
        # Get model preference
        model = FALLBACK_MODEL if USE_FALLBACK else PRIMARY_MODEL
        
        # Get images from files or URLs
        images = []
        
        # Handle file uploads
        if 'images' in request.files:
            files = request.files.getlist('images')
            images.extend([f for f in files if f.filename])
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename:
                images.append(file)
        
        # Handle image URLs (if provided)
        image_urls = request.form.getlist('image_urls') if 'image_urls' in request.form else []
        
        if not images and not image_urls:
            return jsonify({"error": "No images provided. Please provide image files or URLs."}), 400
        
        # Process each image
        all_ingredients = []
        all_notes = []
        diagnostics = {
            "images_processed": 0,
            "images_failed": 0,
            "processing_times": [],
            "model_used": model
        }
        
        # Process file uploads
        for image_file in images:
            try:
                img_start_time = time.time()
                
                # Read image
                image_bytes = image_file.read()
                image_pil = Image.open(io.BytesIO(image_bytes))
                
                # Validate image quality
                is_valid, error_msg = validate_image_quality(image_pil)
                if not is_valid:
                    logger.warning(f"Image validation failed: {error_msg}")
                    diagnostics["images_failed"] += 1
                    continue
                
                # Optimize image size
                image_pil = optimize_image_size(image_pil, max_dimension=1024)
                
                # Convert to base64
                image_base64 = image_to_base64(image_pil)
                
                # Classify with Dedalus Labs API
                result = classify_image_with_openai(image_base64, model=model)
                
                # Extract ingredients and notes
                ingredients = result.get("ingredients", [])
                notes = result.get("notes", "")
                
                if notes:
                    all_notes.append(notes)
                
                all_ingredients.append(ingredients)
                diagnostics["images_processed"] += 1
                
                img_processing_time = time.time() - img_start_time
                diagnostics["processing_times"].append(img_processing_time)
                
                logger.info(f"Processed image in {img_processing_time:.2f}s, found {len(ingredients)} ingredients")
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                diagnostics["images_failed"] += 1
                continue
        
        # Process image URLs (if any)
        # Note: This would require fetching URLs, validating TTL, etc.
        # For now, we'll skip URL processing in the initial implementation
        # but leave the structure for future implementation
        
        if not all_ingredients:
            return jsonify({
                "error": "No images were successfully processed.",
                "diagnostics": diagnostics
            }), 400
        
        # Merge ingredients from multiple images
        if len(all_ingredients) > 1:
            final_ingredients = merge_ingredient_lists(all_ingredients, min_confidence=min_conf)
        else:
            final_ingredients = post_process_ingredients(all_ingredients[0], min_confidence=min_conf, top_k=top_k)
        
        # Combine notes
        combined_notes = " ".join(all_notes).strip() if all_notes else None
        
        total_time = time.time() - start_time
        diagnostics["total_processing_time"] = total_time
        diagnostics["ingredients_returned"] = len(final_ingredients)
        diagnostics["api_provider"] = "Dedalus Labs"
        diagnostics["base_url"] = DEDALUS_BASE_URL
        
        logger.info(f"Classification complete via Dedalus Labs: {len(final_ingredients)} ingredients in {total_time:.2f}s")
        
        response = {
            "ingredients": final_ingredients,
            "diagnostics": diagnostics
        }
        
        if combined_notes:
            response["notes"] = combined_notes
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in classify_fridge: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to classify image: {str(e)}"}), 500

# Keep the old endpoint for backward compatibility, but route to new system
@app.route('/api/recalibrate-recipes', methods=['POST'])
def recalibrate_recipes():
    """
    Generate recipes from existing ingredients list (without image).
    Accepts JSON with ingredients list and preferences.
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Get ingredients list
        ingredients_list = data.get('ingredients', [])
        if not ingredients_list:
            return jsonify({"error": "No ingredients provided"}), 400
        
        # Get preferences
        preferences = data.get('preferences', {})
        
        # Convert ingredient names to ingredient objects with confidence
        ingredients = []
        for ing_name in ingredients_list:
            if isinstance(ing_name, str):
                ingredients.append({
                    "name": ing_name,
                    "confidence": 0.95  # Default confidence for user-provided ingredients
                })
            else:
                ingredients.append(ing_name)
        
        # Generate recipes using Gemini
        recipes = generate_recipes(ingredients, preferences)
        
        # Extract missing ingredients (optimized: O(n) using set instead of O(n²))
        missing_ingredients_set = set()
        for recipe in recipes:
            for missing in recipe.get('missingIngredients', []):
                missing_ingredients_set.add(missing)
        missing_ingredients = list(missing_ingredients_set)
        
        # Generate shopping suggestions
        shopping_suggestions = generate_shopping_suggestions(missing_ingredients)
        
        return jsonify({
            "ingredients": ingredients,
            "recipes": recipes,
            "missingIngredients": missing_ingredients,
            "shoppingSuggestions": shopping_suggestions
        }), 200
        
    except Exception as e:
        logger.error(f"Error in recalibrate_recipes: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to generate recipes: {str(e)}"}), 500

def emit_progress(progress_callback, progress: int, message: str):
    """Helper function to emit progress updates"""
    progress_data = {
        "progress": progress,
        "message": message
    }
    progress_callback(progress_data)

@app.route('/api/analyze-fridge', methods=['POST'])
def analyze_fridge():
    """
    Analyzes fridge and generates recipes with progress updates.
    Uses Server-Sent Events (SSE) to stream progress updates.
    """
    def generate():
        try:
            # Get preferences if provided
            preferences = {}
            if 'preferences' in request.form:
                preferences = json.loads(request.form['preferences'])
            
            # Use new classification endpoint logic
            min_conf = float(request.form.get('min_conf', 0.50))
            
            # Get image
            if 'image' not in request.files:
                yield f"data: {json.dumps({'error': 'No image provided', 'progress': 0})}\n\n"
                return
            
            image_file = request.files['image']
            if image_file.filename == '':
                yield f"data: {json.dumps({'error': 'No image selected', 'progress': 0})}\n\n"
                return
            
            # Progress: 5% - Reading image
            yield f"data: {json.dumps({'progress': 5, 'message': 'Reading image...'})}\n\n"
            
            # Process image through new classification system
            image_bytes = image_file.read()
            image_pil = Image.open(io.BytesIO(image_bytes))
            
            # Progress: 10% - Validating image quality
            yield f"data: {json.dumps({'progress': 10, 'message': 'Validating image quality...'})}\n\n"
            
            # Validate and optimize
            is_valid, error_msg = validate_image_quality(image_pil)
            if not is_valid:
                yield f"data: {json.dumps({'error': error_msg, 'progress': 0})}\n\n"
                return
            
            # Progress: 20% - Optimizing image
            yield f"data: {json.dumps({'progress': 20, 'message': 'Optimizing image for analysis...'})}\n\n"
            
            image_pil = optimize_image_size(image_pil, max_dimension=1024)
            image_base64 = image_to_base64(image_pil, quality=90)
            
            # Progress: 30% - Analyzing image with AI
            yield f"data: {json.dumps({'progress': 30, 'message': 'Analyzing image with AI vision model...'})}\n\n"
            
            # Classify using Dedalus Labs API
            model = FALLBACK_MODEL if USE_FALLBACK else PRIMARY_MODEL
            result = classify_image_with_openai(image_base64, model=model)
            raw_ingredients = result.get("ingredients", [])
            
            # Progress: 60% - Processing ingredients
            yield f"data: {json.dumps({'progress': 60, 'message': f'Processing {len(raw_ingredients)} detected ingredients...'})}\n\n"
            
            # Post-process
            ingredients = post_process_ingredients(raw_ingredients, min_confidence=min_conf)
            
            # Progress: 70% - Generating recipes
            yield f"data: {json.dumps({'progress': 70, 'message': 'Generating personalized recipes...'})}\n\n"
            
            # Generate recipes using Gemini (keep existing recipe generation)
            recipes = generate_recipes(ingredients, preferences)
            
            # Progress: 85% - Processing recipes
            yield f"data: {json.dumps({'progress': 85, 'message': f'Processing {len(recipes)} recipe recommendations...'})}\n\n"
            
            # Extract missing ingredients (optimized: O(n) using set instead of O(n²))
            missing_ingredients_set = set()
            for recipe in recipes:
                for missing in recipe.get('missingIngredients', []):
                    missing_ingredients_set.add(missing)
            missing_ingredients = list(missing_ingredients_set)
            
            # Progress: 90% - Generating shopping suggestions
            yield f"data: {json.dumps({'progress': 90, 'message': 'Generating shopping suggestions...'})}\n\n"
            
            # Generate shopping suggestions
            shopping_suggestions = generate_shopping_suggestions(missing_ingredients)
            
            # Progress: 100% - Complete
            yield f"data: {json.dumps({'progress': 100, 'message': 'Analysis complete!'})}\n\n"
            
            # Send final results
            result_data = {
                "ingredients": ingredients,
                "recipes": recipes,
                "missingIngredients": missing_ingredients,
                "shoppingSuggestions": shopping_suggestions,
                "complete": True
            }
            yield f"data: {json.dumps(result_data)}\n\n"
            
        except Exception as e:
            logger.error(f"Error in analyze_fridge: {str(e)}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': f'Failed to analyze image: {str(e)}', 'progress': 0})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

def generate_recipes(ingredients, preferences):
    """
    Generate recipe suggestions using Google Gemini AI based on detected ingredients.
    Keep existing implementation for now.
    """
    try:
        # Extract ingredient names
        ingredient_names = [ing['name'] for ing in ingredients]
        
        if not ingredient_names:
            logger.info("No ingredients provided, returning empty recipes")
            return []
        
        logger.info(f"Starting recipe generation for ingredients: {ingredient_names}")
        
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
        if preferences.get('halal'):
            pref_strings.append("halal")
        if preferences.get('kosher'):
            pref_strings.append("kosher")
        
        preferences_text = ", ".join(pref_strings) if pref_strings else "no specific dietary restrictions"
        
        # Add cuisine regions if provided
        cuisine_regions = preferences.get('cuisineRegions', [])
        cuisine_text = ""
        if cuisine_regions and len(cuisine_regions) > 0:
            cuisine_text = f"\n\nPreferred cuisine styles: {', '.join(cuisine_regions)}. Prioritize recipes from these cuisines when possible."
        
        # Use Gemini for recipe generation (optional - can be replaced with OpenAI)
        try:
            import requests
            
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            GEMINI_MODEL = os.getenv("GEMINI_MODEL", "google/gemini-2.5-flash")
            
            if not GEMINI_API_KEY:
                logger.warning("GEMINI_API_KEY not set, skipping recipe generation")
                return []
            
            # Extract model name
            GEMINI_MODEL_NAME = GEMINI_MODEL.replace("google/", "") if "/" in GEMINI_MODEL else GEMINI_MODEL
            GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
            
            # Create prompt for recipe generation
            prompt = f"""Generate 4 recipe recommendations based on these available ingredients: {', '.join(ingredient_names)}.

User dietary preferences: {preferences_text}{cuisine_text}

IMPORTANT: Prioritize recipes that can be made PRIMARILY or ENTIRELY with the available ingredients. 
- Include at least 2 recipes that require NO additional ingredients (or only common pantry staples like salt, pepper, oil)
- Include recipes that maximize use of the available ingredients
- Only suggest missing ingredients that are absolutely necessary and not common pantry items
- If cuisine preferences are specified, prioritize recipes from those cuisines

For each recipe, provide:
1. A creative and appealing recipe name
2. A brief description (1-2 sentences)
3. Prep time in minutes
4. Cook time in minutes
5. List of available ingredients used from the provided list: {', '.join(ingredient_names)}
6. List of missing ingredients needed (ONLY if there are missing ingredients beyond common pantry staples like oil, salt, pepper - if no missing ingredients, omit the "missingIngredients" field entirely or use an empty array)
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

IMPORTANT: For recipes that can be made entirely with available ingredients (or only need common pantry items), either:
- Omit the "missingIngredients" field entirely, OR
- Use an empty array: "missingIngredients": []

Only include actual missing ingredients that are NOT common pantry staples (oil, salt, pepper, water, etc.).

Make sure the recipes are practical, ensure at least 3 ingredients are used, use     the available ingredients creatively, and respect the user's dietary preferences and cuisine preferences. The availableIngredients should be a subset of the provided ingredients list.
PRIORITIZE recipes that can be made with the available ingredients with minimal or no additional purchases."""
            
            # Call Gemini API
            headers = {'Content-Type': 'application/json'}
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
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
            
            # Extract text from response
            response_text = None
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                candidate = response_data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    if len(candidate['content']['parts']) > 0:
                        response_text = candidate['content']['parts'][0].get('text', '').strip()
            
            if not response_text:
                logger.warning("No response from Gemini API")
                return []
            
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
            
            # Clean up recipes: filter out common pantry items and normalize missingIngredients
            for recipe in recipes:
                # Normalize missingIngredients field
                if 'missingIngredients' not in recipe:
                    recipe['missingIngredients'] = []
                elif not isinstance(recipe['missingIngredients'], list):
                    recipe['missingIngredients'] = []
                elif len(recipe['missingIngredients']) == 0:
                    recipe['missingIngredients'] = []
                else:
                    # Filter out common pantry items that shouldn't be considered "missing"
                    # Common pantry items (case-insensitive, word-boundary aware)
                    common_pantry_keywords = {
                        'oil', 'salt', 'pepper', 'water', 'flour', 'sugar', 'butter', 
                        'garlic', 'onion', 'vinegar', 'baking soda', 'baking powder',
                        'vanilla', 'cinnamon', 'paprika', 'oregano', 'basil', 'thyme'
                    }
                    
                    filtered_missing = []
                    for ing in recipe['missingIngredients']:
                        if not ing or not isinstance(ing, str):
                            continue
                        ing_lower = ing.lower().strip()
                        # Check if ingredient is a common pantry item
                        # Also check if any word in the ingredient matches common pantry items
                        ing_words = set(ing_lower.split())
                        is_pantry_item = False
                        
                        # Direct match
                        if ing_lower in common_pantry_keywords:
                            is_pantry_item = True
                        # Word match (e.g., "olive oil" contains "oil")
                        elif any(word in common_pantry_keywords for word in ing_words):
                            # But allow specific oils like "olive oil", "coconut oil" if they're explicitly needed
                            # Only filter generic "oil" or if it's clearly a pantry staple
                            if ing_lower in ['oil', 'vegetable oil', 'cooking oil']:
                                is_pantry_item = True
                        
                        if not is_pantry_item:
                            filtered_missing.append(ing)
                    
                    recipe['missingIngredients'] = filtered_missing if filtered_missing else []
            
            logger.info(f"Successfully generated {len(recipes)} recipes")
            return recipes[:10]  # Limit to 10 recipes
            
        except Exception as recipe_error:
            logger.error(f"Error generating recipes with Gemini: {str(recipe_error)}")
            return []
        
    except Exception as e:
        logger.error(f"Error generating recipes: {str(e)}")
        return []

def get_brands(missing_ingredients: List[str]) -> Dict[str, List[str]]:
    """
    Get brand names for each missing ingredient using AI.
    
    Args:
        missing_ingredients: List of ingredient names
        
    Returns:
        Dictionary mapping ingredient names to lists of brand names
    """
    brands_data = {}
    
    for ingredient in missing_ingredients:
        try:
            query = f"""
            1. Find 10 different brands that sell {ingredient}.
            2. Output the results in the form: [brand1, brand2, brand3, brand4, ...]
            3. Return ONLY a JSON array, no other text.
            """
            
            result = run_ai_query(query)
            
            # Try to extract JSON array from the response
            # Remove markdown code blocks if present
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            elif result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            # Try to parse as JSON
            try:
                brands = json.loads(result)
                if isinstance(brands, list):
                    brands_data[ingredient] = brands[:10]  # Limit to 10 brands
                else:
                    logger.warning(f"Expected list for {ingredient}, got {type(brands)}")
                    brands_data[ingredient] = []
            except json.JSONDecodeError:
                # Try to extract list from text
                # Look for patterns like [brand1, brand2, ...]
                match = re.search(r'\[([^\]]+)\]', result)
                if match:
                    brands_str = match.group(1)
                    brands = [b.strip().strip('"\'') for b in brands_str.split(',')]
                    brands_data[ingredient] = brands[:10]
                else:
                    logger.warning(f"Could not parse brands for {ingredient}")
                    brands_data[ingredient] = []
                    
        except Exception as e:
            logger.error(f"Error getting brands for {ingredient}: {str(e)}")
            brands_data[ingredient] = []
    
    return brands_data


def get_ecoscores(brands_data: Dict[str, List[str]]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """
    Get eco scores, environmental impact details, and reasoning for each brand using AI.
    
    Args:
        brands_data: Dictionary mapping ingredient names to lists of brand names
        
    Returns:
        Tuple of (eco_scores, environmental_impact, reasoning) where:
        - eco_scores: Dictionary mapping ingredient names to dictionaries of brand: eco_score
        - environmental_impact: Dictionary mapping ingredient names to dictionaries of brand: impact_description
        - reasoning: Dictionary mapping ingredient names to dictionaries of brand: reasoning_description
    """
    eco_scores = {}
    environmental_impact = {}
    reasoning = {}
    
    for ingredient, brands in brands_data.items():
        if not brands:
            eco_scores[ingredient] = {}
            environmental_impact[ingredient] = {}
            reasoning[ingredient] = {}
            continue
            
        try:
            brands_str = ", ".join(brands)
            query = f"""For the ingredient "{ingredient}", analyze the environmental impact of these brands: {brands_str}

For each brand, provide:
1. An environmental score from 0-100 (higher is better)
2. A detailed description of the brand's environmental impact (2-3 sentences)
3. Reasoning explaining why the brand received that specific eco score (2-3 sentences)

Consider factors such as:
- Carbon footprint and greenhouse gas emissions
- Sustainable sourcing and farming practices
- Packaging and waste reduction
- Water usage and conservation
- Biodiversity impact
- Certifications (organic, fair trade, etc.)
- Corporate sustainability initiatives

Return ONLY a valid JSON object with this exact structure (no markdown, no code blocks, no explanations):
{{
  "brand1": {{
    "score": 85,
    "environmentalImpact": "Detailed description of environmental impact...",
    "reasoning": "Explanation of why this score was assigned..."
  }},
  "brand2": {{
    "score": 72,
    "environmentalImpact": "Detailed description of environmental impact...",
    "reasoning": "Explanation of why this score was assigned..."
  }}
}}

If information is not available for a brand, use:
- score: 50
- environmentalImpact: "Limited information available about this brand's environmental practices."
- reasoning: "Default score assigned due to lack of available environmental data."

Return ONLY the JSON object, no other text or explanations."""
            
            # Try with JSON mode first (if supported by model)
            try:
                result = run_ai_query(query, json_mode=True)
            except Exception as json_mode_error:
                # If JSON mode fails, try without it
                logger.debug(f"JSON mode not supported, trying without: {str(json_mode_error)}")
                result = run_ai_query(query, json_mode=False)
            
            # Try to extract JSON object from the response
            result_clean = result.strip()
            if result_clean.startswith("```json"):
                result_clean = result_clean[7:]
            elif result_clean.startswith("```"):
                result_clean = result_clean[3:]
            if result_clean.endswith("```"):
                result_clean = result_clean[:-3]
            result_clean = result_clean.strip()
            
            # Try to find the outermost JSON object by matching braces
            brace_count = 0
            json_start = -1
            json_end = -1
            
            for i, char in enumerate(result_clean):
                if char == '{':
                    if brace_count == 0:
                        json_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and json_start != -1:
                        json_end = i
                        break
            
            if json_start != -1 and json_end != -1:
                result_clean = result_clean[json_start:json_end + 1]
            
            try:
                data = json.loads(result_clean)
                if isinstance(data, dict):
                    cleaned_scores = {}
                    cleaned_impact = {}
                    cleaned_reasoning = {}
                    
                    for brand in brands:
                        brand_data = data.get(brand, {})
                        if isinstance(brand_data, dict):
                            # Extract score
                            try:
                                score = int(float(brand_data.get("score", 50)))
                                cleaned_scores[brand] = max(0, min(100, score))  # Clamp to 0-100
                            except (ValueError, TypeError):
                                cleaned_scores[brand] = 50
                            
                            # Extract environmental impact
                            impact = brand_data.get("environmentalImpact", "Limited information available about this brand's environmental practices.")
                            cleaned_impact[brand] = str(impact).strip() if impact else "Limited information available about this brand's environmental practices."
                            
                            # Extract reasoning
                            reason = brand_data.get("reasoning", "Default score assigned due to lack of available environmental data.")
                            cleaned_reasoning[brand] = str(reason).strip() if reason else "Default score assigned due to lack of available environmental data."
                        else:
                            # Fallback if brand data is not a dict
                            cleaned_scores[brand] = 50
                            cleaned_impact[brand] = "Limited information available about this brand's environmental practices."
                            cleaned_reasoning[brand] = "Default score assigned due to lack of available environmental data."
                    
                    eco_scores[ingredient] = cleaned_scores
                    environmental_impact[ingredient] = cleaned_impact
                    reasoning[ingredient] = cleaned_reasoning
                else:
                    logger.warning(f"Expected dict for eco scores of {ingredient}, got {type(data)}")
                    eco_scores[ingredient] = {brand: 50 for brand in brands}
                    environmental_impact[ingredient] = {brand: "Limited information available about this brand's environmental practices." for brand in brands}
                    reasoning[ingredient] = {brand: "Default score assigned due to lack of available environmental data." for brand in brands}
            except json.JSONDecodeError as json_error:
                logger.warning(f"Could not parse eco scores for {ingredient}: {str(json_error)}")
                logger.debug(f"Failed to parse JSON. Response was: {result_clean[:500]}")
                eco_scores[ingredient] = {brand: 50 for brand in brands}
                environmental_impact[ingredient] = {brand: "Limited information available about this brand's environmental practices." for brand in brands}
                reasoning[ingredient] = {brand: "Default score assigned due to lack of available environmental data." for brand in brands}
                
        except Exception as e:
            logger.error(f"Error getting eco scores for {ingredient}: {str(e)}")
            eco_scores[ingredient] = {brand: 50 for brand in brands}
            environmental_impact[ingredient] = {brand: "Limited information available about this brand's environmental practices." for brand in brands}
            reasoning[ingredient] = {brand: "Default score assigned due to lack of available environmental data." for brand in brands}
    
    return eco_scores, environmental_impact, reasoning


def get_online_links_and_prices(brands_data: Dict[str, List[str]]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """
    Get online product links and prices for each brand using AI.
    Supports all major online retailers (Amazon, Walmart, Target, Whole Foods, etc.)
    
    Args:
        brands_data: Dictionary mapping ingredient names to lists of brand names
        
    Returns:
        Tuple of (product_links_map, prices_map) where:
        - product_links_map: Dictionary mapping ingredient names to dictionaries of brand: product_link
        - prices_map: Dictionary mapping ingredient names to dictionaries of brand: price
    """
    product_links_map = {}
    prices_map = {}
    
    for ingredient, brands in brands_data.items():
        if not brands:
            product_links_map[ingredient] = {}
            prices_map[ingredient] = {}
            continue
            
        try:
            brands_str = ", ".join(brands)
            query = f"""For the ingredient "{ingredient}", find REAL online product links and CURRENT prices for these brands: {brands_str}

You can search on ANY major online retailer including:
- Amazon (amazon.com)
- Walmart (walmart.com)
- Target (target.com)
- Whole Foods (wholefoodsmarket.com)
- Instacart (instacart.com)
- Kroger (kroger.com)
- Other reputable online grocery stores

CRITICAL REQUIREMENTS:
1. URLs MUST be valid, working product URLs from any reputable online retailer
2. DO NOT use placeholder URLs, search URLs (with ?q= or /s?k=), generic links, or made-up URLs
3. URLs must be direct product pages (not category pages or search results)
4. If you cannot find a real product URL, use empty string "" for that brand's link
5. Prices MUST be in format "$X.XX" (e.g., "$5.99", "$12.50", "$19.99")
6. If price is unavailable, use "$?.??"
7. Only include REAL, VERIFIABLE product URLs that would work when clicked
8. Prefer the retailer with the best price or most availability for each brand

Return ONLY a valid JSON object with this exact structure (no markdown, no code blocks, no explanations):
{{
  "links": {{
    "brand1": "https://www.retailer.com/product/...",
    "brand2": "https://www.retailer.com/product/..."
  }},
  "prices": {{
    "brand1": "$X.XX",
    "brand2": "$Y.YY"
  }}
}}

Example of CORRECT format:
{{
  "links": {{
    "Lundberg": "https://www.amazon.com/dp/B000EDF0R4",
    "Uncle Ben's": "https://www.walmart.com/ip/12345678"
  }},
  "prices": {{
    "Lundberg": "$8.99",
    "Uncle Ben's": "$4.50"
  }}
}}

Return ONLY the JSON object, no other text or explanations."""
            
            # Try with JSON mode first (if supported by model)
            try:
                result = run_ai_query(query, json_mode=True)
            except Exception as json_mode_error:
                # If JSON mode fails, try without it
                logger.debug(f"JSON mode not supported, trying without: {str(json_mode_error)}")
                result = run_ai_query(query, json_mode=False)
            
            # Log the raw response for debugging (first 500 chars)
            logger.debug(f"Raw AI response for {ingredient} (first 500 chars): {result[:500]}")
            
            # Try to extract JSON object from the response
            result_clean = result.strip()
            
            # Remove markdown code blocks
            if result_clean.startswith("```json"):
                result_clean = result_clean[7:]
            elif result_clean.startswith("```"):
                result_clean = result_clean[3:]
            if result_clean.endswith("```"):
                result_clean = result_clean[:-3]
            result_clean = result_clean.strip()
            
            # Try to extract JSON object from response (handle cases where JSON is embedded in text)
            # Method 1: Try to find the outermost JSON object by matching braces
            brace_count = 0
            json_start = -1
            json_end = -1
            
            for i, char in enumerate(result_clean):
                if char == '{':
                    if brace_count == 0:
                        json_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and json_start != -1:
                        json_end = i
                        break
            
            if json_start != -1 and json_end != -1:
                result_clean = result_clean[json_start:json_end + 1]
            
            try:
                data = json.loads(result_clean)
                if isinstance(data, dict):
                    # Extract links and prices
                    links = data.get("links", {})
                    prices = data.get("prices", {})
                    
                    # Ensure links and prices are dictionaries
                    if not isinstance(links, dict):
                        links = {}
                    if not isinstance(prices, dict):
                        prices = {}
                    
                    # Validate and clean prices
                    cleaned_prices = {}
                    for brand in brands:
                        # Get price for this brand
                        price = prices.get(brand, "$?.??")
                        if isinstance(price, str):
                            # Clean up price string
                            price_clean = price.strip()
                            # Ensure it starts with $ or is N/A
                            if not price_clean.startswith("$") and price_clean.upper() != "N/A":
                                # Try to extract price from text
                                price_match = re.search(r'\$[\d,]+\.?\d*', price_clean)
                                if price_match:
                                    price_clean = price_match.group(0)
                                else:
                                    price_clean = "$?.??"
                            # Validate price format
                            if not re.match(r'^\$[\d,]+\.?\d*$|^\$\.\?\?$|^N/A$', price_clean):
                                price_clean = "$?.??"
                            cleaned_prices[brand] = price_clean
                        else:
                            cleaned_prices[brand] = "$?.??"
                    
                    # Validate and clean links
                    cleaned_links = {}
                    for brand in brands:
                        link = links.get(brand, "")
                        if isinstance(link, str) and link.strip():
                            link_clean = link.strip()
                            # Remove quotes if present
                            link_clean = link_clean.strip('"').strip("'")
                            
                            # Validate and normalize URL
                            if link_clean:
                                # Ensure it starts with http:// or https://
                                if not link_clean.startswith(("http://", "https://")):
                                    # Try to construct proper URL
                                    if link_clean.startswith("www."):
                                        link_clean = "https://" + link_clean
                                    elif "/" in link_clean:
                                        # Try to detect domain and construct URL
                                        # Extract domain if possible (e.g., "walmart.com/product" -> "https://www.walmart.com/product")
                                        parts = link_clean.split("/")
                                        if len(parts) > 0 and "." in parts[0]:
                                            domain = parts[0]
                                            path = "/" + "/".join(parts[1:]) if len(parts) > 1 else ""
                                            link_clean = "https://www." + domain + path
                                        else:
                                            # Invalid format, skip
                                            link_clean = ""
                                    else:
                                        # Invalid format, skip
                                        link_clean = ""
                                
                                # Clean up URL - remove any trailing punctuation that might break the link
                                link_clean = link_clean.rstrip('.,;:!?')
                                
                                # Validate it's a proper product URL from a reputable online retailer
                                # Must be a product page (not a search URL or category page)
                                valid_domains = [
                                    "amazon.com", "amazon.", "walmart.com", "target.com",
                                    "wholefoodsmarket.com", "instacart.com", "kroger.com",
                                    "safeway.com", "wegmans.com", "publix.com", "costco.com",
                                    "sprouts.com", "traderjoes.com", "aldi.com"
                                ]
                                
                                is_valid_retailer = any(domain in link_clean.lower() for domain in valid_domains)
                                
                                # Reject search URLs and category pages
                                is_search_url = any(param in link_clean.lower() for param in ["/s?k=", "?q=", "/search?", "/browse?"])
                                
                                is_valid_product_url = (
                                    link_clean.startswith(("http://", "https://")) and 
                                    is_valid_retailer and
                                    not is_search_url
                                )
                                
                                if is_valid_product_url:
                                    cleaned_links[brand] = link_clean
                                else:
                                    cleaned_links[brand] = ""
                                    logger.debug(f"Invalid product URL format for {brand}: {link} (must be from a valid retailer and not a search URL)")
                            else:
                                cleaned_links[brand] = ""
                        else:
                            cleaned_links[brand] = ""
                    
                    # Log cleaned links for debugging
                    logger.debug(f"Cleaned links for {ingredient}: {cleaned_links}")
                    logger.debug(f"Cleaned prices for {ingredient}: {cleaned_prices}")
                    
                    product_links_map[ingredient] = cleaned_links
                    prices_map[ingredient] = cleaned_prices
                    
                    logger.info(f"Successfully parsed online product data for {ingredient}: {len(cleaned_links)} links, {len(cleaned_prices)} prices")
                else:
                    logger.warning(f"Expected dict for product data of {ingredient}, got {type(data)}. Response: {result_clean[:200]}")
                    product_links_map[ingredient] = {}
                    prices_map[ingredient] = {}
            except json.JSONDecodeError as json_error:
                logger.warning(f"Could not parse online product data for {ingredient}: {str(json_error)}")
                logger.warning(f"Failed to parse JSON. Attempted to parse: {result_clean[:300]}...")
                logger.warning(f"Full response (first 1000 chars): {result[:1000]}")
                # Try to extract at least partial data using regex as fallback
                try:
                    # Try to extract links and prices using regex as fallback
                    links_dict = {}
                    prices_dict = {}
                    for brand in brands:
                        # Look for brand-specific patterns in the original result
                        # Try multiple patterns to catch variations
                        link_patterns = [
                            rf'"{re.escape(brand)}"\s*:\s*"([^"]*)"',  # Simple pattern
                            rf'"{re.escape(brand)}"\s*:\s*"(https?://[^"]*)"',  # URL pattern for any retailer
                            rf'{re.escape(brand)}\s*:\s*"(https?://[^"]*)"',  # Without quotes around brand
                        ]
                        price_patterns = [
                            rf'"{re.escape(brand)}"\s*:\s*"(\$[^"]*)"',  # Simple price pattern
                            rf'"{re.escape(brand)}"\s*:\s*"([^"]*)"',  # Any string, we'll validate
                            rf'{re.escape(brand)}\s*:\s*"(\$[^"]*)"',  # Without quotes around brand
                        ]
                        
                        link_found = False
                        for pattern in link_patterns:
                            link_match = re.search(pattern, result, re.IGNORECASE)
                            if link_match:
                                extracted_link = link_match.group(1)
                                # Validate and normalize the extracted link
                                extracted_link = extracted_link.strip().strip('"').strip("'")
                                
                                # Normalize URL
                                if not extracted_link.startswith(("http://", "https://")):
                                    if extracted_link.startswith("www."):
                                        extracted_link = "https://" + extracted_link
                                    elif "/" in extracted_link:
                                        # Try to detect domain and construct URL
                                        parts = extracted_link.split("/")
                                        if len(parts) > 0 and "." in parts[0]:
                                            domain = parts[0]
                                            path = "/" + "/".join(parts[1:]) if len(parts) > 1 else ""
                                            extracted_link = "https://www." + domain + path
                                        else:
                                            extracted_link = ""
                                    else:
                                        extracted_link = ""
                                
                                # Clean up URL
                                extracted_link = extracted_link.rstrip('.,;:!?')
                                
                                # Validate it's a proper product URL from a reputable retailer
                                valid_domains = [
                                    "amazon.com", "amazon.", "walmart.com", "target.com",
                                    "wholefoodsmarket.com", "instacart.com", "kroger.com",
                                    "safeway.com", "wegmans.com", "publix.com", "costco.com",
                                    "sprouts.com", "traderjoes.com", "aldi.com"
                                ]
                                
                                is_valid_retailer = any(domain in extracted_link.lower() for domain in valid_domains)
                                is_search_url = any(param in extracted_link.lower() for param in ["/s?k=", "?q=", "/search?", "/browse?"])
                                
                                is_valid_product_url = (
                                    extracted_link.startswith(("http://", "https://")) and 
                                    is_valid_retailer and
                                    not is_search_url
                                )
                                
                                if is_valid_product_url:
                                    links_dict[brand] = extracted_link
                                    link_found = True
                                    break
                        if not link_found:
                            links_dict[brand] = ""
                        
                        price_found = False
                        for pattern in price_patterns:
                            price_match = re.search(pattern, result, re.IGNORECASE)
                            if price_match:
                                price_val = price_match.group(1)
                                # Validate it looks like a price
                                if "$" in price_val or re.search(r'\d+\.\d{2}', price_val):
                                    # Extract price if embedded in text
                                    dollar_match = re.search(r'\$[\d,]+\.?\d*', price_val)
                                    if dollar_match:
                                        prices_dict[brand] = dollar_match.group(0)
                                    else:
                                        prices_dict[brand] = "$?.??"
                                    price_found = True
                                    break
                        if not price_found:
                            prices_dict[brand] = "$?.??"
                    
                    product_links_map[ingredient] = links_dict
                    prices_map[ingredient] = prices_dict
                    logger.info(f"Partially recovered online product data for {ingredient} using regex fallback: {len(links_dict)} links, {len(prices_dict)} prices")
                except Exception as fallback_error:
                    logger.warning(f"Regex fallback also failed for {ingredient}: {str(fallback_error)}")
                    logger.debug(f"Fallback error traceback: {traceback.format_exc()}")
                    product_links_map[ingredient] = {}
                    prices_map[ingredient] = {}
                
        except Exception as e:
            logger.error(f"Error fetching online product links and prices for {ingredient}: {str(e)}")
            product_links_map[ingredient] = {}
            prices_map[ingredient] = {}
    
    return product_links_map, prices_map


def get_sustainability_label(eco_score: int) -> str:
    """
    Convert eco score to sustainability label.
    
    Args:
        eco_score: Eco score from 0-100
        
    Returns:
        Sustainability label string
    """
    if eco_score >= 85:
        return "Excellent"
    elif eco_score >= 70:
        return "Very Good"
    elif eco_score >= 55:
        return "Good"
    elif eco_score >= 40:
        return "Fair"
    else:
        return "Poor"


def generate_shopping_suggestions(missing_ingredients: List[str]) -> List[Dict[str, Any]]:
    """
    Generate shopping suggestions with eco-scores using AI.
    Falls back to mock data if AI queries fail.
    
    Args:
        missing_ingredients: List of ingredient names
        
    Returns:
        List of suggestion dictionaries with ingredient and stores
    """
    if not missing_ingredients:
        return []
    
    suggestions = []
    
    try:
        # Get brands for each ingredient
        logger.info("Fetching brands for ingredients...")
        brands_data = get_brands(missing_ingredients)
        
        # Get eco scores, environmental impact, and reasoning for brands
        logger.info("Fetching eco scores and environmental impact for brands...")
        eco_scores, environmental_impact, reasoning = get_ecoscores(brands_data)
        
        # Get prices for brands (we still need prices, but not links)
        logger.info("Fetching prices for brands...")
        _, prices_map = get_online_links_and_prices(brands_data)
        
        # Build suggestions
        for ingredient in missing_ingredients:
            stores = []
            ingredient_brands = brands_data.get(ingredient, [])
            ingredient_scores = eco_scores.get(ingredient, {})
            ingredient_impact = environmental_impact.get(ingredient, {})
            ingredient_reasoning = reasoning.get(ingredient, {})
            ingredient_prices = prices_map.get(ingredient, {})
            
            # Create store entries for each brand
            for brand in ingredient_brands:
                eco_score = ingredient_scores.get(brand, 50)
                impact = ingredient_impact.get(brand, "Limited information available about this brand's environmental practices.")
                reason = ingredient_reasoning.get(brand, "Default score assigned due to lack of available environmental data.")
                price = ingredient_prices.get(brand, "$?.??")  # Use fetched price or default
                
                # Final validation: ensure price is not empty or None
                if not price or price == "" or price is None:
                    price = "$?.??"
                price = str(price).strip()
                
                # Ensure impact and reasoning are strings
                impact = str(impact).strip() if impact else "Limited information available about this brand's environmental practices."
                reason = str(reason).strip() if reason else "Default score assigned due to lack of available environmental data."
                
                # Log store entry for debugging
                logger.info(f"Store entry for {ingredient} - Brand: {brand}, Price: {price}, Eco Score: {eco_score}")
                
                store_entry = {
                    "name": brand,
                    "distance": "Online",  # Brands are online, no physical distance
                    "ecoScore": eco_score,
                    "sustainability": get_sustainability_label(eco_score),
                    "rating": f"Eco score: {eco_score}/100",
                    "price": price,  # Ensure price is a non-empty string
                    "environmentalImpact": impact,  # Environmental impact description
                    "reasoning": reason  # Reasoning for eco score
                }
                stores.append(store_entry)
                
                # Log final store entry for verification
                logger.debug(f"Final store entry for {ingredient}/{brand}: price={store_entry['price']}, ecoScore={store_entry['ecoScore']}")
            
            # Sort stores by eco score (highest first)
            stores = sorted(stores, key=lambda x: x['ecoScore'], reverse=True)
            
            # If no stores found, use mock data as fallback
            if not stores:
                logger.warning(f"No stores found for {ingredient}, using mock data")
                stores = get_mock_stores()
            
            suggestions.append({
                "ingredient": ingredient,
                "stores": stores
            })
            
    except Exception as e:
        logger.error(f"Error generating shopping suggestions: {str(e)}")
        logger.info("Falling back to mock data")
        # Fallback to mock data
        for ingredient in missing_ingredients:
            suggestions.append({
                "ingredient": ingredient,
                "stores": get_mock_stores()
            })
    
    return suggestions


def get_mock_stores() -> List[Dict[str, Any]]:
    """
    Get mock store data as fallback.
    
    Returns:
        List of mock store dictionaries
    """
    return [
        {
            "name": "Local Farmers Market",
            "distance": "0.5 mi",
            "ecoScore": 95,
            "sustainability": "Excellent",
            "rating": "Eco score: 95/100",
            "price": "$8.99",
            "environmentalImpact": "This local market sources directly from organic farms, minimizing transportation emissions. They use minimal packaging and prioritize seasonal, locally-grown produce.",
            "reasoning": "Received a high score due to strong local sourcing, organic certifications, and minimal packaging practices that significantly reduce environmental footprint."
        },
        {
            "name": "Whole Foods Market",
            "distance": "2.1 mi",
            "ecoScore": 82,
            "sustainability": "Very Good",
            "rating": "Eco score: 82/100",
            "price": "$12.99",
            "environmentalImpact": "Whole Foods has strong sustainability initiatives including organic product lines, responsible sourcing, and waste reduction programs. However, they source from various locations which increases transportation emissions.",
            "reasoning": "Good score based on comprehensive sustainability programs and organic options, but slightly lower due to broader sourcing that increases carbon footprint."
        },
        {
            "name": "Trader Joe's",
            "distance": "3.5 mi",
            "ecoScore": 75,
            "sustainability": "Good",
            "rating": "Eco score: 75/100",
            "price": "$9.99",
            "environmentalImpact": "Trader Joe's offers many organic and sustainably-sourced products with competitive pricing. They have made progress in reducing packaging, though there's room for improvement in sustainable sourcing.",
            "reasoning": "Solid score reflecting good organic options and packaging improvements, but moderate due to mixed sourcing practices and limited transparency on some environmental metrics."
        }
    ]

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
