# Migration Summary: Roboflow to OpenAI Vision

## Overview

The image analysis system has been completely replaced from Roboflow to OpenAI Vision API (GPT-4o) according to the project specification.

## Key Changes

### 1. Backend Changes

#### Removed:
- Roboflow Inference SDK
- Destructive image preprocessing (noise, cutouts)
- Roboflow model dependency

#### Added:
- OpenAI Vision API integration (GPT-4o / GPT-4o-mini)
- Image quality validation (darkness, blur, resolution)
- Image size optimization (resize to 1024px longest edge)
- Post-processing pipeline:
  - Ingredient name normalization
  - Alias mapping (e.g., scallion → green onion)
  - Singularization (eggs → egg)
  - Non-ingredient filtering (containers, packaging, etc.)
  - Confidence threshold filtering (default: ≥0.90)
  - Deduplication
- New endpoint: `/classify-fridge`
- Legacy endpoint maintained: `/api/analyze-fridge` (uses new system)

### 2. Frontend Changes

#### Added:
- Client-side image resizing (1024px longest edge)
- Image validation (minimum 320x320px)
- Improved error handling

### 3. Dependencies

#### Removed:
- `inference-sdk==0.60.0`

#### Added:
- `openai>=1.0.0`
- `inflect>=7.0.0` (for singularization)

#### Kept:
- `google-generativeai==0.3.2` (for recipe generation)

## Environment Variables

Create a `.env` file in the `backend` directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
OPENAI_MODEL_FALLBACK=gpt-4o-mini
USE_FALLBACK_MODEL=false

# Optional: For recipe generation
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=google/gemini-2.5-flash

PORT=5000
```

## API Endpoints

### 1. `/classify-fridge` (New)

**Method:** POST  
**Purpose:** High-confidence ingredient extraction

**Input:**
- `images` or `image`: Image file(s)
- `min_conf` (optional): Minimum confidence threshold (default: 0.90)
- `top_k` (optional): Maximum number of ingredients to return
- `image_urls` (optional): List of image URLs (future implementation)

**Output:**
```json
{
  "ingredients": [
    {"name": "tomato", "confidence": 0.95},
    {"name": "onion", "confidence": 0.92}
  ],
  "notes": "optional context",
  "diagnostics": {
    "images_processed": 1,
    "images_failed": 0,
    "processing_times": [2.3],
    "model_used": "gpt-4o",
    "total_processing_time": 2.3,
    "ingredients_returned": 2
  }
}
```

### 2. `/api/analyze-fridge` (Legacy - Maintained for Backward Compatibility)

**Method:** POST  
**Purpose:** Full analysis with recipes (uses new classification system)

**Input:**
- `image`: Image file
- `preferences`: JSON string with dietary preferences
- `min_conf` (optional): Minimum confidence threshold (default: 0.80)

**Output:**
```json
{
  "ingredients": [...],
  "recipes": [...],
  "missingIngredients": [...],
  "shoppingSuggestions": [...]
}
```

## Features Implemented

### ✅ Image Quality Validation
- Minimum resolution: 320x320px
- Brightness check (rejects too dark/bright images)
- Blur detection (Laplacian variance)

### ✅ Image Optimization
- Automatic resizing to 1024px longest edge
- Maintains aspect ratio
- Client-side resizing to reduce token costs

### ✅ Post-Processing
- Normalization: lowercase, remove adjectives
- Singularization: eggs → egg, tomatoes → tomato
- Alias mapping: scallion → green onion
- Non-ingredient filtering: containers, packaging, utensils
- Confidence threshold: default ≥0.90
- Deduplication: keep highest confidence

### ✅ Multi-Image Support
- Process multiple images
- Merge and deduplicate results
- Keep highest confidence for duplicates

### ✅ Logging & Observability
- Token usage logging
- Processing time tracking
- Error logging
- Diagnostics in response

### ✅ Privacy
- Images processed in-memory
- No image persistence
- Images discarded after processing

## Configuration

### Model Selection
- **Primary:** `gpt-4o` (default)
- **Fallback:** `gpt-4o-mini` (cost-optimized)
- Set `USE_FALLBACK_MODEL=true` to use fallback model

### Confidence Threshold
- **Default:** 0.90 (90% confidence)
- Only ingredients with confidence ≥ threshold are returned
- Adjustable via `min_conf` parameter

### Image Size
- **Client-side:** Resized to 1024px longest edge
- **Server-side:** Further optimization if needed
- Reduces token costs while maintaining quality

## Testing

### Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables in `.env`
3. Start backend: `python app.py`
4. Start frontend: `npm run dev`

### Test the New Endpoint
```bash
curl -X POST http://localhost:5000/classify-fridge \
  -F "image=@test_fridge.jpg" \
  -F "min_conf=0.90"
```

### Test Legacy Endpoint
```bash
curl -X POST http://localhost:5000/api/analyze-fridge \
  -F "image=@test_fridge.jpg" \
  -F "preferences={\"vegan\": false}"
```

## Performance Targets

- **Response Time:** ≤ 2.5-3.0s median per image at 1024px
- **Accuracy:** High-confidence detections (≥0.90)
- **Cost:** Monitor token usage to optimize model choice

## Migration Notes

1. **Backward Compatibility:** The old `/api/analyze-fridge` endpoint still works but now uses the new OpenAI Vision system
2. **Recipe Generation:** Still uses Gemini API (optional)
3. **No Breaking Changes:** Frontend doesn't need changes (uses legacy endpoint)
4. **New Features:** Use `/classify-fridge` for new features

## Next Steps

1. Test with real fridge images
2. Tune alias mapping based on results
3. Adjust confidence threshold based on precision/recall
4. Monitor token usage and costs
5. Consider fine-tuning based on user feedback

## Troubleshooting

### OpenAI API Key Error
- Ensure `OPENAI_API_KEY` is set in `.env`
- Check API key is valid and has credits

### Image Validation Failures
- Check image resolution (minimum 320x320px)
- Ensure good lighting
- Take clearer photos

### No Ingredients Returned
- Check confidence threshold (default 0.90)
- Try lowering `min_conf` parameter
- Verify image quality

### Recipe Generation Not Working
- Ensure `GEMINI_API_KEY` is set (optional)
- Recipe generation is optional and won't break if unavailable

