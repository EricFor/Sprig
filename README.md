# Sprig

Intelligent Recipe Recommendation from Fridge Image Scanning using OpenAI Vision and Google Gemini AI.

## Overview

Sprig is a full-stack application that:
- Analyzes fridge images using OpenAI Vision (via Dedalus Labs API)
- Detects ingredients in your refrigerator automatically with high confidence
- Enhances images for better recognition accuracy
- Recommends personalized recipes based on available ingredients
- Supports manual ingredient addition and removal
- Provides recipe recalibration based on updated ingredient lists
- Suggests eco-friendly shopping options for missing ingredients
- Real-time progress tracking during analysis

## Tech Stack

- **Frontend**: React 19 + Vite
- **Backend**: Flask (Python 3.8+)
- **AI Vision Model**: OpenAI GPT-4o / GPT-4o-mini (via Dedalus Labs API)
- **Recipe Generation**: Google Gemini AI
- **Image Processing**: OpenCV, Pillow
- **Real-time Updates**: Server-Sent Events (SSE)

## Project Structure

```
EcoMarket/
├── frontend/              # React frontend application
│   ├── src/
│   │   ├── App.jsx       # Main application component
│   │   ├── App.css       # Application styles
│   │   └── main.jsx      # Application entry point
│   ├── package.json      # Frontend dependencies
│   └── index.html        # HTML entry point
├── backend/               # Flask backend API
│   ├── app.py            # Main Flask application
│   ├── requirements.txt  # Python dependencies
│   └── .env              # Environment variables (create this)
└── README.md             # This file
```

## Features

### Image Analysis
- Upload fridge images (JPG, PNG, JPEG)
- Automatic image quality validation
- Image enhancement pipeline for better recognition
- Real-time progress tracking with detailed status messages
- High-confidence ingredient detection

### Ingredient Management
- Automatic ingredient detection from images
- Manual ingredient addition
- Ingredient removal
- Confidence scores for detected ingredients
- User-inputted ingredient tagging

### Recipe Generation
- Personalized recipe recommendations
- Recipe recalibration based on updated ingredients
- Dietary preference support (vegan, vegetarian, halal, kosher, etc.)
- Cuisine region preferences
- Recipe details including prep time, cook time, and tags
- Missing ingredient identification

### Shopping Suggestions
- Eco-friendly store recommendations
- Sustainability scores
- Distance and pricing information
- Multiple store options per ingredient

## Quick Start

### Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **Node.js 16+** and **npm** ([Download](https://nodejs.org/))
- **Dedalus Labs API Key** (Get one at [dedaluslabs.ai](https://dedaluslabs.ai))
- **Google Gemini API Key** (Get one at [Google AI Studio](https://makersuite.google.com/app/apikey))

### Step 1: Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create environment file:**
   
   Create a `.env` file in the `backend` directory:
   ```env
   DEDALUS_API_KEY=your_dedalus_api_key_here
   DEDALUS_BASE_URL=https://api.dedaluslabs.ai/v1
   DEDALUS_MODEL=gpt-4o
   DEDALUS_MODEL_FALLBACK=gpt-4o-mini
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=google/gemini-2.5-flash
   PORT=5000
   ```
   
   Replace the API keys with your actual keys.

5. **Start the backend server:**
   ```bash
   python app.py
   ```
   
   You should see output indicating the server is running on `http://localhost:5000`

### Step 2: Frontend Setup

1. **Open a new terminal** and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Create environment file (optional):**
   
   If your backend is running on a different URL, create a `.env` file in the `frontend` directory:
   ```env
   VITE_API_URL=http://localhost:5000
   ```

4. **Start the development server:**
   ```bash
   npm run dev
   ```
   
   The frontend will be available at `http://localhost:5173` (or another port if 5173 is taken)

### Step 3: Use the Application

1. **Open your browser** to the frontend URL (usually `http://localhost:5173`)
2. **Set your preferences** (optional):
   - Diet preferences: vegan, vegetarian, spicy, low-carb, gluten-free, dairy-free, halal, kosher
   - Cuisine regions: Select from Asian, American, African, European, Middle Eastern, Oceanian cuisines
3. **Upload an image** of your refrigerator by clicking the upload area
4. **Click "Analyze Fridge"** to detect ingredients
5. **View real-time progress** as the image is processed
6. **View results:**
   - **Ingredients Tab**: See detected ingredients with confidence scores
   - Add or remove ingredients manually
   - Recalibrate recipes based on updated ingredients
   - **Recipes Tab**: Browse personalized recipe recommendations
   - **Shopping Tab**: Find eco-friendly shopping options for missing ingredients

## API Documentation

### POST `/api/analyze-fridge`

Analyzes a fridge image and returns detected ingredients, recipes, and shopping suggestions with real-time progress updates via Server-Sent Events (SSE).

**Request:**
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `image`: Image file (required) - JPG, PNG, or JPEG format
  - `preferences`: JSON string with user preferences (optional)
    ```json
    {
      "vegan": false,
      "vegetarian": false,
      "spicy": false,
      "lowCarb": false,
      "glutenFree": false,
      "dairyFree": false,
      "halal": false,
      "kosher": false,
      "cuisineRegions": ["Chinese", "Italian", "Mexican"]
    }
    ```

**Response:**
- **Content-Type**: `text/event-stream` (SSE)
- **Format**: Server-Sent Events with JSON data

Progress events:
```json
{
  "progress": 30,
  "message": "Analyzing image with AI vision model..."
}
```

Final result event:
```json
{
  "progress": 100,
  "message": "Analysis complete!",
  "complete": true,
  "ingredients": [
    {
      "name": "Tomato",
      "confidence": 0.92
    }
  ],
  "recipes": [
    {
      "name": "Chicken Stir-Fry",
      "description": "A quick and healthy stir-fry",
      "prepTime": "15 min",
      "cookTime": "20 min",
      "availableIngredients": ["Tomato", "Onion"],
      "missingIngredients": ["Olive Oil"],
      "tags": ["high-protein", "gluten-free"]
    }
  ],
  "missingIngredients": ["Olive Oil"],
  "shoppingSuggestions": [
    {
      "ingredient": "Olive Oil",
      "stores": [
        {
          "name": "Local Farmers Market",
          "distance": "0.5 mi",
          "ecoScore": 95,
          "sustainability": "Excellent",
          "rating": "Local sourcing, zero-waste packaging",
          "price": "$8.99"
        }
      ]
    }
  ]
}
```

### POST `/api/recalibrate-recipes`

Generates recipes from an existing ingredients list without requiring an image.

**Request:**
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "ingredients": ["Tomato", "Onion", "Chicken"],
    "preferences": {
      "vegan": false,
      "vegetarian": false,
      "cuisineRegions": ["Italian"]
    }
  }
  ```

**Response:**
```json
{
  "ingredients": [
    {
      "name": "Tomato",
      "confidence": 0.95
    }
  ],
  "recipes": [...],
  "missingIngredients": [...],
  "shoppingSuggestions": [...]
}
```

### GET `/health`

Health check endpoint to verify the server is running.

**Response:**
```json
{
  "status": "healthy"
}
```

### POST `/classify-fridge`

Direct image classification endpoint that returns only detected ingredients.

**Request:**
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `image`: Image file (required)
  - `images`: Multiple image files (optional)
  - `min_conf`: Minimum confidence threshold (optional, default: 0.50)
  - `top_k`: Maximum number of ingredients to return (optional)

**Response:**
```json
{
  "ingredients": [
    {
      "name": "Tomato",
      "confidence": 0.92
    }
  ],
  "notes": "Optional context",
  "diagnostics": {
    "images_processed": 1,
    "processing_times": [1.23],
    "model_used": "gpt-4o"
  }
}
```

## Configuration

### Backend Environment Variables

Create a `.env` file in the `backend` directory:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DEDALUS_API_KEY` | Your Dedalus Labs API key | - | Yes |
| `DEDALUS_BASE_URL` | Dedalus Labs API base URL | `https://api.dedaluslabs.ai/v1` | No |
| `DEDALUS_MODEL` | Primary vision model | `gpt-4o` | No |
| `DEDALUS_MODEL_FALLBACK` | Fallback vision model | `gpt-4o-mini` | No |
| `USE_FALLBACK_MODEL` | Use fallback model by default | `false` | No |
| `GEMINI_API_KEY` | Google Gemini API key | - | Yes |
| `GEMINI_MODEL` | Gemini model for recipes | `google/gemini-2.5-flash` | No |
| `PORT` | Backend server port | `5000` | No |

### Frontend Environment Variables

Create a `.env` file in the `frontend` directory (optional):

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | `http://localhost:5000` |

## AI Models

### OpenAI Vision (via Dedalus Labs)

This project uses OpenAI Vision models through the Dedalus Labs API:
- **Primary Model**: `gpt-4o`
- **Fallback Model**: `gpt-4o-mini`
- **Purpose**: Detects food ingredients in refrigerator images with high confidence
- **Features**:
  - High-accuracy ingredient recognition
  - Confidence scores for each detected ingredient
  - Exclusion of non-food items (containers, packaging, utensils)
  - Brand name filtering

### Image Enhancement Pipeline

Images are automatically enhanced before analysis to improve recognition accuracy:
- Auto white balance correction (fixes fridge LED lighting tints)
- Mild brightness and contrast adjustment
- Light sharpening (unsharp mask) for edge definition
- Gentle noise reduction (chroma-focused)
- Resolution normalization (optimized for GPT-4o)
- JPEG quality optimization (85-92%)

### Google Gemini AI

This project uses Google Gemini for intelligent recipe generation:
- **Model**: `gemini-2.5-flash` or `gemini-1.5-pro`
- **Purpose**: Generates personalized recipe recommendations based on detected ingredients and user preferences
- **Features**:
  - Creates creative recipe suggestions
  - Identifies missing ingredients needed
  - Respects dietary preferences (vegan, vegetarian, halal, kosher, etc.)
  - Supports cuisine region preferences
  - Prioritizes recipes using available ingredients
  - Minimizes missing ingredient requirements

## Image Processing

### Image Quality Validation

The system validates images before processing:
- Minimum resolution: 320x320 pixels
- Brightness validation (prevents too dark/bright images)
- Blur detection (Laplacian variance)
- Clear error messages for invalid images

### Image Enhancement

Images are enhanced using a non-destructive pipeline:
- White balance correction for fridge LED lighting
- Brightness adjustment (+3% to +8%)
- Contrast adjustment (+5% to +12%)
- Light sharpening (unsharp mask, 30-60% strength)
- Chroma-focused noise reduction
- Resolution optimization (1024-1600px longest side)

## Development

### Running in Development Mode

**Backend:**
```bash
cd backend
python app.py
```

**Frontend:**
```bash
cd frontend
npm run dev
```

### Building for Production

**Frontend:**
```bash
cd frontend
npm run build
```

The production build will be in the `frontend/dist` directory.

### Code Structure

- **Backend**: 
  - `backend/app.py` - Main Flask application with API endpoints
  - Image processing functions for validation and enhancement
  - Ingredient post-processing and normalization
  - Recipe generation with Gemini AI
  - Server-Sent Events for progress tracking

- **Frontend**: 
  - `frontend/src/App.jsx` - Main React component
  - `frontend/src/App.css` - Application styles
  - Real-time progress tracking with SSE
  - Manual ingredient management
  - Recipe recalibration

## Features in Detail

### Dietary Preferences

The application supports the following dietary preferences:
- Vegan
- Vegetarian
- Spicy
- Low-Carb
- Gluten-Free
- Dairy-Free
- Halal
- Kosher

### Cuisine Regions

Users can select from various cuisine regions:
- **Asian**: Chinese, Japanese, Indian, Thai, Korean, Vietnamese, and more
- **American**: Mexican, Brazilian, Argentinian, American, Canadian, and more
- **African**: Moroccan, Ethiopian, South African, Nigerian, and more
- **European**: Italian, French, Spanish, German, British, and more
- **Middle Eastern**: Turkish, Lebanese, Persian, Israeli
- **Oceanian**: Australian, New Zealand, Hawaiian

### Ingredient Management

- **Automatic Detection**: Ingredients are detected from uploaded images with confidence scores
- **Manual Addition**: Users can add ingredients manually that weren't detected
- **Removal**: Users can remove incorrect or unwanted ingredients
- **Tagging**: User-added ingredients are tagged as "User Inputted"
- **Recalibration**: Recipes can be regenerated based on updated ingredient lists

4. Ensure environment variables are set properly
5. Check that API keys are valid and have sufficient quota
