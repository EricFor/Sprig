# 🌱 EcoFridge

Intelligent Recipe Recommendation from Fridge Image Scanning using Roboflow's refrigerator-food detection model.

## 📋 Overview

EcoFridge is a full-stack application that:
- **Analyzes fridge images** using Roboflow's computer vision model
- **Detects ingredients** in your refrigerator automatically
- **Recommends recipes** based on available ingredients
- **Suggests eco-friendly shopping options** for missing ingredients

## 🛠️ Tech Stack

- **Frontend**: React + Vite
- **Backend**: Flask (Python)
- **AI Model**: Roboflow Inference SDK (refrigerator-food/3)

## 📁 Project Structure

```
EcoMarket/
├── frontend/              # React frontend application
│   ├── src/
│   │   ├── App.jsx       # Main application component
│   │   ├── App.css       # Application styles
│   │   └── main.jsx      # Application entry point
│   └── package.json      # Frontend dependencies
├── backend/               # Flask backend API
│   ├── app.py            # Main Flask application
│   └── requirements.txt  # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **Node.js 16+** and **npm** ([Download](https://nodejs.org/))
- **Roboflow API Key** (Get one at [roboflow.com](https://roboflow.com))

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
   ROBOFLOW_API_KEY=your_api_key_here
   PORT=5000
   ```
   
   Replace `your_api_key_here` with your actual Roboflow API key.

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
2. **Upload an image** of your refrigerator by clicking the upload area
3. **Set your preferences** (optional) - vegan, vegetarian, gluten-free, etc.
4. **Click "Analyze Fridge"** to detect ingredients
5. **View results:**
   - **Ingredients Tab**: See detected ingredients with confidence scores
   - **Recipes Tab**: Browse recipe recommendations
   - **Missing Tab**: View ingredients needed for recipes
   - **Shopping Tab**: Find eco-friendly shopping options

## 📡 API Documentation

### POST `/api/analyze-fridge`

Analyzes a fridge image and returns detected ingredients, recipes, and shopping suggestions.

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
      "dairyFree": false
    }
    ```

**Response:**
```json
{
  "ingredients": [
    {
      "name": "Tomato",
      "confidence": 0.92
    },
    {
      "name": "Onion",
      "confidence": 0.88
    }
  ],
  "recipes": [
    {
      "name": "Chicken Stir-Fry",
      "description": "A quick and healthy stir-fry",
      "prepTime": "15 min",
      "cookTime": "20 min",
      "missingIngredients": ["Olive Oil"],
      "tags": ["high-protein", "gluten-free"]
    }
  ],
  "missingIngredients": ["Olive Oil", "Black Beans"],
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

### GET `/health`

Health check endpoint to verify the server is running.

**Response:**
```json
{
  "status": "healthy"
}
```

## ⚙️ Configuration

### Backend Environment Variables

Create a `.env` file in the `backend` directory:

| Variable | Description | Default |
|----------|-------------|---------|
| `ROBOFLOW_API_KEY` | Your Roboflow API key | Required |
| `PORT` | Backend server port | `5000` |

### Frontend Environment Variables

Create a `.env` file in the `frontend` directory (optional):

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | `http://localhost:5000` |

## 🤖 Roboflow Model

This project uses the **refrigerator-food** model from Roboflow:
- **Model ID**: `refrigerator-food/3`
- **Model URL**: https://universe.roboflow.com/personal-dvpdm/refrigerator-food
- **Purpose**: Detects various food items commonly found in refrigerators

## 🔧 Development

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

- **Backend**: Edit `backend/app.py` to modify API endpoints and recipe generation logic
- **Frontend**: Edit `frontend/src/App.jsx` for UI changes, `frontend/src/App.css` for styling

## 🐛 Troubleshooting

### Backend Issues

**Problem**: `ModuleNotFoundError: No module named 'flask'`
- **Solution**: Make sure you've activated your virtual environment and run `pip install -r requirements.txt`

**Problem**: Backend server won't start
- **Solution**: 
  - Check that port 5000 is not already in use
  - Verify your `.env` file exists and contains `ROBOFLOW_API_KEY`
  - Check the console for error messages

**Problem**: `ERROR: ResolutionImpossible` when installing dependencies
- **Solution**: Make sure you're using Python 3.8+ and try updating pip: `pip install --upgrade pip`

### Frontend Issues

**Problem**: Frontend can't connect to backend
- **Solution**:
  - Ensure the backend server is running on port 5000
  - Check that `VITE_API_URL` in frontend `.env` matches your backend URL
  - Check browser console for CORS errors (CORS should be enabled by default)

**Problem**: `npm install` fails
- **Solution**: 
  - Make sure you have Node.js 16+ installed
  - Try deleting `node_modules` and `package-lock.json`, then run `npm install` again

### Model Inference Issues

**Problem**: No ingredients detected
- **Solution**:
  - Ensure your image is clear and shows food items
  - Verify your Roboflow API key is valid
  - Check that the model ID `refrigerator-food/3` is correct
  - Try a different image

**Problem**: API returns error
- **Solution**:
  - Check your Roboflow API key is set correctly
  - Verify you have API credits/quota available
  - Check backend console for detailed error messages

## 📝 Notes

- The recipe generation is currently using a simplified database. In production, you would integrate with a recipe API or database.
- Shopping suggestions are mock data. In production, integrate with store APIs or location services.
- The model may not detect all food items perfectly. Results depend on image quality and food visibility.

## 📄 License

MIT

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues:
1. Check the Troubleshooting section above
2. Review the error messages in the browser console (frontend) or terminal (backend)
3. Verify all prerequisites are installed correctly
4. Ensure environment variables are set properly
