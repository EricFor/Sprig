import { useState, useEffect } from 'react'
import './App.css'
import { FaBowlFood } from "react-icons/fa6";


function App() {
  const [image, setImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [ingredients, setIngredients] = useState(null)
  const [missingIngredients, setMissingIngredients] = useState(null)
  const [recipes, setRecipes] = useState(null)
  const [shoppingSuggestions, setShoppingSuggestions] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showPreferences, setShowPreferences] = useState(false)
  const [activeTab, setActiveTab] = useState('ingredients')
  
  // User taste preferences
  const [preferences, setPreferences] = useState({
    vegan: false,
    vegetarian: false,
    spicy: false,
    lowCarb: false,
    glutenFree: false,
    dairyFree: false
  })

  const handleImageUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      setError(null)
      setImage(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handlePreferenceChange = (pref) => {
    setPreferences(prev => ({
      ...prev,
      [pref]: !prev[pref]
    }))
  }

  const handleAnalyze = async () => {
    if (!image) {
      setError('Please upload an image first')
      return
    }

    setLoading(true)
    setError(null)
    setIngredients(null)
    setMissingIngredients(null)
    setRecipes(null)
    setShoppingSuggestions(null)
    setActiveTab('ingredients') // Reset to ingredients tab when analyzing

    try {
      // Prepare form data
      const formData = new FormData()
      formData.append('image', image)
      formData.append('preferences', JSON.stringify(preferences))
      
      // Get API URL from environment or use default
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000'
      
      // Call the backend API
      const response = await fetch(`${apiUrl}/api/analyze-fridge`, {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }))
        throw new Error(errorData.error || `Server error: ${response.status}`)
      }
      
      const data = await response.json()
      
      // Log the response for debugging
      console.log('API Response:', data)
      console.log('Ingredients:', data.ingredients)
      console.log('Recipes:', data.recipes)
      console.log('Missing Ingredients:', data.missingIngredients)
      console.log('Shopping Suggestions:', data.shoppingSuggestions)
      
      // Update state with API response
      setIngredients(data.ingredients || [])
      setRecipes(data.recipes || [])
      setMissingIngredients(data.missingIngredients || [])
      setShoppingSuggestions(data.shoppingSuggestions || [])
      
    } catch (err) {
      console.error('Error analyzing fridge:', err)
      setError(err.message || 'Failed to analyze image. Please make sure the backend server is running.')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setImage(null)
    setImagePreview(null)
    setIngredients(null)
    setMissingIngredients(null)
    setRecipes(null)
    setShoppingSuggestions(null)
    setError(null)
    setLoading(false)
  }

  const getEcoScoreColor = (score) => {
    if (score >= 85) return '#6B8E5A'
    if (score >= 70) return '#7A9B6A'
    if (score >= 55) return '#E8C4A0'
    return '#D4A574'
  }

  // Determine which tabs to show based on available data
  const getAvailableTabs = () => {
    const tabs = []
    if (ingredients && ingredients.length > 0) tabs.push({ id: 'ingredients', label: 'Ingredients', icon: '' })
    if (recipes && recipes.length > 0) tabs.push({ id: 'recipes', label: 'Recipes', icon: '' })
    if (shoppingSuggestions && shoppingSuggestions.length > 0) tabs.push({ id: 'shopping', label: 'Shopping', icon: '' })
    return tabs
  }

  const availableTabs = getAvailableTabs()

  // Set active tab to first available tab if current tab has no data
  // Note: activeTab is intentionally not in dependencies - we only want to check validity when data changes
  useEffect(() => {
    const tabs = getAvailableTabs()
    if (tabs.length > 0 && !tabs.find(tab => tab.id === activeTab)) {
      setActiveTab(tabs[0].id)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ingredients, recipes, shoppingSuggestions])

  return (
    <div className="app">
      <header className="header">
        <h1 className="title">Sprig</h1>
        <p className="subtitle">Vision that cooks.</p>
        <p className="tagline">Reduce food waste • Make healthier decisions • Discover delicious recipes</p>
      </header>

      <main className="main-content">
        <div className="preferences-section">
          <button 
            onClick={() => setShowPreferences(!showPreferences)}
            className="preferences-toggle"
          >
            {showPreferences ? '▼' : '▶'} Taste Preferences
          </button>
          {showPreferences && (
            <div className="preferences-panel">
              <div className="preferences-grid">
                <label className="preference-checkbox">
                  <input
                    type="checkbox"
                    checked={preferences.vegan}
                    onChange={() => handlePreferenceChange('vegan')}
                  />
                  <span> Vegan</span>
                </label>
                <label className="preference-checkbox">
                  <input
                    type="checkbox"
                    checked={preferences.vegetarian}
                    onChange={() => handlePreferenceChange('vegetarian')}
                  />
                  <span> Vegetarian</span>
                </label>
                <label className="preference-checkbox">
                  <input
                    type="checkbox"
                    checked={preferences.spicy}
                    onChange={() => handlePreferenceChange('spicy')}
                  />
                  <span> Spicy</span>
                </label>
                <label className="preference-checkbox">
                  <input
                    type="checkbox"
                    checked={preferences.lowCarb}
                    onChange={() => handlePreferenceChange('lowCarb')}
                  />
                  <span> Low-Carb</span>
                </label>
                <label className="preference-checkbox">
                  <input
                    type="checkbox"
                    checked={preferences.glutenFree}
                    onChange={() => handlePreferenceChange('glutenFree')}
                  />
                  <span> Gluten-Free</span>
                </label>
                <label className="preference-checkbox">
                  <input
                    type="checkbox"
                    checked={preferences.dairyFree}
                    onChange={() => handlePreferenceChange('dairyFree')}
                  />
                  <span> Dairy-Free</span>
                </label>
              </div>
            </div>
          )}
        </div>

        <div className="upload-section">
          {!imagePreview ? (
            <div className="upload-container">
              <label className="upload-area" htmlFor="image-upload">
                <div className="upload-icon">📸</div>
                <p className="upload-text">Click to upload or drag and drop</p>
                <p className="upload-hint">Upload a clear image of your fridge interior</p>
                <p className="upload-hint-small">PNG, JPG, JPEG up to 10MB</p>
                <input
                  id="image-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="file-input"
                />
              </label>
            </div>
          ) : (ingredients || recipes || shoppingSuggestions) && availableTabs.length > 0 ? (
            <div className="image-results-wrapper">
              <div className="image-wrapper">
                <div className="image-preview-container">
                  <img src={imagePreview} alt="Uploaded fridge" className="preview-image" />
                </div>
                <button onClick={handleReset} className="remove-image-btn">
                  ✕ Remove Image
                </button>
              </div>

              <div className="results-section">
                <div className="tabs-container">
                  <div className="tab-content">
                    {activeTab === 'ingredients' && ingredients && (
                      <div className="ingredients-card">
                        <h2 className="section-title"> Detected Ingredients</h2>
                        <p className="section-subtitle">Confidence scores from object detection model</p>
                        <div className="ingredients-grid">
                          {ingredients.map((ingredient, index) => (
                            <div key={index} className="ingredient-tag">
                              <span className="ingredient-name">{typeof ingredient === 'string' ? ingredient : ingredient.name}</span>
                              {typeof ingredient === 'object' && (
                                <span className="confidence-score">
                                  {(ingredient.confidence * 100).toFixed(0)}%
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {activeTab === 'recipes' && recipes && recipes.length > 0 && (
                      <div className="recipes-card">
                        <h2 className="section-title"> <FaBowlFood />  Recommended Recipes</h2>
                        <p className="section-subtitle">Personalized based on your preferences and available ingredients</p>
                        <div className="recipes-list">
                          {recipes.map((recipe, index) => (
                            <div key={index} className="recipe-item">
                              <div className="recipe-header">
                                <h3 className="recipe-name">{recipe.name}</h3>
                                {recipe.tags && (
                                  <div className="recipe-tags">
                                    {recipe.tags.map((tag, i) => (
                                      <span key={i} className="recipe-tag">{tag}</span>
                                    ))}
                                  </div>
                                )}
                              </div>
                              <p className="recipe-description">{recipe.description}</p>
                              {recipe.availableIngredients && recipe.availableIngredients.length > 0 && (
                                <div className="recipe-ingredients-section">
                                  <div className="recipe-ingredients-label">✅ Available Ingredients:</div>
                                  <div className="recipe-ingredients-list">
                                    {recipe.availableIngredients.map((ingredient, i) => (
                                      <span key={i} className="recipe-ingredient-tag available">
                                        {ingredient}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}
                              <div className="recipe-meta">
                                <span className="meta-item">⏱️ Prep: {recipe.prepTime}</span>
                                <span className="meta-item">🔥 Cook: {recipe.cookTime}</span>
                                {recipe.missingIngredients && recipe.missingIngredients.length > 0 && (
                                  <div className="recipe-missing-section">
                                    <span className="meta-item missing-label">⚠️ Missing:</span>
                                    <div className="recipe-missing-list">
                                      {recipe.missingIngredients.map((ingredient, i) => (
                                        <span key={i} className="recipe-missing-tag">
                                          {ingredient}
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {activeTab === 'shopping' && shoppingSuggestions && shoppingSuggestions.length > 0 && (
                      <div className="shopping-card">
                        <h2 className="section-title"> Eco-Friendly Shopping Suggestions</h2>
                        <p className="section-subtitle">Stores ranked by sustainability score, carbon footprint, and local sourcing</p>
                        <div className="shopping-suggestions">
                          {shoppingSuggestions.map((suggestion, index) => (
                            <div key={index} className="ingredient-shopping">
                              <h3 className="shopping-ingredient-name">{suggestion.ingredient}</h3>
                              <div className="stores-list">
                                {suggestion.stores.map((store, storeIndex) => (
                                  <div key={storeIndex} className="store-item">
                                    <div className="store-header">
                                      <div className="store-info">
                                        <h4 className="store-name">{store.name}</h4>
                                        <span className="store-distance">{store.distance}</span>
                                      </div>
                                      <div 
                                        className="eco-score-badge"
                                        style={{ backgroundColor: getEcoScoreColor(store.ecoScore) }}
                                      >
                                        <span className="eco-score-value">{store.ecoScore}</span>
                                        <span className="eco-score-label">Eco Score</span>
                                      </div>
                                    </div>
                                    <div className="store-details">
                                      <div className="sustainability-badge">
                                        <span className="sustainability-label">Sustainability:</span>
                                        <span className="sustainability-value">{store.sustainability}</span>
                                      </div>
                                      <p className="store-rating">{store.rating}</p>
                                      <div className="store-footer">
                                        <span className="store-price">{store.price}</span>
                                        <button className="store-action-btn">View Details</button>
                                      </div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="tabs-nav">
                    {availableTabs.map((tab) => (
                      <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
                        title={tab.label}
                      >
                        <span className="tab-icon">{tab.icon}</span>
                        <span className="tab-label">{tab.label}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="upload-container">
              <div className="image-wrapper">
                <div className="image-preview-container">
                  <img src={imagePreview} alt="Uploaded fridge" className="preview-image" />
                </div>
                <button onClick={handleReset} className="remove-image-btn">
                  ✕ Remove Image
                </button>
              </div>
            </div>
          )}

          {imagePreview && !(ingredients || recipes || shoppingSuggestions) && (
            <button
              onClick={handleAnalyze}
              disabled={!image || loading}
              className="analyze-btn"
            >
              {loading ? (
                <>
                  <span className="loading-spinner"></span>
                  Analyzing Fridge...
                </>
              ) : (
                '🔍 Analyze Fridge'
              )}
            </button>
          )}

          {error && <div className="error-message">{error}</div>}
        </div>
      </main>

      <footer className="footer">
        <p>Powered by EcoFridge AI • Made with love • Not a substitute for professional medical advice! </p>
      </footer>
    </div>
  )
}

export default App
