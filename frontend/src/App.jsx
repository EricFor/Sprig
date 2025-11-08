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

    // TODO: Replace with actual API call when computer vision is integrated
    // Example API call structure:
    // const formData = new FormData()
    // formData.append('image', image)
    // formData.append('preferences', JSON.stringify(preferences))
    // 
    // const response = await fetch('/api/analyze-fridge', {
    //   method: 'POST',
    //   body: formData
    // })
    // const data = await response.json()
    // setIngredients(data.ingredients)
    // setRecipes(data.recipes)
    // setMissingIngredients(data.missingIngredients)
    // setShoppingSuggestions(data.shoppingSuggestions)

    // Mock data simulation
    setTimeout(() => {
      setIngredients([
        { name: 'Tomatoes', confidence: 0.92 },
        { name: 'Onions', confidence: 0.88 },
        { name: 'Garlic', confidence: 0.85 },
        { name: 'Chicken Breast', confidence: 0.91 },
        { name: 'Bell Peppers', confidence: 0.87 },
        { name: 'Mushrooms', confidence: 0.83 },
        { name: 'Spinach', confidence: 0.89 }
      ])
      
      setMissingIngredients([
        'Olive Oil',
        'Black Beans',
        'Quinoa'
      ])

      // Filter recipes based on preferences
      const allRecipes = [
        {
          name: 'Chicken Stir-Fry',
          description: 'A quick and healthy stir-fry with your available ingredients',
          prepTime: '15 min',
          cookTime: '20 min',
          missingIngredients: ['Olive Oil'],
          tags: ['high-protein', 'gluten-free']
        },
        {
          name: 'Mediterranean Quinoa Bowl',
          description: 'Fresh and vibrant bowl with quinoa, vegetables, and herbs',
          prepTime: '10 min',
          cookTime: '20 min',
          missingIngredients: ['Quinoa', 'Olive Oil'],
          tags: ['vegetarian', 'gluten-free', 'high-protein']
        },
        {
          name: 'Stuffed Bell Peppers',
          description: 'Hearty bell peppers stuffed with chicken and vegetables',
          prepTime: '20 min',
          cookTime: '45 min',
          missingIngredients: ['Olive Oil'],
          tags: ['high-protein', 'gluten-free']
        },
        {
          name: 'Black Bean and Mushroom Tacos',
          description: 'Flavorful vegetarian tacos with black beans and mushrooms',
          prepTime: '15 min',
          cookTime: '25 min',
          missingIngredients: ['Black Beans', 'Olive Oil'],
          tags: ['vegetarian', 'vegan-option', 'spicy']
        }
      ]

      // Filter recipes based on preferences
      let filteredRecipes = allRecipes
      if (preferences.vegan || preferences.vegetarian) {
        filteredRecipes = filteredRecipes.filter(r => 
          r.tags.includes('vegetarian') || r.tags.includes('vegan-option')
        )
      }
      if (preferences.lowCarb) {
        filteredRecipes = filteredRecipes.filter(r => 
          !r.tags.includes('high-carb')
        )
      }

      setRecipes(filteredRecipes.slice(0, 4))

      // Mock shopping suggestions with eco-scores
      setShoppingSuggestions([
        {
          ingredient: 'Olive Oil',
          stores: [
            {
              name: 'Local Farmers Market',
              distance: '0.5 mi',
              ecoScore: 95,
              sustainability: 'Excellent',
              rating: 'Local sourcing, zero-waste packaging',
              price: '$8.99'
            },
            {
              name: 'Whole Foods Market',
              distance: '2.1 mi',
              ecoScore: 82,
              sustainability: 'Very Good',
              rating: 'Organic options, sustainable packaging',
              price: '$12.99'
            },
            {
              name: 'Trader Joe\'s',
              distance: '3.5 mi',
              ecoScore: 75,
              sustainability: 'Good',
              rating: 'Sustainable sourcing, recyclable packaging',
              price: '$9.99'
            },
            {
              name: 'Amazon Fresh',
              distance: 'Delivery',
              ecoScore: 58,
              sustainability: 'Fair',
              rating: 'Carbon footprint from delivery, mixed sourcing',
              price: '$10.99'
            }
          ]
        },
        {
          ingredient: 'Black Beans',
          stores: [
            {
              name: 'Local Farmers Market',
              distance: '0.5 mi',
              ecoScore: 92,
              sustainability: 'Excellent',
              rating: 'Locally grown, bulk purchase option',
              price: '$3.50/lb'
            },
            {
              name: 'Whole Foods Market',
              distance: '2.1 mi',
              ecoScore: 85,
              sustainability: 'Very Good',
              rating: 'Organic, sustainable farming practices',
              price: '$4.99/lb'
            },
            {
              name: 'Trader Joe\'s',
              distance: '3.5 mi',
              ecoScore: 78,
              sustainability: 'Good',
              rating: 'Sustainable sourcing, minimal packaging',
              price: '$3.99/lb'
            }
          ]
        },
        {
          ingredient: 'Quinoa',
          stores: [
            {
              name: 'Local Co-op',
              distance: '1.2 mi',
              ecoScore: 88,
              sustainability: 'Very Good',
              rating: 'Fair trade, bulk options available',
              price: '$6.99/lb'
            },
            {
              name: 'Whole Foods Market',
              distance: '2.1 mi',
              ecoScore: 80,
              sustainability: 'Good',
              rating: 'Organic, fair trade certified',
              price: '$8.99/lb'
            },
            {
              name: 'Amazon Fresh',
              distance: 'Delivery',
              ecoScore: 65,
              sustainability: 'Fair',
              rating: 'Delivery carbon footprint, standard sourcing',
              price: '$7.49/lb'
            }
          ]
        }
      ])

      setLoading(false)
    }, 2000)
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
    if (ingredients && ingredients.length > 0) tabs.push({ id: 'ingredients', label: 'Ingredients', icon: '📋' })
    if (recipes && recipes.length > 0) tabs.push({ id: 'recipes', label: 'Recipes', icon: '🍳' })
    if (missingIngredients && missingIngredients.length > 0) tabs.push({ id: 'missing', label: 'Missing', icon: '🛒' })
    if (shoppingSuggestions && shoppingSuggestions.length > 0) tabs.push({ id: 'shopping', label: 'Shopping', icon: '🌍' })
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
  }, [ingredients, recipes, missingIngredients, shoppingSuggestions])

  return (
    <div className="app">
      <header className="header">
        <h1 className="title"> EcoFridge</h1>
        <p className="subtitle">Intelligent Recipe Recommendation from Fridge Image Scanning</p>
        <p className="tagline">Reduce food waste • Make sustainable choices • Discover delicious recipes</p>
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
          ) : (ingredients || recipes || shoppingSuggestions || missingIngredients) && availableTabs.length > 0 ? (
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
                              <div className="recipe-meta">
                                <span className="meta-item">⏱️ Prep: {recipe.prepTime}</span>
                                <span className="meta-item">🔥 Cook: {recipe.cookTime}</span>
                                {recipe.missingIngredients && recipe.missingIngredients.length > 0 && (
                                  <span className="meta-item missing">
                                    ⚠️ Missing: {recipe.missingIngredients.join(', ')}
                                  </span>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {activeTab === 'missing' && missingIngredients && missingIngredients.length > 0 && (
                      <div className="missing-ingredients-card">
                        <h2 className="section-title"> Missing Ingredients</h2>
                        <p className="section-subtitle">Ingredients needed for recommended recipes</p>
                        <div className="missing-list">
                          {missingIngredients.map((ingredient, index) => (
                            <div key={index} className="missing-ingredient-tag">
                              {ingredient}
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

          {imagePreview && !(ingredients || recipes || shoppingSuggestions || missingIngredients) && (
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
        <p>Powered by EcoFridge AI • Reducing food waste through intelligent vision-based recipe recommendation</p>
      </footer>
    </div>
  )
}

export default App
