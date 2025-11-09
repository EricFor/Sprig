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
  const [showCuisineRegions, setShowCuisineRegions] = useState(false)
  const [activeTab, setActiveTab] = useState('ingredients')
  const [dietSearchQuery, setDietSearchQuery] = useState('')
  const [cuisineSearchQuery, setCuisineSearchQuery] = useState('')
  const [newIngredient, setNewIngredient] = useState('')
  const [progress, setProgress] = useState(0)
  const [progressMessage, setProgressMessage] = useState('')
  const [storesVisible, setStoresVisible] = useState({}) // Track visible stores per ingredient
  const [expandedStores, setExpandedStores] = useState({}) // Track which store details are expanded
  
  // User diet preferences
  const [preferences, setPreferences] = useState({
    vegan: false,
    vegetarian: false,
    spicy: false,
    lowCarb: false,
    glutenFree: false,
    dairyFree: false,
    halal: false,
    kosher: false
  })

  // Cuisine regions data - organized by larger regional classifications
  const [selectedCuisines, setSelectedCuisines] = useState([])
  
  const cuisineRegions = [
    {
      name: "Asian",
      cuisines: [
        "Chinese", "Japanese", "Indian", "Thai", "Korean", "Vietnamese",
        "Indonesian", "Malaysian", "Singaporean", "Filipino", "Cambodian",
        "Laotian", "Burmese", "Pakistani", "Bangladeshi", "Sri Lankan",
        "Nepali", "Taiwanese", "Mongolian", "Afghan", "Uzbek", "Kazakh"
      ]
    },
    {
      name: "American",
      cuisines: [
        "Mexican", "Brazilian", "Argentinian", "American", "Canadian",
        "Cuban", "Jamaican", "Caribbean", "Cajun", "Creole", "Tex-Mex",
        "Chilean", "Colombian", "Venezuelan", "Ecuadorian", "Guatemalan",
        "Costa Rican", "Peruvian"
      ]
    },
    {
      name: "African",
      cuisines: [
        "Moroccan", "Ethiopian", "South African", "Nigerian", "Ghanaian",
        "Senegalese", "Kenyan", "Tanzanian", "Tunisian", "Algerian", "Egyptian"
      ]
    },
    {
      name: "European",
      cuisines: [
        "Italian", "French", "Spanish", "German", "British", "Portuguese",
        "Russian", "Polish", "Greek", "Irish", "Scottish", "Swiss",
        "Austrian", "Belgian", "Dutch", "Hungarian", "Romanian", "Bulgarian",
        "Ukrainian", "Georgian", "Armenian", "Mediterranean", "Scandinavian"
      ]
    },
    {
      name: "Middle Eastern",
      cuisines: [
        "Turkish", "Middle Eastern", "Lebanese", "Persian", "Israeli"
      ]
    },
    {
      name: "Oceanian",
      cuisines: [
        "Australian", "New Zealand", "Hawaiian"
      ]
    }
  ]

  // Resize image to ~1024px longest edge to reduce token cost
  const resizeImage = (file, maxDimension = 1024) => {
    return new Promise((resolve) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        const img = new Image()
        img.onload = () => {
          const canvas = document.createElement('canvas')
          let width = img.width
          let height = img.height
          
          // Calculate new dimensions
          if (width > height) {
            if (width > maxDimension) {
              height = (height * maxDimension) / width
              width = maxDimension
            }
          } else {
            if (height > maxDimension) {
              width = (width * maxDimension) / height
              height = maxDimension
            }
          }
          
          canvas.width = width
          canvas.height = height
          
          const ctx = canvas.getContext('2d')
          ctx.drawImage(img, 0, 0, width, height)
          
          canvas.toBlob((blob) => {
            const resizedFile = new File([blob], file.name, {
              type: 'image/jpeg',
              lastModified: Date.now()
            })
            resolve(resizedFile)
          }, 'image/jpeg', 0.9)
        }
        img.src = e.target.result
      }
      reader.readAsDataURL(file)
    })
  }

  const handleImageUpload = async (e) => {
    const file = e.target.files[0]
    if (file) {
      setError(null)
      
      // Validate minimum size (320px)
      const img = new Image()
      const url = URL.createObjectURL(file)
      img.onload = async () => {
        URL.revokeObjectURL(url)
        
        if (img.width < 320 || img.height < 320) {
          setError('Image resolution too low. Please use at least 320x320 pixels.')
          return
        }
        
        // Resize image to ~1024px longest edge
        try {
          const resizedFile = await resizeImage(file, 1024)
          setImage(resizedFile)
          
          const reader = new FileReader()
          reader.onloadend = () => {
            setImagePreview(reader.result)
          }
          reader.readAsDataURL(resizedFile)
        } catch (err) {
          console.error('Error resizing image:', err)
          setError('Failed to process image. Please try again.')
        }
      }
      img.onerror = () => {
        URL.revokeObjectURL(url)
        setError('Invalid image file. Please upload a valid image.')
      }
      img.src = url
    }
  }

  const handlePreferenceChange = (pref) => {
    setPreferences(prev => ({
      ...prev,
      [pref]: !prev[pref]
    }))
  }

  const handleCuisineToggle = (cuisine) => {
    setSelectedCuisines(prev => {
      if (prev.includes(cuisine)) {
        return prev.filter(c => c !== cuisine)
      } else {
        return [...prev, cuisine]
      }
    })
  }

  // Diet preferences list for filtering
  const dietPreferencesList = [
    { key: 'vegan', label: 'Vegan' },
    { key: 'vegetarian', label: 'Vegetarian' },
    { key: 'spicy', label: 'Spicy' },
    { key: 'lowCarb', label: 'Low-Carb' },
    { key: 'glutenFree', label: 'Gluten-Free' },
    { key: 'dairyFree', label: 'Dairy-Free' },
    { key: 'halal', label: 'Halal' },
    { key: 'kosher', label: 'Kosher' }
  ]

  // Filter diet preferences based on search query
  const filteredDietPreferences = dietPreferencesList.filter(pref =>
    pref.label.toLowerCase().includes(dietSearchQuery.toLowerCase())
  )

  // Filter cuisine regions based on search query
  const filterCuisineRegions = (regions, searchQuery) => {
    if (!searchQuery.trim()) {
      return regions
    }

    const query = searchQuery.toLowerCase().trim()
    const filtered = []

    regions.forEach(region => {
      const regionMatches = region.name.toLowerCase().includes(query)
      const matchingCuisines = region.cuisines.filter(cuisine =>
        cuisine.toLowerCase().includes(query)
      )

      if (regionMatches || matchingCuisines.length > 0) {
        filtered.push({
          ...region,
          cuisines: regionMatches ? region.cuisines : matchingCuisines
        })
      }
    })

    return filtered
  }

  const filteredCuisineRegions = filterCuisineRegions(cuisineRegions, cuisineSearchQuery)

  const handleAnalyze = async () => {
    if (!image) {
      setError('Please upload an image first')
      return
    }

    setLoading(true)
    setError(null)
    setProgress(0)
    setProgressMessage('Starting analysis...')
    setIngredients(null)
    setMissingIngredients(null)
    setRecipes(null)
    setShoppingSuggestions(null)
    setActiveTab('ingredients') // Reset to ingredients tab when analyzing

    try {
      // Prepare form data
      const formData = new FormData()
      formData.append('image', image)
      formData.append('preferences', JSON.stringify({
        ...preferences,
        cuisineRegions: selectedCuisines
      }))
      
      // Get API URL from environment or use default
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000'
      
      // Call the backend API with streaming support
      const response = await fetch(`${apiUrl}/api/analyze-fridge`, {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      // Read the stream
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        
        if (done) {
          break
        }

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              // Handle progress updates
              if (data.progress !== undefined) {
                setProgress(data.progress)
                if (data.message) {
                  setProgressMessage(data.message)
                }
              }
              
              // Handle errors
              if (data.error) {
                throw new Error(data.error)
              }
              
              // Handle final results
              if (data.complete && data.ingredients) {
                setIngredients(data.ingredients || [])
                setRecipes(data.recipes || [])
                setMissingIngredients(data.missingIngredients || [])
                setShoppingSuggestions(data.shoppingSuggestions || [])
                setProgress(100)
                setProgressMessage('Analysis complete!')
                
                // Debug: Log shopping suggestions data
                if (data.shoppingSuggestions) {
                  console.log('Shopping suggestions received:', JSON.stringify(data.shoppingSuggestions, null, 2))
                  data.shoppingSuggestions.forEach((suggestion, idx) => {
                    console.log(`Suggestion ${idx} for ${suggestion.ingredient}:`, suggestion.stores?.length || 0, 'stores')
                    suggestion.stores?.forEach((store, storeIdx) => {
                      console.log(`  Store ${storeIdx}:`, {
                        name: store.name,
                        price: store.price,
                        environmentalImpact: store.environmentalImpact ? store.environmentalImpact.substring(0, 50) + '...' : 'none',
                        reasoning: store.reasoning ? store.reasoning.substring(0, 50) + '...' : 'none'
                      })
                    })
                  })
                }
              }
            } catch (parseError) {
              console.error('Error parsing SSE data:', parseError, line)
            }
          }
        }
      }
      
    } catch (err) {
      console.error('Error analyzing fridge:', err)
      setError(err.message || 'Failed to analyze image. Please make sure the backend server is running.')
      setProgress(0)
      setProgressMessage('')
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
    setProgress(0)
    setProgressMessage('')
    setSelectedCuisines([])
    setDietSearchQuery('')
    setCuisineSearchQuery('')
    setNewIngredient('')
    setStoresVisible({}) // Reset visible stores
    setExpandedStores({}) // Reset expanded store details
  }

  const handleAddIngredient = () => {
    if (newIngredient.trim() && ingredients) {
      const ingredientName = newIngredient.trim()
      // Check if ingredient already exists
      const existingNames = ingredients.map(ing => 
        typeof ing === 'string' ? ing.toLowerCase() : ing.name.toLowerCase()
      )
      if (!existingNames.includes(ingredientName.toLowerCase())) {
        // Add new ingredient marked as user-inputted
        const newIng = {
          name: ingredientName,
          userInputted: true // Flag to identify user-added ingredients
        }
        setIngredients([...ingredients, newIng])
        setNewIngredient('')
      } else {
        setError('Ingredient already exists')
        setTimeout(() => setError(null), 2000)
      }
    }
  }

  const handleRemoveIngredient = (index) => {
    const updatedIngredients = ingredients.filter((_, i) => i !== index)
    setIngredients(updatedIngredients)
  }

  const handleRecalibrate = async () => {
    if (!ingredients || ingredients.length === 0) {
      setError('Please add at least one ingredient')
      return
    }

    setLoading(true)
    setError(null)
    setRecipes(null)
    setMissingIngredients(null)
    setShoppingSuggestions(null)

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000'
      
      // Prepare ingredient names
      const ingredientNames = ingredients.map(ing => 
        typeof ing === 'string' ? ing : ing.name
      )

      // Call the recalibrate endpoint
      const response = await fetch(`${apiUrl}/api/recalibrate-recipes`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ingredients: ingredientNames,
          preferences: {
            ...preferences,
            cuisineRegions: selectedCuisines
          }
        })
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }))
        throw new Error(errorData.error || `Server error: ${response.status}`)
      }

      const data = await response.json()

      // Update state with new recipes
      setRecipes(data.recipes || [])
      setMissingIngredients(data.missingIngredients || [])
      setShoppingSuggestions(data.shoppingSuggestions || [])
      
      // Debug: Log shopping suggestions data
      if (data.shoppingSuggestions) {
        console.log('Shopping suggestions received (recalibrate):', JSON.stringify(data.shoppingSuggestions, null, 2))
      }
      
      // Switch to recipes tab
      if (data.recipes && data.recipes.length > 0) {
        setActiveTab('recipes')
      }

    } catch (err) {
      console.error('Error recalibrating recipes:', err)
      setError(err.message || 'Failed to generate recipes. Please make sure the backend server is running.')
    } finally {
      setLoading(false)
    }
  }

  const handleIngredientKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleAddIngredient()
    }
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
            {showPreferences ? '▼' : '▶'} Diet Preferences
          </button>
          {showPreferences && (
            <div className="preferences-panel">
              <div className="search-container">
                <input
                  type="text"
                  placeholder="Search diet preferences..."
                  value={dietSearchQuery}
                  onChange={(e) => setDietSearchQuery(e.target.value)}
                  className="preferences-search-input"
                />
              </div>
              <div className="preferences-grid">
                {filteredDietPreferences.length > 0 ? (
                  filteredDietPreferences.map((pref) => (
                    <label key={pref.key} className="preference-checkbox">
                      <input
                        type="checkbox"
                        checked={preferences[pref.key]}
                        onChange={() => handlePreferenceChange(pref.key)}
                      />
                      <span> {pref.label}</span>
                    </label>
                  ))
                ) : (
                  <div className="no-results">No preferences found matching "{dietSearchQuery}"</div>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="preferences-section">
          <button 
            onClick={() => setShowCuisineRegions(!showCuisineRegions)}
            className="preferences-toggle"
          >
            {showCuisineRegions ? '▼' : '▶'} Cuisine Regions
          </button>
          {showCuisineRegions && (
            <div className="preferences-panel">
              <div className="search-container">
                <input
                  type="text"
                  placeholder="Search regions or cuisines..."
                  value={cuisineSearchQuery}
                  onChange={(e) => setCuisineSearchQuery(e.target.value)}
                  className="preferences-search-input"
                />
              </div>
              <div className="cuisine-regions-container">
                {filteredCuisineRegions.length > 0 ? (
                  filteredCuisineRegions.map((region, regionIndex) => (
                    <div key={regionIndex} className="cuisine-region">
                      <h3 className="cuisine-region-name">{region.name}</h3>
                      <div className="cuisines-list">
                        {region.cuisines.map((cuisine, cuisineIndex) => (
                          <label key={cuisineIndex} className="cuisine-checkbox">
                            <input
                              type="checkbox"
                              checked={selectedCuisines.includes(cuisine)}
                              onChange={() => handleCuisineToggle(cuisine)}
                            />
                            <span>{cuisine}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="no-results">No cuisines found matching "{cuisineSearchQuery}"</div>
                )}
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
                        
                        {/* Add Ingredient Input */}
                        <div className="add-ingredient-section">
                          <div className="add-ingredient-input-container">
                            <input
                              type="text"
                              placeholder="Add ingredient manually..."
                              value={newIngredient}
                              onChange={(e) => setNewIngredient(e.target.value)}
                              onKeyPress={handleIngredientKeyPress}
                              className="add-ingredient-input"
                            />
                            <button 
                              onClick={handleAddIngredient}
                              className="add-ingredient-btn"
                              disabled={!newIngredient.trim()}
                            >
                              + Add
                            </button>
                          </div>
                        </div>

                        {/* Ingredients Grid */}
                        <div className="ingredients-grid">
                          {ingredients.map((ingredient, index) => {
                            const ingName = typeof ingredient === 'string' ? ingredient : ingredient.name
                            const isUserInputted = typeof ingredient === 'object' && ingredient.userInputted
                            const hasConfidence = typeof ingredient === 'object' && ingredient.confidence && !isUserInputted
                            
                            return (
                              <div key={index} className="ingredient-tag">
                                <span className="ingredient-name">{ingName}</span>
                                {isUserInputted && (
                                  <span className="user-inputted-tag">
                                    User Inputted
                                  </span>
                                )}
                                {hasConfidence && (
                                  <span className="confidence-score">
                                    {(ingredient.confidence * 100).toFixed(0)}%
                                  </span>
                                )}
                                <button
                                  onClick={() => handleRemoveIngredient(index)}
                                  className="remove-ingredient-btn"
                                  title="Remove ingredient"
                                >
                                  ×
                                </button>
                              </div>
                            )
                          })}
                        </div>

                        {/* Recalibrate Button */}
                        <div className="recalibrate-section">
                          <button 
                            onClick={handleRecalibrate}
                            className="recalibrate-btn"
                            disabled={loading || !ingredients || ingredients.length === 0}
                          >
                            {loading ? 'Generating...' : '🔄 Recalibrate Recipes'}
                          </button>
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
                          {shoppingSuggestions.map((suggestion, index) => {
                            const ingredientKey = suggestion.ingredient
                            const totalStores = suggestion.stores.length
                            const initialVisible = 3 // Show first 3 stores by default
                            const visibleCount = storesVisible[ingredientKey] || initialVisible
                            const visibleStores = suggestion.stores.slice(0, visibleCount)
                            const hasMore = totalStores > visibleCount
                            
                            return (
                              <div key={index} className="ingredient-shopping">
                                <h3 className="shopping-ingredient-name">{suggestion.ingredient}</h3>
                                <div className="stores-list">
                                  {visibleStores.map((store, storeIndex) => (
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
                                          <span className="store-price">{store.price || "$?.??"}</span>
                                          <button 
                                            className="store-action-btn"
                                            onClick={() => {
                                              const storeKey = `${ingredientKey}-${storeIndex}`
                                              setExpandedStores(prev => ({
                                                ...prev,
                                                [storeKey]: !prev[storeKey]
                                              }))
                                            }}
                                          >
                                            {expandedStores[`${ingredientKey}-${storeIndex}`] ? 'Show Less' : 'Show More'}
                                          </button>
                                        </div>
                                        {expandedStores[`${ingredientKey}-${storeIndex}`] && (
                                          <div className="store-expanded-details">
                                            <div className="environmental-impact-section">
                                              <h5 className="detail-section-title">Environmental Impact</h5>
                                              <p className="detail-section-content">{store.environmentalImpact || "Limited information available about this brand's environmental practices."}</p>
                                            </div>
                                            <div className="reasoning-section">
                                              <h5 className="detail-section-title">Eco Score Reasoning</h5>
                                              <p className="detail-section-content">{store.reasoning || "Default score assigned due to lack of available environmental data."}</p>
                                            </div>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  ))}
                                </div>
                                {hasMore && (
                                  <button
                                    onClick={() => setStoresVisible({
                                      ...storesVisible,
                                      [ingredientKey]: totalStores
                                    })}
                                    className="show-more-btn"
                                  >
                                    Show More ({totalStores - visibleCount} more)
                                  </button>
                                )}
                                {visibleCount > initialVisible && (
                                  <button
                                    onClick={() => setStoresVisible({
                                      ...storesVisible,
                                      [ingredientKey]: initialVisible
                                    })}
                                    className="show-less-btn"
                                  >
                                    Show Less
                                  </button>
                                )}
                              </div>
                            )
                          })}
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
            <>
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
              
              {loading && (
                <div className="progress-bar-container">
                  <div className="progress-bar">
                    <div 
                      className="progress-bar-fill" 
                      style={{ width: `${progress}%` }}
                    ></div>
                  </div>
                  <p className="progress-text">
                    {progressMessage || 'Processing image and generating recipes...'}
                  </p>
                  <p className="progress-percentage">{progress}%</p>
                </div>
              )}
            </>
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
