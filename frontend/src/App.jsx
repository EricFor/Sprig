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

  // Cuisine regions data
  const [selectedCuisines, setSelectedCuisines] = useState([])
  
  const cuisineRegions = {
    "regions": [
      {
        "name": "Africa",
        "subregions": [
          {
            "name": "North Africa",
            "cuisines": ["Moroccan", "Algerian", "Tunisian", "Libyan", "Egyptian"]
          },
          {
            "name": "West Africa",
            "cuisines": ["Nigerian", "Ghanaian", "Senegalese", "Ivorian", "Togolese", "Beninese"]
          },
          {
            "name": "Central Africa",
            "cuisines": ["Cameroonian", "Congolese", "Gabonese", "Chadian", "Central African"]
          },
          {
            "name": "East Africa",
            "cuisines": ["Ethiopian", "Eritrean", "Somalian", "Kenyan", "Tanzanian", "Ugandan"]
          },
          {
            "name": "Southern Africa",
            "cuisines": ["South African", "Zimbabwean", "Zambian", "Namibian", "Botswanan"]
          }
        ]
      },
      {
        "name": "Europe",
        "subregions": [
          {
            "name": "Western Europe",
            "cuisines": ["French", "Belgian", "Dutch", "Luxembourgish"]
          },
          {
            "name": "Southern Europe / Mediterranean",
            "cuisines": ["Italian", "Spanish", "Portuguese", "Maltese", "Greek"]
          },
          {
            "name": "Central Europe",
            "cuisines": ["German", "Austrian", "Swiss", "Czech", "Slovak", "Hungarian", "Polish"]
          },
          {
            "name": "Northern Europe",
            "cuisines": ["English", "Scottish", "Welsh", "Irish", "Danish", "Swedish", "Norwegian", "Finnish", "Icelandic"]
          },
          {
            "name": "Eastern / Balkan / Caucasus",
            "cuisines": [
              "Russian", "Ukrainian", "Belarusian",
              "Serbian", "Croatian", "Bosnian", "Montenegrin", "Slovenian", "Albanian", "Macedonian", "Bulgarian", "Romanian",
              "Georgian", "Armenian", "Baltic (Estonian, Latvian, Lithuanian)"
            ]
          }
        ]
      },
      {
        "name": "Middle East & Central Asia",
        "subregions": [
          {
            "name": "Middle Eastern",
            "cuisines": [
              "Turkish", "Kurdish", "Persian", "Azerbaijani", "Iraqi", "Syrian", "Lebanese", "Palestinian",
              "Jordanian", "Saudi", "Gulf Arabian", "Yemeni"
            ]
          },
          {
            "name": "Central Asian",
            "cuisines": ["Afghan", "Uzbek", "Kazakh", "Tajik", "Turkmen"]
          }
        ]
      },
      {
        "name": "South Asia",
        "subregions": [
          {
            "name": "India (Regional)",
            "cuisines": [
              "Punjabi", "Gujarati", "Rajasthani", "Kashmiri", "Bengali", "Assamese",
              "Goan", "Keralan", "Tamil", "Hyderabadi"
            ]
          },
          {
            "name": "Other South Asian",
            "cuisines": ["Pakistani", "Bangladeshi", "Sri Lankan", "Nepali", "Bhutanese", "Maldivian"]
          }
        ]
      },
      {
        "name": "East Asia",
        "subregions": [
          {
            "name": "Chinese (Regional)",
            "cuisines": [
              "Cantonese", "Sichuan", "Hunan", "Fujian", "Shandong", "Anhui", "Jiangsu",
              "Beijing Cuisine", "Dongbei (Northeast)"
            ]
          },
          {
            "name": "Japan",
            "cuisines": ["Kanto", "Kansai", "Okinawan"]
          },
          {
            "name": "Korea",
            "cuisines": ["Seoul", "Jeolla", "Gyeongsang"]
          },
          {
            "name": "Other East Asian",
            "cuisines": ["Taiwanese", "Mongolian"]
          }
        ]
      },
      {
        "name": "Southeast Asia",
        "subregions": [
          {
            "name": "Mainland",
            "cuisines": ["Thai", "Vietnamese", "Cambodian", "Laotian", "Burmese"]
          },
          {
            "name": "Maritime",
            "cuisines": ["Indonesian", "Malaysian", "Singaporean", "Filipino", "Bruneian", "Timorese"]
          }
        ]
      },
      {
        "name": "Oceania / Pacific",
        "subregions": [
          {
            "name": "Australia & New Zealand",
            "cuisines": ["Australian", "New Zealand", "Māori"]
          },
          {
            "name": "Pacific Islands",
            "cuisines": ["Hawaiian", "Samoan", "Tongan", "Fijian", "Papua New Guinean", "Tahitian"]
          }
        ]
      },
      {
        "name": "The Americas",
        "subregions": [
          {
            "name": "North America",
            "cuisines": [
              "American (Southern, Cajun, Creole, Tex-Mex, Californian, Pacific Northwest, New England)",
              "Canadian", "Quebecois"
            ]
          },
          {
            "name": "Mexico",
            "cuisines": ["Oaxacan", "Yucatecan", "Central Mexican", "Baja", "Northern Mexican"]
          },
          {
            "name": "Central America",
            "cuisines": ["Guatemalan", "Honduran", "Salvadoran", "Nicaraguan", "Costa Rican", "Panamanian"]
          },
          {
            "name": "Caribbean",
            "cuisines": ["Jamaican", "Trinidadian", "Haitian", "Cuban", "Puerto Rican", "Dominican", "Barbadian", "Bahamian"]
          },
          {
            "name": "South America",
            "cuisines": ["Brazilian", "Argentinian", "Chilean", "Colombian", "Peruvian", "Venezuelan", "Ecuadorian", "Bolivian", "Paraguayan", "Uruguayan"]
          }
        ]
      }
    ]
  }

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
      const matchingSubregions = []

      region.subregions.forEach(subregion => {
        const subregionMatches = subregion.name.toLowerCase().includes(query)
        const matchingCuisines = subregion.cuisines.filter(cuisine =>
          cuisine.toLowerCase().includes(query)
        )

        if (subregionMatches || matchingCuisines.length > 0) {
          matchingSubregions.push({
            ...subregion,
            cuisines: subregionMatches ? subregion.cuisines : matchingCuisines
          })
        }
      })

      if (regionMatches || matchingSubregions.length > 0) {
        filtered.push({
          ...region,
          subregions: regionMatches ? region.subregions : matchingSubregions
        })
      }
    })

    return filtered
  }

  const filteredCuisineRegions = filterCuisineRegions(cuisineRegions.regions, cuisineSearchQuery)

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
      formData.append('preferences', JSON.stringify({
        ...preferences,
        cuisineRegions: selectedCuisines
      }))
      
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
    setSelectedCuisines([])
    setDietSearchQuery('')
    setCuisineSearchQuery('')
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
                  placeholder="Search cuisine regions, subregions, or cuisines..."
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
                      {region.subregions.map((subregion, subregionIndex) => (
                        <div key={subregionIndex} className="cuisine-subregion">
                          <h4 className="cuisine-subregion-name">{subregion.name}</h4>
                          <div className="cuisines-list">
                            {subregion.cuisines.map((cuisine, cuisineIndex) => (
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
                      ))}
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
