import json
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI web server instance
app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing). 
# This is required because our frontend runs on a different port (e.g., 3000) 
# than our backend (8000), and browsers block cross-port requests by default.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Importance weights assigned to each feature.
# We care more about categories (0.4) and colors (0.3) than price and age.
WEIGHTS = {
    'category': 0.4,
    'color': 0.3,
    'price': 0.2,
    'age': 0.1,
}

# Global variables to store the trained model and dataset context in memory
_global_ctx = {}
_model = None

# ====================================================================
# Data Models (Pydantic)
# These define the exact JSON structure we expect to receive in our API.
# Pydantic will automatically validate incoming JSON against these types.
# ====================================================================

class Purchase(BaseModel):
    id: int
    name: str
    category: str
    price: float
    color: str

class User(BaseModel):
    id: int
    name: str
    age: int
    purchases: List[Purchase]

class TrainRequest(BaseModel):
    users: List[User]

class RecommendRequest(BaseModel):
    user: User

# ====================================================================
# Preprocessing Logic
# ====================================================================

# Utility function to squish continuous values (like price or age) into a 0 to 1 range.
def normalize(value, min_val, max_val):
    diff = max_val - min_val
    if diff == 0:
        diff = 1 # Prevent division by zero
    return (value - min_val) / diff

# Analyzes all users and products to figure out globals like min/max prices, 
# unique categories, and unique colors. This "context" is needed to encode data later.
def make_context(products, users):
    ages = [u.age for u in users]
    prices = [p['price'] for p in products]

    # Find boundaries for continuous values
    min_age = min(ages) if ages else 0
    max_age = max(ages) if ages else 100
    
    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 1000
    
    # Extract unique colors and categories (removing duplicates)
    colors = list(dict.fromkeys(p['color'] for p in products))
    categories = list(dict.fromkeys(p['category'] for p in products))
    
    # Map each color/category to a specific index number for One-Hot encoding later
    colors_index = {c: i for i, c in enumerate(colors)}
    categories_index = {c: i for i, c in enumerate(categories)}
    
    mid_age = (min_age + max_age) / 2
    age_sums = {}
    age_counts = {}
    
    # Calculate the average age of users who bought each specific product
    for u in users:
        for p in u.purchases:
            age_sums[p.name] = age_sums.get(p.name, 0) + u.age
            age_counts[p.name] = age_counts.get(p.name, 0) + 1
            
    product_avg_age_norm = {}
    for product in products:
        name = product['name']
        avg = age_sums[name] / age_counts[name] if name in age_counts else mid_age
        product_avg_age_norm[name] = normalize(avg, min_age, max_age)
        
    return {
        'products': products,
        'users': users,
        'colors_index': colors_index,
        'categories_index': categories_index,
        'product_avg_age_norm': product_avg_age_norm,
        'min_age': min_age,
        'max_age': max_age,
        'min_price': min_price,
        'max_price': max_price,
        'num_categories': len(categories),
        'num_colors': len(colors),
        # Total dimensions = 2 (Price & Age) + number of categories + number of colors
        'dimensions': 2 + len(categories) + len(colors)
    }

# Creates a One-Hot encoded array.
# Example: If length is 3, index is 1, it makes [0, 1, 0]. Then multiplies by the weight.
def one_hot_weighted(index, length, weight):
    arr = np.zeros(length, dtype=np.float32)
    if 0 <= index < length:
        arr[index] = 1.0
    return arr * weight

# Converts a Product into a numerical array (Vector) so the Neural Network can understand it.
def encode_product(product, context):
    price_norm = normalize(product['price'], context['min_price'], context['max_price']) * WEIGHTS['price']
    age_norm = context['product_avg_age_norm'].get(product['name'], 0.5) * WEIGHTS['age']
    
    cat_idx = context['categories_index'].get(product['category'], -1)
    cat_vec = one_hot_weighted(cat_idx, context['num_categories'], WEIGHTS['category'])
    
    color_idx = context['colors_index'].get(product['color'], -1)
    color_vec = one_hot_weighted(color_idx, context['num_colors'], WEIGHTS['color'])
    
    # Join everything into a single flat array
    return np.concatenate([[price_norm, age_norm], cat_vec, color_vec])

# Converts a User into a numerical array (Vector) based on what they bought.
def encode_user(user, context):
    if user.purchases:
        # If the user bought things, their profile is the mathematical average of the products they bought.
        product_vecs = [encode_product(p.model_dump(), context) for p in user.purchases]
        return np.mean(product_vecs, axis=0)
    
    # If the user bought nothing, fallback to generating an empty/baseline vector containing only their age.
    price_norm = 0.0
    age_norm = normalize(user.age, context['min_age'], context['max_age']) * WEIGHTS['age']
    cat_vec = np.zeros(context['num_categories'], dtype=np.float32)
    color_vec = np.zeros(context['num_colors'], dtype=np.float32)
    return np.concatenate([[price_norm, age_norm], cat_vec, color_vec])

# Builds the actual Training data (X, Y matrices) 
def create_training_data(context):
    inputs = []  # Our X
    labels = []  # Our Y (1 for bought, 0 for not bought)
    
    for u in context['users']:
        if not u.purchases:
            continue
        user_vec = encode_user(u, context)
        purchase_names = {p.name for p in u.purchases}
        
        # Test every user against every product to generate training pairs
        for p in context['products']:
            p_vec = encode_product(p, context)
            label = 1.0 if p['name'] in purchase_names else 0.0
            
            # Combine User Vector + Product Vector. The model will learn if this combination equals 1 or 0
            inputs.append(np.concatenate([user_vec, p_vec]))
            labels.append(label)
            
    return np.array(inputs), np.array(labels)

# ====================================================================
# Neural Network Configuration
# ====================================================================

def configure_and_train(X, y, input_dim):
    # A standard Sequential multi-layer Feed-Forward network
    model = tf.keras.Sequential([
        # First layer matches the combined user+product dimension footprint
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        # Output is a single node representing the percentage chance (0.0 to 1.0)
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model quickly on the CPU
    model.fit(X, y, epochs=100, batch_size=32, shuffle=True, verbose=0)
    return model

# ====================================================================
# API Endpoints
# ====================================================================

# This endpoint handles standard browser POST requests targeting http://localhost:8000/train
@app.post("/train")
def train_api(req: TrainRequest):
    global _model, _global_ctx
    
    # 1. Open the internal product catalog
    base_dir = os.path.dirname(__file__)
    products_path = os.path.join(base_dir, 'data', 'products.json')
    with open(products_path, 'r', encoding='utf-8') as f:
        products = json.load(f)
        
    # 2. Re-calculate context boundaries (max/min/unique values)
    context = make_context(products, req.users)
    
    # 3. Cache product vectors heavily in memory to save time later
    product_vectors = []
    for p in products:
        product_vectors.append({
            'name': p['name'],
            'meta': p,
            'vector': encode_product(p, context)
        })
    context['product_vectors'] = product_vectors
        
    _global_ctx = context
    
    # 4. Generate arrays and Train
    X, y = create_training_data(context)
    
    if len(X) > 0:
        d = int(context['dimensions'])
        # The input dim is User length + Product length (which are identical in size)
        _model = configure_and_train(X, y, input_dim=d * 2)
        
    return {"status": "Training complete"}


# This endpoint handles recommendations and targets http://localhost:8000/recommend
@app.post("/recommend")
def recommend_api(req: RecommendRequest):
    # Guard clause against querying before training
    if not _model or not _global_ctx:
        return {"recommendations": []}
        
    # 1. Translate the incoming user into a numerical array
    user_vec = encode_user(req.user, _global_ctx)
    
    inputs = []
    # 2. Pair the single User Vector against every Product Vector
    for p_info in _global_ctx['product_vectors']:
        inputs.append(np.concatenate([user_vec, p_info['vector']]))
        
    # 3. Ask the trained network to evaluate the pairings
    X_test = np.array(inputs)
    predictions = _model.predict(X_test, verbose=0).flatten()
    
    recommendations = []
    # 4. Re-attach the numerical scores back to human-readable Product metadata
    for i, p_info in enumerate(_global_ctx['product_vectors']):
        item = dict(p_info['meta'])
        item['score'] = float(predictions[i])
        recommendations.append(item)
        
    # 5. Sort highest priority predictions at the top
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    return {"recommendations": recommendations}
