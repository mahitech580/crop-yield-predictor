from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import sys
import os
sys.path.append('src')
from predict import CropYieldPredictor
from auth import init_db, register_user, authenticate_user, login_required

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = 'your-secret-key-here'  # Change this in production
init_db()  # Initialize the database
predictor = CropYieldPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_page')
@login_required
def predict_page():
    return render_template('predict.html')

@app.route('/demo')
@login_required
def demo():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Crop Yield Prediction</title>
    <style>
        body { font-family: Arial; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
        h1 { color: #2e7d32; text-align: center; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #4caf50; color: white; padding: 12px 30px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #45a049; }
        .result { margin-top: 20px; padding: 20px; background: #e8f5e9; border-radius: 5px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌾 Crop Yield Prediction with Ensemble Machine Learning</h1>
        <p style="text-align: center; color: #666;">Using Advanced Machine Learning Algorithms Including Decision Tree, LightGBM, XGBoost, AdaBoost, Random Forest, ExtraTree, Gradient Boosting, and Bagging Classifier</p>
        
        <form id="predictionForm">
            <div class="grid">
                <div>
                    <div class="form-group">
                        <label>Rainfall (mm):</label>
                        <input type="number" id="rainfall" value="850" required>
                    </div>
                    <div class="form-group">
                        <label>Temperature (°C):</label>
                        <input type="number" id="temperature" value="28" required>
                    </div>
                    <div class="form-group">
                        <label>Humidity (%):</label>
                        <input type="number" id="humidity" value="70" required>
                    </div>
                    <div class="form-group">
                        <label>Nitrogen (N):</label>
                        <input type="number" id="N" value="45" required>
                    </div>
                    <div class="form-group">
                        <label>Soil pH:</label>
                        <input type="number" id="pH" value="6.5" step="0.1" required>
                    </div>
                </div>
                <div>
                    <div class="form-group">
                        <label>Phosphorus (P):</label>
                        <input type="number" id="P" value="25" required>
                    </div>
                    <div class="form-group">
                        <label>Potassium (K):</label>
                        <input type="number" id="K" value="35" required>
                    </div>
                    <div class="form-group">
                        <label>Farm Area (hectares):</label>
                        <input type="number" id="area" value="5" step="0.1" required>
                    </div>
                    <div class="form-group">
                        <label>Crop Type:</label>
                        <select id="crop">
                            <option value="Rice">Rice</option>
                            <option value="Wheat">Wheat</option>
                            <option value="Maize">Maize</option>
                            <option value="Cotton">Cotton</option>
                            <option value="Sugarcane">Sugarcane</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Soil Type:</label>
                        <select id="soil_type">
                            <option value="Clay">Clay</option>
                            <option value="Sandy">Sandy</option>
                            <option value="Loamy" selected>Loamy</option>
                            <option value="Black">Black</option>
                            <option value="Red">Red</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <div class="form-group">
                <label>Growing Season:</label>
                <select id="season">
                    <option value="Kharif" selected>Kharif</option>
                    <option value="Rabi">Rabi</option>
                    <option value="Zaid">Zaid</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Irrigation Method:</label>
                <select id="irrigation">
                    <option value="Drip">Drip</option>
                    <option value="Sprinkler">Sprinkler</option>
                    <option value="Flood" selected>Flood</option>
                    <option value="Rainfed">Rainfed</option>
                </select>
            </div>
            
            <div style="text-align: center; margin-top: 20px;">
                <button type="submit">🔮 Predict Yield</button>
            </div>
        </form>
        
        <div id="result" style="display: none;"></div>
    </div>

    <script>
        document.getElementById('predictionForm').onsubmit = function(e) {
            e.preventDefault();
            
            const data = {
                rainfall: document.getElementById('rainfall').value,
                temperature: document.getElementById('temperature').value,
                humidity: document.getElementById('humidity').value,
                N: document.getElementById('N').value,
                P: document.getElementById('P').value,
                K: document.getElementById('K').value,
                pH: document.getElementById('pH').value,
                area: document.getElementById('area').value,
                crop: document.getElementById('crop').value,
                soil_type: document.getElementById('soil_type').value,
                season: document.getElementById('season').value,
                irrigation: document.getElementById('irrigation').value
            };
            
            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                document.getElementById('result').innerHTML = `
                    <div class="result">
                        <h3>🎯 Prediction Results</h3>
                        <p><strong>Predicted Yield:</strong> ${result.predicted_yield} tons/hectare</p>
                        <p><strong>Fertilizer Recommendation:</strong> ${result.fertilizer_recommendation}</p>
                        <p><strong>Estimated Cost:</strong> Rs.${result.estimated_cost}/kg</p>
                        <p><strong>Model Performance:</strong> R² = 0.994 (ExtraTree), 100% Accuracy (Fertilizer Recommendations)</p>
                        <p><strong>Confidence Level:</strong> ${result.confidence}%</p>
                    </div>
                `;
                document.getElementById('result').style.display = 'block';
            });
        };
    </script>
</body>
</html>
    '''

@app.route('/crop/<crop_name>')
def crop_detail(crop_name):
    # Crop information dictionary
    crop_info = {
        'rice': {
            'name': 'Rice',
            'icon': 'fa-seedling',
            'temperature_range': '20-35°C',
            'rainfall_range': '1000-2000mm',
            'growing_season': 'Kharif (Monsoon)',
            'humidity_range': '60-80%',
            'soil_type': 'Clay, Loamy',
            'ph_range': '5.5-7.0',
            'drainage': 'Good',
            'fertility': 'High',
            'average_yield': '3-5 tons/hectare',
            'harvest_time': '3-6 months',
            'planting_depth': '2-3 cm',
            'spacing': '20x20 cm',
            'planting_tips': 'Plant in flooded fields with proper transplanting at 25-30 day old seedlings.',
            'water_management': 'Maintain 5-10cm water depth throughout growing season except flowering stage.',
            'pest_control': 'Monitor for stem borer, leaf folder and brown plant hopper. Use IPM approach.',
            'harvesting_tips': 'Harvest when 85% grains turn golden yellow. Cut plants close to ground.'
        },
        'wheat': {
            'name': 'Wheat',
            'icon': 'fa-wheat-alt',
            'temperature_range': '10-25°C',
            'rainfall_range': '300-500mm',
            'growing_season': 'Rabi (Winter)',
            'humidity_range': '50-70%',
            'soil_type': 'Loamy, Clay',
            'ph_range': '5.5-7.5',
            'drainage': 'Well-drained',
            'fertility': 'Medium to High',
            'average_yield': '2-4 tons/hectare',
            'harvest_time': '4-5 months',
            'planting_depth': '3-5 cm',
            'spacing': '22x10 cm',
            'planting_tips': 'Sow in rows with certified seeds at 5-7 cm depth. Optimum sowing time is October-November.',
            'water_management': 'Requires moderate irrigation. Critical stages are crown root initiation and grain filling.',
            'pest_control': 'Watch for aphids, termites and rust diseases. Apply seed treatment with fungicides.',
            'harvesting_tips': 'Harvest when grains become hard and straw turns golden. Cut 15cm above ground level.'
        },
        'maize': {
            'name': 'Maize',
            'icon': 'fa-corn',
            'temperature_range': '18-30°C',
            'rainfall_range': '500-1000mm',
            'growing_season': 'Kharif & Rabi',
            'humidity_range': '60-70%',
            'soil_type': 'Loamy, Sandy loam',
            'ph_range': '5.5-7.0',
            'drainage': 'Well-drained',
            'fertility': 'High',
            'average_yield': '3-6 tons/hectare',
            'harvest_time': '3-4 months',
            'planting_depth': '4-5 cm',
            'spacing': '75x20 cm',
            'planting_tips': 'Plant seeds 5-7cm deep in well-prepared soil. Maintain proper plant population for hybrid varieties.',
            'water_management': 'Critical water requirement during tasseling and grain filling stages. Avoid waterlogging.',
            'pest_control': 'Control stem borer, shoot fly and leaf blight with timely sprays and resistant varieties.',
            'harvesting_tips': 'Harvest when kernels are hard and moisture content is 20-25%. Detach ears manually.'
        },
        'cotton': {
            'name': 'Cotton',
            'icon': 'fa-cloud-sun',
            'temperature_range': '20-30°C',
            'rainfall_range': '500-1000mm',
            'growing_season': 'Kharif',
            'humidity_range': '50-60%',
            'soil_type': 'Black, Loamy',
            'ph_range': '5.5-8.0',
            'drainage': 'Well-drained',
            'fertility': 'Medium to High',
            'average_yield': '300-500 kg/hectare',
            'harvest_time': '5-6 months',
            'planting_depth': '2-3 cm',
            'spacing': '90x60 cm',
            'planting_tips': 'Sow in ridges and furrows with proper spacing. Use quality seeds treated with fungicide.',
            'water_management': 'Moderate irrigation needed. Critical stages are flowering and boll development.',
            'pest_control': 'Monitor for bollworms, aphids and whiteflies. Adopt integrated pest management practices.',
            'harvesting_tips': 'Pick mature bolls when they crack open. Harvest in 3-4 pickings at weekly intervals.'
        },
        'sugarcane': {
            'name': 'Sugarcane',
            'icon': 'fa-leaf',
            'temperature_range': '20-30°C',
            'rainfall_range': '1000-1500mm',
            'growing_season': 'Year-round',
            'humidity_range': '60-80%',
            'soil_type': 'Alluvial, Black',
            'ph_range': '5.5-8.0',
            'drainage': 'Good',
            'fertility': 'High',
            'average_yield': '60-80 tons/hectare',
            'harvest_time': '10-18 months',
            'planting_depth': '6-10 cm',
            'spacing': '90x30 cm',
            'planting_tips': 'Plant healthy setts with 2-3 buds. Prepare land thoroughly and apply organic manures.',
            'water_management': 'Requires heavy irrigation. Critical periods are tillering and grand growth stages.',
            'pest_control': 'Control top borers, scale insects and red rot disease with proper cultural practices.',
            'harvesting_tips': 'Harvest when sugar content peaks (usually 12-18 months). Cut close to ground with sharp knife.'
        }
    }
    
    # Convert crop name to lowercase for matching
    crop_key = crop_name.lower()
    
    # Check if crop exists in our data
    if crop_key not in crop_info:
        # Return a simple error page if crop not found
        return f"<h1>Crop '{crop_name}' not found</h1><a href='/'>Back to Home</a>", 404
    
    # Get crop information
    crop_data = crop_info[crop_key]
    
    # Render the template with crop data
    return render_template('crop_detail.html', 
                         crop_name=crop_data['name'],
                         crop_icon=crop_data['icon'],
                         temperature_range=crop_data['temperature_range'],
                         rainfall_range=crop_data['rainfall_range'],
                         growing_season=crop_data['growing_season'],
                         humidity_range=crop_data['humidity_range'],
                         soil_type=crop_data['soil_type'],
                         ph_range=crop_data['ph_range'],
                         drainage=crop_data['drainage'],
                         fertility=crop_data['fertility'],
                         average_yield=crop_data['average_yield'],
                         harvest_time=crop_data['harvest_time'],
                         planting_depth=crop_data['planting_depth'],
                         spacing=crop_data['spacing'],
                         planting_tips=crop_data['planting_tips'],
                         water_management=crop_data['water_management'],
                         pest_control=crop_data['pest_control'],
                         harvesting_tips=crop_data['harvesting_tips'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['rainfall', 'temperature', 'humidity', 'N', 'P', 'K', 'crop', 'soil_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        result = predictor.get_recommendation(
            float(data['rainfall']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['N']),
            float(data['P']),
            float(data['K']),
            data['crop'],
            data['soil_type'],
            float(data.get('pH', 6.5)),
            float(data.get('area', 5.0)),
            data.get('season', 'Kharif'),
            data.get('irrigation', 'Rainfed')
        )
        
        # Format confidence as percentage
        confidence_percent = int(result['confidence'] * 100)
        
        return jsonify({
            'predicted_yield': f"{result['predicted_yield']:.2f}",
            'fertilizer_recommendation': result['fertilizer_recommendation'],
            'estimated_cost': f"{result['estimated_cost']:.1f}",
            'confidence': confidence_percent
        })
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': 'There was an issue processing your request. Please check your inputs and try again.'
        }), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = authenticate_user(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('predict_page'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if register_user(username, email, password):
            return redirect(url_for('login'))
        else:
            return render_template('register.html', error='Username or email already exists')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    print("Starting Crop Yield Prediction Web App...")
    print("Open: http://localhost:5000")
    app.run(host='localhost', port=5000, debug=True)