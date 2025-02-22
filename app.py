""" app.py """
import numpy as np
import librosa
import obspy
from scipy import signal
from scipy.stats import skew
import pandas as pd
import xml.etree.ElementTree as ET
import joblib
import os
from datetime import datetime, timedelta
import tempfile
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename
import uuid
import json
import threading
import time
import traceback
import logging
import requests
from PIL import Image, ImageDraw  # Fixed import
import io

# Constants (matching train_model.py)
SAMPLE_RATE = 20
FRAME_SIZE = 128  # Added missing constant
HOP_LENGTH = 64   # Added missing constant
N_MELS = 64      # Reduced from 128
FMIN = 0
FMAX = 50        # Adjusted for better coverage of shorter signals

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('usgs_monitor')

# Initialize Flask app
app = Flask(__name__, static_url_path='/static')
app.secret_key = os.environ.get('SECRET_KEY', 'seismic_prediction_key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Increased to 64MB max upload size

# Also add this configuration
app.config['MAX_FILE_SIZE'] = 100 * 1024 * 1024  # 64MB file size limit

# Ensure uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def extract_features(file_path):
    """Extract MFCC features from different types of seismic data files"""
    if file_path.endswith('.mseed'):
        st = obspy.read(file_path)
        tr = st[0]
        y = tr.data.astype(np.float32)
        sr = tr.stats.sampling_rate
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        y = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values.astype(np.float32)
        sr = 100
    elif file_path.endswith('.xml'):
        tree = ET.parse(file_path)
        root = tree.getroot()
        y = np.array([float(child.text) for child in root if child.text.replace('.', '', 1).isdigit()]).astype(np.float32)
        sr = 100
    else:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    if len(y) == 0:
        print(f"Warning: Empty data for file {file_path}")
        return np.zeros(30), np.array([]), sr

    # Adjust frequency parameters based on sampling rate
    # Nyquist frequency is half the sampling rate
    nyquist = sr / 2
    fmax_adjusted = min(FMAX, nyquist * 0.95)  # Stay safely below Nyquist
    
    # Extract MFCCs with adjusted parameters
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS,
        fmin=FMIN, fmax=fmax_adjusted,
        n_fft=FRAME_SIZE, hop_length=HOP_LENGTH
    )
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    
    # Add spectral centroids and RMS
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    
    # Additional features to match the 30 expected by the model
    # Spectral features - with adjusted frequency parameters
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    
    # Try-except for spectral contrast which may fail with very low sampling rates
    try:
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
        contrast_mean = np.mean(spectral_contrast)
    except:
        contrast_mean = 0.0
    
    # Time domain features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0]
    
    # Calculate first derivatives of MFCCs (delta features)
    mfcc_delta = librosa.feature.delta(mfcc)
    
    # Combine all features
    features = np.concatenate([
        np.mean(mfcc, axis=1),  # 13 features
        [np.mean(spectral_centroids)],  # 1 feature
        [np.mean(rms)]  # 1 feature
    ])
    
    # Ensure we have exactly 30 features
    if len(features) < 30:
        # Add additional features if needed
        extra_features = [
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            contrast_mean,
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate),
            # Statistical features of the raw signal
            np.max(y),
            np.min(y),
            np.mean(y),
            np.std(y),
            skew(y)
        ]
        features = np.concatenate([features, extra_features[:30-len(features)]])
    
    # Final check to ensure exactly 30 features
    if len(features) > 30:
        features = features[:30]
    elif len(features) < 30:
        features = np.pad(features, (0, 30 - len(features)), 'constant')
        
    return features, y, sr

def detect_foreshocks(waveform, sr):
    """Identify potential foreshock patterns"""
    frame_length = int(sr * 0.5)
    hop_length = frame_length // 2

    energy = []
    for i in range(0, len(waveform) - frame_length, hop_length):
        frame = waveform[i:i+frame_length]
        energy.append(np.sum(frame**2))

    energy = np.array(energy)

    if len(energy) < 3:
        return False, 0

    energy_smooth = np.convolve(energy, np.ones(3)/3, mode='valid')
    differences = np.diff(energy_smooth)
    increasing_trend = np.mean(differences) > 0

    if increasing_trend:
        positive_diffs = np.sum(differences > 0)
        consistency = positive_diffs / len(differences)
    else:
        consistency = 0

    return increasing_trend, consistency

class EarthquakePredictionApp:
    def __init__(self, model_dir='.'):
        """Initialize the app with pre-trained models"""
        self.models_loaded = False
        try:
            # Load earthquake detection model (LogisticRegression)
            print(f"Loading earthquake model from {os.path.join(model_dir, 'earthquake_model.joblib')}")
            self.eq_model = joblib.load(os.path.join(model_dir, 'earthquake_model.joblib'))
            print(f"Loaded earthquake model type: {type(self.eq_model)}")
            
            self.eq_scaler = joblib.load(os.path.join(model_dir, 'earthquake_scaler.joblib'))
            print(f"Loaded earthquake scaler type: {type(self.eq_scaler)}")
            
            # Verify the model has required methods
            if not hasattr(self.eq_model, 'predict_proba'):
                raise AttributeError(f"Earthquake model (type: {type(self.eq_model)}) doesn't have predict_proba method")
            
            # Load time prediction model (RandomForestRegressor)
            self.time_model = joblib.load(os.path.join(model_dir, 'time_prediction_model.joblib'))
            self.time_scaler = joblib.load(os.path.join(model_dir, 'time_prediction_scaler.joblib'))
            
            # Load precursor detection model (GradientBoostingClassifier)
            self.precursor_model = joblib.load(os.path.join(model_dir, 'precursor_detection_model.joblib'))
            self.precursor_scaler = joblib.load(os.path.join(model_dir, 'precursor_detection_scaler.joblib'))
            
            self.models_loaded = True
            print("All models loaded successfully")
            
        except FileNotFoundError as e:
            print(f"Error: Model file not found: {e}")
            self.models_loaded = False
            raise
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print(f"Error type: {type(e)}")
            print(traceback.format_exc())
            self.models_loaded = False
            raise
    
    def predict_earthquake(self, file_path):
        """Detect if seismic data indicates an earthquake"""
        if not self.models_loaded:
            return {"error": "Models not loaded properly. Please check model files."}
        
        try:
            # Extract features
            features, waveform, sr = extract_features(file_path)
            
            if len(waveform) == 0:
                return {"error": "Empty waveform data"}
            
            # Handle the dimension mismatch
            features_array = np.array(features).reshape(1, -1)
            
            # Check and adjust dimensions
            if features_array.shape[1] > self.eq_scaler.n_features_in_:
                features_array = features_array[:, :self.eq_scaler.n_features_in_]
            elif features_array.shape[1] < self.eq_scaler.n_features_in_:
                features_array = np.pad(features_array, ((0, 0), (0, self.eq_scaler.n_features_in_ - features_array.shape[1])), 
                                    'constant')
            
            # Scale features
            scaled_features = self.eq_scaler.transform(features_array)
            
            # Adjust scaled features for the model
            if scaled_features.shape[1] != 15 and hasattr(self.eq_model, 'n_features_in_') and self.eq_model.n_features_in_ == 15:
                scaled_features = scaled_features[:, :15]
            
            # Get prediction probability
            probability = self.eq_model.predict_proba(scaled_features)[0][1]
            is_earthquake = probability >= 0.5
            
            # Check for foreshock patterns
            has_foreshocks, foreshock_consistency = detect_foreshocks(waveform, sr)
            
            result = {
                "is_earthquake": bool(is_earthquake),
                "probability": float(probability),
                "has_foreshock_pattern": has_foreshocks,
                "foreshock_consistency": float(foreshock_consistency)
            }
            
            # If it's an earthquake, predict time and check for precursors
            if is_earthquake:
                result.update(self.predict_time_to_event(waveform, sr))
                result.update(self.detect_precursors(waveform, sr))
            
            return result
        
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    def predict_time_to_event(self, waveform, sr):
        """Predict time until potential earthquake event"""
        if not self.models_loaded:
            return {"error": "Models not loaded properly"}
        
        try:
            # Extract temporal features
            temporal_features = extract_temporal_features(waveform, sr)
            
            # Scale features
            scaled_features = self.time_scaler.transform([temporal_features])
            
            # Predict hours to event
            hours_prediction = self.time_model.predict(scaled_features)[0]
            
            # Convert to estimated datetime
            now = datetime.now()
            estimated_time = now + timedelta(hours=hours_prediction)
            
            return {
                "hours_to_event": float(hours_prediction),
                "estimated_time": estimated_time.isoformat(),
                "confidence": self._calculate_time_confidence(hours_prediction)
            }
        
        except Exception as e:
            return {"time_prediction_error": str(e)}
    
    def detect_precursors(self, waveform, sr):
        """Detect if the signal contains earthquake precursors"""
        if not self.models_loaded:
            return {"precursor_detection": "unavailable"}
        
        try:
            # Extract temporal features (reused for precursor detection)
            features = extract_temporal_features(waveform, sr)
            
            # Scale features
            scaled_features = self.precursor_scaler.transform([features])
            
            # Predict precursor probability
            precursor_prob = self.precursor_model.predict_proba(scaled_features)[0][1]
            has_precursors = precursor_prob >= 0.6  # Higher threshold for precursors
            
            return {
                "has_precursors": bool(has_precursors),
                "precursor_probability": float(precursor_prob),
                "precursor_confidence": self._calculate_precursor_confidence(precursor_prob)
            }
        
        except Exception as e:
            return {"precursor_detection_error": str(e)}
    
    def _calculate_time_confidence(self, hours_prediction):
        """Calculate confidence level for time prediction"""
        # Confidence decreases as prediction time increases
        if hours_prediction <= 12:
            return "high"
        elif hours_prediction <= 36:
            return "medium"
        else:
            return "low"
    
    def _calculate_precursor_confidence(self, probability):
        """Calculate confidence level for precursor detection"""
        if probability >= 0.8:
            return "high"
        elif probability >= 0.6:
            return "medium"
        else:
            return "low"
    
    def analyze_live_stream(self, stream_source, duration=60, interval=10):
        """Analyze a live data stream at regular intervals"""
        results = []
        
        try:
            # For demonstration, we'll simulate a stream with a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mseed', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # In a real implementation, this would connect to an actual data stream
            print(f"Monitoring stream from {stream_source} at {interval}s intervals")
            
            for i in range(0, duration, interval):
                # Generate longer simulated data to avoid warnings
                st = obspy.Stream([obspy.Trace(data=np.random.randn(SAMPLE_RATE * interval * 2))])
                st.write(temp_path, format='MSEED')
                
                # Analyze the recent data
                result = self.predict_earthquake(temp_path)
                result["timestamp"] = datetime.now().isoformat()
                results.append(result)
                
                print(f"Analysis at {i}s: {'ALERT! ' if result.get('is_earthquake', False) else ''}"\
                      f"Earthquake probability: {result.get('probability', 0):.2f}")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return results
            
        except Exception as e:
            return {"error": f"Stream analysis error: {str(e)}"}

    def generate_alert(self, prediction_result):
        """Generate alert message based on prediction results"""
        if not prediction_result.get("is_earthquake", False):
            return None
        
        probability = prediction_result.get("probability", 0)
        hours = prediction_result.get("hours_to_event", float('inf'))
        has_precursors = prediction_result.get("has_precursors", False)
        precursor_prob = prediction_result.get("precursor_probability", 0)
        
        # Determine alert level
        if probability >= 0.85 and hours < 24 and has_precursors and precursor_prob >= 0.7:
            level = "SEVERE"
        elif probability >= 0.7 and hours < 48:
            level = "WARNING"
        else:
            level = "ADVISORY"
        
        alert = {
            "level": level,
            "message": f"Earthquake {level}: {probability:.1%} probability within {hours:.1f} hours",
            "timestamp": datetime.now().isoformat(),
            "action_required": self._get_action_recommendation(level, hours)
        }
        
        return alert
    
    def _get_action_recommendation(self, alert_level, hours):
        """Get recommended actions based on alert level"""
        if alert_level == "SEVERE":
            return "Immediate evacuation recommended. Activate emergency response protocols."
        elif alert_level == "WARNING":
            return "Prepare for possible evacuation. Review emergency plans and supplies."
        else:
            return "Monitor situation. No immediate action required."

# Global variable for the prediction engine
prediction_engine = EarthquakePredictionApp()

# Dictionary to store monitoring sessions
monitoring_sessions = {}

class USGSEarthquakeMonitor:
    def __init__(self, model_dir='.'):
        """Initialize the app with pre-trained models"""
        self.models_loaded = False
        try:
            # Load earthquake detection model (LogisticRegression)
            self.eq_model = joblib.load(os.path.join(model_dir, 'earthquake_model.joblib'))
            self.eq_scaler = joblib.load(os.path.join(model_dir, 'earthquake_scaler.joblib'))
            
            # Load time prediction model (RandomForestRegressor)
            self.time_model = joblib.load(os.path.join(model_dir, 'time_prediction_model.joblib'))
            self.time_scaler = joblib.load(os.path.join(model_dir, 'time_prediction_scaler.joblib'))
            
            # Load precursor detection model (GradientBoostingClassifier)
            self.precursor_model = joblib.load(os.path.join(model_dir, 'precursor_detection_model.joblib'))
            self.precursor_scaler = joblib.load(os.path.join(model_dir, 'precursor_detection_scaler.joblib'))
            
            # Verify models are correct type with required methods
            if not hasattr(self.eq_model, 'predict_proba'):
                raise AttributeError("Earthquake model doesn't have predict_proba method")
            if not hasattr(self.time_model, 'predict'):
                raise AttributeError("Time model doesn't have predict method")
            if not hasattr(self.precursor_model, 'predict_proba'):
                raise AttributeError("Precursor model doesn't have predict_proba method")
                
            self.models_loaded = True
            print("All models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_loaded = False
        
    def start_monitoring(self, session_id, min_magnitude=2.5, interval_seconds=300):
        """Start continuous monitoring of USGS earthquake data."""
        if self.monitoring:
            return {"error": "Monitoring already in progress"}
        
        self.session_id = session_id
        self.monitoring = True
        
        # Initialize monitoring session data
        monitoring_sessions[session_id] = {
            'stream_source': 'USGS API',
            'started_at': datetime.now().isoformat(),
            'min_magnitude': min_magnitude,
            'interval': interval_seconds,
            'results': [],
            'alerts': [],
            'status': 'running'
        }
        
        # Start monitoring in a separate thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(session_id, min_magnitude, interval_seconds)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info(f"Started USGS monitoring with session ID: {session_id}")
        return {
            "session_id": session_id,
            "status": "monitoring_started",
            "message": f"Monitoring USGS earthquake data (min magnitude: {min_magnitude})"
        }
        
    def stop_monitoring(self, session_id):
        """Stop the monitoring process."""
        if not self.monitoring or self.session_id != session_id:
            return {"error": "No matching monitoring session found"}
        
        self.monitoring = False
        if session_id in monitoring_sessions:
            monitoring_sessions[session_id]['status'] = 'stopped'
        
        logger.info(f"Stopped USGS monitoring for session ID: {session_id}")
        return {"status": "monitoring_stopped"}
    
    def _monitor_loop(self, session_id, min_magnitude, interval_seconds):
        """Continuous monitoring loop that runs until stopped."""
        while self.monitoring and session_id in monitoring_sessions:
            try:
                # Fetch and analyze earthquake data
                earthquakes = self._fetch_recent_earthquakes(min_magnitude)
                results = self._analyze_earthquakes(earthquakes)
                
                # Store results
                if session_id in monitoring_sessions:
                    monitoring_sessions[session_id]['results'].extend(results)
                    
                    # Generate alerts for significant events
                    for result in results:
                        if result.get('is_earthquake', False) and result.get('probability', 0) > 0.7:
                            alert = self.prediction_engine.generate_alert(result)
                            if alert and self._is_new_alert(result['id'], alert):
                                monitoring_sessions[session_id]['alerts'].append(alert)
                
                # Sleep before next check
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in USGS monitoring: {str(e)}")
                if session_id in monitoring_sessions:
                    monitoring_sessions[session_id]['status'] = 'error'
                    monitoring_sessions[session_id]['error'] = str(e)
                break
        
        # Update session status if monitoring was stopped intentionally
        if self.monitoring is False and session_id in monitoring_sessions:
            monitoring_sessions[session_id]['status'] = 'completed'
    
    def _fetch_recent_earthquakes(self, min_magnitude):
        """Fetch recent earthquake data from USGS API."""
        current_time = datetime.now()
        time_window = current_time - timedelta(minutes=5)  # Check last 5 minutes
        
        params = {
            'starttime': time_window.isoformat(),
            'endtime': current_time.isoformat(),
            'minmagnitude': min_magnitude,
            'orderby': 'time'
        }
        
        response = requests.get(self.api_url, params=params)
        if response.status_code != 200:
            logger.error(f"USGS API error: {response.status_code}")
            return []
        
        data = response.json()
        self.last_check_time = current_time
        return data.get('features', [])
    
    def _analyze_earthquakes(self, earthquakes):
        """Analyze earthquake data with the prediction engine."""
        results = []
        
        for quake in earthquakes:
            try:
                # Extract basic earthquake information
                properties = quake.get('properties', {})
                geometry = quake.get('geometry', {})
                
                # Create a simplified data structure for analysis
                waveform = self._simulate_waveform_from_magnitude(properties.get('mag', 0))
                
                # Use the prediction engine to analyze this data
                analysis = self._analyze_event(waveform, 100)  # Assume 100Hz sampling rate
                
                # Enhance the result with USGS metadata
                enhanced_result = {
                    **analysis,
                    'id': quake.get('id', f"usgs_{int(time.time())}"),
                    'magnitude': properties.get('mag', 0),
                    'place': properties.get('place', 'Unknown location'),
                    'time': datetime.fromtimestamp(properties.get('time', 0)/1000).isoformat(),
                    'url': properties.get('url', ''),
                    'coordinates': geometry.get('coordinates', [0, 0, 0]),
                    'source': 'USGS'
                }
                
                results.append(enhanced_result)
                logger.info(f"Analyzed earthquake: {enhanced_result['place']} (M{enhanced_result['magnitude']})")
                
            except Exception as e:
                logger.error(f"Error analyzing earthquake: {str(e)}")
        
        return results
    
    def _simulate_waveform_from_magnitude(self, magnitude):
        """Generate a simulated waveform based on earthquake magnitude."""
        # Parameters adjusted by magnitude
        duration = 20 + (magnitude * 2)  # seconds
        amplitude = magnitude * 0.2
        frequency = 2.0 / (1 + magnitude * 0.1)  # Hz
        
        # Create time array
        sr = 100  # sample rate
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create a decaying sinusoidal waveform
        decay = np.exp(-t / (duration * 0.3))
        waveform = amplitude * decay * np.sin(2 * np.pi * frequency * t)
        
        # Add some realistic noise and P/S wave separation
        noise = np.random.normal(0, 0.05, len(t))
        p_wave = signal.gaussian(len(t), std=len(t)/20) * amplitude * 0.2
        s_wave = np.roll(signal.gaussian(len(t), std=len(t)/15) * amplitude, int(len(t)/5))
        
        combined = waveform + noise + p_wave + s_wave
        return combined
    
    def _analyze_event(self, waveform, sr):
        """Analyze a simulated waveform to predict earthquake characteristics."""
        # Extract features from the waveform
        temporal_features = extract_temporal_features(waveform, sr)
        
        # Determine if this has characteristics of an earthquake
        probability = min(0.5 + (np.max(np.abs(waveform)) * 0.5), 0.99)
        is_earthquake = probability > 0.7
        
        # Check for foreshock patterns
        has_foreshocks, foreshock_consistency = detect_foreshocks(waveform, sr)
        
        result = {
            "is_earthquake": bool(is_earthquake),
            "probability": float(probability),
            "has_foreshock_pattern": has_foreshocks,
            "foreshock_consistency": float(foreshock_consistency)
        }
        
        # If it's an earthquake, predict time and check for precursors
        if is_earthquake:
            # For USGS data, we know the event already happened, so adjust time prediction
            result.update({
                "hours_to_event": 0,
                "estimated_time": datetime.now().isoformat(),
                "confidence": "high"
            })
            
            # Detect precursors (aftershock likelihood)
            precursor_prob = 0.3 + (np.std(waveform) * 0.7)
            has_precursors = precursor_prob > 0.6
            
            result.update({
                "has_precursors": bool(has_precursors),
                "precursor_probability": float(precursor_prob),
                "precursor_confidence": prediction_engine._calculate_precursor_confidence(precursor_prob)
            })
        
        return result
    
    def _is_new_alert(self, event_id, alert):
        """Check if this is a new alert we haven't seen before."""
        if event_id in self.alert_history:
            return False
        
        # Store this alert in history to avoid duplicates
        self.alert_history[event_id] = {
            'timestamp': datetime.now(),
            'level': alert.get('level')
        }
        
        # Clean up old alerts from history (keep last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.alert_history = {
            k: v for k, v in self.alert_history.items()
            if v['timestamp'] > cutoff
        }
        
        return True

# Add the USGS monitor to the Flask app
usgs_monitor = None

@app.route('/api/start_usgs_monitoring', methods=['POST'])
def start_usgs_monitoring():
    """API endpoint to start monitoring USGS earthquake data"""
    global usgs_monitor
    
    if usgs_monitor is None:
        usgs_monitor = USGSEarthquakeMonitor(prediction_engine)
    
    data = request.json
    min_magnitude = float(data.get('min_magnitude', 2.5))
    interval_seconds = int(data.get('interval', 300))
    
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    
    # Start monitoring
    result = usgs_monitor.start_monitoring(
        session_id=session_id,
        min_magnitude=min_magnitude,
        interval_seconds=interval_seconds
    )
    
    return jsonify(result)

@app.route('/api/stop_usgs_monitoring/<session_id>', methods=['POST'])
def stop_usgs_monitoring(session_id):
    """API endpoint to stop USGS monitoring"""
    global usgs_monitor
    
    if usgs_monitor is None:
        return jsonify({"error": "No active monitoring instance"}), 404
    
    result = usgs_monitor.stop_monitoring(session_id)
    return jsonify(result)


@app.route('/api/usgs_data')
def get_usgs_data():
    """API endpoint to fetch USGS earthquake data"""
    # Get query parameters
    time_range = request.args.get('timeRange', 'day')
    min_magnitude = float(request.args.get('minMagnitude', 2.5))
    
    # Calculate date range
    end_time = datetime.now()
    if time_range == 'day':
        start_time = end_time - timedelta(days=1)
    elif time_range == 'week':
        start_time = end_time - timedelta(days=7)
    elif time_range == 'month':
        start_time = end_time - timedelta(days=30)
    else:
        start_time = end_time - timedelta(days=1)  # Default to 1 day
    
    # Format times for API request
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%S")
    
    # Build request URL
    api_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        'format': 'geojson',
        'starttime': start_time_str,
        'endtime': end_time_str,
        'minmagnitude': min_magnitude,
        'orderby': 'time'
    }
    
    try:
        # Make request to USGS API
        response = requests.get(api_url, params=params)
        
        if response.status_code == 200:
            earthquake_data = response.json()
            return jsonify(earthquake_data)
        else:
            return jsonify({
                'error': f'USGS API returned status code {response.status_code}',
                'features': []
            }), 500
            
    except Exception as e:
        logger.error(f"Error fetching USGS data: {str(e)}")
        return jsonify({
            'error': str(e),
            'features': []
        }), 500

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload for analysis"""
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('upload.html', error="No file part")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error="No selected file")
        
        # Check file size before saving
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > app.config['MAX_FILE_SIZE']:
            return render_template('upload.html', error="File size exceeds 64MB limit")
        
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                secure_filename(file.filename))
        file.save(file_path)
        
        # Generate a unique ID for this analysis session
        session_id = str(uuid.uuid4())
        
        # Store the file path in the session
        session['analysis_file'] = file_path
        session['session_id'] = session_id
        
        # Redirect to the prediction dashboard
        return redirect(url_for('prediction_dashboard'))
    
    return render_template('upload.html')

@app.route('/prediction')
def prediction_dashboard():
    """Render the prediction dashboard"""
    try:
        if 'analysis_file' not in session:
            return redirect(url_for('upload_file'))
        
        # Retrieve file path from session
        file_path = session.get('analysis_file')
        
        if not os.path.exists(file_path):
            flash('Analysis file no longer exists. Please upload again.')
            return redirect(url_for('upload_file'))
        
        try:
            # Perform the earthquake prediction
            result = prediction_engine.predict_earthquake(file_path)
            
            # Clean up the file after analysis
            os.remove(file_path)
            
            if 'error' in result:
                flash(f'Analysis error: {result["error"]}')
                return redirect(url_for('upload_file'))
            
            # Generate alert if it's an earthquake
            alert = None
            if result.get('is_earthquake', False):
                alert = prediction_engine.generate_alert(result)
            
            # Ensure all values are JSON serializable
            result = {k: str(v) if not isinstance(v, (bool, int, float, str, list, dict, type(None))) 
                     else v for k, v in result.items()}
            
            return render_template('prediction_dashboard.html', 
                                result=result, 
                                alert=alert,
                                file_name=os.path.basename(file_path))
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            print(traceback.format_exc())
            flash(f'Error analyzing file: {str(e)}')
            return redirect(url_for('upload_file'))
            
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for analyzing seismic data"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                secure_filename(file.filename))
        file.save(file_path)
        
        # Perform the earthquake prediction
        result = prediction_engine.predict_earthquake(file_path)
        
        # Clean up the file after analysis
        os.remove(file_path)
        
        # Ensure all values in result are JSON serializable
        result = {k: str(v) if not isinstance(v, (bool, int, float, str, list, dict, type(None))) 
                 else v for k, v in result.items()}
        
        # Generate alert if it's an earthquake
        if result.get('is_earthquake', False):
            alert = prediction_engine.generate_alert(result)
            if alert:
                result['alert'] = alert
        
        return jsonify(result)
    
    except Exception as e:
        # Log the full error for debugging
        print(f"Error in api_analyze: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "error": "Analysis failed",
            "details": str(e)
        }), 500

@app.route('/monitor', methods=['GET'])
def monitor_page():
    """Render the monitoring page"""
    return render_template('monitor.html')

@app.route('/api/start_monitoring', methods=['POST'])
def start_monitoring():
    """API endpoint to start monitoring a stream"""
    data = request.json
    stream_source = data.get('stream_source', 'default_stream')
    duration = int(data.get('duration', 300))  # 5 minutes default
    interval = int(data.get('interval', 10))   # 10 seconds default
    
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    
    # Start monitoring in a separate thread
    thread = threading.Thread(
        target=run_monitoring_session,
        args=(session_id, stream_source, duration, interval)
    )
    thread.daemon = True
    thread.start()
    
    # Store session info
    monitoring_sessions[session_id] = {
        'stream_source': stream_source,
        'started_at': datetime.now().isoformat(),
        'duration': duration,
        'interval': interval,
        'results': [],
        'status': 'running'
    }
    
    return jsonify({
        'session_id': session_id,
        'message': f'Monitoring started for {stream_source}',
        'duration': duration,
        'interval': interval
    })

def run_monitoring_session(session_id, stream_source, duration, interval):
    """Run a monitoring session in a separate thread"""
    try:
        results = prediction_engine.analyze_live_stream(
            stream_source=stream_source,
            duration=duration,
            interval=interval
        )
        
        # Update session with results
        if session_id in monitoring_sessions:
            monitoring_sessions[session_id]['results'] = results
            monitoring_sessions[session_id]['status'] = 'completed'
            
            # Find any alerts
            alerts = []
            for result in results:
                if result.get('is_earthquake', False):
                    alert = prediction_engine.generate_alert(result)
                    if alert:
                        alerts.append(alert)
            
            monitoring_sessions[session_id]['alerts'] = alerts
    
    except Exception as e:
        if session_id in monitoring_sessions:
            monitoring_sessions[session_id]['status'] = 'error'
            monitoring_sessions[session_id]['error'] = str(e)

@app.route('/api/monitoring_status/<session_id>', methods=['GET'])
def get_monitoring_status(session_id):
    """API endpoint to get the status of a monitoring session"""
    if session_id not in monitoring_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session_data = monitoring_sessions[session_id]
    
    # Determine progress percentage
    if session_data['status'] == 'completed':
        progress = 100
    elif session_data['status'] == 'error':
        progress = 0
    else:
        # Check if this is a USGS monitoring session (has no duration)
        if 'duration' in session_data:
            # Regular monitoring session with specified duration
            start_time = datetime.fromisoformat(session_data['started_at'])
            elapsed_seconds = (datetime.now() - start_time).total_seconds()
            progress = min(100, int((elapsed_seconds / session_data['duration']) * 100))
        else:
            # USGS monitoring session (continuous)
            progress = -1  # Indicate continuous monitoring
    
    response = {
        'status': session_data['status'],
        'progress': progress,
        'started_at': session_data['started_at'],
        'monitoring_type': 'usgs' if 'min_magnitude' in session_data else 'regular'
    }
    
    # Include results if completed
    if session_data['status'] == 'completed':
        response['results'] = session_data['results']
        response['alerts'] = session_data.get('alerts', [])
    elif session_data['status'] == 'error':
        response['error'] = session_data.get('error', 'Unknown error')
    
    return jsonify(response)

@app.route('/api/latest_results/<session_id>', methods=['GET'])
def get_latest_results(session_id):
    """API endpoint to get the latest results from a monitoring session"""
    if session_id not in monitoring_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session_data = monitoring_sessions[session_id]
    results = session_data.get('results', [])
    
    # Get the most recent result
    latest_result = results[-1] if results else None
    
    return jsonify({
        'latest_result': latest_result,
        'total_results': len(results),
        'status': session_data['status']
    })

@app.route('/stream_data')
def stream_data():
    """API endpoint for streaming data analysis"""
    try:
        results = prediction_engine.analyze_live_stream("simulated_stream", duration=30, interval=5)
        
        # Ensure results are JSON serializable
        clean_results = []
        for result in results:
            clean_result = {k: str(v) if not isinstance(v, (bool, int, float, str, list, dict, type(None))) 
                          else v for k, v in result.items()}
            clean_results.append(clean_result)
        
        return jsonify(clean_results)
    
    except Exception as e:
        print(f"Error in stream_data: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "error": "Stream analysis failed",
            "details": str(e)
        }), 500

# Helper function for secure filenames
def secure_filename(filename):
    """Securely process filename before saving"""
    # First, keep only the basename
    filename = os.path.basename(filename)
    
    # Replace potentially dangerous characters
    filename = ''.join(c for c in filename if c.isalnum() or c in '._- ')
    
    # Ensure uniqueness with timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    name, ext = os.path.splitext(filename)
    return f"{name}_{timestamp}{ext}"

@app.errorhandler(500)
def handle_500_error(error):
    """Handle internal server errors with JSON response"""
    return jsonify({
        'error': 'Internal server error',
        'details': str(error)
    }), 500

if __name__ == '__main__':
    # Check if models are loaded
    if not prediction_engine.models_loaded:
        print("Warning: Models could not be loaded. Predictions will not work.")
    
    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)