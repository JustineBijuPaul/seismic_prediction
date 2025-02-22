#train_model.py
import os
import numpy as np
import pandas as pd
import librosa
import joblib
import scipy
from sklearn import metrics         
from sklearn.model_selection import GridSearchCV
import sklearn.preprocessing as preprocessing 
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import obspy                        
import xml.etree.ElementTree as ET  
import tempfile                     
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from scipy import signal
from scipy.stats import skew
from scipy import stats
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', message='n_fft=.* is too large for input signal of length=.*')

# Define lists for file paths - these should be populated based on your data
quake_files = []  # Add paths to earthquake data files
no_quake_files = [] # Add paths to non-earthquake data files

SAMPLE_RATE = 100  
FRAME_SIZE = 512   
HOP_LENGTH = 256   
N_MELS = 128      
FMIN = 0          
FMAX = 19         

# Load pre-generated waveforms if available
try:
    all_waveforms = joblib.load('train/all_waveforms.joblib')
    all_sample_rates = joblib.load('train/all_sample_rates.joblib')
    print(f"Loaded {len(all_waveforms)} waveforms from joblib files")
except FileNotFoundError:
    all_waveforms = []
    all_sample_rates = []
    print("No pre-generated waveforms found")

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
        return np.zeros(13)  

    # Extract MFCCs (proven effective in original implementation)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX,
        n_fft=FRAME_SIZE, hop_length=HOP_LENGTH
    )
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    
    # Add spectral centroids and RMS (from app.py)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    
    # Make sure enhanced_features has a consistent length
    enhanced_features = extract_enhanced_features(y, sr)
    if len(enhanced_features) != 13:  # Assuming expected length is 13
        print(f"Warning: enhanced_features has unexpected length {len(enhanced_features)}")
        # Pad or truncate to make 13 features
        if len(enhanced_features) < 13:
            enhanced_features = np.pad(enhanced_features, (0, 13 - len(enhanced_features)))
        else:
            enhanced_features = enhanced_features[:13]
    
    # Combine features with consistent lengths
    features = np.concatenate([
        np.mean(mfcc, axis=1),  # 13 features
        [np.mean(spectral_centroids), np.std(spectral_centroids)],  # 2 features
        [np.mean(rms), np.std(rms)],  # 2 features
        enhanced_features  # Now guaranteed to be 13 features
    ])
    
    return features, y, sr

def extract_enhanced_features(waveform, sr):
    """Extract earthquake-specific features to reduce false positives"""
    if len(waveform) < FRAME_SIZE:
        return np.zeros(20)  # Return zeros for too short signals
    
    # 1. Frequency domain analysis - earthquakes have specific spectral signatures
    fft_spectrum = np.abs(np.fft.rfft(waveform))
    freqs = np.fft.rfftfreq(len(waveform), 1/sr)
    
    # 2. Earthquake-specific frequency bands energy
    low_freq_energy = np.sum(fft_spectrum[freqs < 10]**2) / np.sum(fft_spectrum**2)
    mid_freq_energy = np.sum(fft_spectrum[(freqs >= 10) & (freqs < 30)]**2) / np.sum(fft_spectrum**2)
    high_freq_energy = np.sum(fft_spectrum[freqs >= 30]**2) / np.sum(fft_spectrum**2)
    
    # 3. P-wave to S-wave energy ratio estimation (simplified)
    # Earthquakes typically have distinct P and S waves
    first_third = waveform[:len(waveform)//3]
    last_two_thirds = waveform[len(waveform)//3:]
    p_energy = np.sum(first_third**2)
    s_energy = np.sum(last_two_thirds**2)
    ps_ratio = p_energy / (s_energy + 1e-10)  # Avoid division by zero
    
    # 4. Signal complexity and entropy
    # Earthquakes tend to have higher signal complexity
    signal_diff = np.diff(waveform)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(signal_diff)))) / len(signal_diff)
    
    # Shannon entropy - earthquakes have higher entropy
    hist, _ = np.histogram(waveform, bins=50)
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # 5. Signal envelope progression (earthquakes build up then decay)
    analytic_signal = signal.hilbert(waveform)
    amplitude_envelope = np.abs(analytic_signal)
    
    # Split envelope into thirds
    thirds = np.array_split(amplitude_envelope, 3)
    third_means = [np.mean(third) for third in thirds]
    
    # Typically earthquakes build up to maximum then decay
    envelope_shape = third_means[1] / (third_means[0] + 1e-10)
    envelope_decay = third_means[1] / (third_means[2] + 1e-10)
    
    # 6. Spectral kurtosis - higher for impulsive signals (earthquakes)
    spec_kurt = stats.kurtosis(fft_spectrum)
    
    # 7. Duration of signal exceeding background level
    background = np.median(np.abs(waveform))
    signal_duration = np.sum(np.abs(waveform) > 3*background) / sr  # in seconds
    
    features = np.array([
        low_freq_energy, mid_freq_energy, high_freq_energy,
        ps_ratio, zero_crossings, entropy,
        third_means[0], third_means[1], third_means[2],
        envelope_shape, envelope_decay, spec_kurt,
        signal_duration
    ])
    
    return features

def extract_temporal_features(waveform, sr):
    """
    Enhanced version combining the best from both implementations
    """
    if len(waveform) == 0:
        return np.zeros(19)
    
    # Calculate Hilbert transform for envelope (from app.py)
    analytic_signal = signal.hilbert(waveform)
    amplitude_envelope = np.abs(analytic_signal)
    
    # Segment the signal (from train_model.py)
    segment_length = len(waveform) // 5
    segments = [waveform[i*segment_length:(i+1)*segment_length] for i in range(5)]

    # Extract per-segment features
    energy_features = [np.sum(segment**2) for segment in segments]
    zcr_features = [librosa.feature.zero_crossing_rate(segment)[0].mean() 
                   if len(segment) > 0 else 0 for segment in segments]

    # Extract spectral centroids
    centroid_features = []
    for segment in segments:
        if len(segment) > FRAME_SIZE:
            centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0].mean()
            centroid_features.append(centroid)
        else:
            centroid_features.append(0)

    # Envelope statistics (combine both approaches)
    envelope_mean = np.mean(amplitude_envelope)
    envelope_std = np.std(amplitude_envelope)
    envelope_max = np.max(amplitude_envelope)
    envelope_skew = skew(amplitude_envelope) if len(amplitude_envelope) > 0 else 0

    all_features = energy_features + zcr_features + centroid_features + [envelope_mean, envelope_std, envelope_max, envelope_skew]
    return np.array(all_features)

def detect_foreshocks(waveform, sr):
    """Identify potential foreshock patterns (from app.py)"""
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

def load_data(quake_files, no_quake_files):
    """Load and process both earthquake and non-earthquake data files"""
    X, y = [], []  
    local_waveforms = []
    local_sample_rates = []
    
    expected_feature_length = None  # Track the expected feature length

    # Process earthquake files
    for file in quake_files:
        if os.path.exists(file):
            try:
                features, waveform, sr = extract_features(file)
                
                # Set expected length if first file or validate consistency
                if expected_feature_length is None:
                    expected_feature_length = len(features)
                elif len(features) != expected_feature_length:
                    print(f"Warning: Inconsistent feature length in {file}. "
                          f"Expected {expected_feature_length}, got {len(features)}")
                    # Pad or truncate to make consistent
                    if len(features) < expected_feature_length:
                        features = np.pad(features, (0, expected_feature_length - len(features)))
                    else:
                        features = features[:expected_feature_length]
                
                print(f"Extracted features for quake file {file}: shape={features.shape}")
                X.append(features)
                y.append(1)
                local_waveforms.append(waveform)
                local_sample_rates.append(sr)
            except Exception as e:
                print(f"Error processing quake file {file}: {e}")
        else:
            print(f"File not found: {file}")

    # Process non-earthquake files
    for file in no_quake_files:
        if os.path.exists(file):
            try:
                features, waveform, sr = extract_features(file)
                
                # Ensure consistent feature lengths
                if expected_feature_length is not None and len(features) != expected_feature_length:
                    print(f"Warning: Inconsistent feature length in {file}. "
                          f"Expected {expected_feature_length}, got {len(features)}")
                    # Pad or truncate to make consistent
                    if len(features) < expected_feature_length:
                        features = np.pad(features, (0, expected_feature_length - len(features)))
                    else:
                        features = features[:expected_feature_length]
                
                print(f"Extracted features for no quake file {file}: shape={features.shape}")
                X.append(features)
                y.append(0)
                local_waveforms.append(waveform)
                local_sample_rates.append(sr)
            except Exception as e:
                print(f"Error processing no quake file {file}: {e}")
        else:
            print(f"File not found: {file}")

    # Convert to numpy arrays with safety check
    if X:
        try:
            X_array = np.array(X)
            y_array = np.array(y)
            print(f"Created feature array with shape: {X_array.shape}")
            return X_array, y_array, local_waveforms, local_sample_rates
        except Exception as e:
            print(f"Error converting to numpy arrays: {e}")
            # Fall back to lists if conversion fails
            return X, y, local_waveforms, local_sample_rates
    else:
        return np.array([]), np.array([]), [], []

def train_with_cross_validation(X, y, n_splits=5):
    """Train with cross-validation to ensure robust model performance"""
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = preprocessing.RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        clf = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            solver='liblinear',
            penalty='l1'
        )
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1)
        
        print(f"Fold {fold+1}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    print("\nCross-validation results:")
    print(f"Mean Precision: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
    print(f"Mean Recall: {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
    print(f"Mean F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

def train_earthquake_detection_model(X, y):
    """Improved earthquake detection model with reduced false positives"""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Apply robust scaling (less sensitive to outliers)
    scaler = preprocessing.RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 2. Feature selection to identify truly discriminative features
    selector = SelectFromModel(
        GradientBoostingClassifier(n_estimators=100, random_state=42), 
        threshold="median"
    )
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # 3. Handle class imbalance if present
    if len(y_train[y_train==0]) > 2*len(y_train[y_train==1]) or len(y_train[y_train==1]) > 2*len(y_train[y_train==0]):
        smote = SMOTE(random_state=42)
        X_train_selected, y_train = smote.fit_resample(X_train_selected, y_train)
        print("Applied SMOTE to balance classes")
    
    # 4. Grid search for optimal parameters with focus on reducing false positives
    param_grid = {
        'C': [0.1, 0.5, 1.0, 2.0],
        'class_weight': [None, 'balanced', {0:1, 1:0.7}],  # Penalize false positives more
        'solver': ['liblinear', 'saga'],
        'penalty': ['l1', 'l2']
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_grid=param_grid,
        cv=5,
        scoring='f1',  # Balance precision and recall
        n_jobs=-1
    )
    
    grid_search.fit(X_train_selected, y_train)
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    
    # Use best model
    clf = grid_search.best_estimator_
    
    # 5. Adjust decision threshold to reduce false positives
    y_scores = clf.predict_proba(X_test_selected)[:, 1]
    
    # Find threshold that maximizes F1 score
    thresholds = np.arange(0.3, 0.8, 0.05)
    f1_scores = []
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred_threshold = (y_scores >= threshold).astype(int)
        f1_scores.append(metrics.f1_score(y_test, y_pred_threshold))
        precisions.append(metrics.precision_score(y_test, y_pred_threshold))
        recalls.append(metrics.recall_score(y_test, y_pred_threshold))
    
    # Plot threshold tuning curves
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, 'b-', label='F1 Score')
    plt.plot(thresholds, precisions, 'g-', label='Precision')
    plt.plot(thresholds, recalls, 'r-', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs. Decision Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig('threshold_tuning.png')
    
    # Find optimal threshold for precision-recall balance
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"Optimal decision threshold: {best_threshold:.2f}")
    
    # Final evaluation with optimal threshold
    y_pred = (y_scores >= best_threshold).astype(int)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    
    print("\nEarthquake Detection Model Performance (with optimal threshold):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (focus on reducing false positives)")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    cm = metrics.confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Store threshold with model
    model_info = {
        'model': clf,
        'selector': selector,
        'threshold': best_threshold
    }

    joblib.dump(clf, 'earthquake_model.joblib')
    joblib.dump(selector, 'earthquake_selector.joblib')
    joblib.dump(best_threshold, 'earthquake_threshold.joblib')
    joblib.dump(scaler, 'earthquake_scaler.joblib')
    
    return clf, scaler

def predict_earthquake(features, model_info, scaler):
    features_scaled = scaler.transform([features])
    features_selected = model_info['selector'].transform(features_scaled)
    probability = model_info['model'].predict_proba(features_selected)[0, 1]
    prediction = int(probability >= model_info['threshold'])
    return prediction, probability


def train_temporal_prediction_model(dataset_path):
    """
    Train a hybrid model to predict time until next seismic event
    Uses RandomForestRegressor as primary model with ARIMA capabilities
    """
    data = pd.read_csv(dataset_path)

    X = data.drop(['hours_to_event', 'event_id'], axis=1, errors='ignore')
    y = data['hours_to_event']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Using RandomForestRegressor (effective for regression problems)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)

    print(f"Time-to-event prediction model performance:")
    print(f"Mean Absolute Error: {mae:.2f} hours")
    print(f"R² Score: {r2:.4f}")

    feature_importance = model.feature_importances_
    feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    importance_df = importance_df.sort_values('importance', ascending=False)
    print("\nTop 5 important features for time prediction:")
    print(importance_df.head(5))

    joblib.dump(model, 'time_prediction_model.joblib')
    joblib.dump(scaler, 'time_prediction_scaler.joblib')

    return model, scaler

def train_precursor_detection_model(data_path):
    """
    Train a model to identify seismic precursors using GradientBoostingClassifier
    """
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading precursor data: {e}")
        return None, None

    X = data.drop(['is_precursor', 'segment_id'], axis=1, errors='ignore')
    y = data['is_precursor']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Using GradientBoostingClassifier (better for pattern detection)
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nPrecursor Detection Model Performance:")
    print(metrics.classification_report(y_test, y_pred))

    y_proba = model.predict_proba(X_test)[:, 1]
    auc_roc = metrics.roc_auc_score(y_test, y_proba)
    print(f"AUC-ROC: {auc_roc:.4f}")

    if hasattr(model, 'feature_importances_'):
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
        importance_df = importance_df.sort_values('importance', ascending=False)
        print("\nTop 5 important features for precursor detection:")
        print(importance_df.head(5))

    joblib.dump(model, 'precursor_detection_model.joblib')
    joblib.dump(scaler, 'precursor_detection_scaler.joblib')

    return model, scaler

def prepare_historical_time_series(data, window_size=7):
    """
    Prepare historical time series data for ARIMA modeling
    """
    # Extract timestamps and create time series
    timestamps = [entry.get('timestamp', datetime.now()) for entry in data 
                if entry.get('is_seismic', False)]
    timestamps.sort()
    
    if len(timestamps) < 2:
        return None
        
    # Calculate intervals between events
    intervals = []
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
        intervals.append(delta)
    
    return intervals

def predict_with_arima(time_series, steps=1):
    """
    Use ARIMA model for time series prediction
    """
    if not time_series or len(time_series) < 3:
        return None
        
    try:
        # Use ARIMA model with auto-determined parameters
        model = ARIMA(time_series, order=(2,1,1))
        model_fit = model.fit()
        
        # Get forecast with confidence intervals
        forecast = model_fit.forecast(steps=steps)
        forecast_value = forecast[0]
        
        return max(0, forecast_value)
    except Exception as e:
        print(f"ARIMA prediction error: {e}")
        return None

def validate_input_files(file_paths):
    """Validate that input files exist and are readable"""
    valid_files = []
    invalid_files = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            invalid_files.append((file_path, "File not found"))
            continue

        if not os.path.isfile(file_path):
            invalid_files.append((file_path, "Not a file"))
            continue

        try:
            if file_path.endswith('.mseed'):
                obspy.read(file_path)
            elif file_path.endswith('.csv'):
                pd.read_csv(file_path)
            elif file_path.endswith('.xml'):
                ET.parse(file_path)
            else:
                librosa.load(file_path, sr=SAMPLE_RATE, duration=1)  

            valid_files.append(file_path)
        except Exception as e:
            invalid_files.append((file_path, f"Error reading file: {str(e)}"))

    return valid_files, invalid_files

def generate_synthetic_temporal_dataset(output_path, n_samples=500):
    """
    Generate synthetic data for temporal prediction model training
    This is useful for initial development until you have real historical data
    """
    np.random.seed(42)

    event_ids = np.arange(1, n_samples + 1)

    energy_features = np.random.exponential(1, size=(n_samples, 5)) * 10

    zcr_features = np.random.normal(500, 150, size=(n_samples, 5))
    zcr_features = np.abs(zcr_features)

    centroid_features = np.random.normal(2000, 500, size=(n_samples, 5))
    centroid_features = np.abs(centroid_features)

    envelope_mean = np.random.exponential(0.5, size=n_samples)
    envelope_std = np.random.uniform(0.1, 0.3, size=n_samples)
    envelope_max = envelope_mean + envelope_std * np.random.normal(2, 0.5, size=n_samples)
    envelope_skew = np.random.normal(0.5, 0.2, size=n_samples)

    base_hours = np.random.gamma(shape=2, scale=12, size=n_samples)

    energy_factor = 1 - 0.2 * (energy_features[:, -1] / np.max(energy_features[:, -1]))

    envelope_factor = 1 - 0.3 * (envelope_max / np.max(envelope_max))

    hours_to_event = base_hours * energy_factor * envelope_factor
    hours_to_event = np.clip(hours_to_event, 1, 72)  

    df = pd.DataFrame()
    df['event_id'] = event_ids

    for i in range(5):
        df[f'energy_segment_{i}'] = energy_features[:, i]
        df[f'zcr_segment_{i}'] = zcr_features[:, i]
        df[f'centroid_segment_{i}'] = centroid_features[:, i]

    df['envelope_mean'] = envelope_mean
    df['envelope_std'] = envelope_std
    df['envelope_max'] = envelope_max
    df['envelope_skew'] = envelope_skew

    df['hours_to_event'] = hours_to_event

    df.to_csv(output_path, index=False)
    print(f"Generated synthetic dataset with {n_samples} samples at {output_path}")

    return df

def generate_synthetic_precursor_dataset(output_path, n_samples=400):
    """
    Generate synthetic data for precursor detection model training
    """
    np.random.seed(42)
    
    segment_ids = np.arange(1, n_samples + 1)
    
    # Create balanced dataset (50% precursors, 50% non-precursors)
    is_precursor = np.concatenate([
        np.ones(n_samples // 2), 
        np.zeros(n_samples - n_samples // 2)
    ])
    np.random.shuffle(is_precursor)
    
    # Generate features that distinguish precursors
    features = {}
    
    # Energy rising trend for precursors
    energy_base = np.random.exponential(1, size=n_samples) * 5
    energy_trend = np.zeros((n_samples, 5))
    
    for i in range(n_samples):
        if is_precursor[i]:
            # Rising energy pattern for precursors
            trend = np.linspace(0.5, 1.5, 5) + np.random.normal(0, 0.1, 5)
        else:
            # Random or declining pattern for non-precursors
            trend = np.random.normal(1, 0.2, 5)
        
        energy_trend[i] = energy_base[i] * trend
    
    for i in range(5):
        features[f'energy_{i}'] = energy_trend[:, i]
    
    # Frequency characteristics
    for i in range(3):
        if i == 0:
            # Low frequency content (higher in precursors)
            features[f'low_freq_{i}'] = np.random.normal(
                0.7 + 0.2 * is_precursor, 0.1, n_samples
            )
        elif i == 1:
            # Mid frequency content
            features[f'mid_freq_{i}'] = np.random.normal(
                0.5, 0.1, n_samples
            )
        else:
            # High frequency content (lower in precursors)
            features[f'high_freq_{i}'] = np.random.normal(
                0.3 - 0.1 * is_precursor, 0.1, n_samples
            )
    
    # Envelope features
    features['envelope_smoothness'] = np.random.normal(
        0.3 + 0.2 * is_precursor, 0.1, n_samples
    )
    
    features['spectral_flux'] = np.random.normal(
        0.4 + 0.3 * is_precursor, 0.15, n_samples
    )
    
    features['waveform_complexity'] = np.random.normal(
        0.6 - 0.2 * is_precursor, 0.1, n_samples
    )
    
    # Create DataFrame
    df = pd.DataFrame(features)
    df['segment_id'] = segment_ids
    df['is_precursor'] = is_precursor
    
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic precursor dataset with {n_samples} samples at {output_path}")
    
    return df

def process_preloaded_waveforms():
    """Process the pre-generated waveforms if available"""
    if not all_waveforms or not all_sample_rates:
        print("No pre-generated waveforms found to process")
        return None, None
    
    print(f"Processing {len(all_waveforms)} pre-generated waveforms...")
    
    X = []
    # Determine labels based on filenames in dataset info
    y = []
    
    for i, (waveform, sr) in enumerate(zip(all_waveforms, all_sample_rates)):
        # Extract features from waveform
        S = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_mels=N_MELS,
            fmin=FMIN, fmax=FMAX,
            n_fft=FRAME_SIZE, hop_length=HOP_LENGTH
        )
        log_S = librosa.power_to_db(S, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
        
        spectral_centroids = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        rms = librosa.feature.rms(y=waveform)[0]
        
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            [np.mean(spectral_centroids), np.std(spectral_centroids)],
            [np.mean(rms), np.std(rms)]
        ])
        
        X.append(features)
        
        # Determine if this is a quake waveform
        # Since we don't have the original filenames, we'll assume the first half are quakes
        # This should be updated with more accurate logic if the dataset structure is known
        if i < len(all_waveforms) // 2:
            y.append(1)  # Earthquake
        else:
            y.append(0)  # Non-earthquake
    
    return np.array(X), np.array(y)

def main():
    """Main function to orchestrate the training process"""
    print("====== Seismic Detection and Prediction Model Training ======")

    print("\n--- PART 1: Training Earthquake Detection Model ---")

    # Check if pre-generated waveforms are available
    if all_waveforms and len(all_waveforms) > 0:
        print(f"Using {len(all_waveforms)} pre-generated waveforms...")
        X, y = process_preloaded_waveforms()
        if X is not None and len(X) > 0:
            # Add cross-validation here before the final training
            train_with_cross_validation(X, y, n_splits=5)
            
            # Then proceed with final model training
            clf, scaler = train_earthquake_detection_model(X, y)
            joblib.dump(clf, 'earthquake_model.joblib')
            joblib.dump(scaler, 'earthquake_scaler.joblib')
            print("\nEarthquake detection model trained with pre-generated waveforms.")
        else:
            print("Failed to process pre-generated waveforms. Continuing with synthetic data.")
            # Generate synthetic data for model development
            n_samples = 100
            n_features = 17
            X_synthetic = np.random.randn(n_samples, n_features)
            y_synthetic = np.random.randint(0, 2, n_samples)
            clf, scaler = train_earthquake_detection_model(X_synthetic, y_synthetic)
            joblib.dump(clf, 'earthquake_model.joblib')
            joblib.dump(scaler, 'earthquake_scaler.joblib')
            print("\nEarthquake detection model trained with synthetic data.")
    else:
        # No pre-generated waveforms available, check for files
        quake_files = ['/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.vea.2018.353.5.mseed',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.02.vma.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.vea.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.vmb.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.vkc.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.vkp.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.vev.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.14.vkp.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.vea.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.vk1.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.vk2.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.vkd.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.vka.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.vea.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.vev.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.vmb.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.vev.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.vea.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.vmc.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.352.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.vk1.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.vev.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.vev.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.vev.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.vk1.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.vev.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.12.uev.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.vev.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.vk2.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.25.vev.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.vk1.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.vev.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.16.vea.2019.001.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.27.vk1.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.vk1.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.vk2.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.vev.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.vev.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.vk2.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.vev.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.vea.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.vev.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.vyo.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.vk2.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.vea.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.vev.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.vev.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.vkp.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.40.vev.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.vkc.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.vea.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.03.uma.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.30.vk1.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.vea.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.vk1.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.vmb.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.vk2.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.vev.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.24.vk2.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.vev.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.vev.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.vea.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.vmc.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.vka.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.vev.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.17.vk2.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.vk1.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.vev.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.44.uev.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.350.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.vea.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.vk1.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.03.vma.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.vmc.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.28.vk2.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.vea.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.vk2.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.vev.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.vev.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.vev.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.vma.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.vev.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.vev.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.vma.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.vk1.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.06.vmb.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.vyo.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.vev.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.vmc.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.39.uev.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.vyo.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.vea.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.vmb.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.vev.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.vev.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.25.uev.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.vev.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.vk1.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.vev.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.vkp.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.vev.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.vea.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/Earthquake Sounds.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.vk1.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.vea.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.vk2.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.vk2.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/Scary sound of earthquake 2018.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.vea.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.vmb.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.vev.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.vk1.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.vmb.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.vkd.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.vk2.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.vk1.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.vmc.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.vk1.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.vkp.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.vea.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.vea.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.vyo.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.vk1.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.vea.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.vkp.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.vma.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.vkp.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.vk1.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.vk2.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.vea.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.vev.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.vk1.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.vev.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.vmc.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.vk1.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.vev.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.vev.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.vea.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.vea.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.vk2.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.vk2.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.vev.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.vev.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.31.vev.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.vev.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.vma.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.vev.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.vev.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.07.vmb.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.vk2.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.350.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.vea.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.vea.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.vev.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.350.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.vmb.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.vmc.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.vk2.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.vev.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.vma.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.04.vev.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.vkp.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.vyo.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.vev.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.vk1.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.vkp.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.18.vev.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/hit1_filtered.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.vk1.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.vev.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.vea.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.vyo.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.vmc.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.vka.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.48.vea.2019.001.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.vev.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.352.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.13.uk1.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.vmb.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.33.vka.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.vkp.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.vea.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.vmc.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.vev.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.vmb.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/noise_filtered.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.vkp.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.vk1.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.vev.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/hit1.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.vma.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.20.uk2.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.vea.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.vea.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.08.vea.2019.001.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.vev.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.22.vk2.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.vyo.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.vma.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.vea.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.30.uk1.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.vk1.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.vkp.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.vma.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.vk2.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.vea.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.19.vk1.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.vev.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.vea.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.vk1.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.vk1.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.40.uev.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.vev.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.vea.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.vkd.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.vev.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.vka.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.vea.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.vea.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.352.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.vk2.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.vev.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.vk1.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.vev.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.vev.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.vev.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.vmb.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.vea.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.vkp.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.vea.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.vev.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.vea.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.vev.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.vk2.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.vev.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.vev.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.08.uea.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.vea.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.vev.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/output.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.vma.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.vk2.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.vkc.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.vk2.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.vmb.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.vev.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.vma.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.vea.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.vea.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.vev.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.vma.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.04.uev.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.vea.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.vk1.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.vkp.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.36.vea.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.vkd.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.vk1.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.vmc.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.32.vev.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.vea.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.vev.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.vev.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.vmc.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.vyo.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.vev.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.vk2.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.vk1.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.vev.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.vk1.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.vev.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.vev.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.vea.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.09.vk1.2019.001.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.vk1.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.vkc.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.vk1.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.vk2.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.vk2.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.vea.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.vk1.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.346.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.vev.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.vma.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.vk1.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.vk1.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.41.vev.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.vmc.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.vk2.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.vmb.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.10.vmc.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.vyo.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.vk2.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.vev.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.vkp.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.vmc.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.vev.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.vk1.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.350.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.37.ukd.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.vea.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.vk1.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.vk2.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.vev.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.23.vk1.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.vk2.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.43.vkc.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.vk2.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.vea.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.vev.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.vev.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.352.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.vea.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.vea.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.vk1.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.33.uka.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.vev.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.vev.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.vea.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.38.vea.2019.001.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.vk2.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.vmb.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.vev.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.vyo.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.vkd.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.21.uea.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.vk2.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.vka.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.vk2.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.vea.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.14.ukp.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.vk2.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.vev.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.vkp.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.vev.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.vmb.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.vev.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.vea.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.vk1.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.37.vkd.2019.001.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.vk1.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.vk2.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.vka.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.vea.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.vyo.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.13.vk1.2019.001.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.vea.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.28.uk2.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.vev.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.vev.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.vea.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.vev.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.vk2.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.41.uev.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.vea.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.vea.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.vk2.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.vk2.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.vea.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.vmb.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.vk2.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.vkd.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.vkd.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.vk1.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.vyo.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.17.uk2.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.vk1.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.vmb.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.vk1.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.vma.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.vk1.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.352.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.vev.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.vea.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.vev.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.vk2.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.vev.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.vk1.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.vea.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.vev.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.vea.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.42.vev.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.352.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.vev.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.vev.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.vk1.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.vkp.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.vma.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.vea.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.vea.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.vyo.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.vev.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.vkc.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.vea.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.vk1.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.vka.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.vea.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.vmc.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.vk1.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.vea.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.vea.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.23.uk1.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.vk1.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.vmc.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.vk1.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.vkc.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.vk2.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.vev.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.16.uea.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.vea.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.vk2.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.vev.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.22.uk2.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.vk1.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.vev.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.vev.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.vea.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.44.vev.2019.001.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.350.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.vev.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.350.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.vev.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.vyo.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.vmb.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.vk2.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.350.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.vka.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.vev.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.vk2.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.vea.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.vea.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.vyo.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.vk2.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.vea.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.vk2.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.vyo.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.vkp.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.vk1.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.vma.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.vyo.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.vk2.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.vk1.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.vk2.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.vk2.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.vea.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.vev.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.vk2.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.vk1.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.vev.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.vk1.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.vk1.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.vev.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.vk1.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.vev.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.vk2.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.02.uma.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.vea.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.vev.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.vmc.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.vev.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.vk2.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.vk2.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.vk1.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.vk1.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.vmb.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.07.umb.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.vmc.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.vea.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.45.uea.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.vkp.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.vkp.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.vev.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.vea.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.24.uk2.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.vk1.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.vev.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.15.vkp.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.vmc.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.vk2.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.vmb.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.15.ukp.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.vma.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.vev.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.vk2.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.vk1.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.32.uev.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.vea.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.vea.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.vea.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.vev.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.vyo.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.vk2.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.vk2.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.vea.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.vea.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.10.umc.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.vkd.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.vk1.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.34.vev.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.46.uea.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.vea.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.vk1.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.vma.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.vk2.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.352.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.vev.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.vkc.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.vev.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.vev.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.vev.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.vka.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.vk1.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.vev.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.vev.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.vev.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.vk1.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.vea.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.352.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.vev.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.vkc.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.vev.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.vk2.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.vev.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.36.uea.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.vk2.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.vma.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.vma.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.vea.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.vea.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.vkp.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.vkp.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.vmc.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.vea.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.vkc.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.vyo.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.vev.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.vk2.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.vk2.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.31.uev.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.vk2.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.vma.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.vea.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.vev.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.vev.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.vea.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.vea.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.vmb.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.vev.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.35.vev.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.vkp.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.vkd.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.vkd.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.vma.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.35.uev.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.vea.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.vk2.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.48.uea.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.vkd.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.vea.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.vev.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.vk1.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.vk2.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.vyo.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.27.uk1.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.vkp.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.vk2.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.34.uev.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.vk2.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.vev.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.vk1.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.vyo.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.vea.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.vev.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.46.vea.2019.001.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.vkd.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.19.uk1.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.vev.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.18.uev.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.vmc.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.vev.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.vea.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.vev.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.vmc.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.vmb.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.vev.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.vea.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.vev.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.vkp.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.vev.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.vea.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.vka.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.vev.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.vev.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.vev.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.vea.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.05.vyo.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.12.vev.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.vma.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.vk2.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.vk1.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.vev.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.26.vk2.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.uev.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.vea.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.38.uea.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.vmb.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.vmb.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.vev.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.06.umb.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.vma.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.vmb.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.ukc.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.vkp.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.vev.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.39.vev.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.vea.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.05.uyo.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.vk1.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/output_filtered.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.vev.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.vev.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.47.vyo.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.vea.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.vea.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.vmc.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.vea.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.vev.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.346.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.vea.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.vev.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/Earthquake Sounds_filtered.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.vka.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.vk2.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.352.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.vea.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.vev.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.vma.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.vyo.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.vev.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.ukd.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.vkc.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.45.vea.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.352.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.vk1.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.vev.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.vk1.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.vev.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.29.uea.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.vkc.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.vmc.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.vea.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.vmc.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.vk2.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.uk2.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.vk1.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.vea.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.uk2.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.27.uk1.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.vkc.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.11.vmc.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.vea.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.vev.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.vev.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.351.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.vk1.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.vk1.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.43.ukc.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.uev.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.29.vea.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.21.vea.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.vev.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.11.umc.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.vk2.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.32.vev.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.47.uyo.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.vev.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.vyo.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.vma.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.vev.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.vka.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.45.uea.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.350.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.vev.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.vk2.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.vev.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.vyo.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.vev.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.vea.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.uea.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.29.uea.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.uk1.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.uyo.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.vea.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.vev.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.vk2.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.vea.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.23.uk1.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.vev.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.vea.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.vev.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.vea.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.vev.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.vmc.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.uev.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.vev.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.ukp.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.vk2.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.vk1.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.uev.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.vea.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.43.vkc.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.05.vyo.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.vev.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.vk2.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.26.uk2.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.41.vev.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.vev.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.uea.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.22.uk2.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.uma.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.vmb.2018.335.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.26.vk2.2018.341.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.vmb.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.40.vev.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.17.uk2.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.vea.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.11.umc.2018.340.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.uk1.2018.352.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.vev.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.umc.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.vka.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.uev.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.vev.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.36.uea.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.uyo.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.339.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.07.umb.2018.346.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.31.uev.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.vea.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.vk2.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.21.vea.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.09.vk1.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.44.vev.2018.351.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.uk1.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.25.uev.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.345.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.344.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.30.vk1.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.uea.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.03.uma.2018.352.7.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.06.umb.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.vk1.2018.349.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.10.vmc.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.42.uev.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.33.uka.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.37.vkd.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.vev.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.42.vev.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.14.vkp.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.04.uev.2018.350.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.vea.2018.340.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.16.uea.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.24.vk2.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.09.uk1.2019.001.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.uev.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.vea.2018.343.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.vk2.2018.345.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.uk2.2018.349.6.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.20.uk2.2018.348.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.02.vma.2018.339.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.38.vea.2018.353.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.47.vyo.2018.347.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.46.uea.2018.338.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.08.vea.2018.338.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.48.uea.2018.342.5.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.18.uev.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyhk.20.vk2.2019.001.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.13.uk1.2018.343.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.353.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.28.vk2.2018.336.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.39.uev.2018.341.3.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.34.vev.2018.334.2.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.15.ukp.2018.337.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.12.uev.2018.347.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.35.uev.2018.336.4.a.csv',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/hit/xb.elyh0.19.vk1.2018.349.3.a.csv',]    
        no_quake_files = ['/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_3.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_5.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/WhatsApp Audio 2024-10-06 at 09.49.31_e730ccb4.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_2.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_30.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_35.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_1.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_12.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_28.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/country.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_17.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_37.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_9.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_15.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_41.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_38.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_23.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_21.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_44.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_31.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_25.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_11.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_22.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_27.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_18.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_32.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_24.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/audio2.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/country_filtered.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_10.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_14.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_8.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/8d.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_29.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_4.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_26.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_6.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_40.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_13.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_42.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_33.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/Traffic Sound, Indian Traffic Sound, Traffic noise, vehicle noise, traffic sound, road traffic_filtered.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_43.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/audio1.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_36.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/Traffic Sound, Indian Traffic Sound, Traffic noise, vehicle noise, traffic sound, road traffic.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_34.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/min_filtered.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/min.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_39.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_19.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/WhatsApp Audio 2024-10-06 at 09.44.57_c2581d3e.mp3',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_20.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_16.wav',
'/home/linxcapture/Desktop/projects/hackathon/seismic_web/train/dataset/train/no-hit/segment_7.wav',
] 

        print("Note: No files specified for earthquake detection model training.")
        print("To train with real data, add file paths to quake_files and no_quake_files lists.")

        if quake_files or no_quake_files:
            all_files = quake_files + no_quake_files
            valid_files, invalid_files = validate_input_files(all_files)

            if invalid_files:
                print("\nWarning: The following files have issues:")
                for file_path, error in invalid_files:
                    print(f"  - {file_path}: {error}")

            quake_files = [f for f in quake_files if f in valid_files]
            no_quake_files = [f for f in no_quake_files if f in valid_files]

            X, y, waveforms, sample_rates = load_data(quake_files, no_quake_files)

            if len(X) > 0 and len(y) > 0:
                # Add cross-validation here
                train_with_cross_validation(X, y, n_splits=5)
                
                # Then train the final model
                clf, scaler = train_earthquake_detection_model(X, y)
                joblib.dump(clf, 'earthquake_model.joblib')
                joblib.dump(scaler, 'earthquake_scaler.joblib')
                print("\nEarthquake detection model trained and saved successfully.")
            else:
                print("No valid data loaded for earthquake detection. Using synthetic data.")
                # Generate synthetic data for model development
                n_samples = 100
                n_features = 17
                X_synthetic = np.random.randn(n_samples, n_features)
                y_synthetic = np.random.randint(0, 2, n_samples)
                clf, scaler = train_earthquake_detection_model(X_synthetic, y_synthetic)
                joblib.dump(clf, 'earthquake_model.joblib')
                joblib.dump(scaler, 'earthquake_scaler.joblib')
                print("\nEarthquake detection model trained with synthetic data.")
        else:
            # Generate synthetic data for development purposes
            print("Generating synthetic earthquake detection data...")
            n_samples = 100
            n_features = 17
            X_synthetic = np.random.randn(n_samples, n_features)
            y_synthetic = np.random.randint(0, 2, n_samples)
            
            # Train with synthetic data
            clf, scaler = train_earthquake_detection_model(X_synthetic, y_synthetic)
            joblib.dump(clf, 'earthquake_model.joblib')
            joblib.dump(scaler, 'earthquake_scaler.joblib')
            print("\nEarthquake detection model trained with synthetic data.")

    print("\n--- PART 2: Training Time-to-Event Prediction Model ---")
    synthetic_data_path = "synthetic_temporal_data.csv"

    print("Generating synthetic temporal data for model development...")
    generate_synthetic_temporal_dataset(synthetic_data_path)

    time_predictor, time_scaler = train_temporal_prediction_model(synthetic_data_path)
    print("\nTime-to-event prediction model trained and saved successfully.")

    print("\n--- PART 3: Training Precursor Detection Model ---")
    precursor_data_path = "synthetic_precursor_data.csv"

    print("Generating synthetic precursor detection data...")
    generate_synthetic_precursor_dataset(precursor_data_path)
    
    precursor_model, precursor_scaler = train_precursor_detection_model(precursor_data_path)
    if precursor_model:
        print("Precursor detection model trained and saved successfully.")

    print("\n====== Model Training Complete ======")
    print("Models saved to current directory:")
    print("  - earthquake_model.joblib (LogisticRegression for binary classification)")
    print("  - earthquake_scaler.joblib (feature scaling)")
    print("  - time_prediction_model.joblib (RandomForestRegressor)")
    print("  - time_prediction_scaler.joblib (feature scaling)")
    print("  - precursor_detection_model.joblib (GradientBoostingClassifier)")
    print("  - precursor_detection_scaler.joblib (feature scaling)")
    print("Note: For production use, replace synthetic data with real historical data.")

if __name__ == '__main__':
    main()