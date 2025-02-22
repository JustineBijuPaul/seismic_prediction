import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import signal
import obspy
from obspy import UTCDateTime
from obspy.core import Stream, Trace
import joblib

def generate_synthetic_quake_waveform(duration=60, sample_rate=100, 
                                      amplitude=1000, p_arrival=15, 
                                      s_arrival=25, noise_level=0.1):
    """
    Generate a synthetic earthquake waveform
    
    Parameters:
    -----------
    duration : int
        Duration of the signal in seconds
    sample_rate : int
        Sampling rate in Hz
    amplitude : float
        Maximum amplitude of the earthquake signal
    p_arrival : float
        P-wave arrival time in seconds
    s_arrival : float
        S-wave arrival time in seconds
    noise_level : float
        Background noise level (0-1)
    
    Returns:
    --------
    waveform : ndarray
        Synthetic earthquake waveform
    sample_rate : int
        Sample rate of the waveform
    """
    # Create time vector
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Generate background noise
    background = np.random.normal(0, noise_level * amplitude, len(t))
    
    # Create P-wave arrival
    p_envelope = amplitude * 0.3 * np.exp(-(t - p_arrival) ** 2 / 4) * (t >= p_arrival)
    p_wave = p_envelope * np.sin(2 * np.pi * 5 * (t - p_arrival)) * (t >= p_arrival)
    
    # Create S-wave arrival (larger amplitude, lower frequency)
    s_envelope = amplitude * np.exp(-(t - s_arrival) ** 2 / 10) * (t >= s_arrival)
    s_wave = s_envelope * np.sin(2 * np.pi * 2 * (t - s_arrival)) * (t >= s_arrival)
    
    # Create coda waves (exponential decay after S-wave)
    coda_start = s_arrival + 5
    coda_envelope = amplitude * 0.5 * np.exp(-(t - coda_start) / 15) * (t >= coda_start)
    coda_wave = coda_envelope * np.sin(2 * np.pi * 1.5 * (t - coda_start)) * (t >= coda_start)
    
    # Combine components
    waveform = background + p_wave + s_wave + coda_wave
    
    return waveform, sample_rate

def generate_synthetic_noise_waveform(duration=60, sample_rate=100, 
                                     base_amplitude=50, num_transients=3):
    """
    Generate a synthetic non-earthquake seismic waveform (noise)
    
    Parameters:
    -----------
    duration : int
        Duration of the signal in seconds
    sample_rate : int
        Sampling rate in Hz
    base_amplitude : float
        Base amplitude of the noise
    num_transients : int
        Number of random transient signals to add
    
    Returns:
    --------
    waveform : ndarray
        Synthetic noise waveform
    sample_rate : int
        Sample rate of the waveform
    """
    # Create time vector
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Generate colored noise (more realistic than white noise)
    white_noise = np.random.normal(0, 1, len(t))
    b, a = signal.butter(3, 0.1)
    colored_noise = signal.filtfilt(b, a, white_noise)
    background = base_amplitude * colored_noise / np.std(colored_noise)
    
    # Add some microseismic noise (ocean waves, ~7 second period)
    microseismic = base_amplitude * 0.3 * np.sin(2 * np.pi * 0.14 * t)
    
    # Add random transients (could be footsteps, vehicles, etc.)
    transients = np.zeros_like(t)
    for _ in range(num_transients):
        start_time = np.random.uniform(5, duration-10)
        trans_duration = np.random.uniform(0.5, 2)
        trans_amp = np.random.uniform(base_amplitude*1.5, base_amplitude*3)
        
        # Create transient envelope
        idx = (t >= start_time) & (t <= start_time + trans_duration)
        envelope = np.zeros_like(t)
        envelope[idx] = trans_amp * np.sin(np.pi * (t[idx] - start_time) / trans_duration)
        
        # Add to transients signal
        transients += envelope * np.sin(2 * np.pi * 8 * t)
    
    # Combine components
    waveform = background + microseismic + transients
    
    return waveform, sample_rate

def save_waveform_as_mseed(waveform, sample_rate, filename, station="SYNTH", 
                         network="SY", is_quake=True):
    """Save a synthetic waveform as an MSEED file using ObsPy"""
    
    # Create stats object
    stats = {
        'network': network,
        'station': station,
        'location': '',
        'channel': 'HHZ',  # High-frequency vertical component
        'npts': len(waveform),
        'sampling_rate': sample_rate,
        'mseed': {'dataquality': 'D'},
        'starttime': UTCDateTime.now()
    }
    
    # Create trace and stream
    trace = Trace(data=waveform, header=stats)
    stream = Stream(traces=[trace])
    
    # Save to file
    stream.write(filename, format='MSEED')
    
    return filename

def save_waveform_as_csv(waveform, filename, add_metadata=True):
    """Save a synthetic waveform as a CSV file"""
    
    df = pd.DataFrame({'amplitude': waveform})
    
    if add_metadata:
        # Add some metadata rows at the top
        with open(filename, 'w') as f:
            f.write(f"# Generated synthetic seismic data\n")
            f.write(f"# Timestamp: {pd.Timestamp.now()}\n")
            f.write(f"# Samples: {len(waveform)}\n")
            f.write("\n")
    
    # Append the data
    df.to_csv(filename, index=False, mode='a' if add_metadata else 'w')
    
    return filename

def plot_waveform(waveform, sample_rate, title, filename=None):
    """Plot a waveform and optionally save to file"""
    
    duration = len(waveform) / sample_rate
    t = np.linspace(0, duration, len(waveform))
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, waveform)
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    if filename:
        plt.savefig(filename, dpi=150)
        plt.close()
    else:
        plt.show()

def generate_dataset(output_dir="seismic_dataset", 
                   num_quakes=5, num_noise=5, 
                   plot_samples=True):
    """Generate a dataset of synthetic seismic waveforms"""
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    quake_dir = os.path.join(output_dir, "quake")
    noise_dir = os.path.join(output_dir, "noise")
    
    if not os.path.exists(quake_dir):
        os.makedirs(quake_dir)
    if not os.path.exists(noise_dir):
        os.makedirs(noise_dir)
    
    # Create train directory for joblib files
    train_dir = os.path.join(output_dir, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
    # Lists to store waveforms and sample rates
    all_waveforms = []
    all_sample_rates = []
    quake_files = []
    noise_files = []
    
    # Generate earthquake waveforms
    for i in range(num_quakes):
        # Randomize parameters for variety
        duration = np.random.uniform(50, 120)
        sample_rate = 100  # Keep consistent with train_model.py
        amplitude = np.random.uniform(800, 1500)
        p_arrival = np.random.uniform(10, 25)
        s_arrival = p_arrival + np.random.uniform(6, 15)
        noise_level = np.random.uniform(0.05, 0.2)
        
        # Generate waveform
        waveform, sr = generate_synthetic_quake_waveform(
            duration=duration,
            sample_rate=sample_rate,
            amplitude=amplitude,
            p_arrival=p_arrival,
            s_arrival=s_arrival,
            noise_level=noise_level
        )
        
        # Store waveform and sample rate
        all_waveforms.append(waveform)
        all_sample_rates.append(sr)
        
        # Save as MSEED
        mseed_filename = os.path.join(quake_dir, f"quake_{i+1}.mseed")
        save_waveform_as_mseed(waveform, sr, mseed_filename, station=f"QK{i+1}")
        quake_files.append(mseed_filename)
        
        # Save a subset as CSV for variety
        if i % 2 == 0:
            csv_filename = os.path.join(quake_dir, f"quake_{i+1}.csv")
            save_waveform_as_csv(waveform, csv_filename)
            quake_files.append(csv_filename)
        
        # Plot sample
        if plot_samples and i < 2:  # Just plot first two for brevity
            plot_waveform(waveform, sr, f"Synthetic Earthquake {i+1}", 
                       os.path.join(output_dir, f"quake_{i+1}_plot.png"))
    
    # Generate noise waveforms
    for i in range(num_noise):
        # Randomize parameters for variety
        duration = np.random.uniform(50, 120)
        sample_rate = 100  # Keep consistent with train_model.py
        base_amplitude = np.random.uniform(30, 80)
        num_transients = np.random.randint(1, 6)
        
        # Generate waveform
        waveform, sr = generate_synthetic_noise_waveform(
            duration=duration,
            sample_rate=sample_rate,
            base_amplitude=base_amplitude,
            num_transients=num_transients
        )
        
        # Store waveform and sample rate
        all_waveforms.append(waveform)
        all_sample_rates.append(sr)
        
        # Save as MSEED
        mseed_filename = os.path.join(noise_dir, f"noise_{i+1}.mseed")
        save_waveform_as_mseed(waveform, sr, mseed_filename, station=f"NS{i+1}", is_quake=False)
        noise_files.append(mseed_filename)
        
        # Save a subset as CSV for variety
        if i % 2 == 0:
            csv_filename = os.path.join(noise_dir, f"noise_{i+1}.csv")
            save_waveform_as_csv(waveform, csv_filename)
            noise_files.append(csv_filename)
        
        # Plot sample
        if plot_samples and i < 2:  # Just plot first two for brevity
            plot_waveform(waveform, sr, f"Synthetic Noise {i+1}", 
                       os.path.join(output_dir, f"noise_{i+1}_plot.png"))
    
    # Save all waveforms and sample rates using joblib for train_model.py
    joblib.dump(all_waveforms, os.path.join(train_dir, 'all_waveforms.joblib'))
    joblib.dump(all_sample_rates, os.path.join(train_dir, 'all_sample_rates.joblib'))
    print(f"Saved {len(all_waveforms)} waveforms to joblib files in {train_dir}")
    
    # Create a README file
    with open(os.path.join(output_dir, "README.txt"), 'w') as f:
        f.write("Synthetic Seismic Dataset\n")
        f.write("========================\n\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
        f.write(f"This dataset contains:\n")
        f.write(f"- {num_quakes} synthetic earthquake waveforms\n")
        f.write(f"- {num_noise} synthetic non-earthquake waveforms\n\n")
        f.write("File formats include MSEED and CSV.\n\n")
        f.write("Earthquake files:\n")
        for file in quake_files:
            f.write(f"- {os.path.basename(file)}\n")
        f.write("\nNon-earthquake files:\n")
        for file in noise_files:
            f.write(f"- {os.path.basename(file)}\n")
        f.write("\nJoblib files for model training:\n")
        f.write(f"- train/all_waveforms.joblib\n")
        f.write(f"- train/all_sample_rates.joblib\n")
    
    # Return dataset information
    dataset_info = {
        'all_waveforms': all_waveforms,
        'all_sample_rates': all_sample_rates,
        'quake_files': quake_files,
        'noise_files': noise_files,
        'output_dir': output_dir,
        'train_dir': train_dir
    }
    
    return dataset_info

def run_training_with_generated_data(dataset_info):
    """
    Run the training process using the generated waveforms
    """
    import importlib.util
    
    # Check if train_model.py exists in the current directory
    if not os.path.exists('train_model.py'):
        print("train_model.py not found in current directory. Cannot run training.")
        return False
    
    print("\nImporting train_model.py...")
    
    # Import train_model.py as a module
    spec = importlib.util.spec_from_file_location("train_model", "train_model.py")
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    
    print("Running model training with generated data...")
    train_module.main()
    
    return True

# Run the generation script with more samples for better training
if __name__ == "__main__":
    print("Generating synthetic seismic dataset...")
    dataset = generate_dataset(num_quakes=20, num_noise=20)
    
    print(f"\nDataset created in: {dataset['output_dir']}")
    print(f"Total waveforms: {len(dataset['all_waveforms'])}")
    print(f"Earthquake files: {len(dataset['quake_files'])}")
    print(f"Non-earthquake files: {len(dataset['noise_files'])}")
    
    # Example of accessing the all_waveforms and all_sample_rates lists
    print("\nSample waveform information:")
    for i in range(min(3, len(dataset['all_waveforms']))):
        waveform = dataset['all_waveforms'][i]
        sr = dataset['all_sample_rates'][i]
        print(f"Waveform {i+1}: Length={len(waveform)}, Duration={len(waveform)/sr:.2f}s, Sample Rate={sr}Hz")
    
    # Ask if user wants to run training
    response = input("\nDo you want to run model training with the generated data? (y/n): ")
    if response.lower() == 'y':
        success = run_training_with_generated_data(dataset)
        if success:
            print("\nTraining completed! The models should now be available in the current directory.")
        else:
            print("\nTraining could not be started.")
    else:
        print("\nYou can run the training separately by running train_model.py")
        print("The generated data has been saved and is ready for use by train_model.py")