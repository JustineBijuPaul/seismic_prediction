document.addEventListener('DOMContentLoaded', function() {
    const startBtn = document.getElementById('start-monitoring');
    const stopBtn = document.getElementById('stop-monitoring');
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    const alertsList = document.getElementById('alerts-list');
    
    let monitoring = false;
    let monitoringInterval;

    startBtn.addEventListener('click', function() {
        startMonitoring();
    });

    stopBtn.addEventListener('click', function() {
        stopMonitoring();
    });

    async function startMonitoring() {
        monitoring = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusDot.classList.add('active');
        statusText.textContent = 'Monitoring Active';
        
        // Start periodic monitoring
        monitoringInterval = setInterval(fetchStreamData, 5000);
    }

    function stopMonitoring() {
        monitoring = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        statusDot.classList.remove('active');
        statusText.textContent = 'Monitoring Inactive';
        
        clearInterval(monitoringInterval);
    }

    async function fetchStreamData() {
        try {
            const response = await fetch('/stream_data');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const contentType = response.headers.get("content-type");
            if (!contentType || !contentType.includes("application/json")) {
                throw new TypeError("Received non-JSON response from server");
            }
            
            const data = await response.json();
            
            if (data.error) {
                console.error('Server error:', data.error);
                throw new Error(data.error);
            }
            
            if (Array.isArray(data)) {
                data.forEach(result => {
                    if (result.is_earthquake) {
                        addAlert(result);
                    }
                });
            } else {
                console.warn('Unexpected data format:', data);
            }
        } catch (error) {
            console.error('Error fetching stream data:', error);
            stopMonitoring();
            
            // Show error to user
            const alertsList = document.getElementById('alerts-list');
            if (alertsList) {
                alertsList.innerHTML = `
                    <div class="alert-item error">
                        <strong>Error</strong>
                        <p>${error.message || 'Failed to fetch stream data'}</p>
                    </div>
                `;
            }
        }
    }

    function addAlert(result) {
        const alertElement = document.createElement('div');
        alertElement.className = 'alert-item';
        
        const timestamp = new Date(result.timestamp).toLocaleString();
        const probability = (result.probability * 100).toFixed(1);
        
        alertElement.innerHTML = `
            <strong>${timestamp}</strong>
            <p>Earthquake Probability: ${probability}%</p>
            <p>Hours to Event: ${result.hours_to_event.toFixed(1)}</p>
            <p>Confidence: ${result.confidence}</p>
        `;
        
        alertsList.insertBefore(alertElement, alertsList.firstChild);
    }
});
