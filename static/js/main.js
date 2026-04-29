/**
 * WINE QUALITY PREDICTION - MAIN JAVASCRIPT
 * Handles form validation, AJAX submission, and UI updates.
 */

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const resultContainer = document.getElementById('resultContainer');
    const alertBox = document.getElementById('alertBox');
    
    // Auto-focus first input
    const firstInput = document.getElementById('fixed_acidity');
    if (firstInput) firstInput.focus();

    // Form Submission Handler
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Hide previous results/alerts
        hideAlert();
        resultContainer.classList.remove('active');
        
        // Collect data
        const formData = new FormData(form);
        const data = {};
        let isValid = true;
        let errors = [];

        // Simple client-side validation
        for (let [key, value] of formData.entries()) {
            data[key] = value;
            
            const numValue = parseFloat(value);
            if (isNaN(numValue)) {
                isValid = false;
                errors.push(`Field <b>${formatFieldName(key)}</b> harus berupa angka.`);
            } else if (value.trim() === '') {
                isValid = false;
                errors.push(`Field <b>${formatFieldName(key)}</b> tidak boleh kosong.`);
            }
        }

        if (!isValid) {
            showAlert('danger', 'Validasi Gagal:<br>' + errors.join('<br>'));
            return;
        }

        // Show loading state
        toggleLoading(true);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            
            // Artificial delay for UI/UX (so loading animation is seen briefly)
            setTimeout(() => {
                toggleLoading(false);

                if (result.success) {
                    displayResult(result.prediction, result.probability, result.message);
                } else {
                    showAlert('danger', '<strong>Error!</strong><br>' + (result.error || 'Terjadi kesalahan pada server.'));
                }
            }, 800);

        } catch (error) {
            console.error('Error:', error);
            toggleLoading(false);
            showAlert('danger', '<strong>Koneksi Gagal!</strong><br>Pastikan server Flask sedang berjalan.');
        }
    });

    /**
     * Toggles the loading overlay
     */
    function toggleLoading(show) {
        if (show) {
            loadingOverlay.classList.add('active');
        } else {
            loadingOverlay.classList.remove('active');
        }
    }

    /**
     * Displays the prediction result in the custom UI
     */
    function displayResult(prediction, probability, message) {
        const resultBox = document.getElementById('resultBox');
        const resultIcon = document.getElementById('resultIcon');
        const resultTitle = document.getElementById('resultTitle');
        const resultDesc = document.getElementById('resultDesc');
        const confidenceValue = document.getElementById('confidenceValue');
        const confidenceBar = document.getElementById('confidenceBar');
        
        // Reset classes
        resultBox.className = 'result-box';
        
        if (prediction === 'GOOD') {
            resultBox.classList.add('good');
            resultIcon.className = 'fas fa-check';
            resultTitle.textContent = 'Wine Kualitas Tinggi';
        } else {
            resultBox.classList.add('bad');
            resultIcon.className = 'fas fa-times';
            resultTitle.textContent = 'Wine Kualitas Rendah';
        }

        resultDesc.textContent = message;
        
        // Calculate confidence percentage
        // If prob > 0.5 it's GOOD, confidence is prob. If prob <= 0.5 it's BAD, confidence is 1 - prob.
        let confidence = (prediction === 'GOOD') ? probability : (1 - probability);
        let confPercent = Math.round(confidence * 100);
        
        confidenceValue.textContent = `${confPercent}%`;
        
        // Show container before animating width
        resultContainer.classList.add('active');
        
        // Animate progress bar
        setTimeout(() => {
            confidenceBar.style.width = `${confPercent}%`;
        }, 100);
        
        // Scroll to result smoothly
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    /**
     * Shows an alert message
     */
    function showAlert(type, htmlContent) {
        alertBox.className = `alert custom-alert alert-${type} alert-dismissible fade show`;
        alertBox.innerHTML = `
            ${htmlContent}
            <button type="button" class="btn-close btn-close-white" onclick="hideAlert()"></button>
        `;
        alertBox.style.display = 'block';
    }

    /**
     * Hides the alert message
     */
    window.hideAlert = function() {
        if(alertBox) {
            alertBox.style.display = 'none';
        }
    }

    /**
     * Helper to format field names for display
     */
    function formatFieldName(name) {
        return name.split('_')
                   .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                   .join(' ');
    }
});
