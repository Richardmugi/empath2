document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const API_URL = window.location.origin;  // This will work for both local and production
    const fileInput = document.getElementById('eegFile');
    const file = fileInput.files[0];
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const resultAlert = document.getElementById('resultAlert');

    if (!file) {
        alert('Please select a file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file, file.name);  // Include filename

    try {
        loading.style.display = 'block';
        results.style.display = 'none';

        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            body: formData,
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Server response:', data);  // Debug log
        
        loading.style.display = 'none';
        results.style.display = 'block';

        if (data.status === 'success') {
            resultAlert.className = 'alert alert-success';
            resultAlert.textContent = `Analysis Result: ${data.message}`;
        } else if (response.status === 503) {
            resultAlert.className = 'alert alert-warning';
            resultAlert.textContent = 'Model is still loading. Please try again in a few moments.';
        } else {
            resultAlert.className = 'alert alert-danger';
            let errorMessage = 'Error processing the file. Please try again.';
            
            if (data.detail && typeof data.detail === 'object') {
                errorMessage = data.detail.error;
                console.error('Error details:', data.detail);
            }
            
            resultAlert.textContent = errorMessage;
        }
    } catch (error) {
        loading.style.display = 'none';
        results.style.display = 'block';
        resultAlert.className = 'alert alert-danger';
        resultAlert.textContent = `Error: ${error.message || 'Failed to process file'}`;
        console.error('Error:', error);
    }
}); 