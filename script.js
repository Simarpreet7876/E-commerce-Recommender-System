document.addEventListener('DOMContentLoaded', () => {
    const API_URL = 'http://127.0.0.1:8000';

    const userIdInput = document.getElementById('userIdInput');
    const recommendBtn = document.getElementById('recommendBtn');
    const recommendationsGrid = document.getElementById('recommendationsGrid');
    const explanationBox = document.getElementById('explanationBox');
    const errorMessage = document.getElementById('errorMessage');

    // --- Event Listener for the main button ---
    recommendBtn.addEventListener('click', handleGetRecommendations);

    async function handleGetRecommendations() {
        const userId = userIdInput.value.trim();
        if (!userId) {
            showError("Please enter a User ID.");
            return;
        }

        setLoading(true);

        try {
            const response = await fetch(`${API_URL}/recommend/${userId}`);
            if (!response.ok) {
                throw new Error('User not found or an error occurred on the server.');
            }
            const data = await response.json();
            displayRecommendations(data.recommendations);
        } catch (err) {
            showError(err.message);
        } finally {
            setLoading(false);
        }
    }

    // --- Renders the recommendation cards ---
    function displayRecommendations(recommendations) {
        recommendationsGrid.innerHTML = ''; // Clear previous results
        if (recommendations.length === 0) {
            showError("No recommendations found for this user.");
            return;
        }

        recommendations.forEach(rec => {
            const card = document.createElement('div');
            card.className = 'recommendation-card';
            card.innerHTML = `
                <div class="product-icon">📦</div>
                <div class="product-name">${rec.product_name.replace(/_/g, ' ')}</div>
                <div class="product-id" title="${rec.product_id}">ID: ${rec.product_id.substring(0, 10)}...</div>
                <button class="explain-btn" data-product-id="${rec.product_id}">Why this?</button>
            `;
            recommendationsGrid.appendChild(card);
        });

        // Add event listeners to the new "Why this?" buttons
        document.querySelectorAll('.explain-btn').forEach(button => {
            button.addEventListener('click', handleGetExplanation);
        });
    }

    // --- Fetches and displays the AI explanation ---
    async function handleGetExplanation(event) {
        const productId = event.target.dataset.productId;
        const userId = userIdInput.value.trim();
        
        explanationBox.classList.remove('hidden');
        explanationBox.innerHTML = '<strong>AI Explanation:</strong> <em>Generating...</em>';

        try {
            const response = await fetch(`${API_URL}/explain/${userId}/${productId}`);
            if (!response.ok) {
                throw new Error('Could not fetch explanation.');
            }
            const data = await response.json();
            explanationBox.innerHTML = `<strong>AI Explanation:</strong> ${data.explanation}`;
        } catch (err) {
            explanationBox.innerHTML = `<strong>AI Explanation:</strong> <span class="error-message">${err.message}</span>`;
        }
    }

    // --- Utility Functions ---
    function setLoading(isLoading) {
        recommendBtn.disabled = isLoading;
        recommendBtn.textContent = isLoading ? 'Loading...' : 'Get Recommendations';
        if (isLoading) {
            errorMessage.textContent = '';
            recommendationsGrid.innerHTML = '';
            explanationBox.classList.add('hidden');
        }
    }

    function showError(message) {
        recommendationsGrid.innerHTML = '';
        explanationBox.classList.add('hidden');
        errorMessage.textContent = message;
    }
});
