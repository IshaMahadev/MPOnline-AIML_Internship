document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');
    
    const formContainer = form;
    const resultContainer = document.getElementById('result-container');
    const resultValue = document.getElementById('result-value');
    const resultDesc = document.getElementById('result-desc');
    const resetBtn = document.getElementById('reset-btn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // 1. Show Loading State
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        submitBtn.disabled = true;

        // 2. Gather Data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            
            if (response.ok) {
                showResult(result);
            } else {
                alert('Error from API: ' + (result.error || 'Unknown error'));
                resetBtn.click();
            }
        } catch(err) {
            console.error(err);
            alert('Failed to connect to the prediction API.');
            resetBtn.click();
        }
    });

    resetBtn.addEventListener('click', () => {
        resultContainer.classList.add('hidden');
        formContainer.classList.remove('hidden');
        form.reset();
        
        // Reset Button State
        btnText.classList.remove('hidden');
        loader.classList.add('hidden');
        submitBtn.disabled = false;
        
        // Remove old classes
        resultValue.classList.remove('high', 'low');
    });

    function mockPredict(data) {
        // Simple deterministic heuristic for demonstration purposes
        let score = 0;
        
        const age = parseInt(data.age);
        if (age > 30 && age < 60) score += 2;
        
        const hours = parseInt(data['hours-per-week']);
        if (hours >= 40) score += 2;
        if (hours > 50) score += 1;

        const edu = data.education;
        const highEdu = ['Bachelors', 'Masters', 'Doctorate', 'Prof-school'];
        if (highEdu.includes(edu)) score += 3;

        const occupation = data.occupation;
        const highIncomeJobs = ['Exec-managerial', 'Prof-specialty'];
        if (highIncomeJobs.includes(occupation)) score += 2;

        const capitalGain = parseInt(data['capital-gain']);
        if (capitalGain > 5000) score += 4;

        // If score is high enough, predict >50K
        if (score >= 6) {
            return {
                label: '>50K',
                probability: (0.7 + Math.random() * 0.25).toFixed(2), // 70-95%
                message: "High likelihood of earning over $50,000 annually based on the provided demographic and employment factors."
            };
        } else {
            return {
                label: '<=50K',
                probability: (0.6 + Math.random() * 0.35).toFixed(2), // 60-95%
                message: "Based on the demographic profile, it is predicted that annual income is $50,000 or below."
            };
        }
    }

    function showResult(prediction) {
        formContainer.classList.add('hidden');
        resultContainer.classList.remove('hidden');
        
        resultValue.textContent = prediction.label;
        if (prediction.label === '>50K') {
            resultValue.classList.add('high');
        } else {
            resultValue.classList.add('low');
        }

        resultDesc.innerHTML = `<strong>${(prediction.probability * 100).toFixed(0)}% Confidence.</strong><br/>${prediction.message}`;
    }
});
