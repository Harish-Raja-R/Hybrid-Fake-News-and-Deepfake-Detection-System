// content.js
const API_URL = "http://127.0.0.1:5000";

async function analyzeText(text) {
    try {
        const response = await fetch(`${API_URL}/predict/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        const data = await response.json();
        return data; // { prediction, confidence }
    } catch (e) {
        console.error("Text API error:", e);
        return { prediction: "Analysis Failed", confidence: 0.0 };
    }
}

async function analyzeImage(imageUrl) {
    try {
        const response = await fetch(imageUrl);
        const imageBlob = await response.blob();

        const formData = new FormData();
        formData.append('file', imageBlob, 'image.jpg');

        const apiResponse = await fetch(`${API_URL}/predict/image`, {
            method: 'POST',
            body: formData
        });
        const data = await apiResponse.json();
        return data; // { prediction, confidence, image_size }
    } catch (error) {
        console.error("Error analyzing image:", error);
        return { prediction: "Analysis Failed", confidence: 0.0 };
    }
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "analyzeContent") {
        (async () => {
            const pageText = document.body.innerText.substring(0, 2000);
            const firstImage = document.querySelector('img');
            const imageUrl = firstImage ? firstImage.src : null;

            let textResult = { prediction: "Skipped", confidence: 0.0 };
            let imageResult = { prediction: "Skipped", confidence: 0.0 };

            if (pageText && pageText.length > 0) {
                textResult = await analyzeText(pageText);
            }

            if (imageUrl) {
                imageResult = await analyzeImage(imageUrl);
            }

            sendResponse({
                status: 'success',
                textResult: textResult,
                imageResult: imageResult
            });
        })();
        return true;
    }
});
