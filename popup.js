// popup.js
document.addEventListener('DOMContentLoaded', function() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const feedback = document.getElementById('feedback');
    const resultsSection = document.getElementById('results');

    const textStatus = document.getElementById('textStatus');
    const textBar = document.getElementById('textBar');
    const textConf = document.getElementById('textConf');

    const imageStatus = document.getElementById('imageStatus');
    const imageBar = document.getElementById('imageBar');
    const imageConf = document.getElementById('imageConf');

    const historyList = document.getElementById('historyList');

    // Render history on load
    chrome.storage.local.get(["analysisHistory"], (result) => {
        const history = result.analysisHistory || [];
        renderHistory(history);
    });

    analyzeBtn.addEventListener('click', async () => {
        feedback.innerHTML = '<span class="spinner"></span> Analyzing...';
        resultsSection.style.display = 'none';

        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        // Directly send message (no need to inject content.js again, it’s already injected via manifest.json)
        chrome.tabs.sendMessage(tab.id, { action: "analyzeContent" }, (response) => {
            feedback.textContent = '';
            if (!response) {
                feedback.textContent = "No response from content script. Reload the page and try again.";
                feedback.style.color = "#b03a2e";
                return;
            }

            if (response && response.status === 'success') {
                resultsSection.style.display = 'block';

                const textResult = response.textResult || {};
                const imageResult = response.imageResult || {};

                const tPred = textResult.prediction || "Skipped";
                const tConf = Math.round((textResult.confidence || 0) * 100);
                textStatus.textContent = `${tPred}`;
                textBar.style.width = `${tConf}%`;
                textConf.textContent = `Confidence: ${tConf}%`;

                const iPred = imageResult.prediction || "Skipped";
                const iConf = Math.round((imageResult.confidence || 0) * 100);
                imageStatus.textContent = `${iPred}`;
                imageBar.style.width = `${iConf}%`;
                imageConf.textContent = `Confidence: ${iConf}%`;

                if ((tPred === "Fake" && tConf >= 60) || (iPred === "Deepfake" && iConf >= 60)) {
                    feedback.textContent = 'Warning: Potentially fake content detected on this page.';
                    feedback.style.color = '#b03a2e';
                } else {
                    feedback.textContent = 'Content appears likely authentic (or low-confidence flags).';
                    feedback.style.color = '#2a7a2a';
                }

                // Save to history
                const record = {
                    time: new Date().toLocaleTimeString(),
                    text: { pred: tPred, conf: tConf },
                    image: { pred: iPred, conf: iConf }
                };

                chrome.storage.local.get(["analysisHistory"], (res) => {
                    let history = res.analysisHistory || [];
                    history.unshift(record);
                    if (history.length > 5) history = history.slice(0, 5);
                    chrome.storage.local.set({ analysisHistory: history }, () => {
                        renderHistory(history);
                    });
                });

            } else {
                feedback.textContent = 'Analysis failed. Make sure the server is running.';
                feedback.style.color = '#b03a2e';
            }
        });
    });

    function renderHistory(history) {
        if (!history.length) {
            historyList.textContent = "No history yet.";
            return;
        }
        historyList.innerHTML = history.map(item => {
            return `<div class="history-item">
                [${item.time}] 
                Text: <span class="${cls(item.text.pred)}">${item.text.pred}</span> (${item.text.conf}%) |
                Image: <span class="${cls(item.image.pred)}">${item.image.pred}</span> (${item.image.conf}%)
            </div>`;
        }).join("");
    }

    function cls(pred) {
        if (pred === "Fake") return "fake";
        if (pred === "Deepfake") return "deepfake";
        if (pred === "Real") return "real";
        return "";
    }
});
