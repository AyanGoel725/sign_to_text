const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const bufferStatusEl = document.getElementById('bufferStatus');
const sentenceEl = document.getElementById('sentence');
const backspaceBtn = document.getElementById('backspaceBtn');
const clearBtn = document.getElementById('clearBtn');

const API_URL = '/predict';

// Buffering config (ported from app.py, optimized for lower latency)
const SMOOTHING_WINDOW = 5;
const SMOOTHING_THRESHOLD = 0.80;
const PAUSE_BETWEEN_SIGNS = 0.5; // seconds
const MIN_CONFIDENCE = 0.55; // Minimum confidence to enter the buffer

let predictionBuffer = [];
let sentence = [];
let confirmedWord = null;
let lastSignTime = Date.now();
let lastHandSeenTime = Date.now();
let isPredicting = false;

let camera = null;
let hands = null;

function updateUI() {
    sentenceEl.textContent = sentence.length > 0 ? sentence.join('') : '—';
}

async function sendToAPI(landmarks) {
    if (isPredicting) return; // Skip frame if still processing previous one

    isPredicting = true;
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ landmarks })
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        handlePrediction(data.letter, data.confidence);

    } catch (err) {
        console.error('API error:', err);
        bufferStatusEl.textContent = `Error: ${err.message}`;
    } finally {
        isPredicting = false;
    }
}

function handlePrediction(predictedLabel, confidence) {
    // Log all predictions for debugging confidence thresholds
    console.log(`Prediction: ${predictedLabel}, Confidence: ${(confidence * 100).toFixed(1)}%`);

    // Filter out low-confidence predictions
    if (confidence < MIN_CONFIDENCE) {
        bufferStatusEl.textContent = `Ignoring low confidence: ${(confidence * 100).toFixed(0)}%`;
        return;
    }

    // Add to buffer
    predictionBuffer.push(predictedLabel);
    if (predictionBuffer.length > SMOOTHING_WINDOW) {
        predictionBuffer.shift();
    }

    // Show buffer status
    bufferStatusEl.textContent = `Buffering ${predictionBuffer.length}/${SMOOTHING_WINDOW}: ${predictedLabel} (${(confidence * 100).toFixed(1)}%)`;

    // Majority-vote logic (ported from app.py)
    if (predictionBuffer.length === SMOOTHING_WINDOW) {
        const counts = {};
        predictionBuffer.forEach(label => {
            counts[label] = (counts[label] || 0) + 1;
        });

        // Find majority
        let majorityLabel = null;
        let majorityCount = 0;
        for (const [label, count] of Object.entries(counts)) {
            if (count > majorityCount) {
                majorityLabel = label;
                majorityCount = count;
            }
        }

        const majorityPct = majorityCount / SMOOTHING_WINDOW;

        if (majorityPct >= SMOOTHING_THRESHOLD) {
            const currentTime = Date.now() / 1000;

            if (confirmedWord !== majorityLabel || (currentTime - lastSignTime) > PAUSE_BETWEEN_SIGNS) {
                confirmedWord = majorityLabel;
                lastSignTime = currentTime;
                predictionBuffer = [];

                // Commit the letter
                if (majorityLabel.toLowerCase() === 'space') {
                    sentence.push(' ');
                } else if (majorityLabel.toLowerCase() === 'del') {
                    if (sentence.length > 0) {
                        sentence.pop();
                    }
                } else {
                    sentence.push(majorityLabel);
                }

                updateUI();
                bufferStatusEl.textContent = `✓ Committed: ${majorityLabel}`;
            } else {
                bufferStatusEl.textContent = `Waiting (same sign, ${majorityPct.toFixed(0)}%)`;
            }
        } else {
            bufferStatusEl.textContent = `No majority (top: ${majorityLabel} ${(majorityPct * 100).toFixed(0)}%)`;
        }
    }
}

function onResults(results) {
    // Draw video frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        lastHandSeenTime = Date.now();

        // Draw hand skeleton
        const handLandmarks = results.multiHandLandmarks[0];
        drawConnectors(ctx, handLandmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 2});
        drawLandmarks(ctx, handLandmarks, {color: '#FF0000', lineWidth: 1, radius: 3});

        // Extract landmarks in training order: x0..x20, y0..y20, z0..z20
        const x_coords = handLandmarks.map(lm => lm.x);
        const y_coords = handLandmarks.map(lm => lm.y);
        const z_coords = handLandmarks.map(lm => lm.z);
        const landmarks = [...x_coords, ...y_coords, ...z_coords];

        sendToAPI(landmarks);
    } else {
        // No hand detected — auto-punctuation after 3 seconds
        const timeSinceLastHand = (Date.now() - lastHandSeenTime) / 1000;
        if (timeSinceLastHand > 3 && sentence.length > 0 && sentence[sentence.length - 1] !== '.') {
            sentence.push('.');
            updateUI();
            lastHandSeenTime = Date.now(); // Reset timer
        }
        bufferStatusEl.textContent = 'No hand detected';
    }
}

async function startCamera() {
    try {
        startBtn.textContent = 'Starting...';
        startBtn.disabled = true;

        hands = new Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`;
            }
        });

        hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.7
        });

        hands.onResults(onResults);

        camera = new Camera(video, {
            onFrame: async () => {
                await hands.send({ image: video });
            },
            width: 640,
            height: 480
        });

        await camera.start();

        canvas.style.display = 'block';
        startBtn.textContent = 'Camera Running';
        bufferStatusEl.textContent = 'Waiting for hand...';

    } catch (err) {
        startBtn.textContent = 'Start Camera';
        startBtn.disabled = false;
        bufferStatusEl.textContent = `Error: ${err.message}`;
    }
}

startBtn.addEventListener('click', startCamera);

backspaceBtn.addEventListener('click', () => {
    if (sentence.length > 0) {
        sentence.pop();
        updateUI();
    }
});

clearBtn.addEventListener('click', () => {
    sentence = [];
    predictionBuffer = [];
    confirmedWord = null;
    updateUI();
    bufferStatusEl.textContent = 'Ready';
});
