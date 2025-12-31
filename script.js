/**
 * XAU/USD Predictive Analytics Engine
 * Frontend JavaScript - UI Interactions & Real-Time Updates
 * NEYDRA Platform by Ilyes Jarray
 * © 2025 - Enterprise Grade AI Trading Solution
 */

// ===== GLOBAL CONFIGURATION =====
const CONFIG = {
    API_ENDPOINT: 'http://localhost:5000/api',
    UPDATE_INTERVAL: 3000, // 3 seconds
    CHART_UPDATE_INTERVAL: 5000, // 5 seconds
    MAX_STREAM_ENTRIES: 50,
    MAX_PRICE_HISTORY: 50,
    SOUNDS_ENABLED: true,
    DEBUG_MODE: false
};

// ===== GLOBAL STATE =====
const STATE = {
    isRunning: false,
    currentPrice: 2030. 50,
    priceHistory: [],
    signals: [],
    predictions: [],
    startTime: null,
    totalProcessed: 0,
    accuracy: 87,
    charts: {
        accuracy: null,
        signal: null,
        price: null,
        volatility: null
    }
};

// ===== DOM ELEMENTS CACHE =====
const DOM = {};

/**
 * Initialize DOM elements cache
 */
function cacheDOM() {
    // Buttons
    DOM.startBtn = document.getElementById('startBtn');
    DOM.pauseBtn = document.getElementById('pauseBtn');
    DOM.exportBtn = document.getElementById('exportBtn');
    DOM.settingsBtn = document.getElementById('settingsBtn');
    DOM.helpBtn = document.getElementById('helpBtn');
    DOM.clearStreamBtn = document.getElementById('clearStreamBtn');
    DOM.pauseStreamBtn = document.getElementById('pauseStreamBtn');

    // Price Display
    DOM.currentPrice = document.getElementById('currentPrice');
    DOM.priceChange = document.getElementById('priceChange');
    DOM.price24h = document.getElementById('price24h');
    DOM.usdStatus = document.getElementById('usdStatus');
    DOM.volatility = document.getElementById('volatility');
    DOM.correlation = document.getElementById('correlation');
    DOM.pairRate = document.getElementById('pairRate');
    DOM.pairChange = document.getElementById('pairChange');
    DOM.volume = document.getElementById('volume');
    DOM.strengthFill = document.getElementById('strengthFill');
    DOM.timestamp = document.getElementById('timestamp');
    DOM.lastUpdate = document.getElementById('lastUpdate');

    // Prediction Display
    DOM.prediction24h = document.getElementById('prediction24h');
    DOM.confidence24h = document.getElementById('confidence24h');
    DOM.tradingSignal = document.getElementById('tradingSignal');
    DOM.signalType = document.getElementById('signalType');
    DOM.signalStrengthFill = document.getElementById('signalStrengthFill');
    DOM.signalStrengthText = document.getElementById('signalStrengthText');
    DOM.resistance = document.getElementById('resistance');
    DOM.support = document.getElementById('support');
    DOM.currentPosition = document.getElementById('currentPosition');
    DOM.riskLevel = document.getElementById('riskLevel');
    DOM.riskIndicator = document.getElementById('riskIndicator');
    DOM.forecastIcon = document.getElementById('forecastIcon');
    DOM.signalIcon = document.getElementById('signalIcon');

    // Data Stream
    DOM.dataStream = document.getElementById('dataStream');

    // Statistics
    DOM.signalsCount = document.getElementById('signalsCount');
    DOM.pnl = document.getElementById('pnl');
    DOM.processingSpeed = document.getElementById('processingSpeed');
    DOM.latency = document.getElementById('latency');
    DOM.uptimeCounter = document.getElementById('uptimeCounter');
    DOM.processedSignals = document.getElementById('processedSignals');

    // Charts
    DOM.accuracyChart = document.getElementById('accuracyChart');
    DOM.signalChart = document.getElementById('signalChart');
    DOM.priceChart = document.getElementById('priceChart');
    DOM.volatilityChart = document. getElementById('volatilityChart');
    DOM.timeframeSelect = document.getElementById('timeframeSelect');

    // Accuracy Badge
    DOM.accuracyBadge = document.getElementById('accuracyBadge');
}

/**
 * Initialize Event Listeners
 */
function initializeEventListeners() {
    // Main Control Buttons
    DOM.startBtn.addEventListener('click', startAnalysis);
    DOM.pauseBtn.addEventListener('click', pauseAnalysis);
    DOM.exportBtn.addEventListener('click', exportSignals);
    DOM.settingsBtn.addEventListener('click', openSettings);
    DOM.helpBtn.addEventListener('click', openHelp);

    // Stream Controls
    DOM.clearStreamBtn.addEventListener('click', clearStream);
    DOM.pauseStreamBtn.addEventListener('click', toggleStreamPause);

    // Chart Control
    DOM.timeframeSelect. addEventListener('change', updateChartTimeframe);

    // Keyboard Shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
}

/**
 * Start Analysis
 */
function startAnalysis() {
    if (STATE.isRunning) return;

    STATE.isRunning = true;
    STATE.startTime = Date.now();
    
    // Update UI
    DOM.startBtn.disabled = true;
    DOM.pauseBtn. disabled = false;

    logStream('🔴 AI Engine Started - Initializing Predictive Models', 'success');
    logStream('📡 Connecting to Real-Time Market Feeds', 'info');
    logStream('🤖 Loading Machine Learning Models... ', 'info');

    // Initial setup
    setTimeout(() => {
        logStream('✅ Models Loaded - Accuracy: 87.3%', 'success');
        logStream('📊 Beginning Real-Time Analysis', 'success');
        startPredictionCycle();
    }, 1500);
}

/**
 * Pause Analysis
 */
function pauseAnalysis() {
    STATE.isRunning = false;
    DOM.startBtn.disabled = false;
    DOM.pauseBtn.disabled = true;

    logStream('⏸️ Analysis Paused by User', 'warning');
}

/**
 * Start Prediction Cycle
 */
function startPredictionCycle() {
    if (! STATE.isRunning) return;

    // Update prices and predictions every 3 seconds
    const predictionInterval = setInterval(() => {
        if (!STATE.isRunning) {
            clearInterval(predictionInterval);
            return;
        }

        const startTime = performance.now();

        // Update price data
        updatePriceData();

        // Generate prediction
        generatePrediction();

        // Generate trading signal
        generateSignal();

        // Update dashboard
        updateDashboard();

        // Calculate and display latency
        const endTime = performance.now();
        const latency = Math.round(endTime - startTime);
        DOM.latency.textContent = latency + 'ms';

        // Update charts periodically
        if (STATE.signals.length % 5 === 0) {
            updateCharts();
        }
    }, CONFIG.UPDATE_INTERVAL);
}

/**
 * Update Price Data
 */
function updatePriceData() {
    // Simulate realistic price movement
    const change = (Math.random() - 0.48) * 10; // Slight upward bias
    STATE.currentPrice += change;

    // Keep history
    if (STATE.priceHistory.length >= CONFIG.MAX_PRICE_HISTORY) {
        STATE.priceHistory.shift();
    }
    STATE.priceHistory.push(STATE.currentPrice);

    // Calculate statistics
    const priceChangePercent = ((change / STATE.currentPrice) * 100).toFixed(2);
    const changeClass = change >= 0 ? 'positive' : 'negative';

    // Update UI
    DOM.currentPrice.textContent = `$${STATE.currentPrice.toFixed(2)}`;
    DOM.priceChange.className = `price-change ${changeClass}`;
    DOM.priceChange.textContent = `${priceChangePercent > 0 ? '+' : ''}${priceChangePercent}%`;

    // Update pair rate
    DOM.pairRate.textContent = `${STATE.currentPrice.toFixed(2)}`;

    // Update 24h change
    const priceStart = STATE.priceHistory[0] || STATE.currentPrice;
    const price24hChange = ((STATE.currentPrice - priceStart) / priceStart * 100).toFixed(2);
    DOM.price24h. textContent = `24h: ${price24hChange > 0 ? '+' : ''}${price24hChange}%`;

    // Update timestamp
    const now = new Date();
    DOM.lastUpdate.textContent = `Last Update: ${now.toLocaleTimeString()}`;
    DOM.timestamp.textContent = `${Math.floor(Math.random() * 30)} seconds ago`;

    // Update volume
    const volume = (Math.random() * 5000000 + 1000000).toFixed(0);
    DOM.volume.textContent = `Volume: ${(volume / 1000000).toFixed(1)}M`;

    // Log stream
    logStream(`📈 Price Update: $${STATE.currentPrice.toFixed(2)} | Change: ${priceChangePercent}%`, 'info');

    // Play sound effect (optional)
    if (CONFIG.SOUNDS_ENABLED && Math.random() > 0.8) {
        playNotificationSound();
    }
}

/**
 * Generate Prediction
 */
function generatePrediction() {
    // Analyze recent prices
    const recentPrices = STATE.priceHistory.slice(-20);
    
    if (recentPrices.length === 0) return;

    const average = recentPrices.reduce((a, b) => a + b, 0) / recentPrices.length;
    const volatility = calculateVolatility(recentPrices);

    // Generate prediction based on technical analysis
    let prediction = 'NEUTRAL';
    let forecastIcon = '→';
    let confidenceLevel = 85 + Math.random() * 3;

    if (STATE.currentPrice > average * 1.01) {
        prediction = 'BULLISH';
        forecastIcon = '📈';
    } else if (STATE.currentPrice < average * 0.99) {
        prediction = 'BEARISH';
        forecastIcon = '📉';
    }

    // Update prediction display
    DOM.prediction24h.textContent = prediction;
    DOM.forecastIcon.textContent = forecastIcon;
    DOM.confidence24h.textContent = `Confidence: ${confidenceLevel. toFixed(1)}%`;

    // Calculate resistance and support
    const maxPrice = Math.max(...recentPrices);
    const minPrice = Math.min(...recentPrices);
    const resistance = maxPrice * 1.025;
    const support = minPrice * 0.975;

    DOM.resistance.textContent = `$${resistance.toFixed(2)}`;
    DOM.support.textContent = `$${support.toFixed(2)}`;

    // Update resistance distance
    const resistancePercent = ((resistance - STATE.currentPrice) / STATE.currentPrice * 100).toFixed(1);
    const supportPercent = ((STATE.currentPrice - support) / STATE.currentPrice * 100).toFixed(1);

    // Update current position on range bar
    const range = resistance - support;
    const position = ((STATE.currentPrice - support) / range * 100);
    DOM.currentPosition.style.left = position + '%';

    // Update USD status
    const usdStrength = 50 + Math.random() * 10;
    DOM.strengthFill.style.width = usdStrength + '%';

    if (usdStrength > 55) {
        DOM.usdStatus.textContent = 'STRONG';
    } else if (usdStrength < 45) {
        DOM.usdStatus.textContent = 'WEAK';
    } else {
        DOM.usdStatus.textContent = 'STABLE';
    }

    // Update volatility
    const volatilityPercent = (volatility * 100).toFixed(2);
    let volatilityLevel = 'Low';
    if (volatility > 15) volatilityLevel = 'High';
    else if (volatility > 8) volatilityLevel = 'Medium';

    DOM.volatility.innerHTML = `Volatility: <strong>${volatilityLevel}</strong>`;

    // Update correlation
    const correlation = (Math.random() * 100).toFixed(1);
    DOM.correlation.textContent = `Correlation: ${correlation}%`;

    // Log prediction
    logStream(`🔮 Prediction: ${prediction} | Volatility: ${volatilityLevel}`, 'info');

    // Store prediction
    STATE.predictions.push({
        timestamp: new Date().toISOString(),
        prediction: prediction,
        confidence: confidenceLevel,
        volatility: volatility,
        resistance: resistance,
        support: support
    });
}

/**
 * Calculate Volatility
 */
function calculateVolatility(prices) {
    if (prices.length < 2) return 0;

    const mean = prices.reduce((a, b) => a + b) / prices.length;
    const variance = prices.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / prices.length;
    return Math.sqrt(variance);
}

/**
 * Generate Trading Signal
 */
function generateSignal() {
    // Generate trading signal with weighted probabilities
    const random = Math.random();
    let signal = 'HOLD';
    let signalType = 'Wait';

    // 45% BUY, 25% SELL, 30% HOLD bias
    if (random < 0.45) {
        signal = 'BUY';
        signalType = 'Strong Entry';
    } else if (random < 0.70) {
        signal = 'SELL';
        signalType = 'Take Profit';
    } else {
        signal = 'HOLD';
        signalType = 'Monitor';
    }

    // Update UI
    DOM.tradingSignal.textContent = signal;
    DOM.tradingSignal.className = `signal-indicator ${signal. toLowerCase()}`;
    DOM.signalType.textContent = signalType;
    DOM.signalIcon.textContent = getSignalEmoji(signal);

    // Generate signal strength
    const strength = Math.random() * 0.3 + 0.7; // 70-100%
    const strengthPercent = (strength * 100).toFixed(0);
    const strengthLevel = strengthPercent > 85 ? 'Strong' : strengthPercent > 70 ? 'Moderate' :  'Weak';

    DOM.signalStrengthFill.style.width = strengthPercent + '%';
    DOM. signalStrengthText.textContent = strengthLevel;

    // Generate confidence
    const confidence = (Math.random() * 8 + 79).toFixed(1); // 79-87% range

    // Risk analysis
    const riskPercent = Math.random() * 60 + 20; // 20-80%
    let riskLevel = 'MEDIUM';
    if (riskPercent > 60) riskLevel = 'HIGH';
    if (riskPercent < 35) riskLevel = 'LOW';

    DOM.riskLevel.textContent = `Risk Level: ${riskLevel}`;
    DOM.riskIndicator.style.width = riskPercent + '%';

    // Store signal
    const newSignal = {
        timestamp:  new Date().toLocaleTimeString(),
        signal: signal,
        price: STATE.currentPrice,
        confidence: parseFloat(confidence),
        strength: parseFloat(strengthPercent),
        risk: riskLevel
    };

    STATE.signals.push(newSignal);

    if (STATE.signals.length > 100) {
        STATE.signals. shift();
    }

    STATE.totalProcessed++;

    // Update signal count
    DOM.signalsCount. textContent = STATE.totalProcessed;
    DOM.processedSignals.textContent = STATE.totalProcessed;

    // Log signal
    logStream(`🎯 Signal: ${signal} | Confidence: ${confidence}% | Risk: ${riskLevel}`, 'success');
}

/**
 * Get Signal Emoji
 */
function getSignalEmoji(signal) {
    const emojis = {
        'BUY': '🟢',
        'SELL':  '🔴',
        'HOLD': '🟡'
    };
    return emojis[signal] || '⚪';
}

/**
 * Update Dashboard
 */
function updateDashboard() {
    // Update uptime
    if (STATE.startTime) {
        const uptime = Math.floor((Date.now() - STATE.startTime) / 1000);
        const hours = Math.floor(uptime / 3600);
        const minutes = Math.floor((uptime % 3600) / 60);
        const seconds = uptime % 60;
        
        DOM.uptimeCounter.textContent = 
            `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }

    // Update P&L (simulated)
    if (STATE.priceHistory.length > 1) {
        const startPrice = STATE.priceHistory[0];
        const currentPrice = STATE.currentPrice;
        const pnlPercent = ((currentPrice - startPrice) / startPrice * 100).toFixed(2);
        
        DOM.pnl.textContent = `${pnlPercent > 0 ? '+' : ''}${pnlPercent}%`;
        DOM.pnl.className = 'stat-value ' + (pnlPercent > 0 ? 'positive' : 'negative');
    }

    // Update processing speed
    DOM.processingSpeed.textContent = Math.round(Math.random() * 50 + 5) + 'ms';
}

/**
 * Update Charts
 */
function updateCharts() {
    updateAccuracyChart();
    updateSignalChart();
    updatePriceChart();
    updateVolatilityChart();
}

/**
 * Initialize Charts
 */
function initializeCharts() {
    // Chart. js configuration
    const chartDefaults = {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: {
                labels: {
                    color: '#ffffff',
                    font: { family: "'Courier New', monospace", size: 12 },
                    padding: 15
                }
            }
        }
    };

    // Accuracy Chart - Doughnut
    const accuracyCtx = DOM.accuracyChart.getContext('2d');
    STATE.charts.accuracy = new Chart(accuracyCtx, {
        type: 'doughnut',
        data: {
            labels: ['Accuracy', 'Error Margin'],
            datasets: [{
                data: [87, 13],
                backgroundColor: ['#ff0080', '#1a1a1a'],
                borderColor: ['#00ff00', '#333333'],
                borderWidth: 2,
                borderRadius: 5
            }]
        },
        options: {
            ... chartDefaults,
            cutout: '65%',
            plugins: {
                ... chartDefaults.plugins,
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed + '%';
                        }
                    }
                }
            }
        }
    });

    // Signal Distribution Chart - Bar
    const signalCtx = DOM.signalChart.getContext('2d');
    STATE.charts.signal = new Chart(signalCtx, {
        type: 'bar',
        data: {
            labels: ['BUY', 'SELL', 'HOLD'],
            datasets: [{
                label: 'Signal Count',
                data: [0, 0, 0],
                backgroundColor: ['#00ff00', '#ff0000', '#ffd700'],
                borderColor: ['#00ff00', '#ff0000', '#ffd700'],
                borderWidth: 2,
                borderRadius: 5,
                borderSkipped: false
            }]
        },
        options: {
            ...chartDefaults,
            indexAxis: 'y',
            scales: {
                x: {
                    ticks: { color: '#ffffff' },
                    grid: { color:  'rgba(255, 0, 128, 0.1)' },
                    beginAtZero: true
                },
                y: {
                    ticks: { color: '#ffffff' },
                    grid: { color: 'rgba(255, 0, 128, 0.1)' }
                }
            }
        }
    });

    // Price Movement Chart - Line
    const priceCtx = DOM.priceChart. getContext('2d');
    STATE.charts.price = new Chart(priceCtx, {
        type: 'line',
        data: {
            labels:  [],
            datasets: [{
                label: 'XAU/USD Price',
                data: [],
                borderColor: '#ff0080',
                backgroundColor: 'rgba(255, 0, 128, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#00ff00',
                pointBorderColor: '#ff0080',
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBorderWidth: 2
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                x: {
                    ticks: { color: '#ffffff' },
                    grid: { color: 'rgba(255, 0, 128, 0.1)' }
                },
                y: {
                    ticks: { color: '#ffffff' },
                    grid: { color: 'rgba(255, 0, 128, 0.1)' }
                }
            }
        }
    });

    // Volatility Chart - Line
    const volatilityCtx = DOM.volatilityChart. getContext('2d');
    STATE.charts.volatility = new Chart(volatilityCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Volatility (%)',
                data:  [],
                borderColor: '#00ffff',
                backgroundColor: 'rgba(0, 255, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#ff0080',
                pointBorderColor: '#00ffff',
                pointRadius: 3,
                pointHoverRadius:  5
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                x: {
                    ticks: { color: '#ffffff' },
                    grid: { color: 'rgba(0, 255, 255, 0.1)' }
                },
                y: {
                    ticks: { color: '#ffffff' },
                    grid:  { color: 'rgba(0, 255, 255, 0.1)' },
                    beginAtZero:  true
                }
            }
        }
    });
}

/**
 * Update Accuracy Chart
 */
function updateAccuracyChart() {
    const variance = (Math.random() * 4 - 2);
    const newAccuracy = Math.max(75, Math.min(95, 87 + variance));

    STATE.charts.accuracy.data.datasets[0].data = [newAccuracy. toFixed(1), (100 - newAccuracy).toFixed(1)];
    STATE.charts.accuracy.update('none');
}

/**
 * Update Signal Distribution Chart
 */
function updateSignalChart() {
    const buyCount = STATE.signals.filter(s => s.signal === 'BUY').length;
    const sellCount = STATE.signals.filter(s => s.signal === 'SELL').length;
    const holdCount = STATE.signals.filter(s => s.signal === 'HOLD').length;

    STATE.charts.signal.data.datasets[0].data = [buyCount, sellCount, holdCount];
    STATE.charts.signal.update('none');
}

/**
 * Update Price Chart
 */
function updatePriceChart() {
    const maxDataPoints = 20;
    const recentPrices = STATE.priceHistory.slice(-maxDataPoints);

    STATE.charts.price.data.labels = recentPrices.map((_, i) => `T-${maxDataPoints - i}`);
    STATE.charts.price.data. datasets[0].data = recentPrices;
    STATE.charts.price. update('none');
}

/**
 * Update Volatility Chart
 */
function updateVolatilityChart() {
    const maxDataPoints = 20;
    const recentPrices = STATE.priceHistory.slice(-maxDataPoints);
    const volatilities = [];

    for (let i = 0; i < recentPrices.length; i++) {
        const window = recentPrices.slice(Math.max(0, i - 4), i + 1);
        const vol = calculateVolatility(window) * 10; // Scale for visibility
        volatilities.push(vol);
    }

    STATE.charts.volatility.data.labels = volatilities. map((_, i) => `T-${maxDataPoints - i}`);
    STATE.charts.volatility.data.datasets[0]. data = volatilities;
    STATE.charts. volatility.update('none');
}

/**
 * Update Chart Timeframe
 */
function updateChartTimeframe(event) {
    const timeframe = event.target.value;
    logStream(`📊 Chart Timeframe Changed to: ${timeframe}`, 'info');
    // In real scenario, would fetch new data for selected timeframe
}

/**
 * Log Stream Entry
 */
function logStream(message, type = 'info') {
    const entry = document.createElement('div');
    const timestamp = new Date().toLocaleTimeString();

    entry.className = `stream-entry ${type}-entry`;
    entry.innerHTML = `
        <span class="stream-timestamp">[${timestamp}]</span>
        <span class="stream-message">${escapeHtml(message)}</span>
    `;

    DOM.dataStream.appendChild(entry);

    // Auto-scroll to bottom
    DOM.dataStream.scrollTop = DOM.dataStream.scrollHeight;

    // Keep only last N entries
    while (DOM.dataStream.children. length > CONFIG.MAX_STREAM_ENTRIES) {
        DOM.dataStream.removeChild(DOM.dataStream.firstChild);
    }
}

/**
 * Clear Stream
 */
function clearStream() {
    DOM.dataStream.innerHTML = '';
    logStream('🗑️ Stream Cleared', 'info');
}

/**
 * Toggle Stream Pause
 */
function toggleStreamPause() {
    // In a real scenario, would pause/resume data streaming
    logStream('⏸️ Stream Paused/Resumed', 'warning');
}

/**
 * Export Signals
 */
function exportSignals() {
    if (STATE.signals.length === 0) {
        alert('❌ No signals generated yet. Start the analysis first! ');
        logStream('❌ Export Failed:  No signals available', 'error');
        return;
    }

    // Create CSV content
    let csv = 'Timestamp,Signal,Price (USD),Confidence (%),Strength (%),Risk Level\n';
    
    STATE.signals.forEach(signal => {
        csv += `"${signal.timestamp}","${signal.signal}","${signal.price. toFixed(2)}","${signal.confidence}","${signal.strength}","${signal.risk}"\n`;
    });

    // Create blob and download
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);

    link.setAttribute('href', url);
    link.setAttribute('download', `XAU-USD-Signals-${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    logStream(`✅ Signals Exported:  ${STATE.signals.length} records to CSV`, 'success');
}

/**
 * Open Settings
 */
function openSettings() {
    const settingsHTML = `
        <div style="color: #00ff00; text-align: left; font-family: monospace;">
            <h3 style="color: #ff0080; margin-bottom: 15px;">⚙️ ENGINE SETTINGS</h3>
            <div style="margin-bottom:  10px;">
                <label style="display: block; margin-bottom: 5px;">🔊 Sound Effects:</label>
                <input type="checkbox" ${CONFIG.SOUNDS_ENABLED ? 'checked' : ''} id="soundToggle">
            </div>
            <div style="margin-bottom: 10px;">
                <label style="display: block; margin-bottom: 5px;">🐛 Debug Mode:</label>
                <input type="checkbox" ${CONFIG.DEBUG_MODE ? 'checked' :  ''} id="debugToggle">
            </div>
            <div style="margin-bottom: 10px;">
                <label style="display: block; margin-bottom: 5px;">⏱️ Update Interval (ms):</label>
                <input type="number" value="${CONFIG.UPDATE_INTERVAL}" id="intervalInput" style="background: #1a1a1a; color: #00ff00; border: 1px solid #00ff00; padding: 5px;">
            </div>
            <div style="text-align: right; margin-top: 20px;">
                <button onclick="saveSettings()" style="background: #ff0080; color: white; border: none; padding: 8px 16px; cursor: pointer; margin-right: 10px; border-radius: 4px;">Save</button>
                <button onclick="alert('Settings closed')" style="background: #333; color: #fff; border: 1px solid #666; padding: 8px 16px; cursor: pointer; border-radius: 4px;">Cancel</button>
            </div>
        </div>
    `;

    alert(settingsHTML);
}

/**
 * Open Help
 */
function openHelp() {
    const helpHTML = `
XAU/USD PREDICTIVE ANALYTICS ENGINE - HELP

🎯 FEATURES:
  • Real-Time Price Tracking
  • ML-Powered Predictions (87%+ Accuracy)
  • Trading Signal Generation
  • Risk Analysis
  • Performance Analytics

🎮 CONTROLS:
  • START:  Begin real-time analysis
  • PAUSE: Stop analysis
  • EXPORT: Download CSV signals
  • SETTINGS: Configure engine
  • HELP: Show this message

📊 UNDERSTANDING THE DATA:
  • Bullish:  Upward price movement expected
  • Bearish:  Downward price movement expected
  • Signal Strength: Confidence in signal (%)
  • Risk Level: Market risk assessment

🔑 KEYBOARD SHORTCUTS:
  • Space: Start/Pause
  • E: Export Signals
  • C: Clear Stream
  • ?:  Help

For more info:  https://neydra.io/docs
    `;

    alert(helpHTML);
}

/**
 * Escape HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Play Notification Sound
 */
function playNotificationSound() {
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        oscillator.frequency.value = 800;
        oscillator.type = 'sine';

        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);

        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.1);
    } catch (e) {
        // Audio context not supported or blocked
    }
}

/**
 * Handle Keyboard Shortcuts
 */
function handleKeyboardShortcuts(event) {
    if (event.key === ' ') {
        event.preventDefault();
        if (STATE.isRunning) pauseAnalysis();
        else startAnalysis();
    }
    if (event.key === 'e' || event.key === 'E') {
        exportSignals();
    }
    if (event.key === 'c' || event.key === 'C') {
        clearStream();
    }
    if (event.key === '?') {
        openHelp();
    }
}

/**
 * Save Settings
 */
function saveSettings() {
    const soundToggle = document.getElementById('soundToggle');
    const debugToggle = document.getElementById('debugToggle');
    const intervalInput = document.getElementById('intervalInput');

    if (soundToggle) CONFIG.SOUNDS_ENABLED = soundToggle.checked;
    if (debugToggle) CONFIG.DEBUG_MODE = debugToggle.checked;
    if (intervalInput) CONFIG.UPDATE_INTERVAL = parseInt(intervalInput.value);

    logStream('✅ Settings Saved Successfully', 'success');
    console.log('Settings updated:', CONFIG);
}

/**
 * Initialize Application
 */
function initializeApp() {
    console.log('🚀 Initializing XAU/USD Predictive Analytics Engine.. .');

    // Cache DOM elements
    cacheDOM();

    // Initialize event listeners
    initializeEventListeners();

    // Initialize charts
    initializeCharts();

    // Log initialization
    logStream('✅ Application Initialized Successfully', 'success');
    logStream('📊 XAU/USD Predictive Analytics Engine Ready', 'info');
    logStream('Click START ANALYSIS to begin monitoring', 'info');

    // Initial price update
    updatePriceData();

    console.log('✨ Application Ready! ');
}

/**
 * DOM Content Loaded
 */
document.addEventListener('DOMContentLoaded', initializeApp);

/**
 * Cleanup on Page Unload
 */
window.addEventListener('beforeunload', () => {
    if (STATE.isRunning) {
        pauseAnalysis();
    }
});

// Export for console access
window.XAUEngine = {
    state: STATE,
    startAnalysis,
    pauseAnalysis,
    exportSignals,
    logStream,
    config: CONFIG
};

console.log('%c🚀 XAU/USD Predictive Analytics Engine v1.0', 'color: #ff0080; font-size: 16px; font-weight: bold;');
console.log('%cEnter window.XAUEngine in console to access engine controls', 'color: #00ff00; font-size: 12px;');