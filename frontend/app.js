// Ski Cam Analytics - Frontend JavaScript

// API Base URL
const API_BASE = window.location.origin;

// State
let ws = null;
let occupancyChart = null;
let crossingsChart = null;
let videoRefreshInterval = null;

// === Initialization ===

document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Ski Cam Analytics Dashboard...');
    
    // Setup event listeners
    document.getElementById('btn-start').addEventListener('click', startPipeline);
    document.getElementById('btn-stop').addEventListener('click', stopPipeline);
    
    // Initialize charts
    initCharts();
    
    // Load initial data
    loadStatus();
    loadMetrics();
    loadTimeseries();
    
    // Connect WebSocket
    connectWebSocket();
    
    // Periodic refresh (fallback if WebSocket fails)
    setInterval(() => {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            loadStatus();
            loadMetrics();
        }
    }, 5000);
    
    // Refresh charts every minute
    setInterval(loadTimeseries, 60000);
    
    // Start video feed refresh
    startVideoRefresh();
});

// === WebSocket ===

function connectWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/live`;
    
    console.log('Connecting to WebSocket:', wsUrl);
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        document.getElementById('stream-status').className = 'status-badge online';
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateStatus(data.status);
        updateMetrics(data.metrics);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting in 5s...');
        document.getElementById('stream-status').className = 'status-badge offline';
        setTimeout(connectWebSocket, 5000);
    };
}

// === API Calls ===

async function loadStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();
        updateStatus(data);
    } catch (error) {
        console.error('Error loading status:', error);
    }
}

async function loadMetrics() {
    try {
        const response = await fetch(`${API_BASE}/api/metrics/latest`);
        const data = await response.json();
        updateMetrics(data);
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

async function loadTimeseries() {
    try {
        const response = await fetch(`${API_BASE}/api/metrics/timeseries?minutes=60`);
        const data = await response.json();
        updateCharts(data);
    } catch (error) {
        console.error('Error loading timeseries:', error);
    }
}

async function startPipeline() {
    const btn = document.getElementById('btn-start');
    btn.disabled = true;
    btn.textContent = '⏳ Spouštím...';
    
    try {
        const response = await fetch(`${API_BASE}/api/pipeline/start`, {
            method: 'POST'
        });
        
        if (response.ok) {
            console.log('Pipeline started');
            setTimeout(() => {
                loadStatus();
                loadMetrics();
            }, 1000);
        } else {
            const error = await response.json();
            alert(`Chyba při spuštění: ${error.detail || 'Neznámá chyba'}`);
            btn.disabled = false;
            btn.textContent = '▶️ START ANALÝZY';
        }
    } catch (error) {
        console.error('Error starting pipeline:', error);
        alert('Chyba při komunikaci se serverem');
        btn.disabled = false;
        btn.textContent = '▶️ START ANALÝZY';
    }
}

async function stopPipeline() {
    const btn = document.getElementById('btn-stop');
    btn.disabled = true;
    btn.textContent = '⏳ Zastavuji...';
    
    try {
        const response = await fetch(`${API_BASE}/api/pipeline/stop`, {
            method: 'POST'
        });
        
        if (response.ok) {
            console.log('Pipeline stopped');
            setTimeout(() => {
                loadStatus();
                loadMetrics();
            }, 1000);
        } else {
            alert('Chyba při zastavování');
            btn.disabled = false;
            btn.textContent = '⏹️ STOP ANALÝZY';
        }
    } catch (error) {
        console.error('Error stopping pipeline:', error);
        alert('Chyba při komunikaci se serverem');
        btn.disabled = false;
        btn.textContent = '⏹️ STOP ANALÝZY';
    }
}

// === UI Updates ===

function updateStatus(status) {
    const statusBadge = document.getElementById('pipeline-status');
    const fpsDisplay = document.getElementById('fps-display');
    const uptimeDisplay = document.getElementById('uptime-display');
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    
    if (status.running) {
        statusBadge.textContent = 'RUNNING';
        statusBadge.className = 'status-badge large running';
        btnStart.disabled = true;
        btnStart.textContent = '▶️ START ANALÝZY';
        btnStop.disabled = false;
    } else {
        statusBadge.textContent = 'STOPPED';
        statusBadge.className = 'status-badge large stopped';
        btnStart.disabled = false;
        btnStart.textContent = '▶️ START ANALÝZY';
        btnStop.disabled = true;
        btnStop.textContent = '⏹️ STOP ANALÝZY';
    }
    
    fpsDisplay.textContent = `FPS: ${status.fps || 0}`;
    
    if (status.uptime_seconds) {
        const hours = Math.floor(status.uptime_seconds / 3600);
        const minutes = Math.floor((status.uptime_seconds % 3600) / 60);
        const seconds = Math.floor(status.uptime_seconds % 60);
        uptimeDisplay.textContent = `Uptime: ${hours}h ${minutes}m ${seconds}s`;
    } else {
        uptimeDisplay.textContent = 'Uptime: --';
    }
}

function updateMetrics(metrics) {
    document.getElementById('occupancy-value').textContent = metrics.occupancy || 0;
    document.getElementById('crossings-total-value').textContent = metrics.crossings_this_run || 0;
    document.getElementById('crossings-1m-value').textContent = metrics.crossings_last_1m || 0;
    document.getElementById('crossings-10m-value').textContent = metrics.crossings_last_10m || 0;
}

// === Charts ===

function initCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 2.5,
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            x: {
                grid: {
                    color: '#334155'
                },
                ticks: {
                    color: '#94a3b8'
                }
            },
            y: {
                beginAtZero: true,
                grid: {
                    color: '#334155'
                },
                ticks: {
                    color: '#94a3b8'
                }
            }
        }
    };
    
    // Occupancy chart
    const occupancyCtx = document.getElementById('occupancy-chart').getContext('2d');
    occupancyChart = new Chart(occupancyCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Occupancy',
                data: [],
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: chartOptions
    });
    
    // Crossings chart
    const crossingsCtx = document.getElementById('crossings-chart').getContext('2d');
    crossingsChart = new Chart(crossingsCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Crossings',
                data: [],
                backgroundColor: '#10b981'
            }]
        },
        options: chartOptions
    });
}

function updateCharts(timeseries) {
    if (!timeseries || timeseries.length === 0) {
        return;
    }
    
    // Extract data
    const labels = timeseries.map(d => {
        const date = new Date(d.timestamp);
        return `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
    });
    
    const occupancyData = timeseries.map(d => d.occupancy_avg);
    const crossingsData = timeseries.map(d => d.crossings);
    
    // Update occupancy chart
    occupancyChart.data.labels = labels;
    occupancyChart.data.datasets[0].data = occupancyData;
    occupancyChart.update();
    
    // Update crossings chart
    crossingsChart.data.labels = labels;
    crossingsChart.data.datasets[0].data = crossingsData;
    crossingsChart.update();
}

// === Video Feed ===

function startVideoRefresh() {
    const videoFeed = document.getElementById('video-feed');
    
    // Refresh video every 200ms (5 FPS)
    videoRefreshInterval = setInterval(() => {
        // Add timestamp to prevent caching
        const timestamp = new Date().getTime();
        videoFeed.src = `${API_BASE}/api/frame/latest?t=${timestamp}`;
    }, 200);
}

function stopVideoRefresh() {
    if (videoRefreshInterval) {
        clearInterval(videoRefreshInterval);
        videoRefreshInterval = null;
    }
}
