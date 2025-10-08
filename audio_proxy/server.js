const express = require('express');
const WebSocket = require('ws');
const http = require('http');
const cors = require('cors');
const path = require('path');

const app = express();
const server = http.createServer(app);

// Enable CORS for all routes
app.use(cors());
app.use(express.static(path.join(__dirname, 'public')));

// WebSocket server for GLaDOS communication
const gladosWs = new WebSocket.Server({ server, path: '/glados' });

// WebSocket server for WebRTC clients
const clientWs = new WebSocket.Server({ server, path: '/client' });

let gladosConnection = null;
let webrtcClients = new Set();

// Handle GLaDOS container connections
gladosWs.on('connection', (ws) => {
    console.log('GLaDOS container connected');
    gladosConnection = ws;

    // Send ready signal to GLaDOS
    ws.send(JSON.stringify({
        type: 'proxy_ready',
        clients_connected: webrtcClients.size
    }));

    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);

            // Forward audio playback to all WebRTC clients
            if (data.type === 'audio_playback' || data.type === 'stop_playback') {
                webrtcClients.forEach(client => {
                    if (client.readyState === WebSocket.OPEN) {
                        client.send(JSON.stringify(data));
                    }
                });
            }
        } catch (error) {
            console.error('Error processing GLaDOS message:', error);
        }
    });

    ws.on('close', () => {
        console.log('GLaDOS container disconnected');
        gladosConnection = null;
    });

    ws.on('error', (error) => {
        console.error('GLaDOS WebSocket error:', error);
    });
});

// Handle WebRTC client connections
clientWs.on('connection', (ws) => {
    console.log('WebRTC client connected');
    webrtcClients.add(ws);

    // Notify client that server is ready
    ws.send(JSON.stringify({
        type: 'server_ready',
        sample_rate: 16000,
        glados_connected: gladosConnection !== null
    }));

    // Notify GLaDOS about new client if connected
    if (gladosConnection && gladosConnection.readyState === WebSocket.OPEN) {
        gladosConnection.send(JSON.stringify({
            type: 'client_connected',
            clients_total: webrtcClients.size
        }));
    }

    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);

            // Forward audio data and client events to GLaDOS container
            if (gladosConnection && gladosConnection.readyState === WebSocket.OPEN) {
                gladosConnection.send(JSON.stringify(data));
            } else {
                console.warn('No GLaDOS connection available for client message');
            }
        } catch (error) {
            console.error('Error processing client message:', error);
        }
    });

    ws.on('close', () => {
        console.log('WebRTC client disconnected');
        webrtcClients.delete(ws);

        // Notify GLaDOS about client disconnect
        if (gladosConnection && gladosConnection.readyState === WebSocket.OPEN) {
            gladosConnection.send(JSON.stringify({
                type: 'client_disconnected',
                clients_total: webrtcClients.size
            }));
        }
    });

    ws.on('error', (error) => {
        console.error('Client WebSocket error:', error);
    });
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        gladosConnected: gladosConnection !== null,
        clientsConnected: webrtcClients.size
    });
});

// Serve the WebRTC client interface
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, '0.0.0.0', () => {
    console.log(`GLaDOS Audio Proxy running on port ${PORT}`);
    console.log(`Web interface: http://localhost:${PORT}`);
    console.log(`GLaDOS WebSocket: ws://localhost:${PORT}/glados`);
    console.log(`Client WebSocket: ws://localhost:${PORT}/client`);
});