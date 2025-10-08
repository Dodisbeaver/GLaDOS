const express = require('express');
const uWS = require('uws');
const cors = require('cors');
const path = require('path');

const app = express();

// Enable CORS for all routes
app.use(cors());
app.use(express.static(path.join(__dirname, 'public')));

let gladosConnection = null;
let webrtcClients = new Set();

// Handle upgrade requests for WebSocket connections
server.on('upgrade', (request, socket, body) => {
    console.log('WebSocket upgrade request for:', request.url);
    console.log('Handshake headers:', request.headers);

    if (WebSocket.isWebSocket(request)) {
        const ws = new WebSocket(request, socket, body);

        if (request.url === '/glados') {
            console.log('GLaDOS container connected');
            console.log('Extensions negotiated: (none with faye-websocket)');
            console.log('WebSocket readyState:', ws.readyState);
            gladosConnection = ws;
            handleGladosConnection(ws);
        } else if (request.url === '/client') {
            console.log('WebRTC client connected');
            webrtcClients.add(ws);
            handleClientConnection(ws);
        } else {
            ws.close();
        }
    }
});

// Handle GLaDOS container connections
gladosWs.on('connection', (ws, req) => {
    console.log('GLaDOS container connected');
    console.log('Extensions negotiated:', ws.extensions);
    console.log('WebSocket readyState:', ws.readyState);
    console.log('Headers:', req.headers);
    gladosConnection = ws;

    // Wait for client to send ready message instead of sending immediately

    ws.on('message', (message) => {
        try {
            console.log('Raw message received:', message);
            console.log('Message type:', typeof message);
            console.log('Message length:', message.length);
            const data = JSON.parse(message);
            console.log('Received from GLaDOS:', data.type);

            if (data.type === 'glados_ready') {
                console.log(`GLaDOS ready with sample rate: ${data.sample_rate}`);
                // Send confirmation
                ws.send(JSON.stringify({
                    type: 'proxy_ready',
                    clients_connected: webrtcClients.size
                }));
            }
            // Forward audio playback to all WebRTC clients
            else if (data.type === 'audio_playback' || data.type === 'stop_playback') {
                console.log(`Forwarding ${data.type} to ${webrtcClients.size} clients`);
                webrtcClients.forEach(client => {
                    if (client.readyState === WebSocket.OPEN) {
                        client.send(JSON.stringify(data));
                    }
                });
            }
        } catch (error) {
            console.error('Error processing GLaDOS message:', error);
            console.error('Raw message was:', message.toString());
        }
    });

    ws.on('close', (code, reason) => {
        console.log('GLaDOS container disconnected:', code, reason.toString());
        console.log('Close event details - wasClean:', code !== 1006);
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