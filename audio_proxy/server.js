const uWS = require('uwebsockets.js');
const path = require('path');
const fs = require('fs');

const PORT = process.env.PORT || 3000;

let gladosConnection = null;
let webrtcClients = new Set();

// Create uWS app with compression completely disabled
const app = uWS.App({
    compression: uWS.DISABLED,
}).ws('/glados', {
    /* Options */
    compression: uWS.DISABLED,
    maxCompressedSize: 64 * 1024,
    maxBackpressure: 64 * 1024,

    /* Handlers */
    message: (ws, message, opCode) => {
        try {
            console.log('Raw message received:', Buffer.from(message));
            console.log('Message type:', typeof message);
            console.log('Message length:', message.byteLength);

            const data = JSON.parse(Buffer.from(message).toString());
            console.log('Received from GLaDOS:', data.type);

            if (data.type === 'glados_ready') {
                console.log(`GLaDOS ready with sample rate: ${data.sample_rate}`);
                // Send confirmation
                const response = JSON.stringify({
                    type: 'proxy_ready',
                    clients_connected: webrtcClients.size
                });
                console.log('Sending proxy_ready response:', response);
                try {
                    ws.send(response);
                    console.log('Response sent successfully');
                } catch (error) {
                    console.error('Failed to send response:', error);
                }
            }
            // Forward audio playback to all WebRTC clients
            else if (data.type === 'audio_playback' || data.type === 'stop_playback') {
                console.log(`Forwarding ${data.type} to ${webrtcClients.size} clients`);
                const message = JSON.stringify(data);
                webrtcClients.forEach(client => {
                    // µWebSockets.js doesn't expose readyState, just try to send
                    try {
                        client.send(message);
                    } catch (error) {
                        console.error('Failed to send to client:', error);
                    }
                });
            }
        } catch (error) {
            console.error('Error processing GLaDOS message:', error);
            console.error('Raw message was:', Buffer.from(message).toString());
        }
    },

    open: (ws) => {
        console.log('GLaDOS container connected');
        console.log('Extensions negotiated: (compression completely disabled)');
        console.log('WebSocket connected successfully');
        gladosConnection = ws;
    },

    close: (ws, code, message) => {
        console.log('GLaDOS container disconnected:', code, Buffer.from(message).toString());
        console.log('Close event details - wasClean:', code !== 1006);
        gladosConnection = null;
    },

    error: (ws, error) => {
        console.error('GLaDOS WebSocket error:', error);
    }

}).ws('/client', {
    /* Options */
    compression: uWS.DISABLED,
    maxCompressedSize: 64 * 1024,
    maxBackpressure: 64 * 1024,

    /* Handlers */
    message: (ws, message, opCode) => {
        try {
            const data = JSON.parse(Buffer.from(message).toString());

            // Forward audio data and client events to GLaDOS container
            if (gladosConnection) {
                try {
                    gladosConnection.send(JSON.stringify(data));
                } catch (error) {
                    console.warn('Failed to send to GLaDOS:', error);
                }
            } else {
                console.warn('No GLaDOS connection available for client message');
            }
        } catch (error) {
            console.error('Error processing client message:', error);
        }
    },

    open: (ws) => {
        console.log('WebRTC client connected');
        webrtcClients.add(ws);

        // Notify client that server is ready
        const response = JSON.stringify({
            type: 'server_ready',
            sample_rate: 16000,
            glados_connected: gladosConnection !== null
        });
        ws.send(response);

        // Notify GLaDOS about new client if connected
        if (gladosConnection) {
            try {
                const notification = JSON.stringify({
                    type: 'client_connected',
                    clients_total: webrtcClients.size
                });
                gladosConnection.send(notification);
            } catch (error) {
                console.error('Failed to notify GLaDOS of client connection:', error);
            }
        }
    },

    close: (ws, code, message) => {
        console.log('WebRTC client disconnected');
        webrtcClients.delete(ws);

        // Notify GLaDOS about client disconnect
        if (gladosConnection) {
            try {
                const notification = JSON.stringify({
                    type: 'client_disconnected',
                    clients_total: webrtcClients.size
                });
                gladosConnection.send(notification);
            } catch (error) {
                console.error('Failed to notify GLaDOS of client disconnect:', error);
            }
        }
    },

    error: (ws, error) => {
        console.error('Client WebSocket error:', error);
    }

}).get('/health', (res, req) => {
    res.writeStatus('200 OK')
       .writeHeader('Content-Type', 'application/json')
       .writeHeader('Access-Control-Allow-Origin', '*')
       .end(JSON.stringify({
           status: 'ok',
           gladosConnected: gladosConnection !== null,
           clientsConnected: webrtcClients.size
       }));

}).get('/*', (res, req) => {
    // Serve static files
    const url = req.getUrl();
    let filePath = path.join(__dirname, 'public', url === '/' ? 'index.html' : url);

    // Check if file exists
    if (fs.existsSync(filePath)) {
        const fileContent = fs.readFileSync(filePath);
        const ext = path.extname(filePath);

        let contentType = 'text/plain';
        if (ext === '.html') contentType = 'text/html';
        else if (ext === '.js') contentType = 'application/javascript';
        else if (ext === '.css') contentType = 'text/css';

        res.writeStatus('200 OK')
           .writeHeader('Content-Type', contentType)
           .writeHeader('Access-Control-Allow-Origin', '*')
           .end(fileContent);
    } else {
        res.writeStatus('404 Not Found')
           .writeHeader('Access-Control-Allow-Origin', '*')
           .end('File not found');
    }

}).listen(PORT, (token) => {
    if (token) {
        console.log(`GLaDOS Audio Proxy (uWS) running on port ${PORT}`);
        console.log(`Web interface: http://localhost:${PORT}`);
        console.log(`GLaDOS WebSocket: ws://localhost:${PORT}/glados`);
        console.log(`Client WebSocket: ws://localhost:${PORT}/client`);
    } else {
        console.log('Failed to listen to port ' + PORT);
    }
});