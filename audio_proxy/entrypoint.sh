#!/bin/bash

# GLaDOS Audio Proxy Entrypoint Script
# This script ensures SSL certificates are generated before starting the server

set -e

echo "🚀 Starting GLaDOS Audio Proxy..."

# Check if SSL certificates exist
if [ ! -f "ssl/cert.pem" ] || [ ! -f "ssl/key.pem" ]; then
    echo "📋 SSL certificates not found, generating new ones..."
    ./generate-ssl.sh
else
    echo "✅ SSL certificates found, using existing ones"
    echo "   - Certificate: ssl/cert.pem"
    echo "   - Private key: ssl/key.pem"
fi

echo ""
echo "🌐 Network Configuration:"
echo "   - HTTP Port: ${PORT:-6080}"
echo "   - HTTPS Port: ${HTTPS_PORT:-6443}"
echo "   - LAN IP: ${SSL_LAN_IP:-192.168.1.100}"
if [ -n "$SSL_EXTERNAL_IP" ]; then
    echo "   - External IP: $SSL_EXTERNAL_IP"
fi
if [ -n "$SSL_DOMAINS" ]; then
    echo "   - Custom domains: $SSL_DOMAINS"
fi

echo ""
echo "🔧 Starting Node.js server..."

# Start the main application
exec npm start