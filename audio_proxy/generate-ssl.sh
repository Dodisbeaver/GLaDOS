#!/bin/bash

# SSL Certificate Generation Script for GLaDOS Audio Proxy
# This script generates self-signed SSL certificates for local, LAN, and external access

set -e

# Default values for certificate details
CERT_COUNTRY="${SSL_COUNTRY:-US}"
CERT_STATE="${SSL_STATE:-California}"
CERT_CITY="${SSL_CITY:-San Francisco}"
CERT_ORG="${SSL_ORG:-GLaDOS}"
CERT_OU="${SSL_OU:-Audio Proxy}"
CERT_CN="${SSL_CN:-audio-proxy}"
CERT_EMAIL="${SSL_EMAIL:-admin@localhost}"

# Network configuration
LAN_IP="${SSL_LAN_IP:-192.168.1.100}"        # Internal LAN IP
EXTERNAL_IP="${SSL_EXTERNAL_IP:-}"            # External/public IP (optional)
CUSTOM_DOMAINS="${SSL_DOMAINS:-}"             # Custom domains (comma-separated)

# Certificate validity period (default 365 days)
CERT_DAYS="${SSL_DAYS:-365}"

# SSL directory
SSL_DIR="$(dirname "$0")/ssl"

echo "Generating SSL certificates for GLaDOS Audio Proxy..."
echo "Certificate details:"
echo "  Country: $CERT_COUNTRY"
echo "  State: $CERT_STATE"
echo "  City: $CERT_CITY"
echo "  Organization: $CERT_ORG"
echo "  Organizational Unit: $CERT_OU"
echo "  Common Name: $CERT_CN"
echo "  Email: $CERT_EMAIL"
echo "  LAN IP: $LAN_IP"
echo "  External IP: ${EXTERNAL_IP:-'Not set'}"
echo "  Custom domains: ${CUSTOM_DOMAINS:-'None'}"
echo "  Valid for: $CERT_DAYS days"

# Create SSL directory if it doesn't exist
mkdir -p "$SSL_DIR"

# Generate private key
echo "Generating private key..."
openssl genrsa -out "$SSL_DIR/key.pem" 2048

# Create certificate request configuration
cat > "$SSL_DIR/cert.conf" << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = $CERT_COUNTRY
ST = $CERT_STATE
L = $CERT_CITY
O = $CERT_ORG
OU = $CERT_OU
CN = $CERT_CN
emailAddress = $CERT_EMAIL

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
DNS.3 = audio-proxy
DNS.4 = *.audio-proxy
DNS.5 = $CERT_CN
IP.1 = 127.0.0.1
IP.2 = ::1
IP.3 = $LAN_IP
EOF

# Add external IP if provided
dns_counter=6
ip_counter=4
if [ -n "$EXTERNAL_IP" ]; then
    echo "IP.$ip_counter = $EXTERNAL_IP" >> "$SSL_DIR/cert.conf"
    ip_counter=$((ip_counter + 1))
fi

# Add custom domains if provided
if [ -n "$CUSTOM_DOMAINS" ]; then
    IFS=',' read -ra DOMAINS <<< "$CUSTOM_DOMAINS"
    for domain in "${DOMAINS[@]}"; do
        domain=$(echo "$domain" | xargs)  # trim whitespace
        echo "DNS.$dns_counter = $domain" >> "$SSL_DIR/cert.conf"
        dns_counter=$((dns_counter + 1))
    done
fi

# Generate certificate
echo "Generating certificate..."
openssl req -new -x509 -key "$SSL_DIR/key.pem" -out "$SSL_DIR/cert.pem" \
    -days "$CERT_DAYS" -config "$SSL_DIR/cert.conf" -extensions v3_req

# Set appropriate permissions
chmod 600 "$SSL_DIR/key.pem"
chmod 644 "$SSL_DIR/cert.pem"

# Clean up temporary config file
rm "$SSL_DIR/cert.conf"

echo ""
echo "✅ SSL certificates generated successfully!"
echo "  Private key: $SSL_DIR/key.pem"
echo "  Certificate: $SSL_DIR/cert.pem"
echo ""
echo "📋 Certificate includes the following names/IPs:"
echo "  🏠 localhost, 127.0.0.1, ::1 (local development)"
echo "  🐳 audio-proxy (Docker service name)"
echo "  🏢 $LAN_IP (LAN access)"
if [ -n "$EXTERNAL_IP" ]; then
    echo "  🌐 $EXTERNAL_IP (external access)"
fi
if [ -n "$CUSTOM_DOMAINS" ]; then
    echo "  🔗 Custom domains: $CUSTOM_DOMAINS"
fi
echo ""
echo "🔧 Environment variables for configuration:"
echo "  SSL_LAN_IP=$LAN_IP           # Your server's LAN IP"
echo "  SSL_EXTERNAL_IP=$EXTERNAL_IP # External IP (optional)"
echo "  SSL_DOMAINS=$CUSTOM_DOMAINS  # Custom domains (optional, comma-separated)"
echo "  SSL_DAYS=$CERT_DAYS          # Certificate validity period"
echo ""
echo "💡 Example usage:"
echo "  SSL_LAN_IP=192.168.1.50 SSL_EXTERNAL_IP=203.0.113.10 ./generate-ssl.sh"