# GLaDOS Universal Deployment Guide

This guide covers deploying GLaDOS for various network configurations including local, LAN, and external access.

## Quick Start

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Configure your network settings** in `.env`:
   ```bash
   # Set your server's LAN IP
   SSL_LAN_IP=192.168.1.50

   # Optional: Set external IP if port forwarding
   SSL_EXTERNAL_IP=203.0.113.10

   # Optional: Custom domains
   SSL_DOMAINS=glados.mydomain.com,voice.mydomain.com
   ```

3. **Deploy:**
   ```bash
   docker-compose up -d
   ```

## Network Configuration

### Local Development
```bash
# .env
SSL_LAN_IP=127.0.0.1
PROXY_PORT=3000
PROXY_HTTPS_PORT=10306
```
- Access: `http://localhost:3000` or `https://localhost:10306`

### LAN Access
```bash
# .env
SSL_LAN_IP=192.168.1.100  # Your server's IP
PROXY_PORT=3000
PROXY_HTTPS_PORT=10306
```
- Access from LAN: `http://192.168.1.100:3000` or `https://192.168.1.100:10306`

### External Access (with port forwarding)
```bash
# .env
SSL_LAN_IP=192.168.1.100
SSL_EXTERNAL_IP=203.0.113.10  # Your public IP
PROXY_PORT=3000
PROXY_HTTPS_PORT=10306
```
- Forward ports 3000 and 10306 on your router
- Access externally: `https://203.0.113.10:10306`

### Custom Domain
```bash
# .env
SSL_LAN_IP=192.168.1.100
SSL_EXTERNAL_IP=203.0.113.10
SSL_DOMAINS=glados.mydomain.com
```
- Set DNS A record: `glados.mydomain.com → 203.0.113.10`
- Access: `https://glados.mydomain.com:10306`

## SSL Configuration

SSL certificates are automatically generated on container startup with the following configurable options:

| Variable | Default | Description |
|----------|---------|-------------|
| `SSL_COUNTRY` | `US` | Certificate country |
| `SSL_STATE` | `California` | Certificate state/province |
| `SSL_CITY` | `San Francisco` | Certificate city |
| `SSL_ORG` | `GLaDOS` | Organization name |
| `SSL_OU` | `Audio Proxy` | Organizational unit |
| `SSL_CN` | `audio-proxy` | Common name |
| `SSL_EMAIL` | `admin@localhost` | Contact email |
| `SSL_LAN_IP` | `192.168.1.100` | **Your server's LAN IP** |
| `SSL_EXTERNAL_IP` | _(none)_ | External/public IP (optional) |
| `SSL_DOMAINS` | _(none)_ | Custom domains (comma-separated) |
| `SSL_DAYS` | `365` | Certificate validity period |

### Manual SSL Generation

You can also generate certificates manually:

```bash
cd audio_proxy
./generate-ssl.sh
```

Or with custom settings:
```bash
SSL_LAN_IP=192.168.1.50 SSL_EXTERNAL_IP=203.0.113.10 ./generate-ssl.sh
```

## Port Configuration

| Service | Default Port | Environment Variable | Description |
|---------|--------------|---------------------|-------------|
| TTS API | 5050 | `TTS_PORT` | Text-to-speech API |
| Audio Proxy (HTTP) | 3000 | `PROXY_PORT` | Web interface & WebSocket |
| Audio Proxy (HTTPS) | 10306 | `PROXY_HTTPS_PORT` | Secure web interface |

## Testing Your Deployment

1. **Health Check:**
   ```bash
   curl http://your-server-ip:3000/health
   ```

2. **HTTPS Access:**
   ```bash
   curl -k https://your-server-ip:10306/health
   ```

3. **From another machine on your LAN:**
   - Open browser to `https://192.168.1.100:10306` (replace with your IP)
   - You'll get a security warning (expected for self-signed certificates)
   - Click "Advanced" → "Proceed to..." to continue

## Security Notes

- Self-signed certificates will show browser warnings
- For production use, consider using Let's Encrypt or proper CA certificates
- The current setup is designed for trusted LAN environments
- Always use HTTPS for external access

## Troubleshooting

### SSL Certificate Issues
- Check container logs: `docker-compose logs audio-proxy`
- Regenerate certificates: `docker-compose exec audio-proxy ./generate-ssl.sh`

### Network Access Issues
- Verify firewall settings on your server
- Check router port forwarding configuration
- Ensure `SSL_LAN_IP` matches your server's actual IP address

### Browser Certificate Warnings
- Expected behavior with self-signed certificates
- Click "Advanced" → "Proceed" to continue
- Consider adding the certificate to your browser's trusted certificates