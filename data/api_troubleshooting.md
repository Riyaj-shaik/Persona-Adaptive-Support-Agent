# API Troubleshooting Guide

## Authentication Errors

### 401 Unauthorized
A 401 Unauthorized error means the API key is missing, invalid, or has been revoked.

**Steps to resolve:**
1. Verify your API key is correctly set in the `Authorization` header as `Bearer YOUR_API_KEY`.
2. Ensure there are no leading/trailing spaces in your API key string.
3. Regenerate a new API key from the Developer Portal under **Settings > API Keys**.
4. Confirm your account subscription is active — expired plans disable API access.

**Required Header Format:**
```
Authorization: Bearer sk-live-xxxxxxxxxxxxxxxxxxxx
Content-Type: application/json
```

### 403 Forbidden
A 403 error means the API key is valid but lacks permission to access a specific endpoint.

**Resolution:**
- Check your plan's feature access level (Basic, Pro, Enterprise).
- Request elevated API scopes from the Admin Panel.

---

## Rate Limiting (429 Too Many Requests)

The API enforces rate limits to ensure fair usage:
- **Basic Plan**: 60 requests per minute
- **Pro Plan**: 300 requests per minute
- **Enterprise**: Custom rate limits

**Best Practice:** Implement exponential backoff on 429 responses. Wait 2^n seconds between retries where n is the retry attempt number.

---

## Database Integration Errors

### Connection Timeout
If your database integration returns connection timeouts:
1. Check that your database host IP is whitelisted in the platform's network settings.
2. Ensure the connection string follows the format: `postgres://user:pass@host:5432/dbname`
3. Verify SSL certificates are properly configured if using encrypted connections.
4. Increase the `connection_timeout` parameter from the default 30 seconds to 60 seconds.

### Internal Server Errors (500)
Internal errors in database integrations are often caused by schema mismatches:
- Validate your request payload against the API schema documentation.
- Check that all required fields are present and correctly typed.
- Review integration logs in the Developer Portal under **Logs > Integration Events**.

---

## Webhook Configuration

### Setting Up Webhooks
1. Navigate to **Settings > Webhooks** in your dashboard.
2. Enter your endpoint URL (must be HTTPS).
3. Select event types to subscribe to.
4. Copy the generated webhook secret for signature verification.

### Verifying Webhook Signatures
All webhook payloads are signed using HMAC-SHA256. Validate using:
```python
import hmac, hashlib
signature = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
```

---

## Cookie and Session Issues

### Clearing Browser Cache and Cookies
If the platform interface is not loading or behaving unexpectedly:
1. Open your browser's developer tools (F12).
2. Go to **Application > Storage > Clear Site Data**.
3. Alternatively, use keyboard shortcut: **Ctrl+Shift+Delete** (Windows) or **Cmd+Shift+Delete** (Mac).
4. Select "Cookies and cached images/files" and click **Clear Data**.
5. Refresh the page and log in again.

### Session Expiry
Sessions expire after 24 hours of inactivity. If you are being logged out unexpectedly:
- Enable "Remember Me" on the login page for 30-day persistent sessions.
- Enterprise accounts can configure custom session durations in SSO settings.
