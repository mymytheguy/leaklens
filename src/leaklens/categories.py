"""Human-readable labels for detector categories."""

from __future__ import annotations

CATEGORY_LABELS: dict[str, str] = {
    "ipv4": "IPv4 address",
    "ipv4_private": "IPv4 address",
    "ipv4_public": "IPv4 address",
    "ipv6": "IPv6 address",
    "gps_coordinates": "GPS coordinate",
    "email": "Email address",
    "phone_number": "Phone number",
    "mac_address": "MAC address",
    "api_key": "API key",
    "access_token": "Access token",
    "jwt": "JWT",
    "cookie_session": "Session ID / cookie",
    "authorization_header": "Authorization header",
    "password": "Password",
    "ssh_key": "Private key",
    "cloud_credential": "Cloud credential",
    "database_connection": "Database connection string",
    "credential_url": "Sensitive URL",
    "private_url": "Sensitive URL",
    "windows_path": "Windows username",
    "unix_home_path": "Unix home path",
    "hostname": "Hostname / device name",
    "username_handle": "Username",
    "full_name": "Possible name",
    "wifi_ssid": "Wi-Fi SSID",
    "license_plate": "License plate",
    "other_sensitive": "Possible secret",
    "notification_popup": "Notification",
    "autofill_suggestion": "Autofill suggestion",
    "face": "Face",
    "qr_code": "QR code",
    "barcode": "Barcode",
}


def category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category.replace("_", " "))
