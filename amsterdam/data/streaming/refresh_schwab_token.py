#!/usr/bin/env python3
"""
Schwab Token Refresh Script

Checks token status and refreshes if needed.
Run this when you see authentication errors.

Usage:
    python refresh_schwab_token.py
    python refresh_schwab_token.py --force  # Force full re-authentication
"""

import asyncio
import os
import sys

# Load .env from project root before any other imports
from dotenv import load_dotenv
project_root = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(project_root, ".env"))

from data.streaming.authenticator import Authenticator


async def main():
    auth = Authenticator()
    force = "--force" in sys.argv

    # Read current token data
    token_data = auth._read_token_file()

    if not token_data:
        print("No token file found. Starting fresh authentication...")
        result = await auth.manual_refresh_token(use_gui=False)
        print(f"Result: {'Success' if result is True else result}")
        return

    # Check token status
    access_expired = auth._is_access_token_expired(token_data)
    refresh_expired = auth._is_refresh_token_expired(token_data)

    print(f"Access token expired:  {access_expired}")
    print(f"Refresh token expired: {refresh_expired}")

    if force:
        print("\n--force flag set, performing full re-authentication...")
        result = await auth.manual_refresh_token(use_gui=False)
        print(f"Result: {'Success' if result is True else result}")
        return

    if refresh_expired:
        print("\nRefresh token expired - need full re-authentication")
        print("You will need to log in via browser.\n")
        result = await auth.manual_refresh_token(use_gui=False)
    elif access_expired:
        print("\nAccess token expired - refreshing automatically...")
        result = await auth.renew_access()
    else:
        print("\nTokens are still valid. No action needed.")
        print("Use --force to force re-authentication anyway.")
        return

    if result is True:
        print("Token refresh successful!")
    else:
        print(f"Token refresh failed: {result}")


if __name__ == "__main__":
    asyncio.run(main())
