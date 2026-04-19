#!/usr/bin/env python3
"""
Refresh Schwab Authentication Token

This script refreshes the Schwab access token by using the refresh token.
If the refresh token is also expired, it will open a browser for full re-authentication.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.streaming.authenticator import Authenticator


def main():
    parser = argparse.ArgumentParser(
        description="Refresh Schwab authentication token",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python refresh_schwab_token.py            # Refresh using refresh token
    python refresh_schwab_token.py --force    # Force full re-authentication
        """,
    )

    parser.add_argument("--force", "-f", action="store_true", help="Force full re-authentication (opens browser)")

    args = parser.parse_args()

    print("=" * 60)
    print("  SCHWAB TOKEN REFRESH")
    print("=" * 60)
    print()

    try:
        auth = Authenticator()

        if args.force:
            print("Force re-authentication requested...")
            print("This will require you to log into Schwab in your browser.")
            print()
            input("Press Enter to continue...")
            print()

            # Force new authentication
            # The authenticator will handle browser authentication flow
            import asyncio

            result = asyncio.run(auth.manual_refresh_token(use_gui=False))

            if result is True:
                print()
                print("✓ Authentication successful!")
            else:
                print()
                print(f"✗ Authentication failed: {result}")
                return 1

        else:
            # Try to refresh using refresh token
            print("Attempting to refresh access token...")

            # Get current access token (will use refresh token if access token expired)
            token = auth.access_token()

            if token:
                print("✓ Token refreshed successfully!")
            else:
                print("✗ Token refresh failed.")
                print()
                print("The refresh token may be expired.")
                print("Run with --force to re-authenticate:")
                print("    python refresh_schwab_token.py --force")
                return 1

    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        print("If refresh token is expired, run with --force:")
        print("    python refresh_schwab_token.py --force")
        return 1

    print()
    print("=" * 60)
    print("  TOKEN STATUS")
    print("=" * 60)

    # Show token expiry info
    try:
        import asyncio

        from core.credential_validator import CredentialValidator

        async def check_status():
            validator = CredentialValidator()
            result = await validator.validate_schwab()

            print(f"Status: {result.status.name}")
            print(f"Message: {result.message}")

            if hasattr(result, "expires_in_hours") and result.expires_in_hours:
                print(f"Expires in: {result.expires_in_hours:.1f} hours")

        asyncio.run(check_status())

    except Exception as e:
        print(f"Could not check token status: {e}")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
