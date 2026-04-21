"""
Tests for token refresh functionality in SchwabStreamingClient.

Coverage:
- Token refresh loop starts with streaming
- Token refresh loop stops with streaming
- Proactive token renewal before expiration
- Error handling in token refresh
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def mock_authenticator():
    """Create a mock authenticator."""
    auth = MagicMock()
    auth._read_token_file.return_value = {
        "access_token": "test_token",
        "refresh_token": "test_refresh",
        "access_time": time.time(),
        "expires_in": 1800,  # 30 minutes
    }
    auth._is_access_token_expired.return_value = False
    auth.renew_access = AsyncMock(return_value=True)
    return auth


@pytest.fixture
def mock_streaming_client(mock_authenticator):
    """Create a mock SchwabStreamingClient."""
    with patch("data.streaming.streamer.Authenticator", return_value=mock_authenticator):
        with patch("data.streaming.streamer.SchwabClient"):
            from data.streaming.streamer import SchwabStreamingClient

            client = SchwabStreamingClient("test_key", "test_secret")
            client.authenticator = mock_authenticator
            return client


class TestTokenRefreshLoop:
    """Tests for the token refresh loop."""

    @pytest.mark.asyncio
    async def test_token_refresh_task_starts_with_run(self, mock_authenticator):
        """Token refresh task should start when run() is called."""
        with patch("data.streaming.streamer.Authenticator", return_value=mock_authenticator):
            with patch("data.streaming.streamer.SchwabClient"):
                from data.streaming.streamer import SchwabStreamingClient

                client = SchwabStreamingClient("test_key", "test_secret")
                # Replace authenticator with our mock after init
                client.authenticator = mock_authenticator
                client._token_refresh_interval = 0.05  # Fast interval for testing

                # Mock websocket_client to exit after brief run
                async def mock_ws(symbols):
                    await asyncio.sleep(0.1)
                    raise Exception("Test exit")

                client.websocket_client = mock_ws
                client._reconnect_enabled = False

                # Run the client - it will exit after websocket fails
                await client.run(["AAPL"])

                # Verify token file was read (token refresh loop ran)
                assert mock_authenticator._read_token_file.called

    @pytest.mark.asyncio
    async def test_token_refresh_renews_expired_token(self, mock_streaming_client):
        """Token refresh should call renew_access when token is expired."""
        client = mock_streaming_client
        client._token_refresh_interval = 0.05  # Fast interval

        # Mark token as expired
        client.authenticator._is_access_token_expired.return_value = True

        # Run the refresh loop briefly
        client._running = True
        refresh_task = asyncio.create_task(client._token_refresh_loop())
        await asyncio.sleep(0.15)
        client._running = False
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass

        # Verify renew_access was called
        client.authenticator.renew_access.assert_called()

    @pytest.mark.asyncio
    async def test_token_refresh_proactive_renewal(self, mock_streaming_client):
        """Token refresh should proactively renew when < 5 minutes remaining."""
        client = mock_streaming_client
        client._token_refresh_interval = 0.05

        # Set token to expire in 4 minutes (< 5 min threshold)
        client.authenticator._read_token_file.return_value = {
            "access_token": "test_token",
            "refresh_token": "test_refresh",
            "access_time": time.time() - 1560,  # 26 minutes ago
            "expires_in": 1800,  # Expires in 4 minutes
        }
        client.authenticator._is_access_token_expired.return_value = False

        # Run the refresh loop briefly
        client._running = True
        refresh_task = asyncio.create_task(client._token_refresh_loop())
        await asyncio.sleep(0.15)
        client._running = False
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass

        # Verify proactive renewal was triggered
        client.authenticator.renew_access.assert_called()

    @pytest.mark.asyncio
    async def test_token_refresh_handles_missing_token_file(self, mock_streaming_client):
        """Token refresh should handle missing token file gracefully."""
        client = mock_streaming_client
        client._token_refresh_interval = 0.05

        # Simulate missing token file
        client.authenticator._read_token_file.return_value = None

        # Run the refresh loop briefly - should not crash
        client._running = True
        refresh_task = asyncio.create_task(client._token_refresh_loop())
        await asyncio.sleep(0.15)
        client._running = False
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass

        # Verify it handled gracefully (no crash, renew_access not called)
        assert not client.authenticator.renew_access.called

    @pytest.mark.asyncio
    async def test_token_refresh_stops_on_cancel(self, mock_streaming_client):
        """Token refresh loop should stop cleanly on cancellation."""
        client = mock_streaming_client
        client._token_refresh_interval = 0.05

        client._running = True
        refresh_task = asyncio.create_task(client._token_refresh_loop())
        await asyncio.sleep(0.1)

        # Cancel the task
        refresh_task.cancel()

        # The loop handles CancelledError internally, so it completes normally
        # We just verify it doesn't hang or crash
        try:
            await asyncio.wait_for(refresh_task, timeout=1.0)
        except asyncio.CancelledError:
            pass  # This is also acceptable

        # Task should be done
        assert refresh_task.done()

    @pytest.mark.asyncio
    async def test_stop_cancels_token_refresh(self, mock_streaming_client):
        """stop() should cancel the token refresh task."""
        client = mock_streaming_client
        client._token_refresh_interval = 0.1

        # Start a mock token refresh task
        client._running = True
        client._token_refresh_task = asyncio.create_task(client._token_refresh_loop())

        await asyncio.sleep(0.05)

        # Stop should cancel the task
        await client.stop()

        assert client._token_refresh_task is None
        assert not client._running


class TestTokenRefreshIntegration:
    """Integration tests for token refresh with streaming."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, mock_streaming_client):
        """Test full start/stop lifecycle with token refresh."""
        client = mock_streaming_client
        client._token_refresh_interval = 0.05

        # Mock websocket to exit after a short time
        call_count = 0

        async def mock_websocket(symbols):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                client._running = False
            await asyncio.sleep(0.1)
            raise Exception("Test disconnect")

        client.websocket_client = mock_websocket
        client._reconnect_enabled = True
        client._max_reconnect_attempts = 1

        # Run the client
        await client.run(["AAPL"])

        # Verify token refresh ran
        assert client.authenticator._read_token_file.call_count >= 1

    @pytest.mark.asyncio
    async def test_token_refresh_continues_through_reconnect(self, mock_streaming_client):
        """Token refresh should continue running during WebSocket reconnects."""
        client = mock_streaming_client
        client._token_refresh_interval = 0.02

        reconnect_count = 0

        async def mock_websocket(symbols):
            nonlocal reconnect_count
            reconnect_count += 1
            await asyncio.sleep(0.05)
            if reconnect_count >= 3:
                client._running = False
            raise Exception("Simulated disconnect")

        client.websocket_client = mock_websocket
        client._reconnect_enabled = True
        client._reconnect_delay = 0.01

        # Run the client
        await client.run(["AAPL"])

        # Token refresh should have run multiple times during reconnects
        assert client.authenticator._read_token_file.call_count >= 2
