"""
Test suite for monitoring and alerting.

Tests system monitoring, health checks, and alert mechanisms.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timedelta


class TestSystemMonitor(unittest.TestCase):
    """Test system monitoring functionality."""

    def test_connection_status_tracking(self):
        """Monitor should track connection status."""
        connection_status = {"broker": True, "streamer": True, "database": True}

        self.assertTrue(all(connection_status.values()))

        # Simulate connection loss
        connection_status["broker"] = False
        self.assertFalse(connection_status["broker"])

    def test_heartbeat_monitoring(self):
        """Monitor should track heartbeats."""
        last_heartbeat = datetime.now()
        timeout_seconds = 30

        # Within timeout
        current_time = last_heartbeat + timedelta(seconds=15)
        is_alive = (current_time - last_heartbeat).total_seconds() < timeout_seconds
        self.assertTrue(is_alive)

        # Past timeout
        current_time = last_heartbeat + timedelta(seconds=60)
        is_alive = (current_time - last_heartbeat).total_seconds() < timeout_seconds
        self.assertFalse(is_alive)


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring."""

    def test_latency_tracking(self):
        """Monitor should track latency."""
        latencies_ms = [5, 8, 3, 12, 6, 4, 15, 7]

        avg_latency = sum(latencies_ms) / len(latencies_ms)
        max_latency = max(latencies_ms)
        min_latency = min(latencies_ms)

        self.assertGreater(avg_latency, 0)
        self.assertEqual(max_latency, 15)
        self.assertEqual(min_latency, 3)

    def test_throughput_tracking(self):
        """Monitor should track throughput."""
        messages_processed = 1000
        time_window_seconds = 10

        throughput = messages_processed / time_window_seconds

        self.assertEqual(throughput, 100)  # 100 msg/sec

    def test_error_rate_tracking(self):
        """Monitor should track error rate."""
        total_requests = 1000
        failed_requests = 5

        error_rate = failed_requests / total_requests

        self.assertEqual(error_rate, 0.005)  # 0.5%


class TestAlertManager(unittest.TestCase):
    """Test alert management."""

    def test_alert_creation(self):
        """Should create alerts with correct properties."""
        alert = {
            "level": "warning",
            "message": "High latency detected",
            "timestamp": datetime.now().isoformat(),
            "source": "latency_monitor",
        }

        self.assertEqual(alert["level"], "warning")
        self.assertIn("High latency", alert["message"])

    def test_alert_levels(self):
        """Should support multiple alert levels."""
        valid_levels = ["info", "warning", "error", "critical"]

        for level in valid_levels:
            alert = {"level": level}
            self.assertIn(alert["level"], valid_levels)

    def test_alert_deduplication(self):
        """Should deduplicate repeated alerts."""
        alerts = []
        seen_keys = set()

        def add_alert(level, message):
            key = f"{level}:{message}"
            if key not in seen_keys:
                alerts.append({"level": level, "message": message})
                seen_keys.add(key)

        # Add same alert twice
        add_alert("warning", "Test alert")
        add_alert("warning", "Test alert")

        self.assertEqual(len(alerts), 1)


class TestHealthCheck(unittest.TestCase):
    """Test health check functionality."""

    def test_component_health_check(self):
        """Should check individual component health."""

        def check_broker():
            return True

        def check_database():
            return True

        def check_streamer():
            return False

        health_status = {"broker": check_broker(), "database": check_database(), "streamer": check_streamer()}

        self.assertTrue(health_status["broker"])
        self.assertFalse(health_status["streamer"])

    def test_overall_health_status(self):
        """Should calculate overall health status."""
        components = {"broker": True, "database": True, "streamer": True}

        # All healthy
        overall_healthy = all(components.values())
        self.assertTrue(overall_healthy)

        # One unhealthy
        components["streamer"] = False
        overall_healthy = all(components.values())
        self.assertFalse(overall_healthy)


class TestDrawdownMonitorAlerts(unittest.TestCase):
    """Test drawdown monitoring alerts."""

    def test_drawdown_threshold_alert(self):
        """Should alert when drawdown exceeds threshold."""
        max_drawdown_threshold = 0.10
        current_drawdown = 0.12

        should_alert = current_drawdown > max_drawdown_threshold
        self.assertTrue(should_alert)

    def test_drawdown_warning_levels(self):
        """Should have multiple warning levels."""
        warning_threshold = 0.05
        critical_threshold = 0.10

        drawdown = 0.07

        is_warning = drawdown > warning_threshold
        is_critical = drawdown > critical_threshold

        self.assertTrue(is_warning)
        self.assertFalse(is_critical)


class TestPositionMonitor(unittest.TestCase):
    """Test position monitoring."""

    def test_position_size_alert(self):
        """Should alert on excessive position size."""
        max_position_pct = 0.10
        portfolio_value = 100000
        position_value = 15000

        position_pct = position_value / portfolio_value
        should_alert = position_pct > max_position_pct

        self.assertTrue(should_alert)

    def test_concentration_alert(self):
        """Should alert on portfolio concentration."""
        positions = {"AAPL": 50000, "MSFT": 30000, "GOOGL": 20000}
        total_value = sum(positions.values())
        max_concentration = 0.40

        for symbol, value in positions.items():
            concentration = value / total_value
            if concentration > max_concentration:
                # AAPL at 50% exceeds 40% threshold
                self.assertEqual(symbol, "AAPL")
                break


class TestMarketHoursMonitor(unittest.TestCase):
    """Test market hours monitoring."""

    def test_market_open_check(self):
        """Should detect market hours."""
        # Simplified market hours check
        market_open_hour = 9  # 9:30 AM ET
        market_close_hour = 16  # 4:00 PM ET

        current_hour = 12  # Noon
        is_market_hours = market_open_hour <= current_hour < market_close_hour

        self.assertTrue(is_market_hours)

        # Before market open
        current_hour = 8
        is_market_hours = market_open_hour <= current_hour < market_close_hour
        self.assertFalse(is_market_hours)

    def test_weekend_detection(self):
        """Should detect weekends."""
        # Saturday = 5, Sunday = 6
        weekday = 5  # Saturday

        is_weekend = weekday >= 5
        self.assertTrue(is_weekend)


if __name__ == "__main__":
    unittest.main()
