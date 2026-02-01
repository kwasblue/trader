#!/usr/bin/env python
"""
Test Runner for Schwab Trader

Runs all tests with coverage reporting.

Usage:
    # Run all tests
    python run_tests.py

    # Run specific test file
    python run_tests.py tests/test_autotrader.py

    # Run with coverage
    python run_tests.py --coverage

    # Run with verbose output
    python run_tests.py -v

    # Run only unit tests (fast)
    python run_tests.py --unit

    # Run integration tests
    python run_tests.py --integration
"""

import subprocess
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "tests"


def run_tests(args):
    """Run pytest with specified options."""
    cmd = [sys.executable, "-m", "pytest"]

    # Add test path
    if args.test_file:
        cmd.append(args.test_file)
    else:
        cmd.append(str(TESTS_DIR))

    # Verbose output
    if args.verbose:
        cmd.append("-v")

    # Coverage
    if args.coverage:
        cmd.extend([
            "--cov=core",
            "--cov=data",
            "--cov=strategies",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_report"
        ])

    # Markers
    if args.unit:
        cmd.extend(["-m", "not integration"])
    elif args.integration:
        cmd.extend(["-m", "integration"])

    # Parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])

    # Show summary
    if args.summary:
        cmd.append("--tb=short")

    # Fail fast
    if args.fail_fast:
        cmd.append("-x")

    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Run tests for Schwab Trader',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'test_file',
        nargs='?',
        help='Specific test file to run'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run with coverage reporting'
    )
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run only unit tests'
    )
    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run only integration tests'
    )
    parser.add_argument(
        '-p', '--parallel',
        action='store_true',
        help='Run tests in parallel'
    )
    parser.add_argument(
        '-s', '--summary',
        action='store_true',
        help='Show short traceback summary'
    )
    parser.add_argument(
        '-x', '--fail-fast',
        action='store_true',
        help='Stop on first failure'
    )

    args = parser.parse_args()

    sys.exit(run_tests(args))


if __name__ == '__main__':
    main()
