#!/usr/bin/env python
"""
Database Schema Migration Tool

Fixes the stock_table schema to include the timeframe column.
This is needed after the multi-timeframe update.

Usage:
    python tools/migrate_database_schema.py
    python tools/migrate_database_schema.py --backup  # Creates backup first
"""

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def migrate_stock_table(db_path: str, create_backup: bool = True):
    """
    Migrate stock_table to include timeframe column.

    Args:
        db_path: Path to stock_base.db
        create_backup: Whether to create a backup first
    """
    db_path = Path(db_path)

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("\nNo migration needed - database will be created with correct schema on first use.")
        return True

    print(f"Migrating database: {db_path}")

    # Create backup if requested
    if create_backup:
        backup_path = db_path.parent / f"{db_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        print(f"\n📦 Creating backup: {backup_path}")
        shutil.copy2(db_path, backup_path)
        print("✓ Backup created")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if stock_table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_table'")
        if not cursor.fetchone():
            print("\n✓ stock_table doesn't exist yet - no migration needed")
            print("  (Table will be created with correct schema on first use)")
            return True

        # Check current columns
        cursor.execute("PRAGMA table_info(stock_table)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}

        print(f"\nCurrent columns in stock_table: {list(columns.keys())}")

        # Check if timeframe column exists
        if "timeframe" in columns:
            print("\n✓ timeframe column already exists - no migration needed")
            return True

        print("\n🔧 Adding timeframe column...")

        # Add timeframe column with default value
        cursor.execute("ALTER TABLE stock_table ADD COLUMN timeframe TEXT DEFAULT 'day'")

        conn.commit()
        print("✓ timeframe column added successfully")

        # Update existing rows to have 'day' as timeframe (legacy data)
        cursor.execute("UPDATE stock_table SET timeframe = 'day' WHERE timeframe IS NULL")
        updated_rows = cursor.rowcount
        conn.commit()

        print(f"✓ Updated {updated_rows} existing rows with default timeframe='day'")

        # Verify the change
        cursor.execute("PRAGMA table_info(stock_table)")
        new_columns = [row[1] for row in cursor.fetchall()]

        print(f"\nNew columns in stock_table: {new_columns}")

        # Check if we need to recreate the unique index
        print("\n🔧 Updating unique index to include timeframe...")

        # Drop old index if it exists
        cursor.execute("DROP INDEX IF EXISTS idx_stock_table_symbol_Date_timeframe")

        # Create new unique index including timeframe
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_stock_table_symbol_date_timeframe
            ON stock_table (symbol, date, timeframe)
        """)

        conn.commit()
        print("✓ Unique index updated")

        print("\n" + "=" * 60)
        print("✅ MIGRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nThe database is now ready for multi-timeframe data.")

        return True

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        conn.rollback()

        if create_backup:
            print(f"\n📦 You can restore from backup: {backup_path}")

        import traceback

        traceback.print_exc()
        return False

    finally:
        conn.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate database schema to support timeframe column",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate with automatic backup
  python tools/migrate_database_schema.py --backup

  # Migrate without backup
  python tools/migrate_database_schema.py --no-backup

  # Custom database path
  python tools/migrate_database_schema.py --db /path/to/stock_base.db
        """,
    )

    parser.add_argument(
        "--db", default="data/stock_base.db", help="Path to database file (default: data/stock_base.db)"
    )
    parser.add_argument(
        "--backup", action="store_true", default=True, help="Create backup before migration (default: True)"
    )
    parser.add_argument("--no-backup", dest="backup", action="store_false", help="Skip backup creation")

    args = parser.parse_args()

    # Resolve database path relative to project root
    db_path = ROOT / args.db

    print("=" * 60)
    print("DATABASE SCHEMA MIGRATION")
    print("=" * 60)

    success = migrate_stock_table(str(db_path), create_backup=args.backup)

    if success:
        print("\n✅ You can now run the optimizer or trader:")
        print("  python tools/optimize_all_symbols.py --save")
        print("  amsterdam start")
        return 0
    else:
        print("\n❌ Migration failed - check errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
