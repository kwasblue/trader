import os
import shutil

PROJECT_ROOT = r"C:\Users\kwasi\OneDrive\Documents\Personal Projects\schwab_trader"

# Map original filenames to new destination paths (relative to PROJECT_ROOT)
file_map = {
    "aggregate.py": "data/",
    "authenticator.py": "data/streaming/",
    "back_test.py": "core/backtester.py",
    "configloader.py": "utils/",
    "datapipeline.py": "data/",
    "datastorage.py": "data/",
    "datautils.py": "data/",
    "eventhandler.py": "core/",
    "framemanager.py": "core/",
    "logger.py": "utils/",
    "monitor.py": "monitoring/",
    "position_sizer.py": "core/",
    "writer.py": "data/output/",
    "risk.py": "utils/risk_metrics.py",
    "indicators.py": "archive/indicators.py",
    "strategy.py": "archive/strategy.py",
    "patern.py": "strategies/patterns/",
    "schwab_client.py": "data/streaming/",
    "streamer.py": "data/streaming/",
}

# Step 1: Scan all files in project tree
discovered_files = {}
for root, _, files in os.walk(PROJECT_ROOT):
    for file in files:
        if file.endswith(".py") and not file.startswith("__"):
            discovered_files[file] = os.path.join(root, file)

# Step 2: Move files based on mapping
for filename, target in file_map.items():
    if filename not in discovered_files:
        print(f"⚠️  {filename} not found in any subdirectory. Skipping.")
        continue

    src = discovered_files[filename]

    # If target includes .py, it's a rename
    if target.endswith(".py"):
        dst = os.path.join(PROJECT_ROOT, target)
        dst_dir = os.path.dirname(dst)
    else:
        dst_dir = os.path.join(PROJECT_ROOT, target)
        dst = os.path.join(dst_dir, filename)

    os.makedirs(dst_dir, exist_ok=True)

    print(f"✅ Moving: {src} → {dst}")
    shutil.move(src, dst)
