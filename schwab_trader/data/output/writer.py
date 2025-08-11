import json
import os
import csv
from schwab_trader.logger.logger import Logger
import pandas as pd

class FileWriter:
    def __init__(self, log_file: str, logger_name: str, log_dir: str = 'logs'):
        self.logger = Logger(log_file, logger_name, log_dir).get_logger()

    def write_json(self, target_path: str, target_file: str, data):
        try:
            # Ensure data is a dictionary or JSON-serializable
            if not isinstance(data, (dict, list)):
                raise TypeError(f"Data must be a dictionary or list, but got {type(data)}")
            
            os.makedirs(target_path, exist_ok=True)

            # Debug: Log data and target file
            self.logger.debug(f"Writing data to {target_file}")

            # Write data to JSON file
            with open(os.path.join(target_path, target_file), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4, default=serialize_obj)

            #self.logger.info(f"JSON file written: {target_file}")
        except TypeError as te:
            self.logger.error(f"JSON serialization failed: {str(te)}")
        except Exception as e:
            self.logger.error(f"Failed to write JSON file: {str(e)}")


    def write_txt(self, target_path: str, target_file: str, data: str):
        try:
            os.makedirs(target_path, exist_ok=True)
            with open(os.path.join(target_path, f'{target_file}.txt'), 'w', encoding='utf-8') as f:
                f.write(data)
            self.logger.info(f"TXT file written: {target_file}.txt at {target_path}")
        except Exception as e:
            self.logger.error(f"Failed to write TXT file: {str(e)}")

    def modify_json(self, target_path: str, target_file: str, new_data: dict):
        try:
            os.makedirs(target_path, exist_ok=True)
            target_file_path = os.path.join(target_path, target_file)

            if os.path.exists(target_file_path):
                with open(target_file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

                for key, value in new_data.items():
                    if isinstance(value, list) and key in existing_data and isinstance(existing_data[key], list):
                        # Merge lists by unique datetime if list of dicts with datetime key
                        existing_items = {item["datetime"]: item for item in existing_data[key] if isinstance(item, dict) and "datetime" in item}
                        new_items = {item["datetime"]: item for item in value if isinstance(item, dict) and "datetime" in item}
                        merged_items = {**existing_items, **new_items}
                        existing_data[key] = sorted(merged_items.values(), key=lambda x: x["datetime"])
                    else:
                        # Shallow update for other keys
                        existing_data[key] = value

                with open(target_file_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=4)

                self.logger.info(f"Modified JSON file: {target_file_path}")
            else:
                with open(target_file_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=4)

                self.logger.info(f"New JSON file created: {target_file_path}")

        except Exception as e:
            self.logger.error(f"Failed to modify JSON file: {str(e)}")


    def write_file(self, target_path: str, target_file: str, data, file_format='json'):
        try:
            os.makedirs(target_path, exist_ok=True)
            file_path = os.path.join(target_path, f'{target_file}.{file_format}')

            with open(file_path, 'w', encoding='utf-8') as f:
                if file_format.lower() == 'json':
                    json.dump(data, f, ensure_ascii=False, indent=4)
                elif file_format.lower() == 'csv':
                    csv_writer = csv.writer(f)
                    for row in data:
                        csv_writer.writerow(row)
                else:
                    f.write(str(data))
            self.logger.info(f"{file_format.upper()} file written: {target_file}.{file_format} at {target_path}")
        except Exception as e:
            self.logger.error(f"Failed to write {file_format.upper()} file: {str(e)}")

    def find(self, name: str, path: str):
        for root, dirs, files in os.walk(path):
            if name in files:
                self.logger.info(f"File found: {name} at {root}")
                return os.path.join(root, name)
        self.logger.warning(f"File not found: {name}")
        return None

def serialize_obj(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()  # Convert Timestamp to string
    raise TypeError(f"Type {type(obj)} not serializable")
    
 

