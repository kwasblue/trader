import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class LogHandler(FileSystemEventHandler):
    def __init__(self, log_dir, on_error_callback=None, on_success_callback=None):
        self.log_dir = log_dir
        self.log_sizes = {file: os.path.getsize(os.path.join(log_dir, file)) for file in os.listdir(log_dir)}
        self.on_error_callback = on_error_callback  # Callback for errors
        self.on_success_callback = on_success_callback  # Callback for successes

    def on_modified(self, event):
        if event.is_directory:
            return

        log_file = event.src_path
        if log_file in self.log_sizes:
            new_size = os.path.getsize(log_file)
            if new_size > self.log_sizes[log_file]:
                try:
                    with open(log_file, 'r') as file:
                        file.seek(self.log_sizes[log_file])  # Read only new lines
                        new_lines = file.readlines()
                        for line in new_lines:
                            self.process_log_line(line.strip(), log_file)
                except Exception as e:
                    print(f"Error reading file {log_file}: {e}")
                self.log_sizes[log_file] = new_size
            elif new_size < self.log_sizes[log_file]:
                # Handle log truncation (e.g., log rotation)
                self.log_sizes[log_file] = new_size
        else:
            # If the log file is new, initialize its size
            self.log_sizes[log_file] = os.path.getsize(log_file)

    def process_log_line(self, line, log_file):
        if "ERROR" in line:
            print(f"Error detected in {log_file}: {line}")
            if self.on_error_callback:
                self.on_error_callback(line)
        elif "SUCCESS" in line or "INFO" in line:
            print(f"Positive update in {log_file}: {line}")
            if self.on_success_callback:
                self.on_success_callback(line)

    def on_created(self, event):
        if not event.is_directory:
            log_file = event.src_path
            self.log_sizes[log_file] = os.path.getsize(log_file)
            print(f"New log file created: {log_file}")

# Sample callback function for errors
def handle_error(error_message):
    print(f"System alert: Handling error - {error_message}")
    # You can send notifications or restart services here
    

# Sample callback function for successes
def handle_success(success_message):
    print(f"System status: All good - {success_message}")
    # You can log this status, send a heartbeat signal, or trigger any normal operation

def monitor_log_directory(log_dir):
    event_handler = LogHandler(log_dir, on_error_callback=handle_error, on_success_callback=handle_success)
    observer = Observer(timeout=0.5)
    observer.schedule(event_handler, log_dir, recursive=False)
    observer.start()
    print('Monitoring started...')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("Monitoring stopped.")
        
    observer.join()
    return 'End of Monitoring. Goodbye!'

if __name__ == "__main__":
    log_directory = r'C:\Users\kwasi\OneDrive\Documents\Personal Projects\schwab_trader\logs'
    print(monitor_log_directory(log_directory))
