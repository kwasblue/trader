import os
from .logger2 import Logger

def get_module_logger(module_name: str, file_key: str, level=None):
    """
    Returns a module-scoped logger.
    - Writes to logs/app.log by default (propagates to root).
    - If DEBUG_MODULES env contains `file_key`, also writes to logs/{file_key}.log.
      Example: export DEBUG_MODULES="broker,executor"
    """
    debug_modules = set(filter(None, (os.getenv("DEBUG_MODULES") or "").split(",")))
    use_file = file_key in debug_modules

    log_file = f"{file_key}.log" if use_file else "app.log"
    return Logger(log_file=log_file, logger_name=module_name, level=level or 20, propagate=True).get_logger()
