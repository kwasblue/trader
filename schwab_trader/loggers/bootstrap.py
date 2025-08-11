import logging, os
from logging.handlers import RotatingFileHandler
from loggers.logger import UniqueMessageFilter

def init_root_logger(
    log_dir: str = "logs",
    root_file: str = "app.log",
    level: int = logging.INFO,
    console: bool = True,
    max_bytes: int = 10_000_000,
    backup_count: int = 10,
):
    """
    Initialize a single root logger that everything can propagate to.
    """
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)

    # avoid duplicate handlers on hot-reloads
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")

        fh = RotatingFileHandler(os.path.join(log_dir, root_file),
                                 maxBytes=max_bytes, backupCount=backup_count)
        fh.setLevel(level)
        fh.addFilter(UniqueMessageFilter())
        fh.setFormatter(fmt)
        root.addHandler(fh)

        if console:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(fmt)
            root.addHandler(ch)
