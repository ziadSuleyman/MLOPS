"""
Logging setup — replaces every print() in the project.

- Console: INFO and above
- File: DEBUG and above, with rotation
- Every prediction request is logged with: input hash, output, latency, model version
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

from src.config import logging_cfg


def setup_logging(name: str = "olist") -> logging.Logger:
    """Create and return a configured logger. Call once at startup."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        logging_cfg.fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
    )

    # ── Console handler (INFO+) ─────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, logging_cfg.level.upper(), logging.INFO))
    console.setFormatter(formatter)
    logger.addHandler(console)

    # ── File handler (DEBUG+, rotating) ─────────────────────────────────
    logging_cfg.file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        logging_cfg.file,
        maxBytes=logging_cfg.max_bytes,
        backupCount=logging_cfg.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Module-level logger — import this from anywhere
log = setup_logging()
