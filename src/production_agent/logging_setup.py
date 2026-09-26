"""
WHY THIS FILE EXISTS:
Your original script used print() for everything - including dumping full
embedding vectors to stdout (`print(f"vector : {vector}")`), which for even
a modest number of chunks is a wall of floats with no way to turn it off.

print() has no levels, no timestamps, and no way to silence "debug" detail
in production while keeping "info" and "error" visible. logging gives you
that for free, and is the standard expectation in any real service.

Call get_logger(__name__) at the top of each module instead of using
print() directly.
"""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers if called more than once
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)  # set to logging.DEBUG to see chunk-level detail
    return logger