import logging
import logging.handlers
import os
import sys
import time

from typing import Tuple
from datetime import datetime


class Formatter(logging.Formatter):

    def __init__(self, format: str, indent: int) -> None:
        super().__init__(format)
        self.indent = indent

    def format(self, record: logging.LogRecord) -> str:
        ret = super().format(record)

        indent = record.indent if hasattr(record, "indent") else self.indent

        if indent == 0:
            return ret
        lines = ret.split("\n")
        if len(lines) == 1:
            return ret
        indent_size = indent \
            if indent is not None else \
            ret.index(record.message)

        return lines[0] + "\n" + "\n".join((" " * indent_size) + line for line in lines[1:])


def log_namer(default_name: str) -> str:
    base_filename = ".".join(default_name.split(".")[:-2])
    ext, date = default_name.split(".")[-2:]
    return f"{base_filename}.{date}.{ext}"


def float_to_time_stamp(value: float):
    time_stamp = datetime.fromtimestamp(value)
    return (
        str(time_stamp.year).rjust(4, '0') +
        str(time_stamp.month).rjust(2, '0') +
        str(time_stamp.day).rjust(2, '0') +
        '_' +
        str(time_stamp.hour).rjust(2, '0') +
        str(time_stamp.minute).rjust(2, '0') +
        str(time_stamp.second).rjust(2, '0'))


def get_logger(output_folder: dict) -> Tuple[logging.Logger, logging.StreamHandler]:
    logger = logging.getLogger("tfs")
    logger.setLevel(logging.DEBUG)
    log_formatter = Formatter("%(asctime)s [%(levelname)8s]: %(message)s", None)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    logger.addHandler(log_stream_handler)

    log_file_handler = logging.handlers.TimedRotatingFileHandler(
        os.path.join(
            output_folder,
            f"tfs_log_{float_to_time_stamp(time.time())}.txt"),
        when='D', interval=3, backupCount=10)
    log_file_handler.setFormatter(log_formatter)
    logger.addHandler(log_file_handler)

    return logger, log_stream_handler, log_file_handler
