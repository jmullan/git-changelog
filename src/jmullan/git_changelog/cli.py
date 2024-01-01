#!/usr/bin/env python3.11
import datetime
import io
import json
import logging
import os
import sys
import traceback
from argparse import ArgumentParser
from signal import signal, SIGPIPE
from typing import Any, Dict, Mapping, Optional

from jmullan.git_changelog.changelog import print_changelog

logger = logging.getLogger(__name__)


def flatten_dict(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Adds dots to all nested fields in dictionaries.

    Entries with different forms of nesting update. (ie {"a": {"b": 1}, "a.b": 2})
    """
    top_level = {}
    for key, val in value.items():
        if not isinstance(val, Mapping):
            top_level[key] = val
        else:
            val = flatten_dict(val)
            for vkey, vval in val.items():
                vkey = f"{key}.{vkey}"
                top_level[vkey] = vval
    return top_level


# inspired by https://github.com/madzak/python-json-logger/blob/master/src/pythonjsonlogger/jsonlogger.py
#
# base list from https://docs.python.org/3/library/logging.html#logrecord-attributes
RECORD_MAPPINGS = {
    "args": "",
    "asctime": "",
    "created": "",
    "exc_info": "",
    "exc_text": "",
    "filename": "",
    "funcName": "",
    "levelname": "log.level",
    "levelno": "",
    "lineno": "",
    "module": "",
    "msecs": "",
    "message": "",
    "msg": "",
    "name": "log.logger",
    "pathname": "",
    "process": "",
    "processName": "",
    "relativeCreated": "",
    "stack_info": "",
    "thread": "",
    "threadName": "",
}


class EasyLoggingFormatter(logging.Formatter):
    def iso_date(self, record):
        iso_minus_timezone = datetime.datetime.utcfromtimestamp(record.created).isoformat()
        return "%sZ" % iso_minus_timezone

    def traceback(self, exception_info):
        """Format and return the specified exception information as a string.

        This default implementation just uses
        traceback.print_exception()
        """
        sio = io.StringIO()
        tb = exception_info[2]
        traceback.print_tb(tb, file=sio)
        # traceback.print_exception(ei[0], ei[1], tb, None, sio)
        s = sio.getvalue()
        sio.close()
        s = s.lstrip("\n")
        return s

    def get_event(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Prepares a flattened dictionary from a LogRecord that includes the basic ECS fields.

        Users of this library are expected to be hygienic about their use of field names.
        """
        event = {"@timestamp": self.iso_date(record), "message": record.getMessage()}

        for from_key, value in record.__dict__.items():
            if from_key in RECORD_MAPPINGS:
                to_key = RECORD_MAPPINGS[from_key]
                if to_key:
                    event[to_key] = value
            else:
                event[from_key] = value

        extra: dict = {}
        if hasattr(record, "extra"):
            extra = record.extra or {}  # type: ignore
        event.update(extra)

        if record.exc_info:
            exception = record.exc_info[1]
            event["error.type"] = type(exception).__name__
            event["error.message"] = str(exception)
            event["error.stack_trace"] = self.traceback(record.exc_info)
        return flatten_dict(event)


class ConsoleFormatter(EasyLoggingFormatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629."""

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    green = "\x1b[38;32m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    COLORS = {
        logging.DEBUG: grey,
        # logging.INFO: blue,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.FATAL: bold_red,
        logging.CRITICAL: bold_red,
    }

    def format_extra(self, value: Any, color: Optional[str] = None):
        if not isinstance(value, str):
            try:
                value = json.dumps(value)
            except Exception:
                value = str(value)
        return self.colorize(value, color)

    def colorize(self, value: Any, color: Optional[str] = None):
        if color is None:
            return value
        return f"{color}{value}{self.reset}"

    def format_field(self, key, value):
        k = self.format_extra(key, self.green)
        v = self.format_extra(value)
        return f"{k}={v}"

    def formatMessage(self, record: logging.LogRecord) -> str:
        event = self.get_event(record)
        color = self.COLORS.get(record.levelno)

        event = flatten_dict(event)

        timestamp = event.pop("@timestamp")
        message = event.pop("message")

        level = self.colorize(event.pop("log.level"), color)
        message = self.colorize(message, color)

        # this method is just formatting the "message". LogFormatter will supply the
        # error message and traceback
        suppress_fields = {"error.type", "error.message", "error.stack_trace"}
        for field in suppress_fields:
            event.pop(field, None)

        extra_pairs = [self.format_field(k, v) for k, v in event.items()]
        if extra_pairs:
            pairs_string = " ".join(extra_pairs)
            message = f"{message} | {pairs_string}"
        line = f"[{timestamp}] [{level}] {message}{self.reset}"

        return line


def easy_initialize_logging(log_level: Optional[str] = None):
    """A very simple way to configure logging, suitable for toy applications."""
    if log_level is None:
        log_level = os.environ.get("LOGLEVEL", "INFO").upper() or "INFO"

    logging.captureWarnings(True)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ConsoleFormatter())

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    root_logger.setLevel(log_level)

def my_except_hook(exctype, value, traceback):
    if exctype == BrokenPipeError:
        pass
    else:
        sys.__excepthook__(exctype, value, traceback)



class Jmullan:
    GO = True


def handle_signal(signum, frame):
    if SIGPIPE == signum:
        sys.stdout.close()
        sys.stderr.close()
        exit(0)
    if not Jmullan.GO:
        exit(0)
    Jmullan.GO = False


def stop_on_broken_pipe_error():
    sys.excepthook = my_except_hook
    signal(SIGPIPE, handle_signal)


def main():
    """Turn a git log into a changelog"""
    stop_on_broken_pipe_error()
    parser = ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="verbose is more verbose",
    )
    parser.add_argument(
        "--after",
        dest="after",
        default=None,
        help="start with this commit, but do not include it",
    )
    parser.add_argument(
        "--since",
        dest="since",
        default=None,
        help="start with this commit, and include it",
    )
    parser.add_argument(
        "--through",
        dest="through",
        default=None,
        help="up to and including this commit",
    )
    parser.add_argument(
        "--until",
        dest="until",
        default=None,
        help="up to but not including this comit",
    )
    parser.add_argument(
        "-t",
        "--tags",
        dest="tags",
        action="store_true",
        default=False,
        help="Use tags and other found versions",
    )
    parser.add_argument("version", default="Current", nargs="?")
    args = parser.parse_args()
    if args.verbose:
        easy_initialize_logging("DEBUG")
    else:
        easy_initialize_logging()

    from_sha = args.since or args.after
    to_sha = args.through or args.until
    logger.debug(args)
    print_changelog(from_sha, to_sha, args.version, args.tags)

if __name__ == "__main__":
    main()
