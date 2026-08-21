"""Structured logging setup with conservative secret redaction."""

import logging
import re

SECRET = re.compile(r"(?i)(api[_-]?key|authorization|token)(\s*[:=]\s*)(\S+)")


class SecretRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = SECRET.sub(r"\1\2***", str(record.msg))
        return True


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(SecretRedactionFilter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
