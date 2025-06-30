"""Functions for manipulating text."""

import logging
import re
import textwrap
from typing import TypeGuard

logger = logging.getLogger(__name__)


def none_as_empty(string: str | None) -> str:
    """Turn that None into an empty string or leave it alone."""
    if string is None:
        return ""
    return string


def none_as_empty_stripped(string: str | None) -> str:
    """Turn Nones into empty strings, and strip other strings."""
    return none_as_empty(string).strip()


def some_string(string: str | None) -> TypeGuard[str]:
    """Determine if the string is None or blank."""
    if string is None:
        return False
    return len(string.strip()) > 0


def fill_text(text: str, width: int, indent: str, initial_indent: str | None = None) -> str:
    """Strip any extra indentation, then reindent and then wrap each line individually."""
    text = textwrap.dedent(text)
    text = "\n".join(x.rstrip() for x in text.split("\n"))
    text = text.strip("\n")
    text = re.sub(r"\n\n\n+", "\n", text)

    texts = text.splitlines()
    if not texts:
        logger.warning("No text to wrap in %s", text)
        return text
    if indent is not None and initial_indent is not None:
        first_line = texts.pop(0)
        first_line = textwrap.fill(first_line, width, initial_indent=initial_indent, subsequent_indent=indent)
        texts = [
            textwrap.fill(line, width, initial_indent=indent, subsequent_indent=indent) for line in texts
        ]  # Wrap each line
        return "\n".join([first_line, *texts])
    if initial_indent is not None:
        first_line = texts.pop(0)
        first_line = textwrap.fill(first_line, width, initial_indent=initial_indent)
        texts = [textwrap.fill(line, width) for line in texts]  # Wrap each line
        return "\n".join([first_line, *texts])
    if indent is not None:
        texts = [
            textwrap.fill(line, width, initial_indent=indent, subsequent_indent=indent) for line in texts
        ]  # Wrap each line
        return "\n".join(texts)
    return "\n".join(textwrap.fill(line, width) for line in texts)
