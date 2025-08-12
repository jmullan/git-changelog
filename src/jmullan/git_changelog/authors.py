#!/usr/bin/env python3.13
"""Functions and cli command to get author information from a git repo."""

import csv
import dataclasses
import logging
import pathlib
import re
import sys
from typing import TextIO, TypeVar

from jmullan.git_changelog import changelog
from jmullan.git_changelog.models import Direction, Inclusiveness, ShaRange
from jmullan.logging import easy_logging

from jmullan.cmd import cmd

logger = logging.getLogger(__name__)

formats = {
    "simple": "{name or username} <{email}>",
    "csv": "{email},{username},{name}",
    "mailmap": "correct name <email> wrong name <email>",
    "pyproject": "The authors stanza of a pyproject.toml",
}
FORMAT_DOC = "\n  ".join(f"{x}: {y}" for x, y in formats.items())
DEFAULT_FORMAT = "simple"
orderings = ["appearance", "email", "name"]
DEFAULT_ORDERING = "appearance"

T = TypeVar("T")


def open_file_or_stdout_for_writing(filename: str | None) -> TextIO:
    """Return a file handle to a file or to stdout."""
    if filename is None or filename == "-":
        return sys.stdout
    return pathlib.Path(filename).open("w", encoding="utf8")


@dataclasses.dataclass
class Author:
    """An author in the repo, possibly with an updated email and name."""

    original_email: str
    email: str
    original_name: str
    name: str
    original_username: str
    username: str

    @property
    def original_address(self) -> str:
        """Get the name-addr without mailmap replacement.

        See: https://datatracker.ietf.org/doc/html/rfc2822#section-3.4
        """
        return f"{self.original_name} <{self.original_email}>"

    @property
    def address(self) -> str:
        """Get the name-addr with any mailmap replacement.

        See: https://datatracker.ietf.org/doc/html/rfc2822#section-3.4
        """
        return f"{self.name} <{self.email}>"

    @property
    def full(self) -> str:
        """Get the full mapping from original to replaced."""
        return f"{self.original_address}:{self.address}:{self.original_username}:{self.username}"


def get_username_from_email(email: str) -> str:
    """Extract just the username from an email."""
    if email is None:
        return ""
    match = re.match("^([^@]+)@", email)
    if match:
        return match.group(1)
    return email


def load_mailmap() -> list[Author]:
    """Load the mailmap into a list of Authors."""
    mail_map_path = pathlib.Path(".mailmap")
    if not mail_map_path.exists():
        return []
    uniques: set[str] = set()
    authors: list[Author] = []
    with mail_map_path.open() as fp:
        for line in fp:
            match = re.match(r"(.*)<([^>]*)>\s*(.*)<([^>]*)>\s*", line.strip())
            if match:
                name = match.group(1).strip()
                email = match.group(2).strip()
                original_name = match.group(3).strip()
                original_email = match.group(4).strip()

                username = get_username_from_email(email)
                original_username = get_username_from_email(original_email)

                author_fields: dict[str, str] = {
                    "original_email": original_email,
                    "email": email,
                    "original_username": original_username,
                    "username": username,
                    "original_name": original_name,
                    "name": name,
                }
                values = [v.strip() for v in author_fields.values() if v is not None]
                unique = " ".join(values)
                if len(unique) == 0 or unique in uniques:
                    continue
                uniques.add(unique)
                author = Author(**author_fields)
                authors.append(author)
    return authors


def extract_authors(
    sha_range: ShaRange,
    files: list[str] | None,
) -> list[Author]:
    """Build a list of authors from git logs."""
    fields = {
        "original_email": "ae",
        "email": "aE",
        "original_username": "al",
        "username": "aL",
        "original_name": "an",
        "name": "aN",
    }
    fields_format = "%h".join(f"{x}:%{x}" for x in fields.values())
    git_format = f"%h {fields_format}"
    uniques: set[str] = set()
    authors: list[Author] = []
    for line in changelog.git_log(git_format, sha_range, Direction.REVERSE, files):
        if line is None or len(line) < 1:
            continue
        parts = line.split(" ", 1)
        if len(parts) != 2:  # noqa: PLR2004
            logger.debug("Weird log line %s", line)
            continue
        sha, remainder = parts
        if sha not in remainder:
            logger.debug("Weird log line %s", line)
            continue
        field_values = {}
        for kv in [x for x in remainder.strip().split(sha) if x and ":" in x]:
            k, v = kv.split(":", 1)
            field_values[k] = v
        unique = " ".join(field_values.values())
        if unique in uniques:
            continue
        uniques.add(unique)
        author_fields = {k: field_values.get(v) or "" for k, v in fields.items()}
        author = Author(**author_fields)
        authors.append(author)

    return authors


def extract_coauthors(sha_range: ShaRange, files: list[str] | None) -> list[Author]:
    """Get coauthors from git logs.

    See: https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/creating-a-commit-with-multiple-authors

    """
    authors = []
    git_format = "%(body)"
    for line in changelog.git_log(git_format, sha_range, Direction.REVERSE, files):
        stripped = line.strip()
        if not stripped.startswith("Co-authored-by:"):
            continue
        co_author = stripped.removeprefix("Co-authored-by:").strip()
        match = re.match(r"(.*)\s*<([^>]+@[^>]+)>", co_author)
        if not match:
            continue
        name = match.group(1).strip()
        email = match.group(2).strip()
        if len(name) and len(email):
            author = Author(
                original_email=email,
                email=email,
                original_name=name,
                name=name,
                original_username=get_username_from_email(email),
                username=get_username_from_email(email),
            )
            authors.append(author)
    return authors


def add_by_lower_key_if_key_is_not_none(to_dict: dict[str, T], key: str | None, value: T) -> None:
    """Add the value to a dictionary if the key is not None."""
    if key is not None:
        key = key.lower().strip()
        if key not in to_dict:
            to_dict[key] = value


def get_by_lower_key_if_key_is_not_none[V](from_dict: dict[str, V], key: str | None) -> V | None:
    """Get a value from a dictionary if the key is not None."""
    if key is not None:
        return from_dict.get(key.lower().strip())
    return None


def resolve_authors(authors: list[Author], mailmap_authors: list[Author]) -> list[Author]:
    """Produce a list of authors for a dot-mail-map.

    Given a list of authors and a list from the mailmap, produce a combined list that
    can be dumped to a .mailmap.

    This should produce at least one line per address, but
    will not produce a line mapping an address to itself if there is already a mapping
    to that address from another address.

    This is just to produce a single list for mailmap purposes -- git already maps items
    from the log.
    """
    email_mappings: dict[str, Author] = {}
    name_mappings: dict[str, Author] = {}
    address_mappings: dict[str, Author] = {}
    for source in [mailmap_authors, authors]:
        for author in source:
            add_by_lower_key_if_key_is_not_none(email_mappings, author.original_email, author)
            add_by_lower_key_if_key_is_not_none(email_mappings, author.email, author)
            add_by_lower_key_if_key_is_not_none(name_mappings, author.original_username, author)
            add_by_lower_key_if_key_is_not_none(name_mappings, author.username, author)
            add_by_lower_key_if_key_is_not_none(name_mappings, author.original_name, author)
            add_by_lower_key_if_key_is_not_none(name_mappings, author.name, author)
            add_by_lower_key_if_key_is_not_none(address_mappings, author.original_address, author)
            add_by_lower_key_if_key_is_not_none(address_mappings, author.address, author)
    combined_authors: list[Author] = []
    for source in [mailmap_authors, authors]:
        for from_author in source:
            to_author = None
            to_address = get_by_lower_key_if_key_is_not_none(address_mappings, from_author.original_address)
            if to_address and to_address.address != from_author.address:
                to_author = to_address
            to_email = get_by_lower_key_if_key_is_not_none(email_mappings, from_author.original_email)
            if not to_author and to_email and to_email.address != from_author.address:
                to_author = to_email
            to_name = get_by_lower_key_if_key_is_not_none(name_mappings, from_author.original_name)
            if not to_author and to_name and to_name.address != from_author.address:
                to_author = to_name
            if not to_author:
                to_author = from_author
            new_author = Author(
                original_email=from_author.original_email,
                email=to_author.email,
                original_name=from_author.original_name,
                name=to_author.name,
                original_username=from_author.original_username,
                username=to_author.username,
            )
            combined_authors.append(new_author)
    finalized_authors: list[Author] = finalize_authors(combined_authors)
    return sorted(finalized_authors, key=lambda a: f"{a.name} {a.email}")


def finalize_authors(combined_authors: list[Author]) -> list[Author]:
    """Remove and resolve duplicate authors."""
    changed_addresses: set[str] = set()
    replacement_addresses: set[str] = set()
    for combined_author in combined_authors:
        if combined_author.address != combined_author.original_address:
            changed_addresses.add(combined_author.original_address)
            replacement_addresses.add(combined_author.address)
    finalized_authors: dict[str, Author] = {}
    for author in combined_authors:
        new_address = author.address
        original_address = author.original_address
        if new_address != original_address or new_address not in replacement_addresses:
            finalized_authors[original_address] = author
    return list(finalized_authors.values())


def output_csv(authors: list[Author], output_handle: TextIO) -> None:
    """Write authors to a CSV file or stdout."""
    seen_lines: set[str] = set()
    writer = csv.writer(output_handle)
    for author in authors:
        csv_tuple = author.email, author.username, author.name
        line = ",".join(f"{x}" for x in csv_tuple if x is not None)
        if line not in seen_lines:
            writer.writerow(csv_tuple)
            seen_lines.add(line)


def output_mailmap(authors: list[Author], output_handle: TextIO) -> None:
    """Write authors to a mail map file or stdout."""
    seen_lines: set[str] = set()
    mailmap_authors = load_mailmap()
    for author in resolve_authors(authors, mailmap_authors):
        line = f"{author.address} {author.original_address}"
        if line not in seen_lines:
            print(f"{line}", file=output_handle)
            seen_lines.add(line)


def output_pyproject(authors: list[Author], output_handle: TextIO) -> None:
    """Write authors as a pyproject author list in toml format to a file or stdout."""
    print("authors = [", file=output_handle)
    lines = {f'    {{name = "{author.name}", email = "{author.email}"}}' for author in authors}
    print(",\n".join(lines), file=output_handle)
    print("]", file=output_handle)


def output_list(authors: list[Author], output_handle: TextIO) -> None:
    """Write authors to a plain list of emails."""
    seen_lines: set[str] = set()
    for author in authors:
        line = f"{author.name} <{author.email}>"
        if line not in seen_lines:
            print(f"{line}", file=output_handle)
            seen_lines.add(line)


def output_authors(
    authors: list[Author],
    output_format: str | None,
    ordering: str | None,
    direction: Direction,
    output: str | None = None,
) -> None:
    """Print the authors in your chosen format."""
    if output_format is None:
        output_format = DEFAULT_FORMAT
    if ordering is None:
        ordering = DEFAULT_ORDERING

    if ordering == "email":
        authors = sorted(authors, key=lambda a: (f"{a.email}", f"{a.name}"))
    elif ordering == "name":
        authors = sorted(authors, key=lambda a: (f"{a.email}", f"{a.email}"))
    if direction == Direction.REVERSE:
        authors.reverse()

    with open_file_or_stdout_for_writing(output) as output_handle:
        if output_format == "csv":
            output_csv(authors, output_handle)
        elif output_format == "mailmap":
            output_mailmap(authors, output_handle)
        elif output_format == "pyproject":
            output_pyproject(authors, output_handle)
        else:
            output_list(authors, output_handle)


class AuthorsMain(cmd.Main):
    """Print out authors in various formats."""

    def __init__(self):
        super().__init__()
        self.parser.add_argument(
            "--after",
            dest="after",
            default=None,
            help="start with this commit, but do not include it",
        )
        self.parser.add_argument(
            "--since",
            dest="since",
            default=None,
            help="start with this commit, and include it",
        )
        self.parser.add_argument(
            "--through",
            dest="through",
            default=None,
            help="up to and including this commit",
        )
        self.parser.add_argument(
            "--until",
            dest="until",
            default=None,
            help="up to but not including this commit",
        )
        self.parser.add_argument(
            "--format",
            dest="format",
            default=DEFAULT_FORMAT,
            choices=list(formats.keys()),
            help=f"How to display the authors.\n{FORMAT_DOC}",
        )
        self.parser.add_argument(
            "--order-by",
            dest="ordering",
            default=DEFAULT_ORDERING,
            choices=orderings,
            help="What order to show the authors.",
        )
        self.parser.add_argument(
            "--reversed",
            dest="reversed",
            action="store_true",
            default=False,
            help="Reverse the order",
        )
        self.parser.add_argument("--file", dest="files", action="append", default=[], required=False)
        self.parser.add_argument(
            "--output",
            dest="output",
            default="-",
            help="Write the output somewhere. Default: stdout",
        )

    def main(self) -> None:
        """Print the authors of this repo."""
        super().main()
        if self.args.verbose:
            easy_logging.easy_initialize_logging("DEBUG", sys.stderr)
        else:
            easy_logging.easy_initialize_logging(stream=sys.stderr)
        logger.debug(self.args)

        from_sha = self.args.after or self.args.since
        from_inclusive = Inclusiveness.if_true(from_sha is None or self.args.since is not None)
        to_sha = self.args.until or self.args.through
        to_inclusive = Inclusiveness.if_true(to_sha is None or self.args.through is not None)
        files = self.args.files or []
        output = self.args.output or None

        sha_range = ShaRange(from_sha, from_inclusive, to_sha, to_inclusive)
        authors = extract_authors(sha_range, files)
        output_authors(
            authors,
            self.args.format,
            self.args.ordering,
            self.args.reversed,
            output,
        )


def main() -> None:
    """Run the command."""
    AuthorsMain().main()


if __name__ == "__main__":
    main()
