#!/usr/bin/env python3.13
import csv
import dataclasses
import logging
import os
import re
import sys
from typing import Any, TextIO

from jmullan.cmd import cmd
from jmullan.logging import easy_logging

from jmullan.git_changelog import changelog

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


def open_file_or_stdout_for_writing(filename: str | None) -> TextIO:
    if filename is None or filename == "-":
        return sys.stdout
    else:
        return open(filename, "w", encoding="utf8")


@dataclasses.dataclass
class Author:
    original_email: str | None
    email: str | None
    original_name: str | None
    name: str | None
    original_username: str | None
    username: str | None

    @property
    def original_address(self):
        """
        A name-addr
        See: https://datatracker.ietf.org/doc/html/rfc2822#section-3.4"""
        return f"{self.original_name} <{self.original_email}>"

    @property
    def address(self):
        """
        A name-addr
        See: https://datatracker.ietf.org/doc/html/rfc2822#section-3.4"""
        return f"{self.name} <{self.email}>"

    @property
    def full(self):
        return f"{self.original_address}:{self.address}:{self.original_username}:{self.username}"


def get_username_from_email(email: str | None) -> str | None:
    if email is None:
        return None
    match = re.match("^([^@]+)@", email)
    if match:
        return match.group(1)
    return email


def load_mailmap() -> list[Author]:
    if not os.path.exists(".mailmap"):
        return []
    uniques: set[str] = set()
    authors: list[Author] = []
    with open(".mailmap") as fp:
        for line in fp.readlines():
            line = line.strip()
            match = re.match(r"(.*)<([^>]*)>\s*(.*)<([^>]*)>\s*", line)
            if match:
                name = match.group(1).strip()
                email = match.group(2).strip()
                original_name = match.group(3).strip()
                original_email = match.group(4).strip()

                username = get_username_from_email(email)
                original_username = get_username_from_email(original_email)

                author_fields: dict[str, str | None] = {
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
    from_sha: str | None = None,
    from_inclusive: bool | None = False,
    to_sha: str | None = None,
    to_inclusive: bool | None = False,
    files: list[str] | None = None,
) -> list[Author]:
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
    for line in changelog.git_log(
        git_format, from_sha, from_inclusive, to_sha, to_inclusive, reversed=True, files=files
    ):
        if line is None or len(line) < 1:
            continue
        parts = line.split(" ", 1)
        if len(parts) != 2:
            logger.debug("Weird log line %s", line)
            continue
        sha, line = parts
        if sha not in line:
            logger.debug("Weird log line %s", line)
            continue
        field_values = {}
        for kv in [x for x in line.strip().split(sha) if x and ":" in x]:
            k, v = kv.split(":", 1)
            field_values[k] = v
        unique = " ".join(field_values.values())
        if unique in uniques:
            continue
        uniques.add(unique)
        author_fields = {k: field_values.get(v) for k, v in fields.items()}
        author = Author(**author_fields)
        authors.append(author)
    return authors


def add_lower_if_not_none(to_dict: dict[str, Any], key: str | None, value: Any):
    if key is not None:
        to_dict[key.lower()] = value


def get_lower_if_not_none(from_dict: dict[str, Any], key: str | None) -> Any:
    if key is not None:
        return from_dict.get(key.lower())
    else:
        return None


def resolve_authors(authors: list[Author], mailmap_authors: list[Author]) -> list[Author]:
    """Given a list of authors and a list from the mailmap, produce a combined list that
    can be dumped to a .mailmap. This should produce at least one line per address, but
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
            add_lower_if_not_none(email_mappings, author.original_email, author)
            add_lower_if_not_none(email_mappings, author.email, author)
            add_lower_if_not_none(name_mappings, author.original_username, author)
            add_lower_if_not_none(name_mappings, author.username, author)
            add_lower_if_not_none(name_mappings, author.original_name, author)
            add_lower_if_not_none(name_mappings, author.name, author)
            add_lower_if_not_none(address_mappings, author.original_address, author)
            add_lower_if_not_none(address_mappings, author.address, author)
    combined_authors: list[Author] = []
    for source in [mailmap_authors, authors]:
        for from_author in source:
            to_author = None
            to_address = get_lower_if_not_none(address_mappings, from_author.address)
            if to_address and to_address.address != from_author.address:
                to_author = to_address
            to_email = get_lower_if_not_none(email_mappings, from_author.email)
            if to_email and not to_author and to_email.address != from_author.address:
                to_author = to_email
            to_name = get_lower_if_not_none(name_mappings, from_author.name)
            if to_name and not to_author and to_name.address != from_author.address:
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
    mapped_addresses: set[str] = set()
    for combined_author in combined_authors:
        if combined_author.address != combined_author.original_address:
            mapped_addresses.add(combined_author.original_address)

    finalized_authors: dict[str, Author] = {}
    for author in combined_authors:
        original_address = author.original_address
        if author.address != original_address or original_address not in mapped_addresses:
            finalized_authors[author.full] = author

    return list(sorted(finalized_authors.values(), key=lambda a: f"{a.name} {a.email}"))


def output_authors(
    from_sha: str | None = None,
    from_inclusive: bool | None = False,
    to_sha: str | None = None,
    to_inclusive: bool | None = False,
    output_format: str | None = DEFAULT_FORMAT,
    ordering: str | None = DEFAULT_ORDERING,
    reversed: bool | None = False,
    files: list[str] | None = None,
    output: str | None = None,
):
    if output_format is None:
        output_format = DEFAULT_FORMAT
    if ordering is None:
        ordering = DEFAULT_ORDERING
    authors = extract_authors(from_sha, from_inclusive, to_sha, to_inclusive, files)

    if ordering == "email":
        authors = sorted(authors, key=lambda author: (f"{author.email}", f"{author.name}"))
    elif ordering == "name":
        authors = sorted(authors, key=lambda author: (f"{author.email}", f"{author.email}"))
    if reversed:
        authors.reverse()
    seen_lines: set[str] = set()

    with open_file_or_stdout_for_writing(output) as output_handle:
        if output_format == "csv":
            writer = csv.writer(output_handle)
            for author in authors:
                csv_tuple = author.email, author.username, author.name
                line = ",".join(f"{x}" for x in csv_tuple if x is not None)
                if line not in seen_lines:
                    writer.writerow(csv_tuple)
                    seen_lines.add(line)
        elif output_format == "mailmap":
            mailmap_authors = load_mailmap()
            for author in resolve_authors(authors, mailmap_authors):
                line = f"{author.address} {author.original_address}"
                if line not in seen_lines:
                    print(f"{line}", file=output_handle)
                    seen_lines.add(line)
        elif output_format == "pyproject":
            print("authors = [", file=output_handle)
            lines = set(
                f'    {{name = "{author.name}", email = "{author.email}"}}' for author in authors
            )
            print(",\n".join(lines), file=output_handle)
            print("]", file=output_handle)
        else:
            for author in authors:
                line = f"{author.name} <{author.email}>"
                if line not in seen_lines:
                    print(f"{line}", file=output_handle)
                    seen_lines.add(line)


class AuthorsMain(cmd.Main):
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
        self.parser.add_argument(
            "--file", dest="files", action="append", default=[], required=False
        )
        self.parser.add_argument(
            "--output",
            dest="output",
            default="-",
            help="Write the output somewhere. Default: stdout",
        )

    def main(self):
        super().main()
        if self.args.verbose:
            easy_logging.easy_initialize_logging("DEBUG", sys.stderr)
        else:
            easy_logging.easy_initialize_logging(stream=sys.stderr)
        logger.debug(self.args)

        from_sha = self.args.after or self.args.since
        from_inclusive = from_sha is None or self.args.since is not None
        to_sha = self.args.until or self.args.through
        to_inclusive = to_sha is None or self.args.through is not None
        files = self.args.files or []
        output = self.args.output or None

        output_authors(
            from_sha,
            from_inclusive,
            to_sha,
            to_inclusive,
            self.args.format,
            self.args.ordering,
            self.args.reversed,
            files,
            output,
        )


def main():
    AuthorsMain().main()


if __name__ == "__main__":
    main()
