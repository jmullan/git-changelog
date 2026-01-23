#!/usr/bin/env python3.13
"""Functions and cli command to get author information from a git repo."""

import csv
import dataclasses
import logging
import pathlib
import re
import sys
from typing import TextIO, TypeVar

import pyinputplus
from simple_term_menu import TerminalMenu

from jmullan.git_changelog import changelog
from jmullan.git_changelog.models import Direction, Inclusiveness, ManualResolution, ShaRange
from jmullan.logging import easy_logging

from jmullan.cmd import cmd

logger = logging.getLogger(__name__)

formats = {
    "simple": "{name or username} <{email}>",
    "csv": "{email},{username},{name}",
    "mailmap": "correct name <email> wrong name <email>",
    "pyproject": "The authors stanza of a pyproject.toml",
    "dump": "All the author information",
}
FORMAT_DOC = "\n  ".join(f"{x}: {y}" for x, y in formats.items())
DEFAULT_FORMAT = "simple"
orderings = ["appearance", "email", "name"]
DEFAULT_ORDERING = "appearance"
ADD_OPTION = "*** ADD ***"
IS_ADDRESS_REGEX = re.compile(r"^[^<]+ <[^@>]+@[^@>]+>$")
T = TypeVar("T")


def is_address(address: str | None) -> bool:
    """Check if a string matches the IS_ADDRESS_REGEX regex."""
    if address is None:
        return False
    return re.match(IS_ADDRESS_REGEX, address) is not None


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
        return f"[{self.username}] {self.address} [{self.original_username}] {self.original_address}"


@dataclasses.dataclass
class AuthorCluster:
    """A group of auther addresses that all belong together."""

    prime_author_address: str
    author_addresses: set[str]

    @property
    def lower_addresses(self) -> set[str]:
        """Get lowercase versions of addresses."""
        all_addresses = [self.prime_author_address, *self.author_addresses]
        return {x.lower() for x in all_addresses}

    def contains(self, address: str) -> bool:
        """Determine if an address is in the cluster."""
        return address.lower() in self.lower_addresses


def get_username_from_email(email: str) -> str:
    """Extract just the username from an email."""
    if email is None:
        return ""
    match = re.match("^([^@]+)@", email)
    if match:
        return match.group(1)
    return email


def load_mail_maps() -> list[Author]:
    """Load the mailmap into a list of Authors."""
    authors = []
    start = pathlib.Path.cwd()
    for directory in (start, *start.parents):
        candidate = directory / ".mailmap"
        if candidate.is_file():
            authors.extend(load_mail_map(candidate))
    return authors


def load_mail_map(mail_map_path: pathlib.Path) -> list[Author]:
    """Load one mailmap file as a list of Authors."""
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


def extract_committers(sha_range: ShaRange, files: list[str] | None) -> list[Author]:
    """Build a list of committers from git logs."""
    fields = {
        "original_email": "ce",
        "email": "cE",
        "original_username": "cl",
        "username": "cL",
        "original_name": "cn",
        "name": "cN",
    }
    return extract_people(sha_range, files, fields)


def extract_authors(sha_range: ShaRange, files: list[str] | None) -> list[Author]:
    """Build a list of authors from git logs."""
    fields = {
        "original_email": "ae",
        "email": "aE",
        "original_username": "al",
        "username": "aL",
        "original_name": "an",
        "name": "aN",
    }
    return extract_people(sha_range, files, fields)


def extract_people(sha_range: ShaRange, files: list[str] | None, fields: dict[str, str]) -> list[Author]:
    """Load Authors from a range of shas in a git repo."""
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


def update_by_lower_key_if_key_is_not_none(to_dict: dict[str, T], key: str | None, value: T) -> None:
    """Add the value to a dictionary if the key is not None."""
    if key is not None:
        key = key.lower().strip()
        to_dict[key] = value


def get_by_lower_key_if_key_is_not_none[V](from_dict: dict[str, V], key: str | None) -> V | None:
    """Get a value from a dictionary if the key is not None."""
    if key is not None:
        return from_dict.get(key.lower().strip())
    return None


def resolve_author(
    from_author: Author,
    email_mappings: dict[str, Author],
    name_mappings: dict[str, Author],
    address_mappings: dict[str, Author],
) -> Author:
    """Find the best author to map a given author to."""
    to_author = None
    to_address = get_by_lower_key_if_key_is_not_none(address_mappings, from_author.original_address)
    if to_address and to_address.address != from_author.address:
        to_author = to_address
    elif to_author is None:
        to_name = get_by_lower_key_if_key_is_not_none(name_mappings, from_author.original_name)
        if not to_author and to_name and to_name.address != from_author.address:
            to_author = to_name
    elif to_author is None:
        to_email = get_by_lower_key_if_key_is_not_none(email_mappings, from_author.original_email)
        if not to_author and to_email and to_email.address != from_author.address:
            to_author = to_email
    if to_author is None:
        to_author = from_author
    return to_author


def build_authors_by_address(*author_lists: list[Author] | None) -> dict[str, Author]:
    """Collect all authors into a dict."""
    authors_by_address: dict[str, Author] = {}
    if author_lists is not None:
        for source in author_lists:
            if source is not None:
                for author in source:
                    authors_by_address[author.original_address] = author
    return authors_by_address


def build_author_clusters(
    address_mappings: dict[str, Author],
    email_mappings: dict[str, Author],
    name_mappings: dict[str, Author],
    *author_lists: list[Author] | None,
) -> list[AuthorCluster]:
    """Group authors by common mappings."""
    author_clusters: list[AuthorCluster] = []
    if author_lists is not None:
        for source in author_lists:
            if source is None:
                continue
            for from_author in source:
                to_author = resolve_author(from_author, email_mappings, name_mappings, address_mappings)
                addresses = {
                    from_author.original_address,
                    from_author.address,
                    to_author.original_address,
                    to_author.address,
                }
                found_author_cluster = None
                for author_cluster in author_clusters:
                    if author_cluster.contains(to_author.address):
                        found_author_cluster = author_cluster
                        break
                if found_author_cluster is None:
                    found_author_cluster = AuthorCluster(to_author.address, addresses)
                    author_clusters.append(found_author_cluster)
                for address in addresses:
                    found_author_cluster.author_addresses.add(address)
    return author_clusters


def fis_cluster(author_cluster: AuthorCluster) -> bool:
    """Give the user the chance to choose a new preferred address for a cluster."""
    options = [*author_cluster.author_addresses, ADD_OPTION]
    menu_entry_index = options.index(author_cluster.prime_author_address)
    terminal_menu = TerminalMenu(options, menu_cursor="    > ", cursor_index=menu_entry_index, skip_empty_entries=True)
    menu_entry_index = terminal_menu.show()
    if menu_entry_index is None:
        logger.debug("No menu entry was chosen")
        return False
    if menu_entry_index < 0 or menu_entry_index > len(options):
        logger.warning("Invalid menu entry was chosen")
        return True
    selected = options[menu_entry_index]
    if selected == ADD_OPTION:
        allowed_regexes = [IS_ADDRESS_REGEX]
        bad_regexes = [re.escape(option) for option in options]
        new_option = pyinputplus.inputStr(allowRegexes=allowed_regexes, blockRegexes=bad_regexes)
        if new_option and is_address(new_option):
            author_cluster.author_addresses.add(new_option)
            author_cluster.prime_author_address = new_option
        else:
            logger.error("    X BAD ADDRESS: %s", new_option)
        return True
    author_cluster.prime_author_address = options[menu_entry_index]
    return False


def manually_resolve_authors(
    addresses_to_resolve: set[str],
    address_mappings: dict[str, Author],
    email_mappings: dict[str, Author],
    name_mappings: dict[str, Author],
    *author_lists: list[Author] | None,
) -> dict[str, Author]:
    """Let user choose the preferred address for each Author."""
    author_clusters = build_author_clusters(address_mappings, email_mappings, name_mappings, *author_lists)
    authors_by_address = build_authors_by_address(*author_lists)

    for author_cluster in sorted(author_clusters, key=lambda x: x.prime_author_address):
        if not addresses_to_resolve.intersection(author_cluster.lower_addresses):
            continue
        logger.info("-> %s", author_cluster.prime_author_address)
        fixing_cluster = True
        while fixing_cluster:
            fixing_cluster = fis_cluster(author_cluster)

        prime_author = None
        if author_cluster.prime_author_address:
            prime_author = authors_by_address.get(author_cluster.prime_author_address)

        for address in sorted(author_cluster.author_addresses):
            if address == author_cluster.prime_author_address:
                logger.info("    + %s", address)
            else:
                if prime_author:
                    update_by_lower_key_if_key_is_not_none(address_mappings, address, prime_author)
                logger.info("    - %s", address)
    return address_mappings


def resolve_authors(
    authors: list[Author],
    mailmap_authors: list[Author],
    all_mailmap_authors: list[Author] | None = None,
    manual_resolution: ManualResolution = ManualResolution.FALSE,
) -> list[Author]:
    """Produce a list of authors for a dot-mail-map.

    Given a list of authors and a list from the mailmap, produce a combined list that
    can be dumped to a .mailmap.

    This should produce at least one line per address, but
    will not produce a line mapping an address to itself if there is already a mapping
    to that address from another address.

    This is just to produce a single list for mailmap purposes -- git already maps items
    from the log.
    """
    if all_mailmap_authors is None:
        all_mailmap_authors = []
    email_mappings: dict[str, Author] = {}
    name_mappings: dict[str, Author] = {}
    address_mappings: dict[str, Author] = {}
    addresses_to_resolve: set[str] = set()
    for source in [authors]:
        for from_author in source:
            addresses_to_resolve.add(from_author.original_address.lower())
            addresses_to_resolve.add(from_author.address.lower())
    for source in [mailmap_authors, all_mailmap_authors, authors]:
        for author in source:
            add_by_lower_key_if_key_is_not_none(address_mappings, author.original_address, author)
            add_by_lower_key_if_key_is_not_none(email_mappings, author.original_email, author)
            add_by_lower_key_if_key_is_not_none(email_mappings, author.email, author)
            add_by_lower_key_if_key_is_not_none(name_mappings, author.original_username, author)
            add_by_lower_key_if_key_is_not_none(name_mappings, author.username, author)
            add_by_lower_key_if_key_is_not_none(name_mappings, author.original_name, author)
            add_by_lower_key_if_key_is_not_none(name_mappings, author.name, author)
            add_by_lower_key_if_key_is_not_none(address_mappings, author.address, author)

    if manual_resolution:
        address_mappings = manually_resolve_authors(
            addresses_to_resolve,
            address_mappings,
            email_mappings,
            name_mappings,
            authors,
            mailmap_authors,
            all_mailmap_authors,
        )

    combined_authors: list[Author] = []
    for source in [mailmap_authors, authors]:
        for from_author in source:
            to_author = resolve_author(from_author, email_mappings, name_mappings, address_mappings)

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


def output_dump(authors: list[Author], output_handle: TextIO) -> None:
    """Write authors to a CSV file or stdout."""
    print("Dumping", file=output_handle)
    for author in authors:
        line = f"""
{author.original_email} {author.email}
{author.original_name} {author.name}
{author.original_username} {author.username}
""".replace("\n", " ")
        print(f"{line}", file=output_handle)


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


def output_mail_map(
    authors: list[Author], output_handle: TextIO, manual_resolution: ManualResolution = ManualResolution.FALSE
) -> None:
    """Write authors to a mail map file or stdout."""
    seen_lines: set[str] = set()
    all_mailmap_authors = load_mail_maps()
    mailmap_authors = load_mail_map(pathlib.Path(".mailmap"))
    for author in resolve_authors(authors, mailmap_authors, all_mailmap_authors, manual_resolution):
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
    manual_resolution: ManualResolution = ManualResolution.FALSE,
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
            output_mail_map(authors, output_handle, manual_resolution)
        elif output_format == "pyproject":
            output_pyproject(authors, output_handle)
        elif output_format == "dump":
            output_dump(authors, output_handle)
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
        self.parser.add_argument(
            "--manual",
            dest="manual",
            action="store_true",
            default=False,
            help="Do a manual resolution step",
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
        committers = extract_committers(sha_range, files)
        coauthors = extract_coauthors(sha_range, files)

        output_authors(
            authors + committers + coauthors,
            self.args.format,
            self.args.ordering,
            self.args.reversed or Direction.REVERSE,
            ManualResolution.if_true(self.args.manual),
            output,
        )


def main() -> None:
    """Run the command."""
    AuthorsMain().main()


if __name__ == "__main__":
    main()
