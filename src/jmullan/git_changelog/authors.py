#!/usr/bin/env python3.11
import csv
import logging
import sys

from jmullan.git_changelog import changelog
from jmullan_cmd import cmd
from jmullan_logging import easy_logging

logger = logging.getLogger(__name__)

formats = {
    "simple": "{name or username} <{email}>",
    "csv": "{email},{username},{name}",
    "mail-map": "correct name <email> wrong name <email>",
    "pyproject": "The authors stanza of a pyproject.toml",
}
FORMAT_DOC = "\n  ".join(f"{x}: {y}" for x, y in formats.items())
DEFAULT_FORMAT = "simple"
orderings = ["appearance", "email", "name"]
DEFAULT_ORDERING = "appearance"


def print_authors(
    from_sha: str | None = None,
    from_inclusive: bool | None = False,
    to_sha: str | None = None,
    to_inclusive: bool | None = False,
    output_format: str | None = DEFAULT_FORMAT,
    ordering: str | None = DEFAULT_ORDERING,
):
    if output_format is None:
        output_format = DEFAULT_FORMAT
    if ordering is None:
        ordering = DEFAULT_ORDERING
    git_format = "%aE %aL %aN"
    emails_to_authors = dict()
    tuples = []
    seen_tuples = set()
    for line in changelog.git_log(
        git_format, from_sha, from_inclusive, to_sha, to_inclusive, reversed=True
    ):
        if line is None or len(line) < 1:
            continue
        logger.debug(line)
        parts = [x for x in line.strip().split(" ") if x]
        email = parts.pop(0)
        username = parts.pop(0)
        if parts:
            name = " ".join(parts)
        else:
            name = username
        emails_to_authors[email] = name
        csv_tuple = (email, username, name)
        if csv_tuple not in seen_tuples:
            seen_tuples.add(csv_tuple)
            tuples.append(csv_tuple)
    if output_format == "csv" or output_format == "mail-map":
        if ordering == "email":
            data = sorted(tuples, key=lambda csv_tuple: csv_tuple[0])
        elif ordering == "name":
            data = sorted(tuples, key=lambda csv_tuple: csv_tuple[2])
        else:
            data = tuples
        if output_format == "csv":
            writer = csv.writer(sys.stdout)
            for csv_tuple in data:
                writer.writerow(csv_tuple)
        else:
            for email, username, name in data:
                print(f"{name} <{email}> {name} <{email}>")
    else:
        if ordering == "email":
            data = sorted(emails_to_authors.items(), key=lambda item: item[0])
        elif ordering == "name":
            data = sorted(emails_to_authors.items(), key=lambda item: item[1])
        else:
            data = list(emails_to_authors.items())
            data.reverse()
        if output_format == "pyproject":
            print("authors = [")
            lines = [
                '    {name = "%s", email = "%s"}' % (name, email) for email, name in data if data
            ]
            print(",\n".join(lines))
            print("]")
        else:
            for email, name in data:
                print(f"{name} <{email}>")


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

        print_authors(
            from_sha, from_inclusive, to_sha, to_inclusive, self.args.format, self.args.ordering
        )


def main():
    AuthorsMain().main()


if __name__ == "__main__":
    main()
