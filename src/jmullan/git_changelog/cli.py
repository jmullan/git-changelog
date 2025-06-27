#!/usr/bin/env python3.13
import logging
import sys

from jmullan.cmd import cmd
from jmullan.logging import easy_logging

from jmullan.git_changelog.changelog import print_changelog

logger = logging.getLogger(__name__)


class ChangeLogMain(cmd.Main):
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
            "--no-use-tags",
            dest="use_tag_names",
            action="store_false",
            default=True,
            help="Use tags and other found versions",
        )
        self.parser.add_argument(
            "--file", dest="files", action="append", default=[], required=False
        )
        self.parser.add_argument("version", default="Current", nargs="?")

    def setup(self):
        super().setup()
        if self.args.verbose:
            easy_logging.easy_initialize_logging("DEBUG", stream=sys.stderr)
        elif self.args.quiet:
            easy_logging.easy_initialize_logging("WARNING", stream=sys.stderr)
        else:
            easy_logging.easy_initialize_logging("INFO", stream=sys.stderr)

    def main(self):
        super().main()
        files = self.args.files or []

        from_sha = self.args.after or self.args.since
        from_inclusive = from_sha is None or self.args.since is not None
        to_sha = self.args.until or self.args.through
        to_inclusive = to_sha is None or self.args.through is not None
        print_changelog(
            from_sha,
            from_inclusive,
            to_sha,
            to_inclusive,
            self.args.version,
            self.args.use_tag_names,
            files,
        )


def main():
    ChangeLogMain().main()


if __name__ == "__main__":
    main()
