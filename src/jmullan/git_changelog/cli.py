#!/usr/bin/env python3.11
import logging

from jmullan.git_changelog.changelog import print_changelog
from jmullan_cmd import cmd
from jmullan_logging import easy_logging

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
            "-t",
            "--tags",
            dest="tags",
            action="store_true",
            default=False,
            help="Use tags and other found versions",
        )
        self.parser.add_argument(
            "--tags",
            dest="tags",
            action="store_true",
            default=False,
            help="Use tags and other found versions",
        )
        self.parser.add_argument("version", default="Current", nargs="?")

    def main(self):
        super().main()
        if self.args.verbose:
            easy_logging.easy_initialize_logging("DEBUG")
        else:
            easy_logging.easy_initialize_logging()
        logger.debug(self.args)

        from_sha = self.args.after or self.args.since
        from_inclusive = from_sha is None or self.args.since is not None
        to_sha = self.args.until or self.args.through
        to_inclusive = to_sha is None or self.args.through is not None

        print_changelog(
            from_sha, from_inclusive, to_sha, to_inclusive, self.args.version, self.args.tags
        )


def main():
    ChangeLogMain().main()


if __name__ == "__main__":
    main()
