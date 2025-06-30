"""Classes and enums mostly devoid of business logic."""

import enum
import logging
import re
import typing
from dataclasses import dataclass, field

from jmullan.git_changelog.text import none_as_empty_stripped, some_string

logger = logging.getLogger(__name__)

B = typing.TypeVar("B", bound="BooleanEnum")


class BooleanEnum(enum.Enum):
    """Extend this to make simple two-value enums to replace boolean arguments."""

    def __bool__(self):
        """Cast me into a boolean."""
        return bool(self.value)

    @classmethod
    def if_true(cls, true_false: bool) -> typing.Self:  # noqa: FBT001
        """Build this enum from something truthy."""
        for item in cls:
            if bool(true_false) == bool(item.value):
                return item
        raise ValueError("Not a boolean")


class Inclusiveness(BooleanEnum):
    """Boolean-ish definitions."""

    EXCLUSIVE = False
    INCLUSIVE = True


class UseTags(BooleanEnum):
    """Boolean-ish definitions."""

    FALSE = False
    TRUE = True


class Direction(enum.Enum):
    """Boolean-ish definitions."""

    REVERSE = enum.auto()
    FORWARD = enum.auto()


@dataclass
class ShaRange:
    """Holds a from, to, and whether to include each in the range."""

    from_sha: str | None
    from_inclusive: Inclusiveness
    to_sha: str | None
    to_inclusive: Inclusiveness


@dataclass
class Tag:
    """A named sha."""

    sha_sha: str = field(metadata={"template": "%(*objectname):%(objectname)"})
    ref_name: str = field(metadata={"template": "%(refname:short)"})
    date: str = field(metadata={"template": "%(taggerdate:iso8601)"})
    subject: str = field(metadata={"template": "%(subject)"})
    body: str = field(metadata={"template": "%(body)"})

    @property
    def sha(self) -> str:
        """Find the sha this tag points to."""
        parts = self.sha_sha.split(":")
        if len(parts[0]):
            return parts[0]
        return parts[1]

    def is_annotated(self) -> bool:
        """Check if this tag is annotated."""
        return any(some_string(x) for x in [self.subject, self.body])


@dataclass
class Commit:
    """The basic unit of a git log."""

    sha: str = field(metadata={"template": "%H"})
    date: str = field(metadata={"template": "%as"})
    email: str = field(metadata={"template": "%aE"})
    original_email: str = field(metadata={"template": "%ae"})
    name: str = field(metadata={"template": "%aN"})
    original_name: str = field(metadata={"template": "%an"})
    refnames: str = field(metadata={"template": "%D"})
    parents: str = field(metadata={"template": "%P"})
    body: str = field(metadata={"template": "%B"})
    notes: str = field(metadata={"template": "%N"})

    additional_jiras: list[str] = field(default_factory=list)
    likely_jiras: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "Commit":
        """Build a commit with nothing in it."""
        return cls(
            sha="",
            date="",
            email="",
            original_email="",
            name="",
            original_name="",
            refnames="",
            parents="",
            body="",
            notes="",
            additional_jiras=[],
            likely_jiras=[],
        )

    @property
    def parent_shas(self) -> list[str]:
        """Get the parents of this commit."""
        if self.parents is None:
            return []
        return self.parents.split(" ")

    @property
    def tag_names(self) -> list[str]:
        """Find any tags associated with this commit."""
        tag_names = []
        refnames = none_as_empty_stripped(self.refnames)
        if some_string(refnames):
            refs = refnames.split(",")
            for reference_name in refs:
                stripped = reference_name.strip()
                if stripped.startswith("tag: "):
                    tag_names.append(stripped.removeprefix("tag: "))
        return tag_names

    @property
    def heads(self) -> list[str]:
        """Find any heads in this commit."""
        heads = []
        refnames = none_as_empty_stripped(self.refnames)
        if refnames:
            refs = refnames.split(",")
            for reference_name in refs:
                stripped_name = reference_name.strip()
                if not stripped_name.startswith("tag: "):
                    heads.append(stripped_name)
        return heads

    def add_jiras(self, jiras: list[str]) -> None:
        """Add jiras if they are provided."""
        if jiras is not None:
            self.additional_jiras.extend(jiras)

    def add_likely_jiras(self, jiras: list[str]) -> None:
        """Add likely jiras if they are provided."""
        if jiras is not None:
            self.likely_jiras.extend(jiras)

    @property
    def subject(self) -> str:
        """Use the first non-empty line from the body as a subject."""
        body = none_as_empty_stripped(self.body)
        return body.split("\n", 1)[0]

    @property
    def description(self) -> str:
        """Cut up the body and haruspice the description."""
        body = none_as_empty_stripped(self.body)
        body_parts = body.split("\n", 1)
        if len(body_parts) > 1:
            return body_parts[1]
        return ""

    @property
    def is_merge_to_main(self) -> bool:
        """Look for evidence that this commit is a merge to main."""
        return bool(re.search("Merge.*to (master|main)", self.body))

    @property
    def version(self) -> str | None:
        """Look for likely version strings in a commit message."""
        matches = re.search(r"pre tag commit.*'(.*)'", self.subject)
        if matches:
            return matches.group(1)
        return None

    @property
    def month(self) -> str:
        """Get the month from a commit."""
        date = self.date
        if date is not None:
            return date[:7]
        return ""

    def is_likely_bot(self) -> bool:
        """Determine if the committer looks like a bot."""
        bots = ["dependabot", "sourcegraph.com", "githubactions-noreply", "jenkins"]
        name_fields = {f.lower() for f in [self.original_email, self.original_name, self.email, self.name]}
        for bot in bots:
            for name_field in name_fields:
                if bot in name_field:
                    return True
        return False


@dataclass
class Month:
    """A month of the year and the commits that seem to live in it."""

    name: str
    commits: list


@dataclass
class Version:
    """A supposed version and the commits it contains."""

    version_name: str
    commits: list[Commit]
