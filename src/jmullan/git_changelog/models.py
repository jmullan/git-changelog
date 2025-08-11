"""Classes and enums mostly devoid of business logic."""

import enum
import logging
import re
import typing
from collections import defaultdict
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
    tagger_date: str = field(metadata={"template": "%(taggerdate:iso8601)"})
    creator_date: str = field(metadata={"template": "%(creatordate:iso8601)"})
    committer_date: str = field(metadata={"template": "%(committerdate:iso8601)"})
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
        # use a dictionary's keys as an ordered set
        shas = {sha.strip(): True for sha in self.parents.split(" ")}
        return list(shas.keys())

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
    version_tags: list[Tag]


class Chain:
    """A series of commits that are connected by single paths."""

    def __init__(self, children: dict[str, set[str]]):
        self._children = children
        self.commits: list[Commit] = []
        self.seen: set[str] = set()

    def add(self, commit: Commit) -> None:
        """Add a commit to the end of the chain."""
        if commit.sha not in self.seen:
            self.commits.append(commit)
            self.seen.add(commit.sha)

    def parents(self) -> list[str]:
        """Find the parents of the first commit in the chain."""
        first = self.first()
        return (first and first.parent_shas) or []

    def children(self) -> set[str]:
        """Find the children of the last commit in the chain."""
        last = self.last()
        return (last and self._children.get(last.sha)) or set()

    def first(self) -> Commit | None:
        """Find the first commit in a chain of commits."""
        if self.commits:
            return self.commits[0]
        return None

    def last(self) -> Commit | None:
        """Find the last commit in a chain of commits."""
        if self.commits:
            return self.commits[-1]
        return None

    def __contains__(self, item: Commit | str | None) -> bool:
        """Determine if the commit or commit sha is in the chain."""
        match item:
            case Commit() as x:
                return x.sha in self.seen
            case str() as x:
                return x in self.seen
            case _:
                return False


@dataclass
class WeightedSha:
    """A distance plus a sha."""

    weight: int
    sha: str


@dataclass
class WeightedShas:
    """A distance plus shas at that distance."""

    distance: int
    shas: set[str]


class CommitTree:
    """Builds information about the entire commit tree."""

    commits_by_sha: dict[str, Commit]
    tags_by_tag_name: dict[str, Tag]
    tags_by_sha: dict[str, list[Tag]]

    parents: dict[str, set[str]]
    children: dict[str, set[str]]
    chains: dict[str, Chain]
    generations: dict[str, int]
    closest_tag_shas: dict[str, set[str]]

    def __init__(self, commits_by_sha: dict[str, Commit], tags_by_tag_name: dict[str, Tag]) -> None:
        self.commits_by_sha = commits_by_sha
        self.tags_by_tag_name = tags_by_tag_name
        self.tags_by_sha = defaultdict(list)

        self.tips: set[str] = set()
        self.roots: set[str] = set()

        self.parents = defaultdict(set)
        self.children = defaultdict(set)
        self.generations = {}
        self.chains = self._build_chains()

        for commit in self.commits_by_sha.values():
            if commit.sha not in self.generations:
                self.generations[commit.sha] = 1
            for parent_sha in commit.parent_shas:
                if parent_sha not in self.generations:
                    self.generations[parent_sha] = 1
                self.generations[commit.sha] = max(self.generations[parent_sha] + 1, self.generations[commit.sha])
                self.parents[commit.sha].add(parent_sha)
                self.children[parent_sha].add(commit.sha)
        for commit in self.commits_by_sha.values():
            if not self.children.get(commit.sha):
                self.tips.add(commit.sha)
            if not self.parents.get(commit.sha):
                self.roots.add(commit.sha)

        self._complete_generations()

        for tag in tags_by_tag_name.values():
            self.tags_by_sha[tag.sha].append(tag)

        self.tag_graph = self.build_tag_graph()
        self.ordered_tag_shas = self.order_tag_shas()
        self.closest_tag_shas = self.build_closest_tag_shas()

    def _build_chains(self) -> dict[str, Chain]:
        """Build chains during setup."""
        chains: dict[str, Chain] = {}
        for sha in self.commits_by_sha:
            if sha in chains:
                continue
            sha_chain = self.walk_chain(sha)
            chain = Chain(self.children)
            for chain_sha in sha_chain:
                chain.add(self.commits_by_sha[sha])
                chains[chain_sha] = chain
        return chains

    def walk_chain(self, sha: str) -> list[str]:
        """Find all shas connected to the start sha by single links."""
        shas = [sha]
        while True:
            parents = self.parents[sha]
            if len(parents) != 1:
                break
            parent = next(iter(parents))
            siblings = self.children[parent]
            if len(siblings) != 1:
                break
            sha = parent
        shas.reverse()
        sha = shas[-1]
        while True:
            children = self.children[sha]
            if len(children) != 1:
                break
            child = next(iter(children))
            siblings = self.parents[child]
            if len(siblings) != 1:
                break
            shas.append(child)
            sha = child
        return shas

    def _complete_generations(self) -> None:
        """Fill in missing generation information."""
        stack: list[str] = list(self.tips)
        seen: set[str] = set()
        while stack:
            sha = stack.pop(0)
            if sha in seen:
                continue
            seen.add(sha)
            children = self.children[sha]
            if not children:
                self.generations[sha] = 1
            else:
                my_depth = self.generations.get(sha) or 1
                max_depth = max(self.generations[child_sha] for child_sha in children)
                self.generations[sha] = max(my_depth, max_depth + 1)
            stack.extend(self.parents.get(sha) or [])

    def is_ancestor(self, child_sha: str, possible_ancestor_sha: str) -> bool:
        """Find all sha ancestors from a given sha."""
        stack: list[str] = [child_sha]
        found = set()
        while stack:
            sha = stack.pop()
            if sha == possible_ancestor_sha:
                return True
            if sha in found:
                continue
            found.add(sha)
            parents = self.parents.get(sha)
            if not parents:
                continue
            for parent in parents:
                if parent == possible_ancestor_sha:
                    return True
                if parent == sha or parent in found:
                    continue
                stack.append(parent)
        return False

    def build_tag_graph(self) -> dict[str, set[str]]:
        """Build a graph of tags and their predecessors."""
        ancestors: dict[str, set[str]] = defaultdict(set)
        for sha, tags in self.tags_by_sha.items():
            for possible_parent in tags:
                possible_parent_sha = possible_parent.sha
                if sha in ancestors[possible_parent_sha] or possible_parent_sha in ancestors[sha]:
                    continue
                if self.is_ancestor(sha, possible_parent_sha):
                    ancestors[sha].add(possible_parent_sha)
                elif self.is_ancestor(possible_parent_sha, sha):
                    ancestors[possible_parent_sha].add(sha)
        for sha, heritage in ancestors.items():
            # remove anything also included in parent heritages
            retained = set(heritage)
            for item in heritage:
                retained = retained - set(ancestors[item])
            ancestors[sha] = retained

        return ancestors

    def order_tag_shas(self) -> list[str]:
        """Build a list of tags ordered by their hierarchy and commit date."""
        ordered_tags = []
        seen_tag_shas: set[str] = set()
        tags_to_check = list(self.tag_graph.keys())
        while tags_to_check:
            leaves = [sha for sha in tags_to_check if not set(self.tag_graph[sha]) - seen_tag_shas]
            for leaf in leaves:
                tags_to_check.remove(leaf)
                seen_tag_shas.add(leaf)
            ordered_tags.extend(sorted(leaves, key=self.get_sha_date))
        return ordered_tags

    def get_sha_date(self, sha: str) -> str:
        """Find the best-ish date for a given sha."""
        dates = []
        commit = self.commits_by_sha.get(sha)
        if commit:
            commit_date = commit.date
            if commit_date:
                dates.append(commit_date)
        tags = self.tags_by_sha[sha]
        for tag in tags:
            dates.append(tag.tagger_date)
            dates.append(tag.committer_date)
            dates.append(tag.creator_date)
        dates.append("The future")
        return min(date for date in dates if date is not None and len(date))

    def get_closest_tags(self, sha: str) -> list[Tag]:
        """Find all tags closest to a given sha."""
        tags = self.tags_by_sha[sha]
        if tags:
            return tags
        closest_tag_shas = self.closest_tag_shas.get(sha)
        closest_tags = []
        if closest_tag_shas:
            for close_tag_sha in closest_tag_shas:
                tags = self.tags_by_sha[close_tag_sha]
                closest_tags.extend(tags)
            closest_tags.extend(tags)
        return closest_tags

    def build_closest_tag_shas(self) -> dict[str, set[str]]:
        """Build a lookup of shas to their closest tag shas."""
        closest_tag_shas: dict[str, WeightedShas] = {}
        for tag_sha in self.ordered_tag_shas:
            self.walk_from_sha(tag_sha, closest_tag_shas)
        return {sha: ws.shas for sha, ws in closest_tag_shas.items()}

    def walk_from_sha(self, tag_sha: str, closest_tag_sha: dict[str, WeightedShas]) -> None:
        """Find all sha ancestors from a given sha."""
        found = set()
        weighted_sha = WeightedSha(0, tag_sha)
        stack: list[WeightedSha] = [weighted_sha]
        while stack:
            weighted_sha = stack.pop()
            if weighted_sha.sha in found:
                continue
            found.add(weighted_sha.sha)
            parents = self.parents.get(weighted_sha.sha)
            if not parents:
                continue
            weight = weighted_sha.weight + 1
            for parent in parents:
                weighted_parent = WeightedSha(weight, parent)
                if parent not in closest_tag_sha:
                    closest_tag_sha[parent] = WeightedShas(weight, {tag_sha})
                    stack.append(weighted_parent)
                else:
                    old_weighted_shas = closest_tag_sha[parent]
                    if old_weighted_shas.distance > weight:
                        closest_tag_sha[parent] = WeightedShas(weight, {tag_sha})
                        stack.append(weighted_parent)
                    elif old_weighted_shas.distance == weight:
                        old_weighted_shas.shas.add(tag_sha)
                        stack.append(weighted_parent)
                    else:
                        # we've already seen this parent in another context
                        # so leave it alone
                        pass
