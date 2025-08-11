"""Functions and classes to support building a CHANGELOG document."""

import csv
import logging
import os
import pathlib
import re
import shlex
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from dataclasses import fields
from typing import IO, TextIO, TypeGuard

from jmullan.git_changelog.models import Commit, CommitTree, Direction, ShaRange, Tag, UseTags, Version
from jmullan.git_changelog.text import fill_text, none_as_empty, none_as_empty_stripped, some_string

logger = logging.getLogger(__name__)

_ignores = [
    "* commit",
    "[Gradle Release Plugin] - new version commit:",
    "[Gradle Release Plugin] - pre tag commit",
    "Merge pull request #",
    "git-p4",
    "integrating changelist",
    "integrate changelist",
    "Integrate changelist",
    "Squashed commit",
    "Merge master",
    "Merge main",
    "merge to master",
    "merge to main",
]

_ignore_matches = [
    r"^ *This reverts commit *[a-z0-9]{40}\. *$",
    r"^ *commit *[a-z0-9]{40} *$",
    r"^ *Author: .*",
    r"^ *Date: .*",
    r"^ *Merge: [a-z0-9]+ [a-z0-9]+ *$",
    r"^ *Merge branch .* into .*",
    r"^ *Merge branch '.*'$",
    r"Merge branch '[^']+' of",
    r"^ *\.\.\. *$",
    r"^ *\.\.\. and [0-9]+ more commits *$",
    r"Merge in",
    r"<!--[^\n]*-->",
]

_delete_regexes = [
    r"^\s*Pull request #[0-9]+\s*:\s*",
    r"^(\** *)*",
    r"^(-* *)*",
    r"^feature/",
    r"^bugfix/",
    r"commit\s*[a-f0-9]+\s*\[formerly\s*[a-f0-9]+\]\s*:?",
    r"Former-commit-id:.*",
    r"^WIP",
    r"\s*WIP$",
    r"^\s*merged\s*$",
    r"^\s*merged master\s*$",
    r"^\s*merged main\s*$",
    r"^\s*Writing version .* to release.properties\s*$",
    r"^\[\]\s*",
    r"^\[Gradle Release Plugin\] - creating tag:\s*'.*'\.",
]

INVALID_NAMES = {"jenkins", "mobileautomation"}

EMAILS_TO_NAMES = {}  # type: dict[str, str]

DEFAULT_WIDTH = 100
LIST_PREFIX = "- "
LIST_CONTINUATION = "- "
MIN_TICKET_FILE_ROW_LENGTH = 2


def load_tickets() -> dict[str, str]:
    """Try to load ticket data from a particular file."""
    environ_path = os.environ.get("JIRA_PATH")
    if environ_path is None:
        return {}
    tickets_file_path = pathlib.Path(environ_path)
    tickets: dict[str, str] = {}
    if tickets_file_path.exists():
        with tickets_file_path.open(encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for record in reader:
                if not record:
                    continue
                if len(record) < MIN_TICKET_FILE_ROW_LENGTH:
                    continue
                key = record[0].strip()
                if not len(key):
                    continue
                text = [x.strip() for x in record[1:]]
                description = "\n".join(x for x in text if len(x))
                if not len(description):
                    continue
                tickets[key] = description
    return tickets


# git-for-each-ref doesn't have a -z option, so we manually add the null character
GIT_TAG_FORMAT = (
    "\n".join(
        f"{tag_field.name} {tag_field.metadata['template']}"
        for tag_field in fields(Tag)
        if tag_field.metadata.get("template") is not None
    )
    + "%00"
)

GIT_COMMIT_FORMAT = "\n".join(
    f"%h {commit_field.name} {commit_field.metadata['template']}"
    for commit_field in fields(Commit)
    if commit_field.metadata.get("template") is not None
)


def include_line(line: str | None) -> bool:
    """Determine if the line is important enough to include."""
    return (
        line is not None
        and not any(x in line for x in _ignores)
        and not any(re.search(regex, line) for regex in _ignore_matches)
    )


def delete_junk(line: str) -> str:
    """Remove junk strings from the line."""
    for regex in _delete_regexes:
        line = re.sub(regex, "", line, flags=re.IGNORECASE).strip()
    return line


def extract_jiras(body: str | None) -> list[str]:
    """Find things that look like jiras in the line."""
    if body is None:
        return []
    return list(set(re.findall("[A-Z]+-[0-9]+", body) or []))


def strip_jiras(line: str, jiras: list[str]) -> str:
    """Remove jiras from the line."""
    line = none_as_empty_stripped(line)
    if not some_string(line):
        return ""
    for jira in jiras:
        escaped = re.escape(jira)
        line = re.sub(rf"{escaped}\s*[-/:]*", "", line)
    return line


def format_tag_names(tags: list[str] | None) -> str:
    """Turn tags into a heading."""
    if not tags:
        return ""
    tags = sorted(tags, key=best_tag)
    tags_line = ", ".join([f"`{tag}`" for tag in tags])
    return f"### Tags: {tags_line}"


def best_tag(tag_name: str) -> tuple[bool, bool, bool, int, str]:
    """Find the most interesting tag."""
    tag_name = none_as_empty_stripped(tag_name)
    has_snapshot = "SNAPSHOT" in tag_name
    has_semantic_version = bool(re.match(r"^[0-9]+(\.[0-9]+){1,2}$", tag_name))
    has_semantic_subversion = bool(re.match(r"^[0-9]+(\.[0-9]+){1,2}(-.*)?$", tag_name))
    return (
        has_snapshot,
        not has_semantic_version,
        not has_semantic_subversion,
        len(tag_name),
        tag_name,
    )


def tags_to_release_version(tags: list[str], found_version: bool) -> str | None:  # noqa: FBT001
    """Build a release version string from available tags."""
    semantic_versions = []
    semantic_sub_versions = []
    other_tags = []
    for tag in tags:
        if found_version and "SNAPSHOT" in tag:
            # ignore snapshot tags
            continue
        if re.match(r"^[0-9]+(\.[0-9]+){1,2}$", tag):
            semantic_versions.append(tag)
        elif re.match(r"^[0-9]+(\.[0-9]+){1,2}(-.*)?$", tag):
            semantic_sub_versions.append(tag)
        else:
            other_tags.append(tag)
    for candidates in [semantic_versions, semantic_sub_versions, other_tags]:
        if candidates:
            candidates.sort(key=best_tag)
            return candidates[0]
    return None


def format_body(body: str | None, jiras: list[str]) -> list[str]:
    """Munge the body into something nice."""
    body = none_as_empty(body)
    if not some_string(body):
        return []
    lines = body.rstrip().split("\n")
    lines = [line for line in lines if include_line(line)]
    lines = [delete_junk(line) for line in lines]
    lines = [strip_jiras(line, jiras) for line in lines]
    lines = [delete_junk(line) for line in lines]
    lines = [line for line in lines if line]
    logger.debug("lines: %s", lines)
    return lines


def clean_body(commit: Commit) -> str:
    """Cast out the impurities."""
    body = none_as_empty(commit.body).strip("\n")
    lines = body.split("\n")
    lines = [line.rstrip() for line in lines]
    lines = [line.strip("\n") for line in lines]
    lines = [line for line in lines if include_line(line)]
    output = "\n".join(lines)
    output = re.sub(r"\n+", "\n", output)
    return output.strip("\n")


def valid_name(name: str | None) -> TypeGuard[str]:
    """Filter out bot names."""
    name = none_as_empty_stripped(name)
    if not some_string(name):
        return False
    name = name.lower()
    if "jenkins builder" in name:
        return False
    return name not in INVALID_NAMES


def smart_name(commit: Commit) -> str | None:
    """Try to get the name or email from a commit."""
    name = none_as_empty_stripped(commit.name)
    if name:
        return name
    email = commit.email
    return EMAILS_TO_NAMES.get(email) or email


def format_names(commits: list[Commit]) -> str | None:
    """Extract committer names from commits."""
    if not commits:
        return None
    names = {smart_name(commit) for commit in commits}
    valid_names = {name for name in names if valid_name(name)}
    return ", ".join(sorted(valid_names))


def make_version_line(release_version: str, commits: list[Commit]) -> str:
    """Use the version string and the first commit to make a version string.

    Example: v.1234 (2024-04)
    """
    version_string = none_as_empty_stripped(release_version)
    if not some_string(version_string):
        return ""
    if commits:
        first_commit = commits[0]
        date = first_commit.date
        parts = ["#"]
        if re.match("^[0-9]", version_string):
            parts.append(f"v.{version_string}")
        else:
            parts.append(version_string)

        if date is not None and not date.startswith(version_string) and date not in version_string:
            parts.append(f"({date})")
        return " ".join(parts)
    return ""


def all_tickets(commit: Commit) -> list[str]:
    """Get all the possible jiras from a commit."""
    return extract_jiras(commit.body) + commit.additional_jiras


def format_commit(commit: Commit) -> list[str]:
    """Build strings from a commit."""
    subject = commit.subject
    if not include_line(subject):
        return []
    commit_lines = []
    subject_lines = format_body(subject, all_tickets(commit))
    if subject_lines:
        commit_lines.extend(subject_lines)
    description = commit.description
    if some_string(description):
        description_lines = format_body(description, all_tickets(commit))
        if description_lines:
            for description_line in description_lines:
                if description_line not in commit_lines:
                    commit_lines.append(description_line)
    if commit.is_likely_bot() and commit_lines:
        commit_line = commit_lines[0]
        return [f"(bot) {commit_line}"]
    return commit_lines


def format_annotated_tag(tag: Tag) -> str | None:
    """Make a string from an annotated tag."""
    if tag is None:
        return None
    ref_name = none_as_empty(tag.ref_name)
    if not some_string(ref_name):
        # this should never happen
        return None
    subject = none_as_empty_stripped(tag.subject).strip()
    subject = delete_junk(subject)
    body = none_as_empty_stripped(tag.body).strip()
    body = delete_junk(body)
    tag_parts = []
    if not some_string(subject):
        tag_parts.append(f"## Tag: `{ref_name}`")
    else:
        tag_parts.append(
            textwrap.fill(
                f"## Tag: `{ref_name}` {subject}",
                DEFAULT_WIDTH,
                initial_indent="",
                subsequent_indent="    ",
            )
        )
    if some_string(body):
        tag_parts.append("")
        body = fill_text(body, DEFAULT_WIDTH, "    ")
        tag_parts.append(body)
    tag_body = "\n\n".join(tag_parts).rstrip()
    return f"{tag_body}"


def fill_body(ticket_string: str, summary_string: str, body: str, list_prefix: str) -> str:
    """Build the full body from possible strings."""
    if ticket_string or summary_string:
        not_body = fill_text(f"{ticket_string} {summary_string}".strip(), DEFAULT_WIDTH, "    ", initial_indent="")
    else:
        not_body = ""

    if not not_body:
        return fill_text(body, DEFAULT_WIDTH, LIST_CONTINUATION, list_prefix)

    if not len(body):
        return not_body

    return fill_text(body, DEFAULT_WIDTH, indent="    ", initial_indent=f"{not_body} ").strip()


def smoosh_commits_into_body(commits: list[Commit]) -> str:
    """Consolidate all commit messages into a single body."""
    lines = []
    seen_commits = set()
    for commit in commits:
        commit_lines = format_commit(commit)
        if commit_lines:
            for commit_line in commit_lines:
                unique_commit_lines = []
                if commit_line not in seen_commits:
                    seen_commits.add(commit_line)
                    unique_commit_lines.append(commit_line)
                lines.extend(unique_commit_lines)
    return "\n".join(lines).strip()


def format_ticket_commits(
    commits: list[Commit], ticket_string: str | None, tickets_to_summaries: dict[str, str]
) -> str:
    """Turn jiras and their commits into a string."""
    body = smoosh_commits_into_body(commits)

    if not some_string(ticket_string):
        return fill_text(body, DEFAULT_WIDTH, LIST_CONTINUATION, LIST_PREFIX)

    tickets = [ticket.strip() for ticket in ticket_string.split(",")]
    summary_string = ""
    tickets_with_summaries = []
    if len(tickets) > 0:
        tickets_with_summaries = [
            f"+ {ticket}: {tickets_to_summaries[ticket]}" for ticket in tickets if tickets_to_summaries.get(ticket)
        ]
        jiras_without_summaries = [ticket for ticket in tickets if ticket not in tickets_to_summaries]
        if len(tickets_with_summaries) > 0:
            summary_string = "\n".join(tickets_with_summaries)
            if jiras_without_summaries:
                ticket_string = ", ".join(jiras_without_summaries)
            else:
                ticket_string = ""
    if ticket_string:
        ticket_string = f"{ticket_string}:"
    if tickets_with_summaries:
        list_prefix = f"    {LIST_PREFIX}"
    else:
        list_prefix = LIST_PREFIX
    return fill_body(ticket_string, summary_string, body, list_prefix)


def format_month_commits(month_commits: list[Commit], tickets_to_summaries: dict[str, str]) -> list[str]:
    """Turn a month and its commits into a list of strings."""
    month_commit_lines = []
    commits_by_ticket: dict[str, list[Commit]] = {"": []}
    for commit in month_commits:
        tickets = all_tickets(commit) or []
        unique_tickets = {ticket.strip() for ticket in tickets if some_string(ticket)}
        if not tickets:
            ticket_string = ""
        else:
            ticket_string = ", ".join(sorted(unique_tickets))
        if ticket_string not in commits_by_ticket:
            commits_by_ticket[ticket_string] = []
        commits_by_ticket[ticket_string].append(commit)
    for ticket_string, commits in commits_by_ticket.items():
        ticket_commits_body = format_ticket_commits(commits, ticket_string, tickets_to_summaries)
        if ticket_commits_body:
            month_commit_lines.append("")
            month_commit_lines.extend(ticket_commits_body.split("\n"))
    return month_commit_lines


def build_tags_by_format(version: Version) -> dict[str, list[str]]:
    """Group tags by a formatted string combination of their names."""
    tags_by_format: dict[str, list[str]] = {}
    for commit in version.commits:
        if commit.tag_names:
            current_tag = format_tag_names(commit.tag_names)
            tags_by_format[current_tag] = commit.tag_names
    return tags_by_format


def build_months_by_tag(
    version: Version,
) -> dict[str, dict[str, list[Commit]]]:
    """Group a version's tag's commits into months."""
    months_by_tag: dict[str, dict[str, list[Commit]]] = {}
    current_tag = ""
    for commit in version.commits:
        maybe_tags = format_tag_names(commit.tag_names)
        if some_string(maybe_tags):
            current_tag = maybe_tags
        current_month = commit.month
        if current_tag not in months_by_tag:
            months_by_tag[current_tag] = {}
        commits_by_month = months_by_tag[current_tag]
        if current_month not in commits_by_month:
            commits_by_month[current_month] = []
        current_month_commits = commits_by_month[current_month]
        current_month_commits.append(commit)
    return months_by_tag


def make_notes(version: Version, tags_by_tag_name: dict[str, Tag], jiras_to_summaries: dict[str, str]) -> str:
    """Make the notes for a Version and its commits."""
    version_line = make_version_line(version.version_name, version.commits)
    release_note = []
    if version_line is not None and len(version_line):
        release_note.append(version_line)
    tags_notes = []
    tags_by_format = build_tags_by_format(version)
    months_by_tag = build_months_by_tag(version)

    for formatted_tags, commits_by_month in months_by_tag.items():
        if some_string(formatted_tags):
            tag_names = tags_by_format[formatted_tags]
            tags = {tag_name: tags_by_tag_name[tag_name] for tag_name in tag_names if tag_name in tags_by_tag_name}
            annotated_tags = {tag_name: tag for tag_name, tag in tags.items() if tag.is_annotated()}
            for tag in annotated_tags.values():
                tag_note = format_annotated_tag(tag)
                if tag_note is not None:
                    tags_notes.append("")
                    tags_notes.append(tag_note)
            remaining_tag_names = [tag_name for tag_name in tag_names if tag_name not in annotated_tags]
            if len(remaining_tag_names) > 0:
                reformatted = format_tag_names(remaining_tag_names)
                tags_notes.append("")
                tags_notes.append(reformatted)
        for month, month_commits in commits_by_month.items():
            tags_notes.extend(build_month_commit_lines(month, month_commits, version_line, jiras_to_summaries))
    if tags_notes:
        release_note.extend(tags_notes)
    return "\n".join(release_note).strip()


def build_month_commit_lines(
    month: str, month_commits: list[Commit], version_line: str, jiras_to_summaries: dict[str, str]
) -> list[str]:
    """Build a new month section."""
    tags_notes = []
    if month not in version_line:
        tags_notes.append("")
        tags_notes.append(f"### {month}")
    formatted_names = format_names(month_commits)
    if some_string(formatted_names) and formatted_names is not None:
        tags_notes.append("")
        tags_notes.append(formatted_names)
    month_commit_lines = format_month_commits(month_commits, jiras_to_summaries)
    if len(month_commit_lines):
        tags_notes.extend(month_commit_lines)
    return tags_notes


def prune_heads(heads: list[str]) -> list[str]:
    """Turn a list of HEAD references into shorter names."""
    new_heads = []
    for head in heads:
        pruned_head = head.strip().removeprefix("HEAD ->").removesuffix("/HEAD")
        if pruned_head:
            new_heads.append(pruned_head)
    return new_heads


def extract_refs(commit: Commit) -> dict[str, list[str]]:
    """Find any refs in a commit."""
    sha = commit.sha
    parent_shas: list[str] = commit.parent_shas or []
    heads = commit.heads or []
    heads = prune_heads(heads)
    shas_to_refs = {sha: heads}

    merge_matches = re.search(r"Merge .* from ([^ ]+) to ([^ ]+)$", commit.subject)
    if merge_matches:
        from_ref = merge_matches.group(1)
        to_ref = merge_matches.group(2)
        if parent_shas and len(parent_shas) == len(merge_matches.groups()):
            left, right = parent_shas
            if right not in shas_to_refs:
                shas_to_refs[right] = []
            shas_to_refs[right].append(from_ref)
            if left not in shas_to_refs:
                shas_to_refs[left] = []
            shas_to_refs[left].append(to_ref)
    return shas_to_refs


def stream_chunks(io: IO[bytes] | None, separator: str = "\n") -> Iterator[str]:
    """Read a stream of bytes and yield strings divided by the separator."""
    separator_bytes = separator.encode("UTF8")
    accumulated = bytearray()
    keep_going = True
    while io is not None and io.readable() and keep_going:
        read_chunk = io.read(1024)
        if read_chunk == b"":
            keep_going = False
        accumulated.extend(read_chunk)
        while separator_bytes in accumulated:
            chunk, accumulated = accumulated.split(separator_bytes, 1)
            yield chunk.decode("UTF8")
    yield accumulated.decode("UTF8")


def chunk_command(args: list[str]) -> Iterator[str]:
    """Run the args as a shell command.

    This is provided to be mockable.
    """
    with subprocess.Popen(args, stdout=subprocess.PIPE) as proc:  # noqa: S603
        yield from stream_chunks(proc.stdout, "\x00")


def get_tags_by_tag_name() -> dict[str, Tag]:
    """Find all the tags in the repo."""
    command = ["git", "for-each-ref", f"--format={GIT_TAG_FORMAT}", "refs/tags"]
    tags_by_tag_name: dict[str, Tag] = {}
    for chunk in chunk_command(command):
        tag = chunk_to_tag(chunk)
        if tag is not None:
            tags_by_tag_name[tag.ref_name] = tag
    return tags_by_tag_name


def git_log(
    git_format: str,
    sha_range: ShaRange,
    direction: Direction,
    files: list[str] | None,
) -> Iterator[str]:
    """Run git log with a particular format."""
    command = ["git", "log", "-z", f"--format={git_format}"]
    from_sha = sha_range.from_sha
    to_sha = sha_range.to_sha
    from_inclusive = sha_range.from_inclusive
    to_inclusive = sha_range.to_inclusive
    if to_inclusive:
        to_caret = ""
    else:
        to_caret = "^"
    if sha_range.from_sha is not None:
        if from_inclusive:
            from_caret = "^"
        else:
            from_caret = ""
        if to_sha is None:
            to_sha = "HEAD"
            to_caret = ""
        dot_sha_range = f"{from_sha}{from_caret}..{to_sha}{to_caret}"
        command.append(dot_sha_range)
    elif sha_range.to_sha is not None:
        from_sha = first_sha()
        dot_sha_range = f"{from_sha}..{to_sha}{to_caret}"
        command.append(dot_sha_range)
    if direction == Direction.REVERSE:
        command.append("--reverse")
    if files:
        command.extend([shlex.quote(f) for f in files])
    yield from chunk_command(command)


def git_commits_by_sha(
    sha_range: ShaRange,
    direction: Direction,
    files: list[str],
) -> dict[str, Commit]:
    """Load commits for a particular range and make them into Commit objects."""
    commits_by_sha = {}  # type: dict[str, Commit]
    for chunk in git_log(GIT_COMMIT_FORMAT, sha_range, direction, files=files):
        commit = chunk_to_commit(chunk)
        if commit is None:
            logger.debug("None commit")
            continue
        commits_by_sha[commit.sha] = commit
    return commits_by_sha


def first_sha() -> str:
    """Find the very first sha in a git repo."""
    command = ["git", "hash-object", "-t", "tree", "/dev/null"]
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE)  # noqa: S603
    return result.stdout.decode("UTF8").strip()


def extract_header_fields(header: str) -> dict[str, str]:
    """Convert a specially formatted string into a dictionary."""
    if not some_string(header):
        return {}
    data = {}
    for line in header.split("\n"):
        if not some_string(line):
            continue
        key, value = line.split(" ", 1)
        data[key] = value
    return data


def chunk_to_tag(chunk: str) -> Tag | None:
    """Build a tag from a string."""
    if chunk is None:
        return None
    if "\nbody" not in chunk:
        logger.debug("body not in tag %r", chunk)
        return None
    header, body = chunk.split("\nbody", 1)
    tag_data = {"body": body.removeprefix(" ").rstrip()}
    tag_data.update(extract_header_fields(header))
    sha_sha = tag_data.get("sha_sha")
    if not some_string(sha_sha):
        logger.debug("No sha_sha in commit %s", chunk)
        return None
    if sha_sha is not None:
        # mypy can't tell that not is_empty means not None :(
        parts = sha_sha.split(":")
        if len(parts) == 0 or not some_string(parts[0]):
            # looks like this is actually not an annotated tag
            tag_data["subject"] = ""
            tag_data["body"] = ""
    return Tag(**tag_data)  # type: ignore[arg-type]


def chunk_to_commit(chunk: str) -> Commit | None:
    """Turn a chunk into a commit object."""
    if chunk is None:
        return None
    abbreviated_sha = chunk.split(" ", 1)[0]
    field_pieces = chunk.removeprefix(f"{abbreviated_sha} ").split(f"\n{abbreviated_sha} ")
    commit_data = {p[0]: p[1] for p in [piece.split(" ", 1) for piece in field_pieces] if len(p) > 1}
    if "body" not in commit_data:
        logger.debug("body not in commit %r", chunk)
        return None

    if not commit_data.get("sha"):
        logger.debug("No sha in commit %s", chunk)
        return None
    try:
        return Commit(**commit_data)  # type: ignore[arg-type]
    except TypeError:
        logger.exception("error processing chunk %s %s", chunk, commit_data)
        raise


def all_ancestors(sha: str, commits_by_sha: dict[str, Commit]) -> set[str]:
    """Find all sha ancestors from a given sha."""
    stack: list[str] = [sha]
    found = set()
    while stack:
        sha = stack.pop()
        if sha in found:
            continue
        found.add(sha)
        commit = commits_by_sha.get(sha)
        if commit is None:
            continue
        parents = commit.parent_shas
        if not parents:
            continue
        for parent in parents:
            if parent == sha or parent in found:
                continue
            stack.append(parent)
    return found


def is_ancestor(child_sha: str, possible_ancestor_sha: str, commits_by_sha: dict[str, Commit]) -> bool:
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
        commit = commits_by_sha.get(sha)
        if commit is None:
            continue
        parents = commit.parent_shas
        if not parents:
            continue
        for parent in parents:
            if parent == possible_ancestor_sha:
                return True
            if parent == sha or parent in found:
                continue
            stack.append(parent)
    return False


def get_commit_tags(commit: Commit, commit_tree: CommitTree) -> tuple[list[Tag], list[str]]:
    """Find tags that apply to a commit."""
    tags = []
    commit_tags = commit.tag_names or []
    if not commit_tags:
        tags = commit_tree.get_closest_tags(commit.sha)
        commit_tags = [tag.ref_name for tag in tags]
    return tags, commit_tags


def build_versions(
    version: str | None,
    use_tags: UseTags,
    commit_tree: CommitTree,
) -> list[Version]:
    """Group commits / shas by version-ish name."""
    version_tree: dict[str, list[Commit]] = {}
    found_version = False
    version_tags: dict[str, list[Tag]] = {}
    for commit in commit_tree.commits_by_sha.values():
        group_name = version or "Unknown"
        if commit.version:
            group_name = commit.version
            found_version = True
        elif use_tags:
            tags, commit_tags = get_commit_tags(commit, commit_tree)
            if commit_tags:
                candidate_version = tags_to_release_version(commit_tags, found_version)
                if candidate_version:
                    group_name = candidate_version
                version_tags[group_name] = tags

        if group_name not in version_tree:
            version_tree[group_name] = []
        version_tree[group_name].append(commit)

    versions: list[Version] = []
    for group_name, commits in version_tree.items():
        if not commits:
            continue
        group_version = Version(group_name, commits, version_tags.get(group_name) or [])
        versions.append(group_version)
    return versions


def populate_jiras_from_parents(commits_by_sha: dict[str, Commit]) -> dict[str, Commit]:
    """Populate a jira from the parent commits of a sha."""
    for sha, commit in commits_by_sha.items():
        parent_shas = commit.parent_shas or []
        if commit.is_merge_to_main and len(parent_shas) in (1, 2):
            parent_sha = parent_shas[-1]
            parent = commits_by_sha.get(parent_sha)

            if parent is not None:
                shares_subject = commit.subject in parent.body or parent.subject in commit.body
                parent_jiras = all_tickets(parent)
                commit_jiras = all_tickets(commit)
                if not parent_jiras:
                    if commit.is_merge_to_main or shares_subject:
                        logger.debug("adding %s from child %s to %s", commit_jiras, sha, parent_sha)
                        parent.add_jiras(commit_jiras)
                    else:
                        parent.add_likely_jiras(commit_jiras)
                elif not commit_jiras:
                    logger.debug("adding %s from parent %s to %s", parent_jiras, parent_sha, sha)
                    if commit.is_merge_to_main or shares_subject:
                        commit.add_jiras(parent_jiras)
                    else:
                        commit.add_likely_jiras(parent_jiras)
    return commits_by_sha


def print_changelog(
    sha_range: ShaRange,
    version: str | None,
    use_tags: UseTags,
    files: list[str],
    out: TextIO | None,
) -> None:
    """Print a nice changelog for the repo."""
    if out is None:
        out = sys.stdout
    commits_by_sha = git_commits_by_sha(sha_range, Direction.FORWARD, files=files)
    tags_by_tag_name = get_tags_by_tag_name()
    tickets_to_summaries = load_tickets()
    commit_tree = CommitTree(commits_by_sha, tags_by_tag_name)

    versions = build_versions(version, use_tags, commit_tree)

    changes = []
    for group_version in versions:
        if not group_version.commits:
            continue
        notes = make_notes(group_version, tags_by_tag_name, tickets_to_summaries)
        if notes:
            changes.append(notes)
    if changes:
        changelog = "\n\n".join(changes)
        out.write(changelog.rstrip())
        out.write("\n")
