#!/usr/bin/env python3.11
import logging
import re
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from typing import IO, Any, TextIO

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
]

_delete_regexes = [
    r"^\s*Pull request #[0-9]+\s*:\s*",
    r"^(\** *)*",
    r"^(-* *)*",
    r"^feature/",
    r"^bugfix/",
]

INVALID_NAMES = {"jenkins", "mobileautomation"}

EMAILS_TO_NAMES = {}  # type: dict[str, str]


def none_as_empty(string: str | None) -> str:
    if string is None:
        return ""
    return string


def none_as_empty_stripped(string: str | None) -> str:
    return none_as_empty(string).strip()


def is_empty(string: str | None) -> bool:
    if string is None:
        return True
    return len(string.strip()) == 0


@dataclass
class Tag:
    sha_sha: str = field(metadata={"template": "%(*objectname):%(objectname)"})
    ref_name: str = field(metadata={"template": "%(refname:short)"})
    tagger_date: str = field(metadata={"template": "%(taggerdate:iso8601)"})
    subject: str = field(metadata={"template": "%(subject)"})
    body: str = field(metadata={"template": "%(body)"})

    @property
    def sha(self):
        parts = self.sha_sha.split(":")
        if len(parts[0]):
            return parts[0]
        else:
            return parts[1]

    def is_annotated(self):
        return any(not is_empty(x) for x in [self.subject, self.body])


@dataclass
class Commit:
    sha: str = field(metadata={"template": "%H"})
    date: str = field(metadata={"template": "%as"})
    email: str = field(metadata={"template": "%aE"})
    name: str = field(metadata={"template": "%aN"})
    refnames: str = field(metadata={"template": "%D"})
    parents: str = field(metadata={"template": "%P"})
    body: str = field(
        metadata={"template": "%B"}
    )  # body must be last since it can have multiple lines

    additional_jiras: list[str] = field(default_factory=list)
    likely_jiras: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls):
        return cls(
            sha="",
            date="",
            email="",
            name="",
            refnames="",
            parents="",
            body="",
            additional_jiras=[],
            likely_jiras=[],
        )

    @property
    def parent_shas(self) -> list[str]:
        if self.parents is None:
            return []
        return self.parents.split(" ")

    @property
    def tag_names(self) -> list[str]:
        tag_names = []
        refnames = none_as_empty_stripped(self.refnames)
        if not is_empty(refnames):
            refs = refnames.split(",")
            for reference_name in refs:
                reference_name = reference_name.strip()
                if reference_name.startswith("tag: "):
                    tag_names.append(reference_name.removeprefix("tag: "))
        return tag_names

    @property
    def heads(self):
        heads = []
        refnames = none_as_empty_stripped(self.refnames)
        if refnames:
            refs = refnames.split(",")
            for reference_name in refs:
                reference_name = reference_name.strip()
                if not reference_name.startswith("tag: "):
                    heads.append(reference_name)
        return heads

    @property
    def jiras(self) -> list[str]:
        return extract_jiras(self.body) + self.additional_jiras

    def add_jiras(self, jiras: list[str]):
        if jiras is not None:
            self.additional_jiras.extend(jiras)

    def add_likely_jiras(self, jiras: list[str]):
        if jiras is not None:
            self.likely_jiras.extend(jiras)

    @property
    def subject(self) -> str:
        body = none_as_empty_stripped(self.body)
        return body.split("\n", 1)[0]

    @property
    def description(self) -> str:
        body = none_as_empty_stripped(self.body)
        if "\n" in body:
            return self.body.split("\n", 1)[1]
        else:
            return ""

    @property
    def clean_body(self) -> str:
        body = none_as_empty(self.body).strip("\n")
        lines = body.split("\n")
        lines = [line.rstrip() for line in lines]
        lines = [line.strip("\n") for line in lines]
        lines = [line for line in lines if include_line(line)]
        output = "\n".join(lines)
        output = re.sub(r"\n+", "\n", output)
        return output.strip("\n")

    @property
    def is_merge_to_main(self) -> bool:
        return bool(re.search("Merge.*to (master|main)", self.body))

    @property
    def version(self) -> str | None:
        matches = re.search(r"pre tag commit.*'(.*)'", self.subject)
        if matches:
            return matches.group(1)
        return None

    @property
    def month(self) -> str:
        date = self.date
        if date is not None:
            return date[:7]
        return ""


@dataclass
class Month:
    name: str
    commits: list


@dataclass
class Version:
    version_name: str
    commits: list[Commit]


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
    f"{commit_field.name} {commit_field.metadata['template']}"
    for commit_field in fields(Commit)
    if commit_field.metadata.get("template") is not None
)


def include_line(line: str | None) -> bool:
    return (
        line is not None
        and not any(x in line for x in _ignores)
        and not any(re.search(regex, line) for regex in _ignore_matches)
    )


def format_for_tag_only(commit: Commit) -> str:
    line = none_as_empty_stripped(commit.subject)
    line = delete_junk(line)
    for x in _ignores:
        line = line.replace(x, " ")
    for x in _ignore_matches:
        line = re.sub(x, " ", line)
    tags = commit.tag_names or []
    tags.sort(key=len, reverse=True)
    for tag in commit.tag_names or []:
        line = line.replace(tag, "")
    if not re.match(r"\w", line):
        line = ""
    line = re.sub(r"\s+", " ", line)
    line = line.strip()
    line = add_jiras(line, commit.jiras)
    return line


def delete_junk(line: str) -> str:
    for regex in _delete_regexes:
        line = re.sub(regex, "", line, flags=re.IGNORECASE)
    return line


def add_star(line: str | None) -> str:
    line = none_as_empty_stripped(line)
    if line:
        return "* %s" % line
    return line


def format_jira(maybe_line: str | None) -> str:
    maybe_line = none_as_empty_stripped(maybe_line)
    if is_empty(maybe_line):
        return ""
    line: str = maybe_line
    jiras = extract_jiras(line)
    if jiras:
        for jira in jiras:
            line = line.replace(jira, "")
        line = re.sub(r"^\W+", "", line)
        joined = ", ".join(sorted(jiras))
        if len(line):
            line = f"* {joined} : {line}"
        else:
            line = f"* {joined}"
    return line


def extract_jiras(body: str | None) -> list[str]:
    if body is None:
        return []
    return list(set(re.findall("[A-Z]+-[0-9]+", body) or []))


def add_jiras(line: str, jiras: list[str]) -> str:
    line = none_as_empty_stripped(line)
    if is_empty(line):
        return ""
    has_jiras = extract_jiras(line)
    if has_jiras:
        return line
    missing_jiras = list(set([jira for jira in jiras if jira not in line]))
    if missing_jiras:
        joined = ", ".join(missing_jiras)
        line = f"{joined} : {line}"
    return line


def unique(items: list | None) -> list:
    if items is None:
        return []
    seen = set()
    output = []
    for item in items or []:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output


def format_tag_names(tags: list[str] | None) -> str:
    if not tags:
        return ""
    tags = sorted(tags, key=best_tag)
    tags_line = ", ".join([f"`{tag}`" for tag in tags])
    return f"### Tags: {tags_line}"


def best_tag(tag_name: str) -> tuple[bool, bool, bool, int, str]:
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


def tags_to_release_version(tags: list[str], found_version) -> str | None:
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


def format_body(body: str | None, jiras: list[str]) -> str | None:
    body = none_as_empty(body)
    if is_empty(body):
        return None
    lines = body.rstrip().split("\n")
    lines = [line for line in lines if include_line(line)]
    lines = [delete_junk(line) for line in lines]
    lines = [add_jiras(line, jiras) for line in lines]
    lines = [add_star(line) for line in lines]
    lines = [format_jira(line) for line in lines]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def valid_name(name: str | None) -> bool:
    name = none_as_empty_stripped(name)
    if is_empty(name):
        return False
    name = name.lower()
    if "jenkins builder" in name:
        return False
    if name in INVALID_NAMES:
        return False
    return True


def smart_name(commit: Commit):
    name = none_as_empty_stripped(commit.name)
    if name:
        return name
    email = commit.email
    if email in EMAILS_TO_NAMES:
        return EMAILS_TO_NAMES[email]


def format_names(commits: list[Commit]) -> str | None:
    if not commits:
        return None
    names = set(smart_name(commit) for commit in commits)
    names = set(name for name in names if valid_name(name))
    if not names:
        return None
    return ", ".join(sorted(names))


def make_version_line(release_version: str, commits: list[Commit]) -> str:
    """Use the version string and the first commit to make a version string like
    v.1234 (2024-04)
    """
    version_string = none_as_empty_stripped(release_version)
    if is_empty(version_string):
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


def fill_text(text: str, width: int, indent: str):
    """Strip any extra indentation, then reindent and then wrap each line individually."""
    text = textwrap.dedent(text)
    text = "\n".join(x.rstrip() for x in text.split("\n"))
    text = text.strip("\n")
    text = re.sub(r"\n\n\n+", "\n\n", text)
    texts = text.splitlines()
    texts = [textwrap.fill(line, width) for line in texts]  # Wrap each line
    text = "\n".join(texts)
    text = textwrap.indent(text, indent)
    return text


def format_commit(commit: Commit) -> list[str]:
    subject = commit.subject
    if not include_line(subject):
        return []

    commit_lines = []
    subject_line = format_body(subject, commit.jiras)
    if subject_line:
        commit_lines.append(subject_line)
    body = format_body(commit.body, commit.jiras)
    if body:
        commit_lines.append(body)
    return "\n".join(commit_lines).split("\n")


def format_annotated_tag(tag: Tag) -> str | None:
    if tag is None:
        return None
    ref_name = none_as_empty(tag.ref_name)
    if is_empty(ref_name):
        # this should never happen
        return None
    subject = none_as_empty_stripped(tag.subject)
    body = none_as_empty_stripped(tag.body).strip()
    tag_parts = []
    if is_empty(subject):
        tag_parts.append(f"## Tag: `{ref_name}`")
    else:
        tag_parts.append(
            textwrap.fill(
                f"## Tag: `{ref_name}` {subject}", 80, initial_indent="", subsequent_indent="    "
            )
        )
    if not is_empty(body):
        body = fill_text(body, 80, "    ")
        tag_parts.append(body)
    return "\n".join(tag_parts).rstrip()


def make_notes(version: Version, tags_by_tag_name: dict[str, Tag]) -> str:
    version_line = make_version_line(version.version_name, version.commits)
    release_note = []
    if version_line is not None and len(version_line):
        release_note.append(version_line)
    tags_notes = []
    tags_by_format = {}
    months_by_tag: dict[str, dict[str, list[Commit]]] = {}
    current_tag = ""
    for commit in version.commits:
        current_month = commit.month
        if commit.tag_names:
            current_tag = format_tag_names(commit.tag_names)
            tags_by_format[current_tag] = commit.tag_names
        if current_tag not in months_by_tag:
            months_by_tag[current_tag] = {}
        commits_by_month = months_by_tag[current_tag]
        if current_month not in commits_by_month:
            commits_by_month[current_month] = []
        current_month_commits = commits_by_month[current_month]
        current_month_commits.append(commit)

    for formatted_tags, commits_by_month in months_by_tag.items():
        if not is_empty(formatted_tags):
            tags_notes.append("")
            tag_names = tags_by_format[formatted_tags]
            tags = {
                tag_name: tags_by_tag_name[tag_name]
                for tag_name in tag_names
                if tag_name in tags_by_tag_name
            }
            annotated_tags = {tag_name: tag for tag_name, tag in tags.items() if tag.is_annotated()}
            for tag in annotated_tags.values():
                tag_note = format_annotated_tag(tag)
                if tag_note is not None:
                    tags_notes.append(tag_note)
                    if "\n" in tag_note:
                        tags_notes.append("")
            remaining_tag_names = [
                tag_name for tag_name in tag_names if tag_name not in annotated_tags
            ]
            if remaining_tag_names:
                reformatted = format_tag_names(remaining_tag_names)
                tags_notes.append(reformatted)
        first_month_in_tag = True
        for month, month_commits in commits_by_month.items():
            month_commit_lines = []
            for commit in month_commits:
                commit_lines = format_commit(commit)
                if len(commit_lines):
                    for commit_line in commit_lines:
                        if commit_line not in tags_notes:
                            month_commit_lines.append(commit_line)
            month_commit_lines = unique(month_commit_lines)
            if month_commit_lines:
                if month not in version_line:
                    if not first_month_in_tag:
                        tags_notes.append("")
                    tags_notes.append(f"### {month}")
                formatted_names = format_names(month_commits)
                if formatted_names is not None:
                    tags_notes.append(formatted_names)
                tags_notes.extend(month_commit_lines)
            first_month_in_tag = False
    if tags_notes:
        release_note.extend(tags_notes)
    return "\n".join(release_note).strip()


def prune_heads(heads: list[str]) -> list[str]:
    new_heads = []
    for head in heads:
        if head.startswith("HEAD ->"):
            head = head[8:]
        elif head.endswith("/HEAD"):
            head = head[:-5]
        if head:
            new_heads.append(head)
    return new_heads


def extract_refs(commit: Commit) -> dict[str, list[str]]:
    sha = commit.sha
    parent_shas = commit.parent_shas or []
    heads = commit.heads or []
    heads = prune_heads(heads)
    shas_to_refs = {sha: heads}

    merge_matches = re.search(r"Merge .* from ([^ ]+) to ([^ ]+)$", commit.subject)
    if merge_matches:
        from_ref = merge_matches.group(1)
        to_ref = merge_matches.group(2)
        if parent_shas and len(parent_shas) == 2:
            left, right = parent_shas
            logger.debug(f"{sha} : {right} {from_ref} + {left} {to_ref}")
            if right not in shas_to_refs:
                shas_to_refs[right] = []
            shas_to_refs[right].append(from_ref)
            if left not in shas_to_refs:
                shas_to_refs[left] = []
            shas_to_refs[left].append(to_ref)
    return shas_to_refs


def print_tree(commits_by_sha: dict[str, Commit], refs_by_sha: dict[str, list[str]]):
    if not commits_by_sha:
        return
    head = list(commits_by_sha.keys())[0]
    current_branches = [head]
    seen = set()
    while current_branches:
        new_current_branches = []
        line = []
        for head in current_branches:
            if head in seen:
                continue
            seen.add(head)
            commit = commits_by_sha.get(head)
            if commit is not None:
                refs = refs_by_sha.get(head) or []
                node = ",".join(refs) or head[:8]
                line.append(node)
                new_current_branches.extend(commit.parent_shas or [])
        current_branches = sorted(
            new_current_branches,
            key=lambda sha: commits_by_sha.get(sha, Commit.empty()).date,
        )
        print(" ".join(line))


def stream_chunks(io: IO[bytes] | None, separator: str = "\n") -> Iterator[str]:
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


def chunk_command(args: Any):
    """A mockable command"""
    with subprocess.Popen(args, stdout=subprocess.PIPE) as proc:
        yield from stream_chunks(proc.stdout, "\x00")


def git_tags() -> dict[str, Tag]:
    command = ["git", "for-each-ref", f"--format={GIT_TAG_FORMAT}", "refs/tags"]
    tags_by_tag: dict[str, Tag] = {}
    for chunk in chunk_command(command):
        tag = chunk_to_tag(chunk)
        if tag is not None:
            tags_by_tag[tag.ref_name] = tag
    return tags_by_tag


def git_log(
    git_format: str,
    from_sha: str | None,
    from_inclusive: bool | None,
    to_sha: str | None,
    to_inclusive: bool | None,
    reversed: bool | None = None,
    files: list[str] | None = None,
) -> Iterator[str]:
    command = ["git", "log", "-z", f"--format={git_format}"]
    if to_inclusive:
        to_caret = ""
    else:
        to_caret = "^"
    if from_sha is not None:
        if from_inclusive:
            from_caret = "^"
        else:
            from_caret = ""
        if to_sha is None:
            to_sha = "HEAD"
            to_caret = ""
        sha_range = f"{from_sha}{from_caret}..{to_sha}{to_caret}"
        command.append(sha_range)
    elif to_sha is not None:
        from_sha = first_sha()
        sha_range = f"{from_sha}..{to_sha}{to_caret}"
        command.append(sha_range)
    if reversed:
        command.append("--reverse")
    if files:
        command.extend(files)
    yield from chunk_command(command)


def git_commits_by_sha(
    from_sha: str | None,
    from_inclusive: bool | None,
    to_sha: str | None,
    to_inclusive: bool | None,
    files: list[str] | None = None,
) -> dict[str, Commit]:
    commits_by_sha = {}  # type: dict[str, Commit]
    for chunk in git_log(
        GIT_COMMIT_FORMAT, from_sha, from_inclusive, to_sha, to_inclusive, files=files
    ):
        commit = chunk_to_commit(chunk)
        if commit is None:
            logger.debug("None commit")
            continue
        commits_by_sha[commit.sha] = commit
    return commits_by_sha


def first_sha() -> str:
    command = ["git", "hash-object", "-t", "tree", "/dev/null"]
    result = subprocess.run(command, stdout=subprocess.PIPE)
    return result.stdout.decode("UTF8").strip()


def extract_header_fields(header: str) -> dict[str, str]:
    if is_empty(header):
        return {}
    data = {}
    for line in header.split("\n"):
        if is_empty(line):
            continue
        key, value = line.split(" ", 1)
        data[key] = value
    return data


def chunk_to_tag(chunk: str) -> Tag | None:
    if chunk is None:
        return None
    if "\nbody" not in chunk:
        logger.debug("body not in tag %r", chunk)
        return None
    header, body = chunk.split("\nbody", 1)
    tag_data = {"body": body.removeprefix(" ").rstrip()}
    tag_data.update(extract_header_fields(header))
    sha_sha = tag_data.get("sha_sha")
    if is_empty(sha_sha):
        logger.debug("No sha_sha in commit %s", chunk)
        return None
    if sha_sha is not None:
        # mypy can't tell that not is_empty means not None :(
        parts = sha_sha.split(":")
        if len(parts) == 0 or is_empty(parts[0]):
            # looks like this is actually not an annotated tag
            tag_data["subject"] = ""
            tag_data["body"] = ""
    return Tag(**tag_data)  # type: ignore[arg-type]


def chunk_to_commit(chunk: str) -> Commit | None:
    if chunk is None:
        return None
    if "\nbody" not in chunk:
        logger.debug("body not in commit %r", chunk)
        return None
    header, body = chunk.split("\nbody", 1)
    commit_data = {"body": body.removeprefix(" ").rstrip()}
    commit_data.update(extract_header_fields(header))
    if not commit_data.get("sha"):
        logger.debug("No sha in commit %s", chunk)
        return None

    return Commit(**commit_data)  # type: ignore[arg-type]


def print_changelog(
    from_sha: str | None = None,
    from_inclusive: bool | None = False,
    to_sha: str | None = None,
    to_inclusive: bool | None = False,
    version: str | None = None,
    use_tags: bool | None = False,
    files: list[str] | None = None,
    out: TextIO | None = sys.stdout,
):
    if out is None:
        out = sys.stdout
    tags_by_tag_name = git_tags()
    commits_by_sha = git_commits_by_sha(from_sha, from_inclusive, to_sha, to_inclusive, files=files)

    shas_to_refs = {}  # type: dict[str, list[str]]
    child_shas = {}  # type: dict[str, list[str]]
    for sha, commit in commits_by_sha.items():
        if sha not in child_shas:
            child_shas[sha] = []
        for parent_sha in commit.parent_shas or []:
            if parent_sha not in child_shas:
                child_shas[parent_sha] = []
            child_shas[parent_sha].append(sha)
        commit_shas_to_refs = extract_refs(commit)
        for commit_sha, refs in commit_shas_to_refs.items():
            if refs:
                if commit_sha not in shas_to_refs:
                    shas_to_refs[commit_sha] = []
                shas_to_refs[commit_sha].extend(refs)
    for sha, commit in commits_by_sha.items():
        refs = shas_to_refs.get(sha) or []
        children = child_shas.get(sha) or []
        walked = []
        while not refs and len(children) == 1:
            child_sha = children[0]
            walked.append(child_sha)
            refs = shas_to_refs.get(child_sha) or []
            children = child_shas.get(child_sha) or []
        if refs:
            for child_sha in walked:
                if not shas_to_refs.get(child_sha):
                    shas_to_refs[child_sha] = refs
        parent_shas = commit.parent_shas or []
        walked = []
        while not refs and len(parent_shas) == 1:
            parent_sha = parent_shas[0]
            walked.append(parent_sha)
            refs = shas_to_refs.get(parent_sha) or []
            parent = commits_by_sha.get(parent_sha)
            parent_shas = (parent and parent.parent_shas) or []
        if refs:
            for parent_sha in walked:
                if not shas_to_refs.get(parent_sha):
                    shas_to_refs[parent_sha] = refs
        shas_to_refs[sha] = refs

    # print_tree(commits_by_sha, shas_to_refs)

    for sha, commit in commits_by_sha.items():
        parent_shas = commit.parent_shas or []
        if commit.is_merge_to_main and len(parent_shas) in (1, 2):
            parent_sha = parent_shas[-1]
            parent = commits_by_sha.get(parent_sha)

            if parent is not None:
                shares_subject = commit.subject in parent.body

                if not parent.jiras:
                    if commit.is_merge_to_main or shares_subject:
                        logger.debug(f"adding {commit.jiras} from child {sha} to {parent_sha}")
                        parent.add_jiras(commit.jiras)
                    else:
                        parent.add_likely_jiras(commit.jiras)
                elif not commit.jiras:
                    logger.debug(f"adding {commit.jiras} from parent {parent_sha} to {sha}")
                    if commit.is_merge_to_main or shares_subject:
                        commit.add_jiras(parent.jiras)
                    else:
                        commit.add_likely_jiras(parent.jiras)

    version_tree: dict[str, list[Commit]] = {}

    group_name = version or "Unknown"

    use_tags = use_tags or False
    found_version = False
    for sha, commit in commits_by_sha.items():
        commit_tags = commit.tag_names
        if commit.version:
            group_name = commit.version
            found_version = True
        elif use_tags and commit_tags:
            candidate_version = tags_to_release_version(commit_tags, found_version)
            if candidate_version:
                group_name = candidate_version

        if group_name not in version_tree:
            version_tree[group_name] = []
        version_tree[group_name].append(commit)

    versions: list[Version] = []
    for group_name, commits in version_tree.items():
        if not commits:
            continue
        group_version = Version(group_name, commits)
        versions.append(group_version)

    changes = []
    for group_version in versions:
        if not group_version.commits:
            continue
        notes = make_notes(group_version, tags_by_tag_name)
        if notes:
            changes.append(notes)
    if changes:
        # changes = reversed(changes)
        changelog = "\n\n".join(changes)
        out.write(changelog.rstrip())
        out.write("\n")
