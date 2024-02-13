#!/usr/bin/env python3.11
import logging
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from typing import IO

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

INVALID_NAMES = {"jenkins", "mobileautomation"}

EMAILS_TO_NAMES = {}  # type: dict[str, str]


@dataclass
class Commit:
    sha: str = field(metadata={"template": "%H"})
    date: str = field(metadata={"template": "%as"})
    email: str = field(metadata={"template": "%ae"})
    name: str = field(metadata={"template": "%an"})
    refnames: str = field(metadata={"template": "%D"})
    parents: str = field(metadata={"template": "%P"})
    body: str = field(
        metadata={"template": "%B"}
    )  # body must be last since it can have multiple lines

    parent_commits: list["Commit"] = field(default_factory=list)
    additional_jiras: list[str] = field(default_factory=list)
    likely_jiras: list[str] = field(default_factory=list)

    @property
    def parent_shas(self) -> list[str]:
        if self.parents is None:
            return []
        return self.parents.split(" ")

    @property
    def tags(self) -> list[str]:
        tags = []
        refnames = self.refnames or ""
        if refnames:
            refs = refnames.split(", ")
            for reference_name in refs:
                if reference_name.startswith("tag: "):
                    tags.append(reference_name[5:])
        return tags

    @property
    def heads(self):
        heads = []
        refnames = self.refnames or ""
        if refnames:
            refs = refnames.split(", ")
            for reference_name in refs:
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
        body = self.body or ""
        return body.strip().split("\n", 1)[0]

    @property
    def description(self) -> str:
        body = self.body or ""
        body = body.strip()
        if "\n" in body:
            return self.body.split("\n", 1)[1]
        else:
            return ""

    @property
    def clean_body(self):
        body = self.body.strip("\n")
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


EMPTY_COMMIT = Commit("", "", "", "", "", "", "", [], [])


GIT_FORMAT = "\n".join(
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
    line = commit.subject or ""
    line = strip_line(line)
    for x in _ignores:
        line = line.replace(x, " ")
    for x in _ignore_matches:
        line = re.sub(x, " ", line)
    tags = commit.tags or []
    tags.sort(key=len, reverse=True)
    for tag in commit.tags or []:
        line = line.replace(tag, "")
    if not re.match(r"\w", line):
        line = ""
    line = re.sub(r"\s+", " ", line)
    line = line.strip()
    line = add_jiras(line, commit.jiras)
    return line


def strip_line(line: str | None) -> str:
    if line is None:
        line = ""
    line = line.strip()
    line = re.sub(r"^(\** *)*", "", line)
    line = re.sub(r"^(-* *)*", "", line)
    line = re.sub(r"^Pull request #[0-9]+: +", "", line, flags=re.IGNORECASE)
    line = re.sub(r"^feature/", "", line, flags=re.IGNORECASE)
    line = re.sub(r"^bugfix/", "", line, flags=re.IGNORECASE)
    return line


def add_star(line: str | None) -> str:
    line = strip_line(line)
    if line:
        return "* %s" % line
    return line


def format_jira(line) -> str:
    if line is None:
        return ""
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
    if line is None:
        return ""
    line = line.strip()
    if len(line) < 1:
        return line
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


def format_tags(tags: list[str] | None) -> str:
    if not tags:
        return ""
    tags = sorted(tags, key=best_tag)
    tags_line = ", ".join([f"`{tag}`" for tag in tags])
    return f"### tags: {tags_line}"


def best_tag(tag: str) -> tuple[bool, bool, bool, int, str]:
    if tag is None:
        tag = ""
    has_snapshot = "SNAPSHOT" in tag
    has_semantic_version = bool(re.match(r"^[0-9]+(\.[0-9]+){1,2}$", tag))
    has_semantic_subversion = bool(re.match(r"^[0-9]+(\.[0-9]+){1,2}(-.*)?$", tag))
    return (
        has_snapshot,
        not has_semantic_version,
        not has_semantic_subversion,
        len(tag),
        tag,
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
    if body is None:
        return None
    lines = body.rstrip().split("\n")
    lines = [line for line in lines if include_line(line)]
    lines = [line for line in lines if line is not None]
    lines = [add_jiras(line, jiras) for line in lines if line is not None]
    lines = [add_star(line) for line in lines if line is not None]
    lines = [format_jira(line) for line in lines if line is not None]
    lines = [line for line in lines if line is not None]
    return "\n".join(lines)


def valid_name(name: str | None) -> bool:
    if name is None:
        return False
    name = name.strip()
    if len(name) < 1:
        return False
    name = name.lower()
    if "jenkins builder" in name:
        return False
    if name in INVALID_NAMES:
        return False
    return True


def smart_name(commit: Commit):
    name = commit.name
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
    version_string = f"{release_version}".strip()

    if commits:
        first_commit = commits[0]
        date = first_commit.date
        parts = ["##"]
        if version_string.startswith("v"):
            parts.append("v.")
        parts.append(version_string.strip())

        if not date.startswith(version_string) and date not in version_string:
            parts.append(f"({date})")
        return " ".join(parts)
    return ""


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


def make_notes(release_version: str, commits: list[Commit]):
    version_line = make_version_line(release_version, commits)
    release_note = []
    if version_line is not None and len(version_line):
        release_note.append(version_line)
    tags_notes = []

    months_by_tag = {}  # type: dict[str, dict[str, list[Commit]]]
    current_tag = ""
    for commit in commits:
        current_month = commit.month
        if commit.tags:
            current_tag = format_tags(commit.tags)
        if current_tag not in months_by_tag:
            months_by_tag[current_tag] = {}
        commits_by_month = months_by_tag[current_tag]
        if current_month not in commits_by_month:
            commits_by_month[current_month] = []
        current_month_commits = commits_by_month[current_month]
        current_month_commits.append(commit)

    for tags, commits_by_month in months_by_tag.items():
        if len(tags):
            tags_notes.append(tags)
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
        release_note.extend("\n".join(tags_notes).split("\n"))
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
            key=lambda sha: commits_by_sha.get(sha, EMPTY_COMMIT).date,
        )
        print(" ".join(line))


def stream_chunks(io: IO[bytes] | None, separator: str = "\n") -> Iterator[str]:
    accumulated = ""
    keep_going = True
    while io is not None and io.readable() and keep_going:
        read_chunk = io.read(1024)
        if read_chunk == b"":
            keep_going = False
        decoded_chunk = read_chunk.decode("UTF8")
        accumulated = f"{accumulated}{decoded_chunk}"
        while separator in accumulated:
            chunk, accumulated = accumulated.split(separator, 1)
            yield chunk
    yield accumulated


def git_log(
    from_sha: str | None, from_inclusive: bool | None, to_sha: str | None, to_inclusive: bool | None
) -> Iterator[str]:
    command = ["git", "log", "-z", f"--format={GIT_FORMAT}"]
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
    with subprocess.Popen(command, stdout=subprocess.PIPE) as proc:
        yield from stream_chunks(proc.stdout, "\x00")


def first_sha() -> str:
    command = ["git", "hash-object", "-t", "tree", "/dev/null"]
    result = subprocess.run(command, stdout=subprocess.PIPE)
    return result.stdout.decode("UTF8").strip()


def chunk_to_commit(chunk: str) -> Commit | None:
    if chunk is None:
        return None
    if "\nbody" not in chunk:
        logger.debug("body not in commit %s", chunk)
        return None
    header, body = chunk.split("\nbody", 1)
    commit_data = {"body": body}  # type: dict[str, str | None]
    for line in header.split("\n"):
        key, value = line.split(" ", 1)
        commit_data[key] = value
    if not commit_data.get("sha"):
        logger.debug("No sha in commit %s", chunk)
        return None

    return Commit(**commit_data)  # type: ignore


def print_changelog(
    from_sha: str | None = None,
    from_inclusive: bool | None = False,
    to_sha: str | None = None,
    to_inclusive: bool | None = False,
    version: str | None = None,
    use_tags: bool | None = False,
):
    commits_by_sha = {}  # type: dict[str, Commit]
    for chunk in git_log(from_sha, from_inclusive, to_sha, to_inclusive):
        commit = chunk_to_commit(chunk)
        if commit is None:
            logger.debug("None commit")
            continue
        commits_by_sha[commit.sha] = commit

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
                commit.parent_commits.append(parent)

    group_name = version or "Unknown"
    group_commits = {group_name: []}  # type: dict[str, list[Commit]]

    groups = [group_name]

    release_dates = {}  # type: dict[str, str]

    use_tags = use_tags or False
    found_version = False
    for sha, commit in commits_by_sha.items():
        version = commit.version
        commit_tags = commit.tags
        if version:
            group_name = version
            found_version = True
        elif use_tags and commit_tags:
            candidate_version = tags_to_release_version(commit_tags, found_version)
            if candidate_version:
                group_name = candidate_version

        if group_name not in group_commits:
            groups.append(group_name)
            group_commits[group_name] = []
        group_commits[group_name].append(commit)
        current_month = commit.month
        release_month = release_dates.get(group_name)
        if release_month is None or current_month > release_month:
            release_dates[group_name] = current_month

    changes = []
    for group_name in sorted(groups, key=lambda v: release_dates.get(v) or v, reverse=True):
        commits = group_commits.get(group_name) or []
        if not commits:
            continue
        notes = make_notes(group_name, commits)
        if notes:
            changes.append(notes)

    if changes:
        # changes = reversed(changes)
        print("\n\n".join(changes))
        print("\n")
