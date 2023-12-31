#!/usr/bin/env python3.11
import logging
import re
import subprocess
from dataclasses import dataclass, field, fields
from typing import Dict, IO, List, Optional, Tuple

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
    "merge to master",
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

EMAILS_TO_NAMES = {}


@dataclass
class Commit:
    sha: str = field(metadata={"template": "%H"})
    date: str = field(metadata={"template": "%as"})
    email: str = field(metadata={"template": "%ae"})
    name: str = field(metadata={"template": "%an"})
    refnames: str = field(metadata={"template": "%D"})
    parents: str = field(metadata={"template": "%P"})
    body: str = field(metadata={"template": "%B"})  # body must be last since it can have multiple lines

    parent_commits: List["Commit"] = field(default_factory=list)
    additional_jiras: List[str] = field(default_factory=list)
    likely_jiras: List[str] = field(default_factory=list)

    @property
    def parent_shas(self) -> List[str]:
        if self.parents is None:
            return []
        return self.parents.split(" ")

    @property
    def tags(self) -> List[str]:
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
    def jiras(self) -> List[str]:
        return extract_jiras(self.body) + self.additional_jiras

    def add_jiras(self, jiras: List[str]):
        if jiras is not None:
            self.additional_jiras.extend(jiras)

    def add_likely_jiras(self, jiras: List[str]):
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
    def version(self) -> Optional[str]:
        matches = re.search(r"pre tag commit.*'(.*)'", self.subject)
        if matches:
            return matches.group(1)

    @property
    def month(self) -> Optional[str]:
        date = self.date
        if date is not None:
            return date[:7]


GIT_FORMAT = "\n".join(
    f"{field.name} {field.metadata['template']}"
    for field in fields(Commit)
    if field.metadata.get('template') is not None
)


def include_line(line: Optional[str]) -> bool:
    return (
        line is not None
        and not any(x in line for x in _ignores)
        and not any(re.search(regex, line) for regex in _ignore_matches)
    )


def test_include_line():
    includes = ["yes", "me", ""]
    for line in includes:
        assert include_line(line)

    not_includes = [
        None,
        "* commit to ignore",
        "git-p4",
        "Merge branch 'CONTIN-5792-refactor-backfill'",
    ]
    for line in not_includes:
        assert not include_line(line)


def format_for_tag_only(commit: Commit) -> str:
    line = commit.subject
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


def strip_line(line: str) -> str:
    if line:
        line = line.strip()
        line = re.sub(r"^(\** *)*", "", line)
        line = re.sub(r"^(-* *)*", "", line)
        line = re.sub(r"^Pull request #[0-9]+: +", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^feature/", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^bugfix/", "", line, flags=re.IGNORECASE)
        return line


def add_star(line: Optional[str]) -> Optional[str]:
    line = strip_line(line)
    if line:
        return "* %s" % line


def format_jira(line) -> Optional[str]:
    if line:
        jiras = extract_jiras(line)
        if jiras:
            for jira in jiras:
                line = line.replace(jira, "")
            line = re.sub(r"^\W+", "", line)
            jiras = ", ".join(sorted(jiras))
            if len(line):
                line = f"* {jiras} : {line}"
            else:
                line = f"* {jiras}"
    return line


def test_format_jira():
    expectations = {
        "* FOOBAR-1637 last": "* FOOBAR-1637 : last",
        "* BAZZ-2733 :     ": "* BAZZ-2733",
        "* PIRATE-6206 - New ": "* PIRATE-6206 : New ",
        "* PIRATE-6206- New ": "* PIRATE-6206 : New ",
        "* PIRATE-6206 -New ": "* PIRATE-6206 : New ",
        "* PIRATE-6206-New ": "* PIRATE-6206 : New ",
        "* LEAF-5410, LEAF-5316 :   More cleanup, tests": "* LEAF-5316, LEAF-5410 : More cleanup, tests",
        "* A-5316, B-5316 : sorting": "* A-5316, B-5316 : sorting",
        "* B-5316, A-5316 : sorting": "* A-5316, B-5316 : sorting",
        "* LEAF-5410 :   More cleanup, LEAF-5316 ,tests": "* LEAF-5316, LEAF-5410 : More cleanup,  ,tests",
    }
    for line, expected in expectations.items():
        assert format_jira(line) == expected


def extract_jiras(body):
    return list(set(re.findall("[A-Z]+-[0-9]+", body) or []))


def add_jiras(line: str, jiras: List[str]) -> str:
    if not line:
        return line
    has_jiras = extract_jiras(line)
    if has_jiras:
        return line
    missing_jiras = list(set([jira for jira in jiras if jira not in line]))
    if missing_jiras:
        jiras = ", ".join(missing_jiras)
        line = f"{jiras} : {line}"
    return line


def unique(items: List) -> List:
    seen = set()
    output = []
    for item in items or []:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output


def format_tags(tags: List[str]) -> str:
    if not tags:
        return ""
    tags = sorted(tags, key=best_tag)
    tags_line = ", ".join([f"`{tag}`" for tag in tags])
    return f"### tags: {tags_line}"


def best_tag(tag: str) -> Tuple[bool, bool, bool, int, str]:
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


def tags_to_release_version(tags: List[str], found_version) -> Optional[str]:
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


def format_body(body: Optional[str], jiras: List[str]) -> Optional[str]:
    if body is None:
        return None
    lines = body.rstrip().split("\n")
    lines = [line for line in lines if include_line(line)]
    lines = [line for line in lines if line is not None]
    lines = [add_jiras(line, jiras) for line in lines]
    lines = [add_star(line) for line in lines]
    lines = [format_jira(line) for line in lines]
    lines = [line for line in lines if line is not None]
    return "\n".join(lines)


def valid_name(name: Optional[str]) -> bool:
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



def format_names(commits: List[Commit]) -> Optional[str]:
    if not commits:
        return None
    names = set(smart_name(commit) for commit in commits)
    names = set(name for name in names if valid_name(name))
    if not names:
        return None
    return ", ".join(sorted(names))


def make_version_line(release_version: str, commits: List[Commit]) -> str:
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


def format_commit(commit: Commit) -> List[str]:
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


def make_notes(release_version: str, commits: List[Commit]):
    version_line = make_version_line(release_version, commits)
    release_note = [version_line]
    tags_notes = []

    months_by_tag = {}
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
        tags_notes = "\n".join(tags_notes).split("\n")
        # tags_notes = unique(tags_notes)
        release_note.append("\n".join(tags_notes))
        release_note.append("")
        return "\n".join(release_note).strip()


def prune_heads(heads: List[str]) -> List[str]:
    new_heads = []
    for head in heads:
        if head.startswith("HEAD ->"):
            head = head[8:]
        elif head.endswith("/HEAD"):
            head = head[:-5]
        if head:
            new_heads.append(head)
    return new_heads


def extract_refs(commit: Commit) -> Dict[str, List[str]]:
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


def test_extract_refs():
    commit_data = {"sha": "abcd", "subject": ""}
    refs = extract_refs(Commit(**commit_data))
    assert {"abcd": []} == refs

    commit_data = {
        "sha": "abcd",
        "parent_shas": ["aaaa", "bbbb"],
        "tags": [],
        "heads": [],
        "subject": "Pull request #21: anything",
    }
    refs = extract_refs(Commit(**commit_data))
    assert {"abcd": []} == refs

    commit_data = {
        "sha": "abcd",
        "parent_shas": [
            "aaaa",
            "bbbb",
        ],
        "tags": [],
        "heads": [],
        "subject": "Merge branch 'feature/branch_name'",
    }
    refs = extract_refs(Commit(**commit_data))
    assert {"abcd": []} == refs

    commit_data = {
        "sha": "abcd",
        "parent_shas": [
            "aaaa",
            "bbbb",
        ],
        "tags": [],
        "heads": [],
        "subject": "Merge pull request #1 in anything from branch_name to master",
    }
    refs = extract_refs(Commit(**commit_data))
    assert {"abcd": []} == refs


def print_tree(commits_by_sha: Dict[str, Commit], refs_by_sha: Dict[str, List[str]]):
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
            key=lambda sha: commits_by_sha.get(sha, {}).get("date"),
        )
        print(" ".join(line))


def stream_chunks(io: IO, separator: str = "\n"):
    accumulated = ""
    keep_going = True
    while io.readable() and keep_going:
        read_chunk = io.read(1024)
        if read_chunk == b"":
            keep_going = False
        read_chunk = read_chunk.decode("UTF8")
        accumulated = f"{accumulated}{read_chunk}"
        while separator in accumulated:
            chunk, accumulated = accumulated.split(separator, 1)
            yield chunk
    yield accumulated


def git_log(from_sha: str, to_sha: str):

    command = ["git", "log", "-z", f"--format={GIT_FORMAT}"]
    if from_sha is not None:
        if to_sha is None:
            to_sha = "HEAD"
        sha_range = f"{from_sha}^..{to_sha}"
        command.append(sha_range)
    elif to_sha is not None:
        from_sha = first_sha()
        sha_range = f"{from_sha}..{to_sha}"
        command.append(sha_range)
    with subprocess.Popen(command, stdout=subprocess.PIPE) as proc:
        for chunk in stream_chunks(proc.stdout, "\x00"):
            yield chunk


def first_sha() -> str:
    command = ["git", "hash-object", "-t", "tree", "/dev/null"]
    result = subprocess.run(command, stdout=subprocess.PIPE)
    return result.stdout.decode("UTF8").strip()


def chunk_to_commit(chunk: str) -> Optional[Commit]:
    if chunk is None:
        return None
    if "\nbody" not in chunk:
        print(repr(chunk))
        logger.debug("body not in commit %s", chunk)
        return None
    header, body = chunk.split("\nbody", 1)
    commit_data = {"body": body}
    for line in header.split("\n"):
        key, value = line.split(" ", 1)
        commit_data[key] = value
    if not commit_data.get("sha"):
        logger.debug("No sha in commit %s", chunk)
        return None

    return Commit(**commit_data)


def print_changelog(
        from_sha: Optional[str] = None,
        to_sha: Optional[str] = None,
        version: Optional[str] = None,
        use_tags: Optional[bool] = False):
    commits_by_sha = {}  # type: Dict[str, Commit]
    for chunk in git_log(from_sha, to_sha):
        commit = chunk_to_commit(chunk)
        if commit is None:
            logger.debug("None commit")
            continue
        commits_by_sha[commit.sha] = commit

    shas_to_refs = {}
    child_shas = {}
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
            refs = shas_to_refs.get(child_sha)
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
                        logger.debug(
                            f"adding {commit.jiras} from child {sha} to {parent_sha}"
                        )
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
    group_commits = {group_name: []}

    groups = [group_name]

    release_dates = {}

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
