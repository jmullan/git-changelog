import io
import logging
from dataclasses import MISSING, fields
from pathlib import Path

from jmullan.git_changelog import changelog
from jmullan.git_changelog.models import Inclusiveness, ShaRange, UseTags

logger = logging.getLogger(__name__)


def test_include_line():
    includes = ["yes", "me", ""]
    for line in includes:
        assert changelog.include_line(line)

    not_includes = [
        None,
        "* commit to ignore",
        "git-p4",
        "Merge branch 'TICKET-5792-refactor-backfill'",
    ]
    for line in not_includes:
        assert not changelog.include_line(line)


def make_commit_data(data) -> dict[str, str | None]:
    commit_template = {
        commit_field.name: None for commit_field in fields(changelog.Commit) if commit_field.default_factory is MISSING
    }
    commit_data = {}  # type: dict[str, str | None]
    commit_data.update(commit_template)
    commit_data.update(data)
    return commit_data


def test_extract_refs():
    commit_data = make_commit_data({"sha": "abcd", "body": ""})
    refs = changelog.extract_refs(changelog.Commit(**commit_data))
    assert refs == {"abcd": []}


def test_extract_refs2():
    commit_data = make_commit_data(
        {
            "sha": "abcd",
            "parents": "aaaa bbbb",
            "refnames": "",
            "body": "Pull request #21: anything",
        }
    )
    refs = changelog.extract_refs(changelog.Commit(**commit_data))
    assert refs == {"abcd": []}


def test_extract_refs3():
    commit_data = make_commit_data(
        {
            "sha": "abcd",
            "parents": "aaaa bbbb",
            "refnames": "",
            "body": "Merge branch 'feature/branch_name'",
        }
    )
    refs = changelog.extract_refs(changelog.Commit(**commit_data))
    assert refs == {"abcd": []}


def test_extract_refs4():
    commit_data = make_commit_data(
        {
            "sha": "abcd",
            "parents": "aaaa bbbb",
            "refnames": "",
            "body": "Merge pull request #1 in anything from branch_name to main",
        }
    )
    refs = changelog.extract_refs(changelog.Commit(**commit_data))
    assert refs == {"abcd": [], "bbbb": ["branch_name"], "aaaa": ["main"]}


def test_make_version_line():
    commit_data = make_commit_data(
        {
            "sha": "abcd",
            "parents": "aaaa bbbb",
            "refnames": "",
            "body": "Merge branch 'feature/branch_name'",
        }
    )
    commit = changelog.Commit(**commit_data)
    version_line = changelog.make_version_line("", [commit])
    assert version_line == ""

    version_line = changelog.make_version_line("v1234", [commit])
    assert version_line == "# v1234"

    version_line = changelog.make_version_line("1234", [commit])
    assert version_line == "# v.1234"

    version_line = changelog.make_version_line("Current", [commit])
    assert version_line == "# Current"


def test_print_changelog():
    sha_range = ShaRange(
        None, Inclusiveness.INCLUSIVE, "01783c84fd85acfcde57e1ac7d2ae933ba75de6f", Inclusiveness.INCLUSIVE
    )

    with io.StringIO() as handle:
        changelog.print_changelog(
            sha_range,
            "Current",
            UseTags.TRUE,
            [],
            handle,
        )
        content = handle.getvalue()
    file_path = Path(__file__).parent / "through_01783c84fd85acfcde57e1ac7d2ae933ba75de6f.txt"
    with file_path.open("r") as f:
        expected = f.read()
    assert content is not None
    assert len(content)
    assert expected == content


def test_format_body():
    assert changelog.format_body("", []) == []
    assert changelog.format_body("body", []) == ["body"]
    assert changelog.format_body("body\nody", []) == ["body", "ody"]
    assert changelog.format_body("body\nody", ["ABC-123"]) == ["body", "ody"]
    assert changelog.format_body("body\nABC-123: ody", ["ABC-123"]) == ["body", "ody"]
