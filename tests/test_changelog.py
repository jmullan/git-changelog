import io
from dataclasses import MISSING, fields

from jmullan.git_changelog import changelog


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


def test_format_jira():
    expectations = {
        "* FOOBAR-1637 last": "* FOOBAR-1637 : last",
        "* BAZZ-2733 :     ": "* BAZZ-2733",
        "* PIRATE-6206 - New ": "* PIRATE-6206 : New",
        "* PIRATE-6206- New ": "* PIRATE-6206 : New",
        "* PIRATE-6206 -New ": "* PIRATE-6206 : New",
        "* PIRATE-6206-New ": "* PIRATE-6206 : New",
        "* LEAF-5410, LEAF-5316 :   More tests": "* LEAF-5316, LEAF-5410 : More tests",
        "* A-5316, B-5316 : sorting": "* A-5316, B-5316 : sorting",
        "* B-5316, A-5316 : sorting": "* A-5316, B-5316 : sorting",
        "* LEAF-5410 :   More, LEAF-5316 ,tests": "* LEAF-5316, LEAF-5410 : More,  ,tests",
    }
    for line, expected in expectations.items():
        assert changelog.format_jira(line) == expected


def make_commit_data(data) -> dict[str, str | None]:
    commit_template = {
        commit_field.name: None
        for commit_field in fields(changelog.Commit)
        if commit_field.default_factory is MISSING
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
    assert "" == version_line

    version_line = changelog.make_version_line("v1234", [commit])
    assert version_line == "# v1234"

    version_line = changelog.make_version_line("1234", [commit])
    assert version_line == "# v.1234"

    version_line = changelog.make_version_line("Current", [commit])
    assert version_line == "# Current"


def test_print_changelog():
    with io.StringIO() as handle:
        changelog.print_changelog(
            None,
            True,
            "c97a2b73b2128c114eda9d310e91be3ccce6c18f",
            True,
            "Current",
            False,
            [],
            handle,
        )
        content = handle.getvalue()
    with open("tests/through_c97a2b73b2128c114eda9d310e91be3ccce6c18f.txt") as f:
        expected = f.read()
    assert content is not None
    assert len(content)
    assert content == expected
    print(content)


def chunk_from_file(filename: str):
    print(f"pretending to run chunk_command but really loading from file {filename}")
    with open(filename, "rb") as handle:
        yield from changelog.stream_chunks(handle, "\x00")


# @mock.patch("jmullan.git_changelog.changelog.chunk_command")
# def test_print_changelog_complex(mock_chunk_command):
#     mock_chunk_command.side_effect = lambda *_: chunk_from_file("tests/big.bin")
#     with io.StringIO() as handle:
#         changelog.print_changelog(
#             None,
#             True,
#             "c97a2b73b2128c114eda9d310e91be3ccce6c18f",
#             True,
#             "Current",
#             False,
#             [],
#             handle,
#         )
#         content = handle.getvalue()
#     with open("tests/big.txt") as f:
#         expected = f.read()
#     assert content is not None
#     assert len(content)
#     print(content)
#     assert content == expected
