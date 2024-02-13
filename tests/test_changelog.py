from dataclasses import fields, MISSING

from jmullan.git_changelog import changelog


def test_include_line():
    includes = ["yes", "me", ""]
    for line in includes:
        assert changelog.include_line(line)

    not_includes = [
        None,
        "* commit to ignore",
        "git-p4",
        "Merge branch 'CONTIN-5792-refactor-backfill'",
    ]
    for line in not_includes:
        assert not changelog.include_line(line)


def test_format_jira():
    expectations = {
        "* FOOBAR-1637 last": "* FOOBAR-1637 : last",
        "* BAZZ-2733 :     ": "* BAZZ-2733",
        "* PIRATE-6206 - New ": "* PIRATE-6206 : New ",
        "* PIRATE-6206- New ": "* PIRATE-6206 : New ",
        "* PIRATE-6206 -New ": "* PIRATE-6206 : New ",
        "* PIRATE-6206-New ": "* PIRATE-6206 : New ",
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
    assert {"abcd": []} == refs


def test_extract_refs2():
    commit_data = make_commit_data(
        {
            "sha": "abcd",
            "parents": "aaaa bbbb",
            "refnames": [],
            "body": "Pull request #21: anything",
        }
    )
    refs = changelog.extract_refs(changelog.Commit(**commit_data))
    assert {"abcd": []} == refs


def test_extract_refs3():
    commit_data = make_commit_data(
        {
            "sha": "abcd",
            "parents": "aaaa bbbb",
            "refnames": [],
            "body": "Merge branch 'feature/branch_name'",
        }
    )
    refs = changelog.extract_refs(changelog.Commit(**commit_data))
    assert {"abcd": []} == refs


def test_extract_refs4():
    commit_data = make_commit_data(
        {
            "sha": "abcd",
            "parents": "aaaa bbbb",
            "refnames": [],
            "body": "Merge pull request #1 in anything from branch_name to main",
        }
    )
    refs = changelog.extract_refs(changelog.Commit(**commit_data))
    print(refs)
    assert {"abcd": [], "bbbb": ["branch_name"], "aaaa": ["main"]} == refs
