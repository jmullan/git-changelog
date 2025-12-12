from jmullan.git_changelog import authors

AUTHORS = [
    ["Jan Reed", "jan.reed@example.com", "Jannett Reed", "jreed@example.com"],
    ["Bettie Bianco", "bb@example.com", "Bettie Bianco", "betty@example.com"],
    ["Bettie Bianco", "bb@example.com", "Bettie Bianco", "betty.bianco@example.com"],
    ["Tatsuo Chase", "tchase@example.com", "T Chase", "tchase@example.com"],
    ["Richard Lundin", "rlundin@example.com", "Richard Lundin", "rlundin@example.com"],
    ["Celia Moore", "calia@example.com", "Celia Moore", "celia@local.example.com"],
    ["Luke  Bethel", "luke.b@example.com", "Luke  Bethel", "lbethel@example.com"],
]

MAILMAP_AUTHORS = [
    ["Jan Reed", "jan.reed@example.com", "Jannett Reed", "jreed@example.com"],
    ["Bettie Bianco", "bb@example.com", "Bettie Bianco", "betty@example.com"],
]


def make_author(data: list[str]):
    name = data[0]
    email = data[1]
    original_name = data[2]
    original_email = data[3]

    original_username = authors.get_username_from_email(original_email)
    username = authors.get_username_from_email(email)

    return authors.Author(original_email, email, original_name, name, original_username, username)


def test_resolve_authors():
    sample_authors = [make_author(a) for a in AUTHORS]
    mailmap_authors = [make_author(a) for a in MAILMAP_AUTHORS]
    resolved = authors.resolve_authors(sample_authors, mailmap_authors)
    expected = [
        "[bb] Bettie Bianco <bb@example.com> [betty] Bettie Bianco <betty@example.com>",
        "[bb] Bettie Bianco <bb@example.com> [betty.bianco] Bettie Bianco <betty.bianco@example.com>",
        "[calia] Celia Moore <calia@example.com> [celia] Celia Moore <celia@local.example.com>",
        "[jan.reed] Jan Reed <jan.reed@example.com> [jreed] Jannett Reed <jreed@example.com>",
        "[luke.b] Luke  Bethel <luke.b@example.com> [lbethel] Luke  Bethel <lbethel@example.com>",
        "[rlundin] Richard Lundin <rlundin@example.com> [rlundin] Richard Lundin <rlundin@example.com>",
        "[tchase] Tatsuo Chase <tchase@example.com> [tchase] T Chase <tchase@example.com>",
    ]
    assert expected == [a.full for a in resolved]


def test_dupe_authors():
    jans = [
        ["Jan Reed", "jan.reed@example.com", "Jannett Reed", "jreed@example.com"],
        ["Jan Reed", "jan.reed@example.com", "Jannett Reed", "jan.reed@example.com"],
        ["Jan Reed", "jan.reed@example.com", "Jan Reed", "jan.reed@example.com"],
        ["Bettie Bianco", "bb@example.com", "Bettie Bianco", "betty@example.com"],
        ["Tatsuo Chase", "tchase@example.com", "Tatsuo Chase", "tchase@example.com"],
        ["Tatsuo Chase", "tchase@example.com", "T Chase", "tchase@example.com"],
    ]

    mailmap_jans = [
        ["Jan Reed", "jan.reed@example.com", "Jannett Reed", "jreed@example.com"],
        ["Jan Reed", "jan.reed@example.com", "Jan Reed", "jreed@example.com"],
        ["Jan Reed", "jan.reed@example.com", "Jannett Reed", "jan.reed@example.com"],
        ["Bettie Bianco", "bb@example.com", "Bettie Bianco", "betty@example.com"],
    ]

    sample_authors = [make_author(a) for a in jans]
    mailmap_authors = [make_author(a) for a in mailmap_jans]
    resolved = authors.resolve_authors(sample_authors, mailmap_authors)

    # we expect T Chase because that is a mapped to-address and the other has an
    # identical from and to

    expected = [
        "[bb] Bettie Bianco <bb@example.com> [betty] Bettie Bianco <betty@example.com>",
        "[jan.reed] Jan Reed <jan.reed@example.com> [jreed] Jannett Reed <jreed@example.com>",
        "[jan.reed] Jan Reed <jan.reed@example.com> [jreed] Jan Reed <jreed@example.com>",
        "[jan.reed] Jan Reed <jan.reed@example.com> [jan.reed] Jannett Reed <jan.reed@example.com>",
        # is ok?
        # "[jan.reed] Jan Reed <jan.reed@example.com> [jan.reed] Jan Reed <jan.reed@example.com>",
        "[tchase] Tatsuo Chase <tchase@example.com> [tchase] T Chase <tchase@example.com>",
    ]
    assert expected == [a.full for a in resolved]


def test_resolve_names_authors():
    jans = [
        ["foo", "a@b", "foo", "a@b"],
        ["bar", "a@b", "bar", "a@b"],
        ["baz", "a@b", "baz", "a@b"],
        ["qqq", "a@b", "qqq", "a@b"],
        ["rrr", "a@b", "rrr", "a@b"],
        ["sss", "a@b", "sss", "a@b"],
    ]

    mailmap_jans = []

    sample_authors = [make_author(a) for a in jans]
    mailmap_authors = [make_author(a) for a in mailmap_jans]
    resolved = authors.resolve_authors(sample_authors, mailmap_authors)

    # we expect T Chase because that is a mapped to-address and the other has an
    # identical from and to

    expected = [
        "[a] foo <a@b> [a] bar <a@b>",
        "[a] foo <a@b> [a] baz <a@b>",
        "[a] foo <a@b> [a] qqq <a@b>",
        "[a] foo <a@b> [a] rrr <a@b>",
        "[a] foo <a@b> [a] sss <a@b>",
    ]
    assert expected == [a.full for a in resolved]


def test_resolve_names_authors2():
    jans = [
        ["foo", "a@b", "foo", "a@b"],
        ["bar", "a@b", "bar", "a@b"],
        ["baz", "a@b", "baz", "a@b"],
        ["qqq", "a@b", "qqq", "a@b"],
        ["rrr", "a@b", "rrr", "a@b"],
        ["sss", "a@b", "sss", "a@b"],
    ]

    mailmap_jans = [
        ["foo", "foo@b", "foo", "a@b"],
        ["bar", "bar@b", "bar", "a@b"],
        ["baz", "baz@b", "baz", "a@b"],
        ["qqq", "qqq@b", "qqq", "a@b"],
        ["rrr", "rrr@b", "rrr", "a@b"],
        ["sss", "sss@b", "sss", "a@b"],
    ]

    sample_authors = [make_author(a) for a in jans]
    mailmap_authors = [make_author(a) for a in mailmap_jans]
    resolved = authors.resolve_authors(sample_authors, mailmap_authors)

    # we expect T Chase because that is a mapped to-address and the other has an
    # identical from and to

    expected = [
        "[bar] bar <bar@b> [a] bar <a@b>",
        "[baz] baz <baz@b> [a] baz <a@b>",
        "[foo] foo <foo@b> [a] foo <a@b>",
        "[qqq] qqq <qqq@b> [a] qqq <a@b>",
        "[rrr] rrr <rrr@b> [a] rrr <a@b>",
        "[sss] sss <sss@b> [a] sss <a@b>",
    ]
    assert expected == [a.full for a in resolved]
