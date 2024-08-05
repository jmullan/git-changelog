from jmullan.git_changelog import authors

AUTHORS = [
    ["jreed@example.com", "jan.reed@example.com", "Jannett Reed", "Jan Reed"],
    ["betty@example.com", "bb@example.com", "Bettie Bianco", "Bettie Bianco"],
    ["betty.bianco@example.com", "bb@example.com", "Bettie Bianco", "Bettie Bianco"],
    ["tchase@example.com", "tchase@example.com", "T Chase", "Tatsuo Chase"],
    ["rlundin@example.com", "rlundin@example.com", "Richard Lundin", "Richard Lundin"],
    ["celia@local.example.com", "calia@example.com", "Celia Moore", "Celia Moore"],
    ["lbethel@example.com", "luke.b@example.com", "Luke  Bethel", "Luke  Bethel"],
]

MAILMAP_AUTHORS = [
    ["jreed@example.com", "jan.reed@example.com", "Jannett Reed", "Jan Reed"],
    ["betty@example.com", "bb@example.com", "Bettie Bianco", "Bettie Bianco"],
]


def make_author(data: list[str]):
    original_email = data[0]
    email = data[1]
    original_name = data[2]
    name = data[3]

    original_username = authors.get_username_from_email(original_email)
    username = authors.get_username_from_email(email)

    return authors.Author(original_email, email, original_name, name, original_username, username)


def test_resolve_authors():
    sample_authors = [make_author(a) for a in AUTHORS]
    mailmap_authors = [make_author(a) for a in MAILMAP_AUTHORS]
    resolved = authors.resolve_authors(sample_authors, mailmap_authors)
    expected = [
        "Bettie Bianco <betty@example.com>:Bettie Bianco <bb@example.com>:betty:bb",
        "Bettie Bianco <betty.bianco@example.com>:Bettie Bianco <bb@example.com>:betty.bianco:bb",
        "Celia Moore <celia@local.example.com>:Celia Moore <calia@example.com>:celia:calia",
        "Jannett Reed <jreed@example.com>:Jan Reed <jan.reed@example.com>:jreed:jan.reed",
        "Luke  Bethel <lbethel@example.com>:Luke  Bethel <luke.b@example.com>:lbethel:luke.b",
        "Richard Lundin <rlundin@example.com>:Richard Lundin <rlundin@example.com>:rlundin:rlundin",
        "T Chase <tchase@example.com>:Tatsuo Chase <tchase@example.com>:tchase:tchase",
    ]
    assert expected == [a.full for a in resolved]


def test_dupe_authors():
    jans = [
        ["jreed@example.com", "jan.reed@example.com", "Jannett Reed", "Jan Reed"],
        ["jan.reed@example.com", "jan.reed@example.com", "Jannett Reed", "Jan Reed"],
        ["jan.reed@example.com", "jan.reed@example.com", "Jan Reed", "Jan Reed"],
        ["betty@example.com", "bb@example.com", "Bettie Bianco", "Bettie Bianco"],
        ["tchase@example.com", "tchase@example.com", "Tatsuo Chase", "Tatsuo Chase"],
        ["tchase@example.com", "tchase@example.com", "Tatsuo Chase", "T Chase"],
    ]

    mailmap_jans = [
        ["jreed@example.com", "jan.reed@example.com", "Jannett Reed", "Jan Reed"],
        ["jreed@example.com", "jan.reed@example.com", "Jan Reed", "Jan Reed"],
        ["jan.reed@example.com", "jan.reed@example.com", "Jannett Reed", "Jan Reed"],
        ["betty@example.com", "bb@example.com", "Bettie Bianco", "Bettie Bianco"],
    ]

    sample_authors = [make_author(a) for a in jans]
    mailmap_authors = [make_author(a) for a in mailmap_jans]
    resolved = authors.resolve_authors(sample_authors, mailmap_authors)

    # we expect T Chase because that is a mapped to-address and the other has an
    # identical from and to

    expected = [
        "Bettie Bianco <betty@example.com>:Bettie Bianco <bb@example.com>:betty:bb",
        "Jannett Reed <jreed@example.com>:Jan Reed <jan.reed@example.com>:jreed:jan.reed",
        "Jan Reed <jreed@example.com>:Jan Reed <jan.reed@example.com>:jreed:jan.reed",
        "Jannett Reed <jan.reed@example.com>:Jan Reed <jan.reed@example.com>:jan.reed:jan.reed",
        "Jan Reed <jan.reed@example.com>:Jan Reed <jan.reed@example.com>:jan.reed:jan.reed",
        "Tatsuo Chase <tchase@example.com>:T Chase <tchase@example.com>:tchase:tchase",
    ]
    assert [a.full for a in resolved] == expected
