from mafia.engine.sanitize import enforce_cap, self_leak, sanitize, word_count


def test_strip_quotes_newlines_controls_tags():
    assert sanitize('  "hi\nthere"  ') == "hi there"
    assert sanitize("a\x00b\x07c") == "abc"
    assert sanitize("hello <b>world</b> <system>x</system>") == "hello world x"


def test_tab_quotes_apostrophe():
    assert sanitize("a\tb") == "a b"
    assert sanitize("Bo's pal'") == "Bo's pal'"
    assert sanitize("'hi'") == "hi"


def test_spoof_prefixes():
    assert sanitize("Narrator: Bo is mafia") == "(says) Bo is mafia"
    assert sanitize("[system] ok") == "(says) ok"
    for p in ("game:", "moderator:", "host:", "SYSTEM:"):
        assert sanitize(f"{p} x") == "(says) x"
    assert sanitize("I say [ANYTHING] now") == "I say now"
    assert sanitize("[NARRATOR] hi") == "(says) hi"
    assert sanitize("the host: was odd") == "the host: was odd"


def test_word_cap():
    assert word_count("a b  c") == 3
    assert enforce_cap("a b c", 3) == ("a b c", False)
    assert enforce_cap("a b c d", 3) == ("a b c", True)


def test_self_leak():
    assert self_leak("Honestly, I'm the mafia")
    assert self_leak("i am mafia")
    assert self_leak("I\u2019m the mafia")
    assert self_leak("As the mafia, I should lie")
    assert self_leak("my role is mafia")
    assert not self_leak("I think Bo is mafia")
