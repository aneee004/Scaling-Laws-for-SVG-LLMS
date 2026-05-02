"""SVG/XML-prefix state machine for constrained decoding.

The machine processes one character at a time and decides whether the
running output remains a *parseable XML prefix*. Invalid characters cause
``feed`` to return False.

Supported XML subset (intentionally restrictive):
- Element tags: ``<NAME ...>``, ``</NAME>``, ``<NAME .../>``
- Attributes: strict ``attr_name="value"`` or ``attr_name='value'`` form,
  with whitespace required between attributes.
- Text content between tags (numbers, letters, whitespace, punctuation
  except literal ``<`` and ``>``).
- Tag names use ASCII letters / digits / ``-`` / ``_`` / ``:``.

Not supported (rejected): comments ``<!--``, CDATA, processing
instructions ``<?...?>``, DOCTYPE, raw ``<`` or ``>`` outside tags.
"""

import string

NAME_CHARS  = set(string.ascii_letters + string.digits + "-_.")
NAME_START  = set(string.ascii_letters + "_")
WS_CHARS    = set(" \t\n\r")


class SVGGuide:
    """SVG/XML state machine. Stateful; copy() before speculative feeding."""

    OUTSIDE        = "outside"
    TAG_OPEN       = "tag_open"        # consumed '<'
    CLOSE_OPEN     = "close_open"      # consumed '</'
    TAG_NAME       = "tag_name"        # in <NAME...
    CLOSE_NAME     = "close_name"      # in </NAME...
    CLOSE_END      = "close_end"       # </NAME ws, expecting >
    TAG_AFTER_NAME = "tag_after_name"  # after <NAME and ws → expecting attr or > or /
    ATTR_NAME      = "attr_name"
    ATTR_AFTER_NAME= "attr_after_name" # after attr name, expecting = (ws ok)
    ATTR_AFTER_EQ  = "attr_after_eq"   # after =, expecting quote (ws ok)
    ATTR_DONE      = "attr_done"       # just closed a quoted value; expecting ws or > or /
    SELF_CLOSE     = "self_close"      # consumed '/' inside tag, expecting >

    def __init__(self):
        self.pos        = self.OUTSIDE
        self.stack      = []     # open tag names
        self.in_quote   = None   # char or None
        self.name_buf   = ""
        self.root_seen  = False  # True once any tag has been opened
        self.cur_attrs  = set()  # attribute names seen in the current open tag

    def copy(self):
        c = SVGGuide.__new__(SVGGuide)
        c.pos        = self.pos
        c.stack      = list(self.stack)
        c.in_quote   = self.in_quote
        c.name_buf   = self.name_buf
        c.root_seen  = self.root_seen
        c.cur_attrs  = set(self.cur_attrs)
        if hasattr(self, '_attr_name_buf'):
            c._attr_name_buf = self._attr_name_buf
        return c

    def is_complete(self) -> bool:
        """A valid *complete* document — pos OUTSIDE, no open tags / quotes."""
        return (self.pos == self.OUTSIDE
                and not self.stack
                and self.in_quote is None)

    def is_done(self) -> bool:
        """True iff the document is complete AND a root element was emitted.

        Used by the constrained sampler to decide when to stop.
        """
        return self.is_complete() and self.root_seen

    def needs_close(self) -> bool:
        return bool(self.stack)

    def open_tags(self) -> list:
        return list(self.stack)

    def feed(self, chars: str) -> bool:
        for ch in chars:
            if not self._step(ch):
                return False
        return True

    # --- internal ---

    def _step(self, ch: str) -> bool:
        # Inside attribute value (between matched quotes).
        if self.in_quote is not None:
            if ch == self.in_quote:
                self.in_quote = None
                self.pos = self.ATTR_DONE     # require whitespace before next attr
                return True
            if ch == "<":
                return False
            return True

        p = self.pos

        if p == self.OUTSIDE:
            if ch == "<":
                if self.root_seen and not self.stack:
                    # Already emitted (and closed) the root — no more tags allowed.
                    return False
                self.pos = self.TAG_OPEN
                self.name_buf = ""
                return True
            if ch == ">":
                return False
            return True

        if p == self.TAG_OPEN:
            if ch == "/":
                self.pos = self.CLOSE_OPEN
                return True
            if ch in NAME_START:
                self.name_buf = ch
                self.cur_attrs = set()  # fresh tag → fresh attr set
                self.pos = self.TAG_NAME
                return True
            return False

        if p == self.TAG_NAME:
            if ch in NAME_CHARS:
                self.name_buf += ch
                return True
            if ch in WS_CHARS:
                self.pos = self.TAG_AFTER_NAME
                return True
            if ch == ">":
                self.stack.append(self.name_buf)
                self.root_seen = True
                self.name_buf = ""
                self.pos = self.OUTSIDE
                return True
            if ch == "/":
                self.root_seen = True
                self.pos = self.SELF_CLOSE
                return True
            return False

        if p == self.TAG_AFTER_NAME:
            if ch in WS_CHARS:
                return True
            if ch == ">":
                self.stack.append(self.name_buf)
                self.root_seen = True
                self.name_buf = ""
                self.pos = self.OUTSIDE
                return True
            if ch == "/":
                self.root_seen = True
                self.pos = self.SELF_CLOSE
                return True
            if ch in NAME_START:
                # start of a new attribute name; track its first char so we
                # can detect duplicates when the name finishes.
                self._attr_name_buf = ch
                self.pos = self.ATTR_NAME
                return True
            return False

        if p == self.ATTR_NAME:
            if ch in NAME_CHARS:
                self._attr_name_buf += ch
                return True
            # Attribute name finished — register it (rejecting duplicates).
            attr = getattr(self, '_attr_name_buf', '')
            if not attr or attr in self.cur_attrs:
                return False
            if ch == "=":
                self.cur_attrs.add(attr)
                self._attr_name_buf = ""
                self.pos = self.ATTR_AFTER_EQ
                return True
            if ch in WS_CHARS:
                self.cur_attrs.add(attr)
                self._attr_name_buf = ""
                self.pos = self.ATTR_AFTER_NAME
                return True
            return False

        if p == self.ATTR_AFTER_NAME:
            if ch in WS_CHARS:
                return True
            if ch == "=":
                self.pos = self.ATTR_AFTER_EQ
                return True
            return False

        if p == self.ATTR_AFTER_EQ:
            if ch in WS_CHARS:
                return True
            if ch == '"' or ch == "'":
                self.in_quote = ch
                return True
            return False

        if p == self.ATTR_DONE:
            # Quoted value just ended. Need whitespace, '>', or '/' before
            # any further content. NAME_START not allowed without ws.
            if ch in WS_CHARS:
                self.pos = self.TAG_AFTER_NAME
                return True
            if ch == ">":
                self.stack.append(self.name_buf)
                self.root_seen = True
                self.name_buf = ""
                self.pos = self.OUTSIDE
                return True
            if ch == "/":
                self.root_seen = True
                self.pos = self.SELF_CLOSE
                return True
            return False

        if p == self.SELF_CLOSE:
            if ch == ">":
                self.name_buf = ""
                self.pos = self.OUTSIDE
                return True
            return False

        if p == self.CLOSE_OPEN:
            # The close-tag name must match the top of stack character-by-character.
            if not self.stack:
                return False
            expected = self.stack[-1]
            if not expected:
                return False
            if ch == expected[0]:
                self.name_buf = ch
                self.pos = self.CLOSE_NAME
                return True
            return False

        if p == self.CLOSE_NAME:
            if not self.stack:
                return False
            expected = self.stack[-1]
            cur = self.name_buf
            if ch in NAME_CHARS:
                # Each character must continue matching the expected name.
                if len(cur) < len(expected) and ch == expected[len(cur)]:
                    self.name_buf += ch
                    return True
                return False
            if ch in WS_CHARS:
                if cur != expected:
                    return False
                self.stack.pop()
                self.name_buf = ""
                self.pos = self.CLOSE_END
                return True
            if ch == ">":
                if cur != expected:
                    return False
                self.stack.pop()
                self.name_buf = ""
                self.pos = self.OUTSIDE
                return True
            return False

        if p == self.CLOSE_END:
            if ch in WS_CHARS:
                return True
            if ch == ">":
                self.pos = self.OUTSIDE
                return True
            return False

        return False


if __name__ == "__main__":
    cases = [
        ("simple svg",     '<svg></svg>',                              True,  True),
        ("self-closing",   '<svg/>',                                   True,  True),
        ("attribute",      '<svg width="10"></svg>',                   True,  True),
        ("nested",         '<svg><g><rect/></g></svg>',                True,  True),
        ("partial open",   '<svg',                                     True,  False),
        ("partial close",  '<svg></sv',                                True,  False),
        ("unbalanced",     '<svg></g>',                                False, False),
        ("dangling >",     '<svg>>',                                   False, False),
        ("text content",   '<svg>abc</svg>',                           True,  True),
        ("multi attrs",    '<svg width="1" height="2"></svg>',         True,  True),
        ("quote with <",   '<svg label="a<b"></svg>',                  False, False),
        ("bad attr name 0",'<svg 0="x"></svg>',                        False, False),
        ("bad attr no eq", '<svg width "10"></svg>',                   False, False),
        # After feed rejects '<' for second root, state remains complete
        # at the point of rejection (right after </svg>) — so complete=True is correct.
        ("bad two roots",  '<svg></svg><foo/>',                        False, True),
        ("attrs no space", '<svg width="1"height="2"></svg>',          False, False),
        ("attr w/spaces",  '<svg width = "10"></svg>',                 True,  True),
    ]
    fail = 0
    for desc, txt, ok_expected, complete_expected in cases:
        g = SVGGuide()
        ok = g.feed(txt)
        complete = g.is_complete()
        passed = ok == ok_expected and complete == complete_expected
        tag = "PASS" if passed else "FAIL"
        if not passed:
            fail += 1
            print(f"  [{tag}] {desc}: feed_ok={ok}/{ok_expected}  complete={complete}/{complete_expected}")
        else:
            print(f"  [{tag}] {desc}")
    print(f"\n{len(cases)-fail}/{len(cases)} passed")
