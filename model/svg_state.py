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

    # Attribute names that must contain a numeric value.
    NUMERIC_ATTRS = {
        'x', 'y', 'width', 'height',
        'rx', 'ry',
        'cx', 'cy', 'r',
        'x1', 'y1', 'x2', 'y2',
        'stroke-width',
        'opacity',
        'font-size',
    }
    # Strict single-number value chars (digits + at most one decimal + leading sign).
    # No spaces or commas — that prevents space-separated lists like "0 273 357" that
    # cairosvg would silently parse as just the leading number.
    # Permissive set used for non-first chars of a numeric value.
    NUMERIC_VALUE_CHARS = set("0123456789")

    # Attributes that MUST be present before a tag can close. Without them
    # the resulting element is degenerate (zero-size shapes etc.) and won't
    # contribute visible pixels. Tags not in this dict have no requirements.
    REQUIRED_ATTRS = {
        'rect':     {'width', 'height'},
        'circle':   {'r'},
        'ellipse':  {'rx', 'ry'},
        'line':     {'x1', 'y1', 'x2', 'y2'},
        'path':     {'d'},
        'polygon':  {'points'},
        'polyline': {'points'},
    }

    # Per-tag attribute-name whitelists. Only these attribute names are
    # accepted on the corresponding tags when ``attr_whitelists`` is enabled.
    # Built from the SVG specification's commonly-rendered attributes.
    DEFAULT_ATTR_WHITELISTS = {
        'svg':      {'viewBox', 'xmlns', 'width', 'height', 'fill', 'stroke', 'stroke-width', 'version', 'preserveAspectRatio'},
        # `transform`, `fill`, `stroke`, `opacity` are intentionally excluded:
        # the model emits broken values for them that collapse rendering.
        # Color attributes are inherited from the parent <svg>'s fill/stroke,
        # which the user supplies as part of the seed prompt.
        'rect':     {'x', 'y', 'width', 'height', 'rx', 'ry'},
        'circle':   {'cx', 'cy', 'r'},
        'ellipse':  {'cx', 'cy', 'rx', 'ry'},
        'line':     {'x1', 'y1', 'x2', 'y2', 'stroke-width'},
        'path':     {'d'},
        'polygon':  {'points'},
        'polyline': {'points'},
        'g':        set(),
        'text':     {'x', 'y', 'font-size'},
    }

    def __init__(self, tag_whitelist=None, forbid_bare_text=False,
                 attr_whitelists=None):
        """
        tag_whitelist: optional set/list of allowed tag names. If provided,
            opening tags can only have names that are *prefixes* of one of
            these strings during construction, and must equal one of them
            on completion. Closing tags are unaffected (they always match
            the open stack).
        forbid_bare_text: if True, when the stack is non-empty (i.e. the
            cursor is inside an open tag's body), only whitespace and ``<``
            are accepted in OUTSIDE state — the model is forced to open a
            child tag rather than emit free-form text content.
        """
        self.pos             = self.OUTSIDE
        self.stack           = []     # open tag names
        self.in_quote        = None   # char or None
        self.name_buf        = ""
        self.root_seen       = False  # True once any tag has been opened
        self.cur_attrs       = set()  # attribute names seen in the current open tag
        self.tag_whitelist    = set(tag_whitelist) if tag_whitelist else None
        self.forbid_bare_text = forbid_bare_text
        self.attr_whitelists  = attr_whitelists  # dict[tag_name -> set[attr_name]] or None
        self._last_attr_name  = None  # name of the attr currently being valued (mid-quote)
        self._in_numeric_quote = False  # True when in_quote AND _last_attr_name is numeric
        self._quote_chars     = 0     # number of chars consumed in current quote
        self._numeric_max     = 3     # cap on chars inside a numeric attribute value (max 999)

    def copy(self):
        c = SVGGuide.__new__(SVGGuide)
        c.pos              = self.pos
        c.stack            = list(self.stack)
        c.in_quote         = self.in_quote
        c.name_buf         = self.name_buf
        c.root_seen        = self.root_seen
        c.cur_attrs        = set(self.cur_attrs)
        c.tag_whitelist    = self.tag_whitelist
        c.forbid_bare_text = self.forbid_bare_text
        c.attr_whitelists  = self.attr_whitelists
        c._last_attr_name  = self._last_attr_name
        c._in_numeric_quote = self._in_numeric_quote
        c._quote_chars     = self._quote_chars
        c._numeric_max     = self._numeric_max
        if hasattr(self, '_attr_name_buf'):
            c._attr_name_buf = self._attr_name_buf
        return c

    def _is_valid_tag_prefix(self, prefix: str) -> bool:
        """True iff `prefix` could grow into one of the whitelisted tags."""
        if self.tag_whitelist is None:
            return True
        return any(t.startswith(prefix) for t in self.tag_whitelist)

    def _is_complete_tag(self, name: str) -> bool:
        """True iff `name` is one of the whitelisted tag names (or whitelist is off)."""
        if self.tag_whitelist is None:
            return True
        return name in self.tag_whitelist

    def _allowed_attrs(self):
        """The allowed attribute-name set for the currently-open tag, or None
        if the attribute whitelist is disabled / the tag has no entry."""
        if self.attr_whitelists is None:
            return None
        return self.attr_whitelists.get(self.name_buf, None)

    def _is_valid_attr_prefix(self, prefix):
        """True iff `prefix` could grow into an allowed attribute that is
        not already used on the current tag. The cur_attrs filter prevents
        the model from committing characters into a doomed partial-attr
        name that would later fail the duplicate-detection check."""
        allowed = self._allowed_attrs()
        if allowed is None:
            return True
        return any(a.startswith(prefix) and a not in self.cur_attrs for a in allowed)

    def _is_complete_attr(self, name):
        allowed = self._allowed_attrs()
        if allowed is None:
            return True
        return name in allowed

    def _required_attrs_satisfied(self):
        """True iff the current tag has all attributes required for non-degenerate rendering.
        Only enforced when ``attr_whitelists`` is active (the constraint stack as a whole)."""
        if self.attr_whitelists is None:
            return True
        required = self.REQUIRED_ATTRS.get(self.name_buf, None)
        if not required:
            return True
        return required.issubset(self.cur_attrs)

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
                self._in_numeric_quote = False
                self._quote_chars = 0
                self.pos = self.ATTR_DONE     # require whitespace before next attr
                return True
            if ch == "<":
                return False
            # If we're inside a numeric attribute's value, restrict chars and length.
            if self._in_numeric_quote:
                if ch not in self.NUMERIC_VALUE_CHARS:
                    return False
                if self._quote_chars >= self._numeric_max:
                    return False  # forces the closing quote next
            self._quote_chars += 1
            return True

        p = self.pos

        if p == self.OUTSIDE:
            # If the root has been closed already, only whitespace is permitted.
            if self.root_seen and not self.stack:
                if ch in WS_CHARS:
                    return True
                return False
            if ch == "<":
                self.pos = self.TAG_OPEN
                self.name_buf = ""
                return True
            if ch == ">":
                return False
            # Inside an open tag's body, optionally forbid bare text content
            # (forces the model to open a child tag rather than emit prose).
            if self.forbid_bare_text and self.stack and ch not in WS_CHARS:
                return False
            return True

        if p == self.TAG_OPEN:
            if ch == "/":
                self.pos = self.CLOSE_OPEN
                return True
            if ch in NAME_START:
                # Whitelist check: ch must be a valid first char of some allowed tag.
                if not self._is_valid_tag_prefix(ch):
                    return False
                self.name_buf = ch
                self.cur_attrs = set()  # fresh tag → fresh attr set
                self.pos = self.TAG_NAME
                return True
            return False

        if p == self.TAG_NAME:
            if ch in NAME_CHARS:
                # Whitelist check: name_buf + ch must still be a prefix of some allowed tag.
                if not self._is_valid_tag_prefix(self.name_buf + ch):
                    return False
                self.name_buf += ch
                return True
            # Tag name is finished — must be a complete whitelisted tag.
            if not self._is_complete_tag(self.name_buf):
                return False
            if ch in WS_CHARS:
                self.pos = self.TAG_AFTER_NAME
                return True
            if ch == ">":
                if not self._required_attrs_satisfied():
                    return False
                self.stack.append(self.name_buf)
                self.root_seen = True
                self.name_buf = ""
                self.pos = self.OUTSIDE
                return True
            if ch == "/":
                if not self._required_attrs_satisfied():
                    return False
                self.root_seen = True
                self.pos = self.SELF_CLOSE
                return True
            return False

        if p == self.TAG_AFTER_NAME:
            if ch in WS_CHARS:
                return True
            if ch == ">":
                if not self._required_attrs_satisfied():
                    return False
                self.stack.append(self.name_buf)
                self.root_seen = True
                self.name_buf = ""
                self.pos = self.OUTSIDE
                return True
            if ch == "/":
                if not self._required_attrs_satisfied():
                    return False
                self.root_seen = True
                self.pos = self.SELF_CLOSE
                return True
            if ch in NAME_START:
                # Whitelist check: ch must start a valid attribute name for the current tag.
                if not self._is_valid_attr_prefix(ch):
                    return False
                self._attr_name_buf = ch
                self.pos = self.ATTR_NAME
                return True
            return False

        if p == self.ATTR_NAME:
            if ch in NAME_CHARS:
                # Whitelist check: name_buf + ch must still be a prefix of some allowed attr.
                if not self._is_valid_attr_prefix(self._attr_name_buf + ch):
                    return False
                self._attr_name_buf += ch
                return True
            # Attribute name finished — must be in whitelist (if active) and not a duplicate.
            attr = getattr(self, '_attr_name_buf', '')
            if not attr or attr in self.cur_attrs:
                return False
            if not self._is_complete_attr(attr):
                return False
            if ch == "=":
                self.cur_attrs.add(attr)
                self._last_attr_name = attr
                self._attr_name_buf = ""
                self.pos = self.ATTR_AFTER_EQ
                return True
            if ch in WS_CHARS:
                self.cur_attrs.add(attr)
                self._last_attr_name = attr
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
                self._in_numeric_quote = (self._last_attr_name in self.NUMERIC_ATTRS)
                return True
            return False

        if p == self.ATTR_DONE:
            # Quoted value just ended. Need whitespace, '>', or '/' before
            # any further content. NAME_START not allowed without ws.
            if ch in WS_CHARS:
                self.pos = self.TAG_AFTER_NAME
                return True
            if ch == ">":
                if not self._required_attrs_satisfied():
                    return False
                self.stack.append(self.name_buf)
                self.root_seen = True
                self.name_buf = ""
                self.pos = self.OUTSIDE
                return True
            if ch == "/":
                if not self._required_attrs_satisfied():
                    return False
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
    # --- semantic-constraint tests ---
    sem_cases = [
        # (description, kwargs, text, expected_feed_ok, expected_complete)
        ("whitelist accepts allowed",
            {'tag_whitelist': ['svg', 'rect']},
            '<svg><rect/></svg>',  True, True),
        ("whitelist rejects unknown tag",
            {'tag_whitelist': ['svg', 'rect']},
            '<svg><foo/></svg>',  False, False),
        ("whitelist rejects partial-match-only",
            {'tag_whitelist': ['svg', 'rectangle']},
            '<svg><rect/></svg>',  False, False),  # rect not in {svg, rectangle}
        ("forbid bare text in body",
            {'forbid_bare_text': True, 'tag_whitelist': ['svg', 'rect']},
            '<svg>hello</svg>',  False, False),
        ("ws between child tags ok",
            {'forbid_bare_text': True, 'tag_whitelist': ['svg', 'rect']},
            '<svg>\n  <rect/>\n</svg>',  True, True),
    ]
    print('--- semantic-constraint tests ---')
    sf = 0
    for desc, kw, txt, ok_e, comp_e in sem_cases:
        g = SVGGuide(**kw)
        ok = g.feed(txt)
        comp = g.is_complete()
        passed = ok == ok_e and comp == comp_e
        if not passed:
            sf += 1
            print(f'  [FAIL] {desc}: feed_ok={ok}/{ok_e}  complete={comp}/{comp_e}')
        else:
            print(f'  [PASS] {desc}')
    print(f'{len(sem_cases)-sf}/{len(sem_cases)} semantic tests passed')
    print()
    print('--- existing baseline tests ---')

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
