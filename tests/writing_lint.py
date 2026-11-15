"""writing_lint.py - checks a document against Writing_requirements_2026."""
import re, sys

BANNED = {
 "read(verb)": r"\bre(?:ad|ads|ading)\b",
 "carry": r"\bcarr(?:y|ies|ied|ying)\b",
 "discipline": r"\bdisciplin\w*",
 "cripple": r"\bcrippl\w*",
 "bear": r"\bbear(?:s|ing)?\b|\bborne\b",
 "novel": r"\bnovel(?:ty|ties)?\b",
 "comprehensive": r"\bcomprehensiv\w*",
 "leverage": r"\bleverag\w*",
 "delve": r"\bdelv\w*",
 "underscore": r"\bunderscor\w*",
 "genuinely": r"\bgenuinely\b",
 "honestly": r"\bhonestly\b",
 "straightforward": r"\bstraightforward\b",
}
CONSTR = {
 "cleft is-what/why/how": r"\bis (?:what|why|how)\b|\bwhich is (?:why|what|how)\b|\bthat is how\b",
 "which-is (other)": r", which is\b",
 "em dash": r"---|\u2014",
 "rather than": r"\brather than\b",
 "not-X-but": r"\bnot (?:a |an |the )?\w+,? but\b",
 "trailing tag , which": r", which [^.\n]{0,60}\.\s",
 "trailing tag , and": r", and [^.\n]{0,70}\.\s",
 "trailing tag , a": r", a [^.\n]{0,40}\.\s",
}

def strip_tex(t):
    t = re.sub(r"%.*", "", t)
    t = re.sub(r"\\(?:cite|ref|label|includegraphics|input)\{[^}]*\}", "", t)
    return t

for path in sys.argv[1:]:
    t = open(path, encoding="utf8", errors="ignore").read()
    if path.endswith(".tex"):
        t = strip_tex(t)
    print(f"\n== {path}")
    total = 0
    for name, pat in {**BANNED, **CONSTR}.items():
        hits = re.findall(pat, t, flags=re.I)
        if hits:
            total += len(hits)
            print(f"  {name:24s} {len(hits):3d}")
    print(f"  TOTAL {total}")
