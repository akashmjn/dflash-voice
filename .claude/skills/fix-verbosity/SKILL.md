---
name: fix-verbosity
description: Tighten verbose docstrings and comments that were written during an agent session. Use when the user says comments/docstrings are too long, too specific to the session, or asks to make them crisp/concise.
---

# Fixing session-verbose docstrings and comments

Prose written while working carries the *argument* for a decision — the alternatives weighed, the bug that motivated it, the reassurance that an odd-looking line is intentional. That belongs in the commit message and the PR, not in the file. A reader six months from now needs to know what the code does and which non-obvious constraint it respects. They do not need the debate.

## Scope

Default to **the prose in the current change** — `git diff` plus any commits from this session's work that haven't been reviewed yet. Session-written comments elsewhere in a touched file are fair game; unrelated pre-existing prose is not, unless the user asks.

Do not change code. This is a comments-and-docstrings pass, and behavior must be provably identical afterward.

## What to cut

- **Re-litigation.** "This exists because X cannot Y, so Z is what turns A into B." Keep the constraint, drop the derivation.
- **Repetition across levels.** The same rationale restated in a module constant, the function docstring, and an inline comment. State it once, at the broadest level that needs it; the others get a short pointer or nothing.
- **Defensive justification.** "This is legal, since…", "nobody should fix this later", "that is the check nothing else can make." If a line looks wrong but isn't, one clause saying why is enough.
- **Narration of the obvious.** A docstring restating the signature, or a comment restating the line under it.
- **Session artifacts.** References to what was tried, what an earlier version did, or how something was verified.

## What to keep

- The non-obvious constraint itself — magic numbers traced to their source, invariants a caller must respect, a genuine "this looks wrong but isn't."
- Cross-references to the authoritative definition (`see T3CondEnc.forward`).
- Anything that becomes a real bug if a future editor doesn't know it.

Aim for roughly a third of the original length. A 15-line module docstring is usually 4 lines; a 12-line function docstring is usually 5. If cutting loses a fact a maintainer needs, keep the fact and cut the sentences around it.

Match the surrounding file's existing density and idiom — the goal is prose indistinguishable from the rest of the codebase, not uniformly terse prose.

## Watch for stale references

Session prose often describes mechanisms that later changed. Check that each surviving comment still refers to something that exists — a field being removed, a helper since inlined, a parameter renamed. Fix or drop those rather than shortening them in place.

## Verify

Prose-only edits can still break things: a docstring may be asserted in a doctest, a golden, or a `--help` snapshot. Before reporting done, run the test suites the touched files affect — including any that are deselected by default (in this repo, `-m deprecated` and `-m expensive`) if the change touches code they cover.

Then confirm the behavior you were relying on is intact. If the change under review had a verification story (a corruption matrix, a round-trip over stored artifacts), re-run it rather than assuming a comment edit was inert.

Report what was cut and confirm behavior is unchanged. Do not commit unless asked.
