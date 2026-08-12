---
name: release
description: Promote finished work from dev onto main in this repo. Use when the user wants to publish/release work that's ready on dev, or asks to update main. Never merge dev into main directly — this skill does an explicit path-level promotion instead. Triggers on "promote this to main", "publish this on main", "release X".
---

# Promoting dev -> main

`main` is a curated public subset of `dev`, not a merge target. See "Branch policy: dev/main" in `CLAUDE.md` for why: some paths (e.g. `train/`) are meant to stay dev-only, and a plain `git merge dev` would eventually resurrect them on `main`.

**Never run `git merge dev` (or `git merge origin/dev`) while on `main`.** Always promote via explicit path checkout.

## Steps

1. Confirm what's ready to promote — ask the user which paths/changes if not obvious. Check what's currently dev-only and should stay excluded:
   ```bash
   git switch main && git pull
   git diff main dev --stat        # see everything that differs
   ```
2. Pull only the intended paths from `dev` onto `main`, as an explicit allowlist (never a wildcard that could catch dev-only dirs like `train/`):
   ```bash
   git checkout dev -- dataprep/ mlx_decode/ experiments/ demo/ README.md pyproject.toml
   ```
   Adjust the path list to what's actually changing — do not blanket-copy the whole tree.
3. Verify nothing dev-only leaked in:
   ```bash
   git status
   git diff --cached --stat
   ```
   If `train/` or other dev-only paths appear staged, unstage them (`git restore --staged <path>`) before continuing.
4. Commit with a message describing what was promoted (not a generic "sync with dev"):
   ```bash
   git commit -m "promote: <summary of what shipped>"
   ```
5. Confirm with the user before pushing `main` — this is a public-facing branch.

## If a direct edit was made on main

Trivial fixes (typos, small doc corrections) sometimes land directly on `main`. If so, cherry-pick that commit onto `dev` immediately in the same session so it isn't lost on the next promotion:
```bash
git switch dev
git cherry-pick <commit-sha-from-main>
```
If the edit turns out to be non-trivial, redo it properly on a `dev` feature branch instead of cherry-picking.
