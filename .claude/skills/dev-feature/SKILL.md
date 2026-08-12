---
name: dev-feature
description: Start or land a feature/investigation branch off dev in this repo. Use when the user wants to start new work that should be preserved (not just agent-workspace scratch), or wants to land finished work back into dev. Triggers on "start a branch for X", "land this on dev", "squash merge into dev".
---

# Feature/investigation branches on dev

This repo's `dev` branch is the working branch for all feature and investigation work. See the "Branch policy: dev/main" section in `CLAUDE.md` for the full policy — `main` is a separate, explicitly-curated public subset and is never merged into or from directly.

## Starting new work

1. Confirm the base is `dev`, not `main`:
   ```bash
   git fetch origin
   git switch dev && git pull
   ```
2. Branch off `dev` with a descriptive name, suffixed with the date as `-MMDD` (e.g. `akash/<topic>-0811` or `claude/<topic>-0811`):
   ```bash
   git switch -c <branch-name>-MMDD dev
   ```
3. Work normally. Prefer `agent-workspace/` for throwaway scratch; put anything meant to persist under a real module directory or `experiments/<name>/`.

## Landing finished work back into dev

Land as **one clean squash commit** — `dev`'s history should read as a sequence of coherent units, since path-promotion to `main` later depends on `dev` being legible.

```bash
git switch dev && git pull
git merge --squash <branch-name>
git commit   # write a real commit message summarizing the change, not the branch's raw history
```

Do not use `git merge --no-ff` or preserve the branch's individual commits on `dev` — squash only. Confirm with the user before pushing (`git push origin dev`) unless they've already asked for that explicitly.

After landing, the branch can be deleted locally; don't delete it without checking with the user first if it might still be referenced elsewhere.
