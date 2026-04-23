# Worktrees And Branches

This project can be used with or without Git worktrees. If you use them, keep
in mind that each worktree is a separate working folder tied to a branch.

## What this means

- One Git repository can have multiple working folders.
- Each working folder is tied to one branch.
- Changes made in one worktree are visible in Git status from that worktree only.

## Why Finder looks confusing

- `.worktrees` starts with a dot, so Finder hides it by default.
- Press `Cmd + Shift + .` in Finder to show hidden files/folders.

## Useful commands

Run these from `<repo-root>`:

```bash
git worktree list
```

```bash
git worktree add ../sumo-accident-simulation-feature <branch-name>
cd ../sumo-accident-simulation-feature
git branch --show-current
git status
```

Use whichever clone or worktree you intentionally opened. Do not assume a
specific `.worktrees/<name>` folder exists in every local setup.

## Should branches be separate folders?

Yes. With worktrees, each branch is a separate folder on disk.
That is expected and is the safest way to avoid mixed changes.
