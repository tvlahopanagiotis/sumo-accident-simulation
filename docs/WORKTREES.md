# Worktrees And Branches (Practical Guide)

This project uses Git worktrees so multiple branches can be checked out at the same time.

## What this means

- One Git repository can have multiple working folders.
- Each working folder is tied to one branch.
- Changes made in one worktree are visible in Git status from that worktree only.

## Recommended folder to use

For day-to-day Thessaloniki work, use:

`<repo-root>/.worktrees/main-clean`

## Why Finder looks confusing

- `.worktrees` starts with a dot, so Finder hides it by default.
- Press `Cmd + Shift + .` in Finder to show hidden files/folders.

## Useful commands

Run these from `<repo-root>`:

```bash
git worktree list
```

```bash
cd .worktrees/main-clean
git branch --show-current
git status
```

## Should branches be separate folders?

Yes. With worktrees, each branch is a separate folder on disk.
That is expected and is the safest way to avoid mixed changes.
