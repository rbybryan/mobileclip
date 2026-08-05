# CLAUDE.md

Guidance for Claude Code (and any future AI session) working in this repo.

This project is a replication and improvement effort for **MobileViCLIP** — *An Efficient
Video-Text Model for Mobile Devices* (ICCV 2025, [arXiv:2508.07312](https://arxiv.org/abs/2508.07312),
upstream [MCG-NJU/MobileViCLIP](https://github.com/MCG-NJU/MobileViCLIP)).

---

## 0. Operating model — READ FIRST

The maintainer runs **all real work on a remote GPU server that has _no_ Claude Code access.**
Claude Code runs here (in the cloud / another machine) purely to figure things out and to
**write the instructions down**. The server only ever does `git pull` and then runs what the
markdown files say.

**This means the durable output of every session is not the chat — it's the committed markdown.**
Anything the maintainer is expected to *run* or *do* on the server must live in a tracked `.md`
file and be pushed, or it is lost the moment the session ends.

> 中文：所有实际操作都在一台**没有 Claude Code**的远程服务器上进行。Claude 在别处运行，只负责把
> 结论和要执行的命令写进仓库里的 markdown，然后提交推送。服务器端只做 `git pull` 再照着 md 执行。
> 因此**每次对话真正的产出是被提交的 markdown，而不是聊天记录** —— 凡是需要在服务器上执行的命令/步骤，
> 都必须落到某个被 git 跟踪的 `.md` 里并推送，否则会话结束即丢失。

### The instruction-logging convention (the core rule)

Whenever a session produces **recommended commands, step-by-step instructions, fixes, or a
plan the maintainer will act on**, do this before ending the turn:

1. **Write it into the right tracked markdown file** (see the map below) — not just into chat.
   - New/changed shell commands or run recipes → **`COMMANDS.md`**
   - A diagnosed error + its fix → **`TROUBLESHOOTING.md`** (new `## Symptom / Cause / Fix` section)
   - Checkpoint / dataset / download details → **`CHECKPOINTS.md`**
   - Design decisions, experiment logs, multi-step plans → **`notes/<topic>.md`**
   - Long-form explainers / reports → **`docs/<topic>.md`**
   - **Anything about recaption / data audit → `recaption/docs/<topic>.md`** (design, plans,
     reports, setup all live there; the code is `recaption/scripts/`)
   Keep entries copy-pasteable: exact commands in fenced ```bash blocks, run from repo root
   unless noted, with any `CUDA_VISIBLE_DEVICES` / path edits called out.
2. **Update, don't duplicate.** If a relevant section already exists, edit it in place and keep
   it consistent with the rest of the file. Add cross-links between the docs where useful.
3. **Commit and push** so the server can pull it. Use a clear message, e.g.
   `docs: log <what> to COMMANDS.md`.

If the chat recommends a command but it never lands in a tracked `.md`, the job is not done.

### Branches / where the server pulls from

Two kinds of branch, so no rotating name has to be remembered:

- **`handoff`** — the standing, long-lived branch for the live server⇄Claude loop
  (`notes/live/from-*.md`). Both sides always pull and push *this* branch.
  See [`notes/git-handoff-workflow.md`](notes/git-handoff-workflow.md).
- **`claude/<topic>`** — short-lived feature branches for substantive work (code changes,
  new docs, experiments). One per topic, merged into `main` by PR, then deleted.

Push to whichever branch is checked out. For the server to pull instructions from `main`, the
branch has to be **merged** into it.

**Merging is pre-authorized (standing instruction, 2026-07-21).** Do not stop to ask, and do
not end a session with work parked on a branch waiting for permission: once the work is
committed and pushed, open a PR and merge it into `main` yourself. This applies to every
session — `handoff` and `claude/<topic>` alike.

Two constraints remain:

- **Merge by PR** (`gh pr create` → `gh pr merge`), never `git push origin main`. The PR is
  the audit trail; a direct push to `main` is still off-limits.
- **Merge finished work only.** If something is half-done, known-broken, or awaiting a server
  result before it can be verified, leave it on its branch and say so — the standing
  authorization covers the *asking*, not the *judgement*.

The maintainer can also just `git pull origin <branch>` on the server to read new instructions
before any merge.

---

## 1. Documentation map (what lives where)

| File | Purpose |
|------|---------|
| `README.md` / `README_zh.md` | Project intro; `_zh` is the full Chinese step-by-step run guide |
| `COMMANDS.md` | **Command cheat sheet** — setup, downloads, eval, train, git |
| `CHECKPOINTS.md` | Checkpoint & dataset download links and expected layout |
| `TROUBLESHOOTING.md` | Known errors → cause → fix |
| `notes/` | Replication log, ablations, architecture ideas, plans (some `_zh` variants) |
| `docs/` | Upstream README, landscape survey, long-form guides |
| `recaption/` | **Self-contained**: the InternVid-zh recaption + data-audit line — `docs/` (six live docs; superseded/not-yet-needed ones sit in `docs/archive/` and are not maintained) and `scripts/` (all code, entry point `run_v100.sh`). Start at [`recaption/README.md`](recaption/README.md) |

When in doubt about *how* to run something, `COMMANDS.md` is the source of truth; when something
breaks, check `TROUBLESHOOTING.md` first and append to it. Anything about recaption/data-audit
goes under `recaption/` — do not add recaption docs back into `notes/` or `docs/`.

---

## 2. Repo layout

```
configs/            upstream configs
models/             model defs (MobileCLIP backbone, InternVideo2, BERT, BEATs)
scripts/            training / evaluation entry points (.sh wrappers)
tasks/ tasks_clip/  training + eval task loops (retrieval.py, pretrain.py)
dataset/ preprocess/ tools/ utils/   data + helpers
recaption/          InternVid-zh 重述+数据审计全线（docs/ 文档 · scripts/ 代码，入口 run_v100.sh）
mobileviclip.yml    conda environment (internal env name: `internvideo`)
requirements.lock.txt   validated `pip freeze` from the working env
```

Assets (`checkpoints/`, `anno_downstream/`, `data/`) are **gitignored** — they are downloaded on
the server per `CHECKPOINTS.md`, never committed.

---

## 3. Environment & common commands (pointers)

Full details in `COMMANDS.md`. The short version:

```bash
# validated one-shot env setup
ENV_NAME=mobileviclip bash scripts/setup_env.sh
conda activate mobileviclip

# zero-shot eval (edit CUDA_VISIBLE_DEVICES inside the .sh first)
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_msrvtt.sh

# pretraining
bash scripts/pretraining/clip/mobileviclip_tiny/run.sh
```

Stack is pinned to **Python 3.10 / PyTorch 1.13.1 + CUDA 11.7**. Two recurring traps (see
`COMMANDS.md` / `TROUBLESHOOTING.md`): install `open_clip_torch` with `--no-deps` (else torch is
silently upgraded to 2.x), and keep `torch`/`torchvision`/`torchaudio` versions matched.

---

## 4. Working conventions

- Run commands from the repo root unless a doc says otherwise.
- Prefer editing existing docs over adding new ones; keep the doc map above accurate.
- Match each file's existing style; keep `_zh` variants in sync when you touch their counterparts.
- Don't commit large binaries or downloaded assets — they stay gitignored.
- Commit messages: short imperative summary of what changed and which doc it landed in.
