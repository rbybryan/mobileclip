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
   Keep entries copy-pasteable: exact commands in fenced ```bash blocks, run from repo root
   unless noted, with any `CUDA_VISIBLE_DEVICES` / path edits called out.
2. **Update, don't duplicate.** If a relevant section already exists, edit it in place and keep
   it consistent with the rest of the file. Add cross-links between the docs where useful.
3. **Commit and push** so the server can pull it. Use a clear message, e.g.
   `docs: log <what> to COMMANDS.md`.

If the chat recommends a command but it never lands in a tracked `.md`, the job is not done.

### "Merge" / where the server pulls from

Development happens on a feature branch (currently `claude/claude-md-auto-update-merge-b4muwd`;
in general whatever branch is checked out). Push there. For the server to pull instructions from
`main`, the branch has to be **merged** — do that via a PR **only when the maintainer asks**, or
tell them the branch is ready to merge. Never push directly to `main` without explicit permission.
The maintainer can also just `git pull origin <feature-branch>` on the server to read new
instructions before any merge.

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

When in doubt about *how* to run something, `COMMANDS.md` is the source of truth; when something
breaks, check `TROUBLESHOOTING.md` first and append to it.

---

## 2. Repo layout

```
configs/            upstream configs
models/             model defs (MobileCLIP backbone, InternVideo2, BERT, BEATs)
scripts/            training / evaluation entry points (.sh wrappers)
tasks/ tasks_clip/  training + eval task loops (retrieval.py, pretrain.py)
dataset/ preprocess/ tools/ utils/   data + helpers
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
