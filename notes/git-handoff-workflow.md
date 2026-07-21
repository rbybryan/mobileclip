# Git-mediated handoff workflow (server ⇄ Claude)

How to hold a working "conversation" with Claude Code through git, given that the
GPU server has **no Claude Code access** and Claude runs elsewhere. See
[CLAUDE.md](../CLAUDE.md) §0 for the operating model this builds on.

> 中文：服务器没有 Claude Code，Claude 在别处跑。双方**通过 git 传递消息**来轮流对话——
> 服务器把输出/报错写进文件 push，Claude 拉下来读、回写指令再 push，服务器再拉。
> 这不是字面意义的实时，而是**回合制的异步循环**;对话状态活在 git 里，不在聊天窗口里。

---

## The loop

```
服务器 (no CC):  跑东西 → 写 from-server.md → git add/commit → git push
        ↓  (你在聊天里 ping Claude，或让它定时拉)
Claude Code:     git fetch + pull → 读 from-server.md → 分析 → 写 from-claude.md → commit → push
        ↓
服务器:          git pull → 照 from-claude.md 执行 → 再写 from-server.md → push …循环
```

It is **turn-based, not real-time**. Latency = how often each side pulls.

## Direction-split files (avoid merge conflicts)

Each file is written by **one** side only, so the two directions never edit the same
file and merges stay clean.

| File | Written by | Read by | Content |
|------|-----------|---------|---------|
| [`notes/live/from-server.md`](live/from-server.md) | **Server** | Claude | current output, errors, which step you're on, questions |
| [`notes/live/from-claude.md`](live/from-claude.md) | **Claude** | Server | next-step instructions, diagnosis, commands to run |

Convention for both files: **append a new dated block at the top**, newest first, so
history is preserved and the latest turn is always visible first. Keep commands in
copy-pasteable ```bash fences.

## Which branch

Use the current feature branch (or a dedicated `handoff` branch). Both sides pull/push
the **same** branch. To also reach the server via `main`, merge the branch by PR (ask
the maintainer first — see CLAUDE.md).

---

## Server side — the two commands you actually run

**1) Send your current state / output to Claude:**

```bash
# edit notes/live/from-server.md (paste output, errors, questions), then:
git add notes/live/from-server.md
git commit -m "handoff: server update $(date +%F_%H%M)"
git push origin <branch>          # e.g. claude/claude-md-auto-update-merge-b4muwd
```

Then ping Claude in chat: “拉一下 from-server.md”.

**2) Read Claude's reply and act:**

```bash
git pull origin <branch>
cat notes/live/from-claude.md      # newest block is at the top
```

---

## Claude side — what Claude does each turn

```bash
git fetch origin <branch> && git pull origin <branch>
# read notes/live/from-server.md, do the work / diagnose,
# write the reply as a new top block in notes/live/from-claude.md, then:
git add notes/live/from-claude.md
git commit -m "handoff: claude reply <topic>"
git push origin <branch>
```

If the reply is a reusable command/fix/plan, ALSO log it into the proper doc
(`COMMANDS.md` / `TROUBLESHOOTING.md` / `notes/…`) per CLAUDE.md — `from-claude.md`
is the live scratch channel, the durable docs are the record.

---

## Automation (optional, not set up yet)

- **A — manual (current):** server pushes, you ping Claude, Claude pulls & replies. Simplest.
- **B — poll in-session:** Claude periodically `git fetch`es `from-server.md` and reacts to
  new commits. Needs the chat session kept open.
- **C — scheduled trigger:** a cron spawns a fresh Claude session that pulls the branch and
  processes new content. Hands-off across sessions.

Ask Claude to wire up B or C when you want it.
