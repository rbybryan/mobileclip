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

The loop runs on the standing **`handoff`** branch — long-lived, so neither side has to
track a rotating feature-branch name. Both sides pull/push the **same** branch.

> 中文：固定用 `handoff` 分支，双方都 pull/push 这一个分支，不用记会变的分支名。

### ⚠️ 但**不要**直接把 `handoff` 合进 `main`

`handoff` 是长期分支，容易落后 `main` 很多（截至 2026-08-05 已落后 7 天、树里少了
约 1.1 万行 / 66 个文件）。**用陈旧的树去合并 `main`，会把 `main` 上后来加的文件删掉** ——
本仓库已经因此翻过一次车：`8699a94` "恢复被 #363 陈旧树 squash 覆盖删除的 10 个文件"。

要把 `from-server.md` 送进 `main`，走**从最新 `origin/main` 切出来的一次性分支**，
不要走 `handoff`。完整可复制命令见
[COMMANDS.md §6.1](../COMMANDS.md#61-把改好的-from-servermd-推上-main服务器端复制即用)：

```bash
git fetch origin main
git stash push -m from-server-wip -- notes/live/from-server.md
git checkout -B server/from-server-$(date +%m%d) origin/main
git stash pop
git add notes/live/from-server.md
git commit -m "handoff: server update $(date +%F_%H%M)"
git push -u origin server/from-server-$(date +%m%d)
# 然后 gh pr create --base main --fill && gh pr merge --squash --delete-branch
# 或者在聊天里让 Claude 开 PR 合并
```

如果哪天想让 `handoff` 本身重新可用，先把它对齐到 `main`
（`git checkout handoff && git merge origin/main`，解冲突时**保留 main 侧的文件**），
再继续用。

---

## Server side — the two commands you actually run

**1) Send your current state / output to Claude:**

```bash
# edit notes/live/from-server.md (paste output, errors, questions), then:
git add notes/live/from-server.md
git commit -m "handoff: server update $(date +%F_%H%M)"
git push origin handoff
```

First time on the server: `git fetch origin && git checkout handoff`.

Then ping Claude in chat: “拉一下 from-server.md”.

**2) Read Claude's reply and act:**

```bash
git pull origin handoff
cat notes/live/from-claude.md      # newest block is at the top
```

---

## Claude side — what Claude does each turn

```bash
git fetch origin handoff && git pull origin handoff
# read notes/live/from-server.md, do the work / diagnose,
# write the reply as a new top block in notes/live/from-claude.md, then:
git add notes/live/from-claude.md
git commit -m "handoff: claude reply <topic>"
git push origin handoff
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
