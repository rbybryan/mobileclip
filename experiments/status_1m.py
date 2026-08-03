#!/usr/bin/env python3
"""1M 验证层排队现状 —— **排新任务之前先跑这个**。

回答三个在排队前必须知道、但从 `nvidia-smi` 和 `ls` 都看不出来的问题:

  1. **现在还有臂在跑吗?** `run_verify_1m.sh` 是**串行**的,再起一个会和在跑的抢同一组卡。
  2. **已"跑完"的臂,墙钟对不对?** 参考值 ~15 h/臂(`sub1m_r0_s42` 实测 15:11:58)。
     远短于它的臂 = **没跑满 epoch 就退出了**,它的结论是**无效**的,不是"结果为负"。
     ⚠️ 这不是假想:`sub1m_r1b_s42` 曾只跑 2:12:39 就出现在"完成"清单里。
  3. **哪些臂真的有结果?** 有 ckpt ≠ 有结论;要有 `eval_res*.json` 才算测过。

只依赖标准库,不 import torch,不碰 GPU —— 在跑的任务旁边随时可以跑。

用法:
    python experiments/status_1m.py                 # 全部 sub1m_* 臂
    python experiments/status_1m.py --runs-dir experiments/runs
    python experiments/status_1m.py --expect-hours 15.2   # 改墙钟参考值

判读见 notes/verification-tier-1m-zh.md §8(既有配方臂)与 §9(新架构臂)。
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time
from datetime import datetime

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
EPOCH_RE = re.compile(r"Train Epoch: \[(\d+)\]")
CKPT_EPOCH_RE = re.compile(r"^ckpt_(\d+)\.pth$")
RESULT_FILES = ("eval_res.json", "eval_res_best.json", "eval_res_latest.json")

# 「在跑」的判定窗口。训练日志按 log_freq 一行,间隔远小于 10 分钟。
#
# ⚠️ **两个时钟都要新才算在跑**,取二者里更旧的那个:
#   · 文件 mtime —— 会被 `git pull` / `cp` / `rsync` / 编辑器无意刷新,单看它会把
#     一份三天前就死掉的日志报成「运行中」,于是**挡住本来可以排的队**;
#   · 日志里最后一行自己的时间戳 —— 训练进程真正写下的时间,不会被外部触碰改动,
#     但没有时间戳的日志(极早期崩溃)读不到。
# 单独任一个都能骗人,取 max(两者的陈旧秒数) ⇒ 必须**同时**新才判定在跑。
LIVE_WINDOW_S = 600


def tail_lines(path, n=400):
    """读文件末尾若干行,不把整份日志读进内存(训练日志可以有几百 MB)。"""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = min(size, 256 * 1024)
            f.seek(size - block)
            data = f.read(block)
    except OSError:
        return []
    return data.decode("utf-8", "replace").splitlines()[-n:]


def head_lines(path, n=200):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [next(f) for _ in range(n)]
    except (OSError, StopIteration) as e:
        if isinstance(e, StopIteration):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    return f.readlines()
            except OSError:
                return []
        return []


def first_ts(path):
    for ln in head_lines(path):
        m = TS_RE.match(ln)
        if m:
            return datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
    return None


def last_ts(path):
    for ln in reversed(tail_lines(path)):
        m = TS_RE.match(ln)
        if m:
            return datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
    return None


def last_epoch_in_log(path):
    for ln in reversed(tail_lines(path, 2000)):
        m = EPOCH_RE.search(ln)
        if m:
            return int(m.group(1))
    return None


def ckpt_state(out_dir):
    """return (max_epoch_ckpt, has_latest, has_best, n_files)"""
    if not os.path.isdir(out_dir):
        return None, False, False, 0
    names = os.listdir(out_dir)
    epochs = [int(m.group(1)) for m in map(CKPT_EPOCH_RE.match, names) if m]
    return (max(epochs) if epochs else None,
            "ckpt_latest.pth" in names,
            "ckpt_best.pth" in names,
            len([n for n in names if n.endswith(".pth")]))


def eval_rmean(out_dir):
    """任一协议下的 r_mean;找不到返回 None。优先 raw 口径(_emb)。"""
    best = None
    for dirpath, _d, files in os.walk(out_dir):
        hit = next((f for f in RESULT_FILES if f in files), None)
        if not hit:
            continue
        try:
            with open(os.path.join(dirpath, hit)) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        for proto, metrics in data.items():
            if not isinstance(metrics, dict):
                continue
            v = metrics.get("r_mean")
            if v is None:
                comps = ["txt_r1", "txt_r5", "txt_r10", "img_r1", "img_r5", "img_r10"]
                vals = [metrics.get(c) for c in comps]
                if all(isinstance(x, (int, float)) for x in vals):
                    v = sum(vals) / 6.0
            if v is None:
                continue
            # raw(_emb)口径优先做展示值
            if best is None or proto.endswith("_emb"):
                best = (proto, float(v))
    return best


def live_procs():
    """在跑的 runner / 训练进程。串行 runner 的前提是同一时刻只有一组。"""
    out = []
    for pat in ("run_verify_1m.sh", "run_pretrain.sh", "tasks_clip/pretrain.py"):
        try:
            r = subprocess.run(["pgrep", "-af", pat], capture_output=True, text=True,
                               timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        for ln in r.stdout.splitlines():
            if "status_1m.py" in ln or "pgrep" in ln:
                continue
            out.append(ln.strip())
    return out


def hhmm(seconds):
    if seconds is None:
        return "—"
    h, rem = divmod(int(seconds), 3600)
    return f"{h}:{rem // 60:02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="experiments/runs")
    ap.add_argument("--config-dir", default="experiments/configs")
    ap.add_argument("--pattern", default="sub1m_*",
                    help="臂名通配,默认全部 1M 臂")
    ap.add_argument("--expect-hours", type=float, default=15.2,
                    help="每臂参考墙钟(sub1m_r0_s42 实测 15:11:58);短于其一半判为可疑")
    args = ap.parse_args()

    cfgs = sorted(os.path.basename(p)[:-3]
                  for p in glob.glob(os.path.join(args.config_dir, args.pattern + ".py")))
    if not cfgs:
        print(f"没有匹配 {args.pattern} 的 config,检查 --config-dir")
        return 1

    now = time.time()
    rows, suspicious, running, queueable = [], [], [], []

    for name in cfgs:
        log = os.path.join(args.runs_dir, name + ".log")
        out_dir = os.path.join(args.runs_dir, name)
        has_log = os.path.isfile(log)
        mx_ep, has_latest, has_best, n_ckpt = ckpt_state(out_dir)
        ev = eval_rmean(out_dir) if os.path.isdir(out_dir) else None

        wall = None
        ep = None
        stale = None
        if has_log:
            t0, t1 = first_ts(log), last_ts(log)
            if t0 and t1:
                wall = (t1 - t0).total_seconds()
            ep = last_epoch_in_log(log)
            # 两个时钟取更旧的那个(见 LIVE_WINDOW_S 上方的说明)
            stale = now - os.path.getmtime(log)
            if t1 is not None:
                stale = max(stale, now - t1.timestamp())

        # ---- 状态判定 ----
        if not has_log and n_ckpt == 0:
            state, note = "未排队", "还没跑过"
            queueable.append(name)
        elif stale is not None and stale < LIVE_WINDOW_S:
            state, note = "🟢 运行中", f"日志 {int(stale)}s 前还在写"
            running.append(name)
        elif ev:
            short = wall is not None and wall < args.expect_hours * 3600 * 0.5
            if short:
                state = "🔴 可疑"
                note = f"有结果但墙钟仅 {hhmm(wall)} ≪ 参考 {args.expect_hours}h ⇒ 结论无效"
                suspicious.append((name, n_ckpt > 0))
            else:
                state, note = "✅ 已完成", f"{ev[0]} r_mean={ev[1]:.2f}"
        elif n_ckpt:
            state = "⚠️ 中断"
            note = f"有 ckpt 无 eval 结果(墙钟 {hhmm(wall)});auto_resume 可续"
            suspicious.append((name, True))
        else:
            state = "⚠️ 起步即死"
            note = f"有日志无 ckpt(墙钟 {hhmm(wall)});看 {log} 末尾"
            suspicious.append((name, False))

        rows.append((name, state, "—" if ep is None else str(ep),
                     hhmm(wall), "—" if mx_ep is None else f"e{mx_ep:02d}",
                     "有" if ev else "无", note))

    w = max(len(r[0]) for r in rows) + 1
    print("=" * 100)
    print(f"1M 验证层现状 · {args.runs_dir}/ · 参考墙钟 {args.expect_hours} h/臂")
    print("=" * 100)
    print(f"{'臂':<{w}} {'状态':<10} {'日志epoch':<9} {'墙钟':<7} {'ckpt':<6} {'eval':<5} 备注")
    print("-" * 100)
    for r in rows:
        print(f"{r[0]:<{w}} {r[1]:<10} {r[2]:<9} {r[3]:<7} {r[4]:<6} {r[5]:<5} {r[6]}")
    print("-" * 100)

    procs = live_procs()
    print()
    if procs:
        print("🟢 检测到在跑的进程 —— **不要**再起一个 run_verify_1m.sh(串行 runner,会抢同一组卡):")
        for p in procs[:8]:
            print(f"   {p}")
        print("   要追加臂:等它跑完,或用另一组 CUDA_VISIBLE_DEVICES 起第二条 runner。")
    else:
        print("⚪ 没检测到在跑的训练进程 ⇒ 可以排新任务。")

    if suspicious:
        print()
        print("🔴 排新任务之前先处理这些(它们的结论现在**不可用**):")
        for s, resumable in suspicious:
            how = "auto_resume 从 ckpt_latest 续" if resumable else "无 ckpt ⇒ 从头重跑"
            print(f"   {s}  —  bash experiments/run_verify_1m.sh <GPUS> {s}   # {how}")
    if queueable:
        print()
        print(f"⚪ 从未跑过的臂({len(queueable)}):{' '.join(queueable)}")
    print()
    print("判读规矩见 notes/verification-tier-1m-zh.md:1M 档**只有否决权,不能排名**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
