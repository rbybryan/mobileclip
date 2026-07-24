# from-claude → Server

Claude writes here in reply to [`from-server.md`](from-server.md). **Newest block at the TOP.**
Server reads this and acts. See [git-handoff-workflow.md](../git-handoff-workflow.md).

Reusable commands/fixes/plans also get logged into `COMMANDS.md` / `TROUBLESHOOTING.md` /
`notes/…` per [CLAUDE.md](../../CLAUDE.md) — this file is the live scratch channel.

<!-- Template — Claude copies this block to the top for each reply:

## YYYY-MM-DD HH:MM — <short topic>

**Diagnosis:** <what's going on>

**Do this next:**
```bash
<exact commands to run from repo root>
```

**Then report back** in from-server.md: <what output Claude needs to see next>

-->

---

## 2026-07-24 (4) — 重字幕主计划定稿：执行以 notes/recaption-master-plan.md 为准

**Diagnosis:** 应维护者要求，对 `caption-fusion-scheme.md` 与 `data-quality-caprl-scheme.md`
两份重字幕方案做了全面评估（含两路独立评审：VLM 方法论 + 统计设计），裁决合并为唯一执行层
**[`notes/recaption-master-plan.md`](../recaption-master-plan.md)**（已并入 main，#221）。要点：
S0 五样本重述照跑不变；**训练门 G1 前置**（不需要任何人力标注），150–300 片人评标定收缩为
~600 单元判定（≈1 人·日）且**后置**到 G1 通过之后；融合（CFU）降为门控假设（外部评审先验
偏低，但保留 §6.4 可证伪测试）；RETRANSLATE 并入 RECAPTION；两份旧方案的分阶段清单不再照做
（各自文首已加取代声明）。

**Do this next（V100 审计机 101vector；= 主计划 Stage 0；与 b100 的 (3) 互不阻塞）:**
```bash
# ① 五样本重述主批（前置: s3+s4 已产 $OUT/v0_2/recaption_tasks.jsonl; 2,532 片当天级）
bash tools/data_audit/run_v100.sh recap
# ② 跑完记录: 总生成数 + 起止时间 + 卡数（吞吐是全量排期的唯一可信输入）
# ③ 两个侧批诊断（各 300 片, 主批完再跑; 精确命令见主计划 §3 Stage 0）
```
b100 侧照 **(3)** 继续英文线 anno → 预检 → §6 冒烟/R0（即主计划 §1 的 EN 论文线，两线并行）。

**Then report back** in from-server.md：recap5.* 行数、吞吐三元组、抽 3 片贴 `verdict.recaps`
（我要目检五条是否真有差异 = 温度坍缩检查）。

---

## 2026-07-24 (3) — 覆盖确认为全量 10.6M，放行：建 11 片 anno → 预检 → §6

**结论：这份 jsonl 就是作者 InternVid-10M-FLT 全量，非子集。** 你的计数：total
**10,647,458**、`video` 前缀 0 不匹配、FLT_1..10 各 100 万、FLT_11 64.7 万。可直接替代 gdown。

**Do this next（`rby/mobileclip` 根，接着方案 C 往下）：**
1. 照 §4「方案 C ②」建 11 片 anno（`video=abs_path`、`caption=caption`）。
2. 照 §4「方案 C ③」建软链——`$FULL=/workspace/mount/b100_zaip_data_luan/InternVid-10M-FLT`
   （**根是 `InternVid`，分片是 `InternVId-FLT_i`，大小写别写反**，以 abs_path 为准），
   `export MVC_DATA_DIR="$WORK/data"`，再 `python experiments/check_readiness.py --decode`。
3. 11 个 `internvid_v*` 全 `[OK ]`（含真解码）后，进 §6：先 2 卡冒烟，再 R0 4 卡。

**Then report back**：② 每片生成条数（应对上上面的计数）、③ 预检 11 行、§6 冒烟头几百步
loss 走势。若某片 `data_root 缺失` 就是软链没对上 `$FULL/InternVId-FLT_i`；若 `全部缺失`
则是软链目标层级不对（abs_path 已是绝对路径，通常不该出现）。

---

## 2026-07-24 (2) — probe 结果确认可用；定稿：video 用 abs_path，caption 用 caption

**结论：可以用 zyc 这份 jsonl 替代 gdown。** 你贴的 probe 已确认：
- **源英文 caption = `caption`**（`'woman using a computer mouse and keyboard'`），中文在
  `raw_caption_zh_translated`——英文 R0–R3 用 `caption`。
- 每条带绝对路径 **`abs_path`**，且 `video` 是**深度嵌套**（`InternVId-FLT_i/shard_xxxxx/InternVId-FLT_i/<file>.mp4`）。
- 因为 loader 是 `os.path.join(data_root, rec["video"])` 且 `os.path.join(root, 绝对路径)=绝对路径`，
  **直接把 `abs_path` 当 `video` 字段最稳**，跟软链/嵌套层级无关。已把 §4「方案 C」改成定稿版。

**还差一件事**：你上一次 probe 的**分片计数那段被 `^S`（Ctrl-S）卡住没打出来**。先补跑它，
我要用 total / 每片条数判断这是作者 FLT 全量还是子集（子集=换了训练集，得有意为之）。

**Do this next（`rby/mobileclip` 项目根，conda 环境已激活）：**
```bash
SRC=/workspace/mount/b100_zaip_data/zyc/vibe_coding/low_resource_cv/MobileViCLIP/outputs/internvid_alignment/caption_translation_qwen36_v0_1/20_outputs/aligned_internvid_zh_translated_retry_merged.jsonl
# ① 覆盖核查（若终端再卡按 Ctrl-Q 解冻）
python - <<PY
import json,re,collections
rx=re.compile(r"^(InternVId-FLT_(?:1[01]|[1-9]))/"); c=collections.Counter(); n=bad=nocap=0
for l in open("$SRC"):
    if not l.strip(): continue
    n+=1; r=json.loads(l); m=rx.match(r.get("video",""))
    c[m.group(1)]+=1 if m else 0; bad+=0 if m else 1
    cap=r.get("caption"); nocap+=0 if (isinstance(cap,str) and cap.strip()) else 1
print("total:",n,"| video 前缀不匹配:",bad,"| caption 空/非串:",nocap)
for i in range(1,12): print(f"  InternVId-FLT_{i}: {c[f'InternVId-FLT_{i}']}")
PY
```
① 出数后，直接照 §4「方案 C」的 ②（生成 11 片，`video=abs_path`）+ ③（建软链过预检）跑；
命令都在 md 里、已定稿，不用再填占位。

**Then report back** in from-server.md：贴 ① 的 total + 每片条数（判断全量/子集）、②
每片生成条数、③ `check_readiness --decode` 的 11 行 `internvid_v*` 结果。全 `[OK ]` 就进 §6 冒烟。

---

## 2026-07-24 (1) — 英文线 anno：能否用 zyc 合并 jsonl 的源 caption 免 gdown？先 probe/test

**Diagnosis:** zyc 的翻译合并文件里**同时含英文源 caption**，理论上可直接生成 11 个训练
anno，省掉 §4 的 gdown。但两件事我这边看不到、必须你在 b100 上 probe 确认：**(a)** 源英文
caption 是哪个 key；**(b)** `video` 前缀是否覆盖 `InternVId-FLT_1..11`、条数是不是你要的规模
（远小于作者 FLT 全量=子集，等于换训练集，要有意为之）。英文 R0–R3 要的是**源英文 caption**，
不是中文 `raw_caption_zh_translated`。详见 `experiments/SETUP_B100.md` §4「方案 C」。

**Do this next（在 `rby/mobileclip` 项目根，已 `source .gitenv` 且 conda 环境激活）：**
```bash
SRC=/workspace/mount/b100_zaip_data/zyc/vibe_coding/low_resource_cv/MobileViCLIP/outputs/internvid_alignment/caption_translation_qwen36_v0_1/20_outputs/aligned_internvid_zh_translated_retry_merged.jsonl
# ① probe：认出源英文 caption 字段名 + 看分片覆盖/条数
python - <<PY
import json,re,collections
r=json.loads(next(l for l in open("$SRC") if l.strip()))
print("keys:",list(r.keys()))
for k,v in r.items(): print(f"  {k!r:34s} {type(v).__name__:6s} {repr(v)[:150]}")
rx=re.compile(r"^(InternVId-FLT_(?:1[01]|[1-9]))/"); c=collections.Counter(); n=bad=0
for l in open("$SRC"):
    if not l.strip(): continue
    n+=1; m=rx.match(json.loads(l).get("video",""))
    c[m.group(1)]+=1 if m else 0; bad+=0 if m else 1
print("total:",n,"| video 前缀不匹配:",bad)
for i in range(1,12): print(f"  InternVId-FLT_{i}: {c[f'InternVId-FLT_{i}']}")
PY
```
拿到 ① 输出后，按 §4「方案 C」的 ②（生成 shard 1 + `check_readiness` 验 `internvid_v1`，
用来确定 `CAPKEY` 和 `STRIP`），③（验好再一次性生成 11 片）执行——那两段命令都在 md 里，照抄。

**Then report back** in from-server.md：贴 ① 的完整输出（keys 那几行 + 每分片条数）、
② 里 `internvid_v1` 那一行是 `[OK ]` 还是 `全部缺失`、以及你最终用的 `CAPKEY`/`STRIP`。
据此我把 §4「方案 C」的占位改成对准你真实字段的定稿脚本，并判断是否算“全量”。

---

## 2026-07-23 (6) — §2 返工：先清理误装进 conda base 的包，再用 conda py3.10 重建

**Diagnosis:** 你的 §2 输出显示两个叠加故障：**(a)** `python -m venv` 建的环境没带
pip，`pip` 落到共用 conda base——torch 2.6+cu124 等全装进了 `/root/miniforge3`，venv
是空的（所以 arch 自检报 `No module named 'torch'`；提示符 `(mvc_env) (base)` 叠着就是
警报）；**(b)** miniforge 是 py3.12，tokenizers 0.13.3 没有 cp312 轮子、源码编译要
Rust，所以 transformers 那整条 pip 一个包都没装上。已把 §2 配方改成 conda python=3.10
+ 全程 `python -m pip`（`SETUP_B100.md` §2），故障记入 `TROUBLESHOOTING.md`
「b100 §2：pip 装错解释器」。

**为什么不换个方式继续用 venv：** venv 只能复制机器上已有的解释器——本机唯一的
python 是 miniforge 的 3.12，venv 建出来永远还是 3.12，而 tokenizers 0.13.3 没有
cp312 轮子（源码编译要 Rust），装到它必死；`ensurepip` 只能修好"venv 没 pip"这一层，
修不了 3.12 这一层。要拿到仓库钉的 python 3.10，本机只有 conda 能提供——所以
conda create 是唯一同时解决两个问题的路，不是偏好。

**Do this next**（照顺序；③④ 需要代理开着）：

```bash
# ① 退出坏 venv 并删除
deactivate 2>/dev/null || true
rm -rf /workspace/mount/b100_zaip_data/rby/mvc_env

# ② 清理误装进共用 base 的包（这些装之前 base 里都没有，卸载=还原，不影响别人已有的东西）
/root/miniforge3/bin/python -m pip uninstall -y \
  torch torchvision torchaudio triton sympy mpmath numpy networkx jinja2 MarkupSafe \
  fsspec pillow open_clip_torch deepspeed py-cpuinfo hjson msgpack ninja einops \
  psutil pydantic pydantic-core annotated-types typing-inspection \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12

# ③ conda 建 py3.10 环境（仍在 rby/ 内）并自检解释器归属
source /root/miniforge3/etc/profile.d/conda.sh
conda create -y -p /workspace/mount/b100_zaip_data/rby/mvc_env310 python=3.10
conda activate /workspace/mount/b100_zaip_data/rby/mvc_env310
which python && which pip   # 必须都在 .../rby/mvc_env310/bin/ 下；不是就停下来贴回来

# ④ 重装（全程 python -m pip；torch 768MB 走隧道约 20 分钟，正常）
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install "numpy<2"
python -m pip install open_clip_torch --no-deps
python -m pip install transformers==4.28.1 tokenizers==0.13.3 timm==0.5.4 openai-clip==1.0.1 \
    einops decord==0.6.0 av==12.0.0 opencv-python ftfy regex pandas scipy scikit-learn \
    pyyaml easydict yacs tabulate tensorboard wandb webdataset imageio \
    accelerate safetensors huggingface-hub setuptools
python -m pip install "deepspeed>=0.14" || true

# ⑤ arch 自检（这回 python = py3.10 环境里的）
python - <<'ARCH'
import torch
a = torch.cuda.get_arch_list()
print("torch", torch.__version__, "| archs:", a, "| dev:", torch.cuda.get_device_name(0))
assert "sm_90" in a, "torch 无 Hopper(sm_90) 内核 -> 换 cu118+/cu12x 的 torch 2.x"
print(torch.zeros(1).cuda())   # 必须打印 tensor([0.], device='cuda:0')
ARCH
```

> 若 ③ 的 `conda create` 走不通（频道被墙/代理不认），把报错贴回来，我再给
> rustup+py3.12 的备选路线。

**Then report back** in from-server.md：② 卸载是否顺利、③ 的 `which python`/`which pip`
输出、⑤ arch 自检整段输出（torch 版本 + arch 列表 + tensor 打印）。

---

## 2026-07-23 (5) — 并排 clone 英文改进线：`rby/mobileclip`（中文线原目录不动）

**Diagnosis:** 快照 `87b812e` 已收到并逐行审读完毕（记录：`notes/b100-local-changes.md`；
通用修复已按方案 B 移植进 fork main，PR #195）。现在把两条线在盘上分开：
现有 `rby/MobileViCLIP` **保留给中文 NebulaCV 线**（上游 clone，wky 协作，别动）；
英文原版模型改进线（R0–R3 全量预训练）**另 clone 本 fork** 到并排的 `rby/mobileclip`。
`SETUP_B100.md` §0/§1 已按此更新，venv 也改到 rby/ 内（$HOME 共用）。

**Do this next**（代理开着；PAT 上次 push 已存进 `rby/.git-credentials`，不会再问）：

```bash
source /workspace/mount/b100_zaip_data/rby/.gitenv
cd /workspace/mount/b100_zaip_data/rby
git clone https://github.com/rbybryan/mobileclip.git mobileclip
cd mobileclip
git log --oneline -3          # 应在 main，且能看到 PR #195 的合并提交
git config core.fileMode false
```

之后照 [`experiments/SETUP_B100.md`](../../experiments/SETUP_B100.md) **§2 起**逐节走
（arch 自检 → 权重 → 数据软链 → `check_readiness.py` 预检 → §6 冒烟）。注意：
- venv 用 `/workspace/mount/b100_zaip_data/rby/mvc_env`（§2 已更新，别装进 ~）；
- 数据软链的 `WORK=/workspace/mount/b100_zaip_data/rby/mvc_stage` 与项目目录无关，照旧；
- 中文线目录 `rby/MobileViCLIP` 里的任何东西都不要挪/删。

**Then report back** in from-server.md：clone 是否成功 + `git log -3` 输出；到 §2 后按
`SETUP_B100.md` §8 的清单回报（arch 自检、探盘、预检、首日墙钟）。

---

## 2026-07-23 (4) — 改动清单已收到；push 403 的原因 + 重做快照（排除权重）推到 fork

**Diagnosis:** 三个新事实：
1. b100 的 `MobileViCLIP` 是**上游 `MCG-NJU/MobileViCLIP` 的 clone**（`origin` 指向上游），
   所以快照推送被 `403 Permission denied` 拒了——推错了远端，不是 token 问题。
2. clone 我们的 fork（`rbybryan/mobileclip`，私有库）时输的是账号密码——GitHub 已废弃
   密码认证，必须用 **PAT**（个人访问令牌）。
3. 快照 `1dabf71` **作废**：`git add -A` 把 `checkpoints/nebulacv-clip-24m.pt` 权重也收
   进去了（上游 .gitignore 不覆盖它）。重做前先用 `.git/info/exclude` 排除。

文件级改动清单已据你回传的 `git status`/diffstat 记入
[`notes/b100-local-changes.md`](../b100-local-changes.md)（主题：NebulaCV 中文 CLIP
文本塔 + 中文 InternVid 标注管线）；还差逐行 diff，按下面走完就齐了。

**Do this next**（需要代理开着；PAT 在手边——就是你在别处 push 这个 fork 用的那个 token）：

```bash
source /workspace/mount/b100_zaip_data/rby/.gitenv
cd /workspace/mount/b100_zaip_data/rby/MobileViCLIP

# ① origin 是上游（只读）。加一个 fork 远端指向我们的私有仓库：
git remote add fork https://github.com/rbybryan/mobileclip.git 2>/dev/null
git remote -v    # 应看到 origin=MCG-NJU（保留）、fork=rbybryan

# ② 凭据也圈在 rby/ 内（不落共用账号的 $HOME）。首次 push 会问一次：
#    Username 填 rbybryan，Password 填 **PAT**（不是账号密码）
git config --global credential.helper "store --file=/workspace/mount/b100_zaip_data/rby/.git-credentials"

# ③ 权重不进快照（只写本仓库的 exclude，不动工作区、不改 tracked 的 .gitignore）
printf 'checkpoints/\n' >> .git/info/exclude

# ④ 重做快照并推到 fork
git add -A
git status -s | grep -c '^A.*checkpoints/' || true   # 必须是 0
STASH=$(git stash create "b100 nebulacv local changes snapshot $(date +%F_%H%M)")
git reset -q
echo "snapshot commit: $STASH"
git push fork "$STASH":refs/heads/server/b100-local-opts
```

> push 时会把上游那份历史一并传到 fork（两边没有共同历史，git 会补齐对象）——预期行为，
> 别慌。若 PAT 不在手边、push 实在走不通，替代：
> ```bash
> git diff HEAD > ../b100_nebulacv.patch
> tar czf ../b100_new_files.tgz $(git ls-files --others --exclude-standard)
> ```
> 把 patch 全文贴进 `from-server.md`（新文件先只贴文件名，逐个要）。

**Then report back**：`snapshot commit` 新哈希 + push 是否成功。成功后 Claude 直接
`git fetch` 读快照，把 `notes/b100-local-changes.md` 的「待确认」全部填实。

---

## 2026-07-23 (3) — 候选字幕融合 F0：`s2-merge` 融合 + 可视化人审（同 better_zh）

**背景：** 你已人工复核 `better_zh`（P0 通过）：新候选更准更全、3 条各带不同细节，
融合有望更好。完整融合方案见 [`notes/caption-fusion-scheme.md`](../caption-fusion-scheme.md)。
**好消息：F0（最简版）不用新写代码** —— 仓库里 `s2-merge`（`vlm_audit_hf.py --mode merge`
+ `MERGE_PROMPT`）已实现「单次 LLM 调用把 2–3 条候选融合成一条、**闭集防幻觉**（绝不新增
候选里没有的细节）、保时序」。它**还没有** support 计票／单例验真／CSER 双向验收（那些是
方案的 F1+）。**F0 的目的：用 300 片人审判断「融合是否真比单条更全、且幻觉 <10%」**，
达标才值得建 F1。人审用**和 better_zh 完全同款的离线可视化包**（视频+候选+融合稿同屏点按钮，
见第 3 步），不用 Gradio、不用手拼 TSV。

**做这几步（repo 根目录，数据机）：**

```bash
git pull                                   # 取到 caption-fusion-scheme.md
# 1) 跑融合（不解码视频、8 卡、可续跑；产出 outputs/audit_10k/s2_merged.*.jsonl）
bash tools/data_audit/run_v100.sh s2-merge
bash tools/data_audit/run_v100.sh status   # 看进度；跑完 8 分片行数之和 ≈ 有 ≥2 候选的片数
```

```bash
# 2) 融合体检（只看健康、不建表；不健康就别急着审）
source ~/audit_env/bin/activate && cd ~/mobileclip
python - <<'PY'
import json, glob, random
random.seed(0)
OUT="outputs/audit_10k"
cand={}
for f in glob.glob(f"{OUT}/s2_verdicts.*.jsonl"):
    for l in open(f,encoding="utf-8"):
        r=json.loads(l); v=(r.get("verdict") or {})
        cs=[c for c in (v.get("better_zh") or []) if isinstance(c,str) and c.strip()]
        if r.get("video"): cand[r["video"]]=cs
tot=real=passthru=nullc=err=0; ex=[]
for f in glob.glob(f"{OUT}/s2_merged.*.jsonl"):
    for l in open(f,encoding="utf-8"):
        r=json.loads(l); v=(r.get("verdict") or {}); tot+=1
        cs=cand.get(r.get("video")) or []; m=v.get("merged_zh")
        if v.get("error"): err+=1; continue
        if len(cs)>=2 and isinstance(m,str) and m.strip(): real+=1; ex.append((cs,m))
        elif len(cs)==1: passthru+=1
        else: nullc+=1
print(f"总行数 {tot} | 真融合(≥2候选) {real} | 单候选透传 {passthru} | 空(0候选/黑屏) {nullc} | 判定失败 {err}")
print("--- 抽 3 例肉眼看融合质量 ---")
for cs,m in random.sample(ex, min(3,len(ex))):
    print(f"候选: {' ／ '.join(cs)}\n融合: {m}\n")
PY
```

**先看这几个数**（不健康就别急着审）：真融合应 ≈9000+（`better_zh` 是 9951×3）；
空的应 ≈49（黑屏片，`better_zh` 为空）；**判定失败应接近 0**——若很多，先别审，把数字贴 Claude 查。
抽的 3 例也扫一眼：融合稿有没有把候选的细节合起来、读着通顺。

```bash
# 3) 可视化人审：一条命令建表 + 打离线包（视频+候选+融合稿同屏，和 better_zh 完全同款）
bash tools/data_audit/run_v100.sh merge-review        # 默认抽 300；MERGE_REVIEW_N=100 可调
# 取回本地看画面：
#   服务器  tar czf ~/bundle_merge.tgz -C outputs/audit_10k bundle_merge
#   本地    scp vca1@10.235.31.101:~/bundle_merge.tgz . && tar xzf bundle_merge.tgz
#           浏览器打开 bundle_merge/index.html  （视频 + 3候选 + 绿色融合稿 + 按钮同屏，进度自动存）
```

**在浏览器里每条点两个按钮**（页头有提示）：
- **幻觉 [y/n]**：绿色融合稿有没有说**任何候选都没提、且画面里也没有**的东西？有=y（最要紧，验闭集守没守）。
- **更全 [y/n]**：融合稿比**其中写得最好的那一条候选**含更多**真实**细节？是=y（验融合有增益）。
- 进度存浏览器本地，关掉再开不丢；填一半也能先统计。

```bash
# 4) 统计决策门（导出的 TSV 传回服务器后跑；复用现成 review-score，自动判两个门）
bash tools/data_audit/run_v100.sh review-score outputs/audit_10k/review_merge.tsv
```

**决策门（`review-score` 会自动打印，对齐方案 §6.2 F0）：**
- **幻觉率 <10%** 且 **更全率 >50%** → F0 通过。落地：`bash tools/data_audit/run_v100.sh s4-merged`
  把 `merged_zh` 写进训练 JSON，再按 §6.3 做 **融合 vs best-single vs uniform-K** 消融（这才是「融合有用」的硬判据）。
- **幻觉率 ≥10%** → 现成 `MERGE_PROMPT` 的闭集约束不够，需上方案 F1 的 support 计票 + 单例验真
  （`support≥2` 采信、单例回看 8 帧 yes/no）。这要新写 `tools/fuse_verify.py`，跟我说一声我来写。

> 纯文本快速版（不想看画面时）：`merge-review` 建的 `review_merge.tsv` 也能直接在 Excel 里填
> `幻觉[y/n]`/`更全[y/n]` 两列，再跑同一个 `review-score`。但**判「更全」看画面更靠谱**——能直接核对某个细节到底在不在视频里。

**Then report back** in from-server.md：贴第 4 步 `review-score` 的「幻觉率 / 更全率 + 判定」，我据此定 F1 要不要做、怎么做。

---

## 2026-07-23 (2) — b100 目录不是 git 仓库：嫁接 `.git` 再采集；git 配置全部圈在 rby/ 内

**Diagnosis:** 上一块的命令全报 `fatal: not a git repository` ⇒
`/workspace/mount/b100_zaip_data/rby/MobileViCLIP` **没有 `.git`**——代码当年不是按
`SETUP_B100.md` §1 clone 的，而是直接拷贝过去的，所以 git 无法对比。修法：用
`git clone --no-checkout` 只取 `.git` 嫁接进现有目录，把它变成"基于 origin/main 的脏
工作区"，改动就全部现形。另外账号是**共用的**：所有 git "全局"配置一律通过
`GIT_CONFIG_GLOBAL` 圈在 `rby/` 目录内，不碰共用的 `~/.gitconfig`。
（已同步记入 `TROUBLESHOOTING.md`。）

**Do this next**（① 每次开工都要 source；②需要代理，见 `SETUP_B100.md` §0）：

```bash
# ① git 环境隔离（只此一次生成；以后每个要用 git 的 shell 先 source 它）
cat > /workspace/mount/b100_zaip_data/rby/.gitenv <<'EOF'
export GIT_CONFIG_GLOBAL=/workspace/mount/b100_zaip_data/rby/.gitconfig
export GIT_CONFIG_SYSTEM=/dev/null
EOF
source /workspace/mount/b100_zaip_data/rby/.gitenv
git config --global safe.directory /workspace/mount/b100_zaip_data/rby/MobileViCLIP
git config --global user.name  "rby (b100)"
git config --global user.email "rbybryan1995@gmail.com"

# ② 嫁接 .git（代理开着；只下载版本库，不碰现有工作区任何文件）
cd /workspace/mount/b100_zaip_data/rby
git clone --no-checkout https://github.com/rbybryan/mobileclip.git MobileViCLIP_gitref
mv MobileViCLIP_gitref/.git MobileViCLIP/.git
rm -rf MobileViCLIP_gitref
cd MobileViCLIP
git config core.fileMode false      # 挂载盘的权限位噪声不算改动（写进仓库自己的 .git/config）
git reset -q origin/main            # 索引对齐 main；工作区一个字节都不动

# ③ 现在改动全部可见了——把这些输出贴回 from-server.md：
git status -sb | head -60
git diff --stat
git ls-files --others --exclude-standard | head -60

# ④ 快照并推回（不动工作区、不切分支、不影响正在跑的任务）
git add -A
STASH=$(git stash create "b100 local optimizations snapshot $(date +%F_%H%M)")
git reset -q
echo "snapshot commit: $STASH"
git push origin "$STASH":refs/heads/server/b100-local-opts
```

> 注意：③ 的 diff 是相对**当前 main HEAD** 的——如果这份拷贝当初基于旧版本，里面会混入
> "仓库后来的演进"。不用管：Claude 拿到快照后会自己在历史里找最小 diff 的基准 commit，
> 把真正的本地优化和版本漂移分开。
> `git add -A` 遵守仓库 `.gitignore`，`data/`、`checkpoints/` 等大文件不会进快照。

**Then report back** in from-server.md：③ 的三段输出 + `snapshot commit` 哈希 + ④ push
是否成功。之后这个目录就是正规 git 仓库，`SETUP_B100.md` §1 的 `git pull origin main`
也能用了（记得先 `source /workspace/mount/b100_zaip_data/rby/.gitenv`）。

---

## 2026-07-23 — b100 本地"优化"改动盘点：先把 diff 完整传回来

**Diagnosis:** b100 上 `/workspace/mount/b100_zaip_data/rby/MobileViCLIP` 的代码相对
远程仓库做过少量优化，但这些改动**只存在于服务器工作区**——仓库里没有任何记录
（`from-server.md` 为空、也没有对应分支；`exp/opt-vs-baseline` 是早已合入 main 的旧
分支，与此无关）。Claude 连不上 b100，所以第一步是在服务器上把改动完整采集回来；
收到后 Claude 会把「改了哪里、为什么」记录到 `notes/b100-local-changes.md`，再评审
是否合入 main。

**Do this next**（①–③ 纯本地 git 操作，不需要外网；只有 ④ 的 push 需要隧道+代理，
见 `experiments/SETUP_B100.md` §0）：

```bash
cd /workspace/mount/b100_zaip_data/rby/MobileViCLIP

# ① 全景：基于哪个 commit、改了哪些文件、有没有本地 commit / stash
git status -sb
git log --oneline -5
git log --oneline origin/main..HEAD
git stash list
git diff HEAD --stat

# ② 未跟踪的新文件（数据/权重等 gitignored 的不会出现在这里）
git ls-files --others --exclude-standard

# ③ 把完整改动（含未跟踪新文件）做成一个快照 commit —— 不动工作区、不切分支、
#    不影响正在跑的任务：
git add -A
STASH=$(git stash create "b100 local optimizations snapshot $(date +%F_%H%M)")
git reset -q
echo "snapshot commit: $STASH"

# ④ 推回（需要代理开着；SETUP_B100.md §0 的 http_proxy 设好后执行）
git push origin "$STASH":refs/heads/server/b100-local-opts
```

> ④ 推不动（隧道没起）时的替代：`git diff HEAD > /workspace/mount/b100_zaip_data/rby/b100_local_changes.patch`，
> 把 patch 内容连同 ① 的输出一起贴进 `from-server.md` 顶部新块再推 handoff；
> patch 太长贴不下就先只贴 ① ② 的输出。

**Then report back** in from-server.md：① ② 的完整输出 + `snapshot commit` 哈希 +
④ push 是否成功。Claude 拿到 `server/b100-local-opts` 后会 `git diff <snapshot>^1 <snapshot>`
读出全部改动并记录。

---

## 2026-07-22 (20) — 评审包视频全空：**路径 bug 已修**，重打包即可

**Diagnosis:** 你本地 `videos/` 全空 = 服务器打包时一个视频都没定位到。根因是**重复
实现漏洞**：抽检表用的是简化版路径解析（直接用 `abs_path` 或盲拼），而 `abs_path`
是**别的主机**的绝对路径、`video` 字段又带 `videos/` 这类本机没有的前缀，拼出来的
路径本机不存在。审查流水线（S1/S2）当初能成功解码，靠的是 `vlm_audit.resolve_video_path`
——它会剥掉前缀、在多个候选里返回**第一个真实存在**的。抽检表没复用它。**已修：改成
复用同一个解析器，并在建表时打印 `[videos] X/N` 命中数。**

**Do this next:**

```bash
cd ~/mobileclip && git pull

# ① 重新生成表——现在会打印 [videos] X/15，先确认不是 0
bash tools/data_audit/run_v100.sh review
```

**看那行 `[videos] X/15`：**

- **是 15/15** → 路径修好了，继续 ②。
- **还是 0/15** → 视频不在默认 `$CLIPS`（`/home/vca1/datasets/internvid_zh_10k/videos`）下。
  把下面三行的输出贴到 `from-server.md`，我据此调 `CLIPS`：
  ```bash
  head -1 outputs/audit_10k/s2_sample.jsonl | python3 -m json.tool | grep -iE 'video|abs_path'
  ls /home/vca1/datasets/internvid_zh_10k/videos/ | head -3
  find /home/vca1/datasets/internvid_zh_10k -name '*.mp4' | head -3
  ```

```bash
# ② 命中正常后打包
bash tools/data_audit/run_v100.sh review-bundle
tar czf ~/review_bundles.tgz -C outputs/audit_10k $(cd outputs/audit_10k && ls -d bundle_*)
```

```bash
# ③ 本地（Linux 用 xdg-open，不是 open；包内附 README.txt）
scp vca1@10.235.31.101:~/review_bundles.tgz . && tar xzf review_bundles.tgz
xdg-open bundle_conflicts/index.html
```

页头会显示 **「视频 X/N」**——一眼确认有没有打包进去；缺失的行会显示占位+路径，
不再是静默坏掉的播放器。

**Then report back** in from-server.md：`review` 那行 `[videos] X/15` 的数字。
若是 0，连同上面三行诊断输出一起贴。

---

## 2026-07-22 (19) — 抽检怎么开跑：**别装 gradio**，用离线评审包

**Diagnosis:** 你那个 `readonly database` 报错**与 gradio 无关，是权限**——
`gradio_video_text_review.py` 把标注库写在**它自己所在的目录**
（`/home/vca1/...` 下），你是 `rby` 只有读权限。报错落在
`PRAGMA journal_mode=WAL` 而不是 `connect()` 正是佐证：SQLite 开 WAL 要在
**库文件所在目录**建 `-wal`/`-shm`，库文件本身可读也没用。

而且**就算装上也不好用**：那个 Gradio 是**浏览数据集**的，没有抽检表的填写字段——
你得一边看 Gradio 一边在 Excel 里按 video id 对着填。

所以走离线评审包：视频 + 自带填写界面的 HTML 打成一个目录，本地浏览器里评，
**服务器上什么都不用装**。

**Do this next:**

```bash
git pull

# ① 打包（纯 stdlib；默认跳过 random 那 300 条视频，太大）
bash tools/data_audit/run_v100.sh review-bundle
tar czf ~/review_bundles.tgz -C outputs/audit_10k $(cd outputs/audit_10k && ls -d bundle_*)
```

```bash
# ② 本地机器上（不是服务器）
scp vca1@10.235.31.101:~/review_bundles.tgz . && tar xzf review_bundles.tgz
open bundle_conflicts/index.html     # 先过这 15 条，几分钟
open bundle_better_zh/index.html     # 再过 50 条
```

页面里视频／原字幕／候选字幕／单选按钮**同屏**；进度存浏览器本地，关掉再开不丢；
按 `n` 跳下一条未填；填完点**「导出 TSV」**。

```bash
# ③ 把导出的 TSV 传回服务器统计（脚本自己判决策门，不用手算比例）
scp review_conflicts.tsv vca1@10.235.31.101:~/mobileclip/outputs/audit_10k/
bash tools/data_audit/run_v100.sh review-score outputs/audit_10k/review_conflicts.tsv
bash tools/data_audit/run_v100.sh review-score outputs/audit_10k/review_better_zh.tsv
```

### 三张表填什么、为什么

| 表 | 填的列 | 判什么 | 为什么要紧 |
|---|---|---|---|
| `conflicts`（15 条） | `谁对` = `s0`/`vlm`/`unclear` | S0 说该重翻、实测说翻译没问题，谁对 | 了结「语言污染 3.3%」的**待核**标记（问题 12） |
| `better_zh`（50 条） | `准确性`/`具体性` = `better`/`same`/`worse`；**`幻觉` = `y`/`n`** | 重写的字幕到底有没有变好 | **v0.2 落地前的硬前置**——`apply_buckets.py` 会把未验证的 `better_zh` 直接写进训练数据（问题 13） |
| `random`（300 条） | `同意打分` = `y`/`n` | 单一质量分准不准 | 顺带能澄清裁判判 `recaption` 63% vs S3 给 25.32% 的分歧 |

**`better_zh` 那张表，「幻觉」那一列最要紧**：准确性/具体性变差顶多是浪费算力，
幻觉是直接往对比学习的**正样本**里掺假信息。两道门脚本会自动判：
幻觉率 >10% → 不能直接入库；`worse` 多于 `better` → 重写是净损失。

> **填一半就跑统计完全可以**，未填的行自动跳过。

### 非要用 Gradio 的话

装到**独立 venv**（你已经建了 `~/gradio_env`，对的，别进 `audit_env`），
然后二选一绕开权限：① `python gradio_video_text_review.py --help` 看有没有
`--result-dir` 之类的参数，指到自己家目录；② 把脚本拷到 `~/review_gui/` 再跑。
**别 `chmod` 别人的目录、也别 `sudo`。** 完整条目见 `TROUBLESHOOTING.md` 首条。

**Then report back** in from-server.md：`review-score` 的输出（它会直接打印
「幻觉率 X% → 能不能入库」「更准确 vs 更差」这类判决），我据此决定 v0.2 能不能落地。

---

## 2026-07-22 (18) — B0 锁定 ✅ **MSR-VTT t2v R@1 = 42.50,与论文 42.5 分毫不差**

四个基准全部跑完。**唯一有外部数字可对照的 MSR-VTT 打出 42.50,论文 ≈42.5** ——
整条评测链路(权重、解码、划分、prompt)验证通过,另外三个基准的数字现在可信了。

| 基准 | 指标 | DSL | raw | 对照 |
|---|---|---|---|---|
| `msrvtt` | **t2v R@1** | **42.50** | 40.80 | 论文 ≈42.5 ✅ |
| `msrvtt` | v2t R@1 | 43.30 | 42.50 | — |
| `ucf101` | **Top-1** | **80.97** | 80.94 | Tiny 75.44 |
| `hmdb51` | **Top-1** | **53.47** | 53.87 | Tiny 48.76 |
| `k400` | **Top-1** | **63.13** | 63.49 | Tiny 56.25 |

### ⚠️ 但你那份 BASELINE.md 有个**读数陷阱**,会把分类数字读高 20 个点

**分类任务的 Top-1 是 `txt_r1`,不是 `img_r1`。** `README_zh.md` §6.2–6.4 有四处写成
"`img_r1` = Top-1",**是错的,我已更正**。

- `txt_r*` = Images→Text:逐**视频**给类别文本排序,分母 = 视频数 ⇒ **这才是零样本 Top-1**
- `img_r*` = Text→Images:逐**类别**给视频排序,分母 = **类别数** ⇒ 问的是"这个类至少有一个
  视频排第一吗",**不是准确率**

证据(都在代码里,不是猜):`itm_eval()` 两个方向分母不同(`retrieval_utils.py:463`);
`ret_dataset.py:295-296` 里 `img2txt` 把视频映到类标签、`txt2img` 长度 = 类别数;
`retrieval_utils.py:125` 把 29 个 prompt 集成掉,所以打分矩阵是 视频 × 类别。

**最直观的自检**:6 个 `img_r1` 全部恰好是 1/类别数 的整数倍 ——
`90.10=91/101`、`60.78=31/51`、`83.00=332/400`,连 README_zh 里 Tiny 的
`83.17=84/101`、`62.75=32/51`、`80.75=323/400` 也一样。视频数根本除不出这些值。

**后果有多大**:K400 按 `img_r1` 读是 **83.00**,真实零样本 Top-1 是 **63.13**,差 20 个点 ——
而且 83% 的零样本 K400 会离谱地超过 SOTA,这本身就该起疑。另一个反证:按 `img_r1` 读,
HMDB51 的 Small(60.78)会**低于** Tiny(62.75);按 `txt_r1` 读则 53.47 > 48.76,方向才对。

**顺带两条**:① 日志里 `number of classes: 11600` 打印的其实是 类别数×prompt 数(400×29),
不是类别数;② 我在块 (12) 里把试点的 t2v/v2t 标签写反了(应为 r0 **83.8 / 83.3**、
r1 **77.2 / 71.9**),已更正 —— 结论不受影响,差距仍是 6.5 个 r_mean 点。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main
sed -n '/### 3.4 B0 基线/,/^### /p' experiments/README.md    # 锁定的基线表 + 读数说明
```

B0 到此完成。**下一步只剩一件事能在这台机器上做:§2b 的 r1a/r1b 归因拆包**
(用那 1 万条,数据已在),把 r1 落后的 6.5 点拆到 sigmoid / 注意力池化 头上:

```bash
export MVC_DATA_DIR=/home/vca1/datasets      # 注意:试点语料要 export,和 B0 相反
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8 \
    model.vision_encoder.temporal_pooling mean \
    output_dir experiments/runs/pilot_10k_r1a
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8 \
    model.use_sigmoid_loss False \
    output_dir experiments/runs/pilot_10k_r1b
```

各约 40 min。跑完把两份 `train.log` 的末轮 `ret_test/` 贴回来。

**§4 全量预训练仍然卡在数据**(InternVid 视频要逐条抓 YouTube,见块 (14)),那是独立的一步。

**Then report back**:上面 r1a/r1b 的结果;或者你决定先解决 §4 的数据获取。

---

## 2026-07-22 (17) — 四基准全绿 ✅ **B0 现在就能跑,零操作**;附 §4 的一条线索

预检确认:`unset MVC_DATA_DIR` 后 msrvtt / k400 / ucf101 / hmdb51 **全部 `[OK ]`**,
连那条软链都不用建 —— `data/MSRVTT_Videos` 本来就在(它是**软链**,所以上一轮
`find` 没跟进去,才没出现在聚合结果里)。**没有任何前置操作了,直接跑:**

```bash
cd ~/mobileclip && unset MVC_DATA_DIR
CKPT=./checkpoints/mobileviclip_small.pt
for b in msrvtt hmdb51 ucf101 k400; do
  CUDA_VISIBLE_DEVICES=0 \
    bash experiments/run_eval.sh small $b $CKPT experiments/evals/baseline_official/$b
done
python experiments/collect_results.py --dirs experiments/evals \
       --baseline baseline_official --out experiments/BASELINE.md
```

> `nvidia-smi` 先看哪张卡空着再定 `CUDA_VISIBLE_DEVICES`(卡是共享的)。
> **先盯 MSR-VTT 的 `ret_test/` t2v R@1 是否 ≈42.5** —— 唯一有外部数字可对照的,
> 它对上了,另外三个的数字才可信。K400 有 1 万条视频,是四个里最慢的。

### 你那份输出里的两条附带信息

**① §4 线索:shard 1 的抓取开过头,没抓完。** 11 个分片的报错**不一样**:

```
internvid_v1   anno 5000 条,抽查 20 条视频全部缺失(检查 MVC_DATA_DIR / 目录层级)
internvid_v2   anno 5000 条,但 data_root 缺失: data/video_sample/InternVId-FLT_2
```

v2–v11 是**目录都不存在**(没开始过);v1 是**目录存在但视频一条都不在** ——
有人跑过 `tools/download_internvid.py`(它会 `mkdir -p` 出 `--out`)但没抓成,
或者抓下来的文件名和 anno 对不上。查一下,别重复劳动:

```bash
ls data/video_sample/InternVId-FLT_1 | wc -l
tail -20 data/video_sample/InternVId-FLT_1/_download_log.tsv 2>/dev/null
python -c "import json;print([e['video'] for e in json.load(open('anno_downstream/InternVid/InternVid-10M-flt_1.json'))[:3]])"
```

大量 `fail` = YouTube 永久损耗;文件在但命名不同 = 只是对不齐,**不必重抓**。

**② 最后那行提示是误导,我已经改掉了。** 它写"pilot anno 未生成的话,先跑
`make_pilot_anno.py`" —— 但你的 pilot anno 好好的(9000 条读入正常),缺的是
**视频目录**,因为这一遍没 export `MVC_DATA_DIR`。照着那句去重建 anno 就白干了。
现在它会分清死因:anno 真缺才劝重建,否则直接告诉你换个根再跑一遍。
同时"已解锁的步骤"标题会带上本次用的根,提醒你这份结论只对这个根成立。

**Then report back**:`BASELINE.md` 的内容,尤其 MSR-VTT 的
`ret_test/` 与 `ret_test_emb/` 两行。

---

## 2026-07-22 (16) — 数据**全都在** ✅ B0 四个基准零下载,只差一条软链

搜索结果很干脆:**四个基准的视频一个都没丢**,全在 `~/mobileclip/data/` 下。
我之前让你下的 15 GB **完全不用下**,那个说法是错的(错因就是预检带着
`MVC_DATA_DIR=/home/vca1/datasets` 跑,而数据在仓库的 `data/`)。

先说个读数陷阱:你那条 `find ~ /home/vca1 ...` 里 `~` = `/home/vca1/rby`,
**正是 `/home/vca1` 的子目录**,同一棵树被遍历了两遍 —— 所以第 3 步每条命中
打印两次、第 4 步的计数**全部翻倍**。折半后:

| 基准 | 盘上目录 | 实际数量 | 与 config |
|---|---|---|---|
| `hmdb51` | `data/hmdb51/videos_test` | 6754 | ✅ 完全一致 |
| `ucf101` | `data/ucf101/videos` | 13320 | ✅ 完全一致 |
| `k400` | `data/kinetics_400_val_10s_320p` | 10119 | ✅ 完全一致(anno 10117) |
| `msrvtt` | `data/MSRVTT/videos/all` | 10004 | ⚠️ 只差软链名 `MSRVTT_Videos` |

`hmdb51_hf/hmdb51.zip` 也在,字节数 `2117581749`,与脚本 `EXPECTED_SIZE` 分毫不差。
另有全量 `data/kinetics-dataset/k400/{train,val,test}`(train 约 24 万条)。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main
unset MVC_DATA_DIR                    # 关键:评测数据在仓库 data/,不是 /home/vca1/datasets

# 唯一要做的事:给 MSR-VTT 补一条软链
ls data/MSRVTT/videos/all | head -3   # 确认是 videoXXXX.mp4 这种裸文件名
ls -ld data/MSRVTT_Videos 2>/dev/null || \
  ln -s "$(pwd)/data/MSRVTT/videos/all" "$(pwd)/data/MSRVTT_Videos"

python experiments/check_readiness.py --sample 50 --decode    # 四行应全部 [OK ]
```

四行都亮了就直接跑 **B0 四基准**(零下载):

```bash
CKPT=./checkpoints/mobileviclip_small.pt
for b in msrvtt hmdb51 ucf101 k400; do
  CUDA_VISIBLE_DEVICES=0 \
    bash experiments/run_eval.sh small $b $CKPT experiments/evals/baseline_official/$b
done
python experiments/collect_results.py --dirs experiments/evals \
       --baseline baseline_official --out experiments/BASELINE.md
```

先看 **MSR-VTT 的 `ret_test/` t2v R@1 是否≈42.5** —— 这是唯一有外部数字可对照的,
对上了才说明整条评测链路(权重、解码、划分)是通的,另外三个的数字才可信。

**两个提醒:**

1. 盘上有 `data/MSRVTT/MSRVTT/videos/all`(同样 10004 个),疑似解压嵌套多出的
   副本。**先别删** —— 等软链指向的那份确认能解码,再回收空间。
2. `/home/vca1/datasets` **没出现**在按目录聚合的结果里,但预检确认那 1 万条试点
   视频存在 —— 多半因为它是**软链**而 `find` 默认不跟随(要 `find -L`)。
   §2b 的数据没问题,别被这个吓到。

**Then report back**:`check_readiness.py` 的四行,以及 `BASELINE.md` 的内容
(尤其 MSR-VTT 那两行 `ret_test/` 与 `ret_test_emb/`)。

---

## 2026-07-22 (15) — HMDB51 **多半没丢**:是预检看错了根(这坑是我埋的)

`tools/download_hmdb51.sh` 写死的是**仓库相对路径**:

```
RAW_DIR="data/hmdb51_hf"            ->  ~/mobileclip/data/hmdb51_hf
DEST_DIR="data/hmdb51/videos_test"  ->  ~/mobileclip/data/hmdb51/videos_test
```

而你跑预检时按我 §3.0 写的 `export MVC_DATA_DIR=/home/vca1/datasets`,它就去
`/home/vca1/datasets/hmdb51/videos_test` 找了 —— **那儿从来没放过评测数据**。
**这台机器上两批数据不在一个根下**:评测集在仓库 `data/`,试点语料在
`/home/vca1/datasets`。我上一轮 §3.0 只给了 export 那一种跑法,是我的错,已修成跑两遍。

**旁证(挺硬的):`hmdb51_act_test.json` 有 1511 条,而 split-1 测试集应是
51×30 = 1530 条。** 少的 19 条正是 `build_act_rec_anno.py` 的 `--videos`
磁盘过滤丢掉的("Keep only entries whose video exists in videos_dir")。
**能过滤出 1511 条,说明生成 anno 的那一刻视频就在盘上。**

**Do this next(按顺序,大概率第 1 步就完事):**

```bash
cd ~/mobileclip && git pull origin main

# 1) 换个根再看一眼 —— 评测集不要 export MVC_DATA_DIR
unset MVC_DATA_DIR
ls -ld data/hmdb51/videos_test data/hmdb51_hf 2>/dev/null
find data/hmdb51/videos_test -maxdepth 1 -name '*.avi' 2>/dev/null | wc -l   # 期望 ~6766
python experiments/check_readiness.py            # hmdb51 应该就亮 [OK ] 了
```

如果目录在、但文件数是 0 或校验仍报缺失,是**第 2 种情况:软链悬空**
(`videos_test/` 里全是指向 `data/hmdb51_hf/` 的软链;若为回收 2.1 GB 把
`hmdb51_hf/` 删了就全断):

```bash
# 2) 悬空检查 + 就地修复(不用重下,只要 hmdb51_hf 里的 .avi 还在)
find data/hmdb51/videos_test -maxdepth 1 -xtype l | wc -l        # >0 = 悬空
find data/hmdb51/videos_test -maxdepth 1 -xtype l -delete
find "$(cd data/hmdb51_hf && pwd)" -name '*.avi' -exec ln -sf {} data/hmdb51/videos_test/ \;
```

两步都没找到,再全盘搜(拿 anno 里的**真实文件名**做外科式搜索,比翻目录快):

```bash
# 3) 外科式搜索:自动从 anno 取真实文件名再全盘找(不用手动粘贴)
NAME=$(python -c "import json;print(json.load(open('anno_downstream/hmdb51_act_test.json'))[0]['video'])")
echo "找这个文件:$NAME"
find ~ /home/vca1 -name "$NAME" 2>/dev/null

# 4) 或者按目录聚合看视频都堆在哪
for d in ~ /home/vca1 /data /mnt; do
  [ -d "$d" ] && find "$d" \( -name '*.avi' -o -name '*.mp4' \) -printf '%h\n' 2>/dev/null
done | sort | uniq -c | sort -rn | head -20

# 5) 原始 zip 还在的话不用重下(脚本会校验字节数后跳过下载)
find ~ /home/vca1 -name 'hmdb51*.zip' -size +1G -printf '%10s  %p\n' 2>/dev/null
# 期望 2117581749 字节;放回 data/hmdb51_hf/hmdb51.zip 再跑 bash tools/download_hmdb51.sh
```

**Then report back**:第 1 步那三条命令的输出(`ls -ld` + `.avi` 计数 +
不带 MVC_DATA_DIR 的 `check_readiness.py`)。若已亮 [OK ],直接跑 hmdb51 的 B0:

```bash
CKPT=./checkpoints/mobileviclip_small.pt
CUDA_VISIBLE_DEVICES=0 \
  bash experiments/run_eval.sh small hmdb51 $CKPT experiments/evals/baseline_official/hmdb51
```

> 同理,MSR-VTT / UCF101 下载前也先用**不带 `MVC_DATA_DIR`** 的预检确认一遍 ——
> 说不定也早就下过了,别白下 13 GB。

---

## 2026-07-22 (14) — 预检坐实了诊断,但多出两条:**B0 能做三个基准**;**§4 我上轮说错了**

预检输出确认了 (13) 的判断,同时给出两条新信息 —— 一条把 B0 做大,一条推翻我自己。

### ① B0 不止 MSR-VTT:**四个基准的 anno 都在盘上**,三个视频是公开源

我上轮只按 §3 那个 5 基准循环去看,漏了 `ucf101` / `hmdb51` —— 预检显示它们
anno 齐(3783 / 1511 条),而且下载量比 K400 小得多:

| 基准 | 任务 | anno | 下载量 | 怎么拿 |
|---|---|---|---|---|
| `hmdb51` | 分类 | ✅ 1511 | **~2.1 GB** | `bash tools/download_hmdb51.sh` 一条命令,幂等 |
| `msrvtt` | 检索 | ✅ 1000 | ~6.7 GB | 公开 wget(主指标) |
| `ucf101` | 分类 | ✅ 3783 | ~6.5 GB | CRCV 官网整包 |
| `k400` | 分类 | ✅ 10117 | 很大 | 可选,源不稳,不做 |

⇒ **B0 = msrvtt + hmdb51 + ucf101**(约 15 GB,全公开源,零授权),检索+分类都覆盖。
**先跑 HMDB51** —— 2.1 GB 一条命令,是验证整条评测链路最便宜的一次。

⚠️ 但只有 **MSR-VTT 有外部数字可对照**(t2v R@1≈42.5)。HMDB51/UCF101 是锁我们
自己的基线,接线错了不会被外部数字兜住 —— 所以先让 MSR-VTT 对上,再信另外两个。

### ② 更正:§4 不是"换台机器"能解决的,我上轮写错了

我在 (13) 里写"§4 需要先解决 InternVid **全量数据**的落盘位置(H100 机或换机)"。
**两处都不准**,查了 `docs/internvid-download-zh.md` 才发现:

1. **盘上那 11 个 anno 不是全量**,是本仓库自己生成的子集 —— 每片恰好 5000 条、
   合计 55000 条。生成脚本里明写着 `LIMIT = 5000  # 先做子集;要全量把这行设为 None`
   和 `PER_SHARD = 5000`。"全量 10M"从来没进过这份 anno。
2. **不存在"别处有一份视频、搬过来就行"**。InternVid 只发 URL 列表,视频要
   **逐条从 YouTube 抓**(`tools/download_internvid.py`),而且**必然有永久损耗**
   —— `Private video` / `Video unavailable` 的条目是真没了。

正确说法:**§4 是一个数据获取任务,不是机器搬迁任务。** 换 H100 也一样要从头抓,
只在带宽/并发/磁盘上占优。开跑前要先定三件事:抓多少条(现有 anno 是 55k,要更多
得重跑转换脚本)、在哪抓、能接受多大损耗率。**建议单独排期,别混进 §4 的训练排期。**

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main
unset MVC_DATA_DIR          # B0 全程不要 export,让视频落回仓库默认的 data/
df -h .                     # 三个合计 ~15 GB,解压时并存,留 >25 GB 更稳

# 最便宜的一步:HMDB51 2.1 GB,一条命令,先把评测链路跑通
bash tools/download_hmdb51.sh
python experiments/check_readiness.py --sample 50 --decode   # 期望 hmdb51 [OK ]

CKPT=./checkpoints/mobileviclip_small.pt
CUDA_VISIBLE_DEVICES=0 \
  bash experiments/run_eval.sh small hmdb51 $CKPT experiments/evals/baseline_official/hmdb51
```

这一步跑通了,再下 MSR-VTT(~6.7 GB)和 UCF101(~6.5 GB),三个一起汇总 ——
完整命令见 `experiments/README.md` §3.2 / §3.3。

**另外:把盘上那份 `experiments/BASELINE.md` 删掉。** 它是 (13) 说的那个陷阱产物,
内容是 10k 试点自检索数字、文件名却叫 BASELINE。重新生成时脚本会在顶部写明出处。

**Then report back**:`bash tools/download_hmdb51.sh` 之后的
`check_readiness.py` 输出,以及 hmdb51 的 `eval_res.json`。

---

## 2026-07-22 (13) — §3 B0 全崩 ❌ 不是配置问题:**本机没有下游评测数据**

5 个评测一个数字都没出。**别再重跑一遍** —— 重跑还是全崩。报错分两类:

- **A 类 anno 就缺**(`didemo` / `anet` / `ssv2_mc`):秒崩在
  `FileNotFoundError: anno_downstream/didemo_ret_test.json`。
- **B 类 anno 有、视频没有**(`msrvtt` / `k400`):走得很远 —— 配置打印、标注读入
  (`1000/1000`)、权重载入、**文本特征都算完了**,才崩在第一批视频:
  `ERROR opening: /home/vca1/datasets/MSRVTT_Videos/video7020.mp4`。
  B 类特别容易误判成偶发解码错误,因为 `data_root` 只在真正读视频那一刻才被触碰。

**根因**:`tools/data_audit/SETUP_V100.md` 开篇就写了 —— 这台机器
**只有 `/home/vca1/datasets/internvid_zh_10k` 那 1 万条抽样,没有全量数据**。
MSR-VTT / K400 / DiDeMo / ANet / SSv2 的视频一个都不在。

**连带影响(更要紧):§4 的 R0→R1 全量 A/B 在这台机器上同样跑不了。**
`internvid_v1..v11` 的 `data_root` 是 `$MVC_DATA_DIR/video_sample/InternVId-FLT_{i}`,
同一个"没有全量数据"的结论对它一样成立。所以 101vector 上 §2b 之后**还能做的只有两件**:
① r1a/r1b 归因拆包(数据已在);② 补完 MSR-VTT 视频后的 B0 单基准。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main

# 1) 先跑预检(10 秒)。仓库里一直有这个脚本,这次就是省掉它换来 5 次崩溃
export MVC_DATA_DIR=/home/vca1/datasets
python experiments/check_readiness.py
```

把这份输出贴回来 —— 它会一次性说清哪些 anno/视频在、哪些步骤已解锁,
我据此定 §4 到底在哪台机器上跑。

```bash
# 2) 想继续 B0 的话:MSR-VTT 是唯一近在咫尺的(anno 已在,只差视频,公开 wget ~6.7GB)
#    注意:跑 MSR-VTT 时不要 export MVC_DATA_DIR,让它落回仓库默认的 data/
cd ~/mobileclip && unset MVC_DATA_DIR
df -h .                     # 先确认 >8 GB 余量
mkdir -p data && cd data
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
unzip -q MSRVTT.zip && ln -s "$(pwd)/MSRVTT/videos/all" "$(pwd)/MSRVTT_Videos"
cd .. && ls data/MSRVTT_Videos/ | wc -l        # 期望 10000
python experiments/check_readiness.py --sample 50 --decode   # 期望 [OK ] msrvtt

CKPT=./checkpoints/mobileviclip_small.pt
CUDA_VISIBLE_DEVICES=0 \
  bash experiments/run_eval.sh small msrvtt $CKPT experiments/evals/baseline_official/msrvtt
python experiments/collect_results.py --dirs experiments/evals \
       --baseline baseline_official --out experiments/BASELINE.md
```

`didemo` / `anet` / `ssv2_mc` **近期不做**(anno 在受限 Drive + 视频要大量下载),
`k400` 可选。B0 收窄成 MSR-VTT 单基准,足够支撑 §6 的判据(主指标本来就是它)。

**两个新发现,已写进文档:**

1. **`BASELINE.md` 会骗人。** 那次 5 个评测全崩、`evals/` 下一个结果都没有,
   `collect_results.py` 照样从 `runs/` 捞到试点数字,写出一个名叫 **BASELINE.md**、
   内容是 **10k 试点自检索** r_mean 93.37 / 86.83 的文件,退出码 0、零警告。
   那**不是** MSR-VTT 基线(基线 t2v R@1≈42.5)。已加两条防线:`--dirs experiments/evals`,
   以及脚本现在会把「扫了哪些目录、含哪些 run」写进产物顶部、基线缺失时显式报警。
   **你现在盘上那份 BASELINE.md 是误导性的,建议直接删掉重生成。**
2. **官方 `mobileviclip_small.pt` 不含文本塔主体**(从加载日志读出):`missing_keys`
   恰好是整个 `text_encoder.{embedding_layer, positional_embedding, transformer.*,
   final_layer_norm}`,而 `text_encoder.projection_layer` **在**权重里,视觉塔和
   `temp` 也都在。文本塔是建模时从 `mobileclip_s2.pt` 载入的(上游 `freeze_text=True`
   只训投影,自洽)。**含义:B0 的数字由这两个权重共同决定**,复现时
   `mobileclip_s2.pt` 也必须是对的那一份。

**Then report back**:`check_readiness.py` 的完整输出;以及若下了 MSR-VTT,
`experiments/evals/baseline_official/msrvtt/eval_res.json` 的内容。

---

## 2026-07-22 (12) — §2b 完成 ✅ 试点跑通了;r1 落后 6.5 点但**不能**据此否定组件

**两臂都跑完,全程零 `OVERFLOW`**(r0 38:10 / r1 37:46,和我估的 ≈40 min 对上)。
判据两条都过:

- **损失曲线**:r0 `1.014 → 0.272`、r1 `7.093 → 4.165`,都单调下降后走平,不发散。
  ⚠️ **两臂 loss 不可互比** —— r0 是 softmax InfoNCE,r1 是 sigmoid,量纲不同。
- **组件存活**:不用开 ckpt,`train.log` 里就能读到 ——
  `diff_names: ['logit_scale','logit_bias'], diff_lr: 0.001` 加
  `optimizer -- lr=0.001 wd=0.2 len(p)=2`(different_lr 生效);
  `vision_encoder.temporal_pool.{q,k,v,proj}` 在优化器参数表里(注意力池化已建起);
  `train_file` 带 `caption_idx: True`。

### r1 的检索明显低:93.37 vs 86.83(r_mean)。**先别下结论。**

末轮 t2v/v2t R@1:r0 **83.8 / 83.3**,r1 **77.2 / 71.9**,两个口径同向。差距 6.5 点
远超 ~3.5pp 噪声带 —— 但**这正是上一轮就写明「这个评测判不了组件优劣」的场景**,
理由没变(开训前已近饱和 + BN running stats 混淆),而且:
r1 是**四个改动的捆绑包**,试点无法归因;r1 从 ep2 起还在**单调回升**
(85.07 → 86.83)、末轮仍在涨,**根本没收敛**,而 r0 早在 93.3 走平。

**我先前猜「注意力池化是随机初始化所以拖后腿」—— 查了源码,这个猜测是错的。**
`TemporalAttentionPool` 的 `proj` 是**零初始化**(`models/mobileviclip_small.py:1351`),
`forward` 返回 `mean + proj(out)`,初始时刻**精确等于 mean 池化**。
更像是 **sigmoid 的 SigLIP 默认初值对「已经对齐好的模型」失配**:
`t=10, b=−10`,正样本 sim≈0.5 ⇒ `t·sim+b≈−5` ⇒ `sigmoid≈0.007`,
等于开局宣布「所有正样本都判错」(首步 loss 7.4531 正好对上),
于是对预训练好的嵌入空间来了一记大推力,然后再慢慢标定回来。**这是假说,没验证。**

### 另外两处更正 / 新发现

1. **我上一轮说「`temp` 被 clamp 钉死」是错的。** 真跑里 r0 的 temp **往上走**:
   `0.0100 → 0.0101 → 0.0102 → 0.0103`,然后随 lr 衰减走平。clamp 是下界,梯度
   根本没往下顶。所以「冻结 temp 近似空操作」**收回** —— 冻结会改变 r0。
   (r1 全程 0.0100 是另一回事:sigmoid 分支传的是 `temp=1.0`,`self.temp`
   **完全没有梯度**,是个没接线的参数。)
2. **`no_weight_decay()` 是死代码。** `utils/optimizer.py:99-107` 把
   `no_decay = model.no_weight_decay()` **注释掉了**、只留 `pass`;而
   `temp`/`logit_scale`/`logit_bias` 是 **0 维**标量(`torch.Size([])`,长度 0),
   也躲过了 `len(shape)==1` 的 bias/BN 兜底。结果三者都吃到 **wd 0.2**。
   量级:`logit_scale`/`logit_bias` 在 lr=1e-3 下 170 步被拉向 0 约 **3.3%**。
   二阶效应,但「声明了 no-decay 却被静默忽略」正是会悄悄污染消融的那类缺陷。
   上游代码,**我没改**。

### Do this next

试点到此为止,别再在 10k 上追组件结论。下一步是 §3 的 B0(需要 MSR-VTT 到位):

```bash
cd ~/mobileclip && git pull origin main
ls -la anno_downstream/msrvtt_1k_test.json checkpoints/mobileviclip_small.pt   # 到位了吗
```

**两个都在** → 走 §3;**缺** → 先补下载(见 `CHECKPOINTS.md`)。

真要在试点上再挖一手,唯一值得跑的是**拆包归因**(各 ≈40 min),验证上面那个假说:

```bash
export MVC_DATA_DIR=/home/vca1/datasets
# 关掉注意力池化(只剩 sigmoid + different_lr)
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8 \
    model.vision_encoder.temporal_pooling mean \
    output_dir experiments/runs/pilot_10k_r1a
# 关掉 sigmoid(只剩注意力池化 + different_lr)
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8 \
    model.use_sigmoid_loss False \
    output_dir experiments/runs/pilot_10k_r1b
```

> 这两条是拿 r1 配置**关掉**其中一项,`different_lr` 仍开着,不是严格单因子;
> 严格拆分要用 §4 的 `r1a_sigmoid.py` / `r1b_attnpool.py`。
> `merge_list` 只认**已存在**的键 —— `output_dir` 存在,可以这样覆盖。
> 若假说成立:r1a(仍带 sigmoid)应该复现这次的掉点,r1b(无 sigmoid)不该掉。

**⚠️ 跑 §4 之前务必先做 `experiments/README.md` §4 顶部标的两项预检**
(训练中评测被静默 Skip、`ckpt_best.pth` 不存在),否则几小时白烧。

---

## 2026-07-22 (11) — 冒烟通过 ✅ 可以跑真 A/B 了(但评测已近饱和,判据要改)

**链路全通了。** 8 卡起来、数据读到、前向反向都过、epoch 末评测出表、下一个 epoch
接着跑。`initial_scale_power 5` 全程**零** `OVERFLOW`,anomaly detection 按预期
被跳过(日志里有那行中文说明)。这是试点第一次真正跑出损失。

**但先别把 1.3846 → 0.9709 读成「在学习」。** 那两个 epoch 的
`Averaged stats` 差异**几乎肯定是批次差异,不是训练效果**,理由有三条:

1. `debug` 每个 epoch 只跑 **5 步**,lr 还在 warmup(1e-6 → 1e-5)。Adam 的单步
   位移约等于 lr,5 步累计把权重挪动 ~5e-6 的相对量级 —— 对 35M 参数的模型
   等于没动。
2. 日志里每行 loss 是**原始单步值**(`pretrain.py:55` 的 `fmt="{value:.4f}"`,
   不是滑窗中位数)。epoch 0 单步在 [1.13, 1.66] 之间跳,epoch 1 在 [0.95, 1.18],
   两个区间是重叠的 —— n=5 的样本判不出趋势。
3. 每 epoch 只有 17 个 batch、抽 5 个,批次组成本身的方差就够解释这个差。

**真正的损失曲线要等去掉 `debug` 的那次(17 步 × 10 epoch = 170 步)。**

**先纠正我上一轮的预测。** 我说开局 loss 应该在 13–14,实际是 **1.1257**。我错了,
而且错的原因很重要:我那个数是按**随机初始化**的归一化嵌入算的,但两个塔都从
`mobileclip_s2.pt` 载入、**开训前就已经对齐**。证据就在你这份日志里 —— 第 0 个
epoch(只跑 5 步、lr≈2e-6,等于没训)held-out 检索已经是 **R@1 79.6 / 83.0**。
所以 `|dL/dtemp|` 远小于我估的 1.2e3,`scale=54 上限`是随机初始化的最坏情况,
不是这个模型的工作点。已在 `TROUBLESHOOTING.md` 和 base config 注释里更正。

**顺带查清了 `temperature: 0.0100` 为什么一步不动:** `temp` 和 `temp_min` 都是
1/100,而 `clip_contrastive_temperature()` 每次 forward 前都 `clamp_(min=temp_min)`。
对比学习想让温度更低(更锐),每步都被夹回去 —— 所以 `temp` **事实上已经是冻结的**,
却还在贡献完整的 `1/temp²` 放大梯度。付了 fp16 的代价,什么也没买到。这让上一轮
留给你的那个决定从「权衡」变成了「一边倒」:冻结 `temp` 在语义上几乎是空操作。
仍然归你拍板,但不着急 —— 真跑没有 OVERFLOW 就先不用动。

### ⚠️ 一个会影响结论的发现:这个 held-out 评测判不了组件优劣

开训前就 79.6 / 83.0,天花板 100,只剩 ~20 个点;而 1000 条 query 的二项标准误
就有 1.3pp,两次运行差 <3.5pp 全在噪声里。**再加一个混淆项**:视觉塔没冻结,
MobileOne 主干全是 BatchNorm,训练期 running stats 会被这批视频重估一遍,而评测
用的正是 running stats —— R@1 变化可能纯粹来自 BN 统计量适应,不是学到了东西。
两臂同样受影响,A/B 仍然公平,但**「R@1 涨了」不等于「这个组件有用」**。

所以试点判据收窄成两条(已写进 `experiments/README.md` §2b):
**(a) 损失曲线形态;(b) 组件存活检查**(`logit_scale`/`logit_bias` 离开
exp(t')=10、b=−10;`temporal_pool.proj` 权重非零)。R@1 只当参考。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main
nvidia-smi          # 先确认 8 张卡是空的:峰值 res mem 27.2 GiB,余量不多

export MVC_DATA_DIR=/home/vca1/datasets

# 真 A/B 一对(等步数 ~170 步)。不要加 debug,不要传 initial_scale_power ——
# 两个配置继承同一个 base(默认 8),不覆盖才是对称的。单臂 ≈40 min。
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8
```

**Then report back** in from-server.md:

```bash
# 1) 两条损失曲线(每个 epoch 的 Averaged stats 就够)
grep "Averaged stats" experiments/runs/pilot_10k_r0/train.log
grep "Averaged stats" experiments/runs/pilot_10k_r1/train.log

# 2) 跳步对称性 —— 两个数字差得多就说明不是等步数对照
grep -c OVERFLOW experiments/runs/pilot_10k_r0/train.log
grep -c OVERFLOW experiments/runs/pilot_10k_r1/train.log

# 3) r1 的组件存活检查(这才是试点真正要回答的)
#    注意:pilot 配置 stop_key=None ⇒ **不会**生成 ckpt_best.pth;
#    DeepSpeed 存的是**目录** experiments/runs/pilot_10k_r1/ckpt_09.pth/,
#    真正的权重在里面的 mp_rank_00_model_states.pt(和 tasks/pretrain.py:521 同款)。
python - <<'PY'
import glob, torch
d = sorted(glob.glob("experiments/runs/pilot_10k_r1/ckpt_*.pth"))[-1]
print("读取:", d)
sd = torch.load(f"{d}/mp_rank_00_model_states.pt", map_location="cpu")["module"]
for k in ("logit_scale", "logit_bias"):
    if k in sd:
        print(k, "=", sd[k].float().item())
print("  ^ 初值 logit_scale=ln10≈2.3026, logit_bias=-10;没动=different_lr 没生效")
pk = [k for k in sd if "temporal_pool" in k]
print("temporal_pool 参数数:", len(pk), "(0 = 注意力池化没建起来)")
for k in pk[:4]:
    print("  ", k, "absmax =", sd[k].float().abs().max().item())
PY
```

以及最后一个 epoch 的评测表(两臂各一份)。
## 2026-07-22 (10) — 裁判（中转机）：清掉重复进程和重复行，单进程跑完 500

> 这一条只关 **S2 裁判**（中转机 `10.55.87.68`），与上面第 (9) 条的训练崩溃无关，可并行。

**现状：链路已全通，卡的是并发不是端点。** 已实测通过的：端点连通
（`172.30.85.198:8000` / `172.30.62.177:8000` 返回 200，4 卡那个不通）、
conda base 环境零安装（用 `cv2` 后端）、`CLIPS` 指向 b100 上的 10k 子集、
`enable_thinking=False` 生效、冒烟 5/5 无失败。

**踩到两个坑，都已修：**

1. **两个进程在写同一个文件** —— 500 条的输入却有 644 行。
   信号是 `nohup … &` 返回 `[2]`（说明 `[1]` 还在跑）。
2. **看起来卡在 `[progress] 200/476`** —— 不是端点慢，是 `vlm_audit.py` 的两个缺陷：
   `pool.map` 按提交顺序产出，一条慢请求把后面所有已完成结果堵住不写盘
   （`--timeout 600` + 重试 ≈ 堵 20 分钟）；进度又写死每 200 条才打一次。
   **已改成 `as_completed` + `--progress-every 25`**（实测首条落盘从 3.05s → 0.16s）。

**Do this next**（逐条，注意标注的机器）：

```bash
# ① [中转] 只留一个进程
jobs -l; pgrep -af vlm_audit.py
pkill -f vlm_audit.py

# ② [101] 送新版脚本（上面两处修复都在里面，旧版跑还会再卡一次）
cd ~/mobileclip && git pull origin main
scp -P 31179 tools/data_audit/vlm_audit.py root@10.55.87.68:~/judge/
```

```bash
# ③ [中转] 去重：按 video 唯一，成功的判定覆盖失败的
cd ~/judge
python3 - <<'PY'
import json
rows=[json.loads(l) for l in open("s2_verdicts_judge.jsonl")]
best={}
for r in rows:
    v=r.get("video")
    if v is None: continue
    ver=r.get("verdict") or {}
    ok = "action" in ver and "error" not in ver
    if v not in best or (ok and not best[v][1]):
        best[v]=(r, ok)
with open("s2_verdicts_judge.dedup.jsonl","w",encoding="utf-8") as f:
    for r,_ in best.values(): f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(f"原 {len(rows)} 行 -> 去重后 {len(best)} 条唯一 video")
PY
mv s2_verdicts_judge.jsonl s2_verdicts_judge.raw.bak
mv s2_verdicts_judge.dedup.jsonl s2_verdicts_judge.jsonl
```

```bash
# ④ [中转] 单进程跑完剩下的（已完成的按 video 自动跳过）
export CLIPS=/workspace/mount/b100_zaip_data/zyc/vibe_coding/low_resource_cv/MobileViCLIP/outputs/internvid_alignment/caption_translation_qwen36_v0_1/20_outputs/internvid_zh_10k
export JUDGE_EP=http://172.30.85.198:8000/v1
pgrep -af vlm_audit.py || echo "确认无残留进程，可以开跑"
nohup python vlm_audit.py \
  --input s2_judge_sample.jsonl --video-root "$CLIPS" \
  --api-base "$JUDGE_EP" --api-key EMPTY \
  --model /workspace/mount/b100_zaip_data/model_zoo/Qwen3.6-27B \
  --frames 8 --concurrency 8 --frame-backend opencv \
  --output s2_verdicts_judge.jsonl > judge.log 2>&1 &
watch -n 30 'wc -l ~/judge/s2_verdicts_judge.jsonl; tail -3 ~/judge/judge.log'
# 判断是否真卡住：看 wc -l 一分钟内涨没涨，别看进度行
```

```bash
# ⑤ [中转] 跑完先自查（这决定裁判有没有价值），再回传
python3 - <<'PY'
import json, collections
rows=[json.loads(l) for l in open("s2_verdicts_judge.jsonl")]
v=[r["verdict"] for r in rows if "action" in (r.get("verdict") or {})]
def same(a,b): return sum(1 for x in v if x.get(a)==x.get(b))/max(len(v),1)
print(f"n={len(v)}  总行数={len(rows)}（两者应相等，不等说明又有重复）")
print(f"en==trans: {same('en_score','trans_score'):.1%}   (主力 8B 92.5%)")
print(f"zh==trans: {same('zh_score','trans_score'):.1%}   (主力 8B 98.9%)")
print("action 分布:", dict(collections.Counter(x.get("action") for x in v)))
PY

# ⑥ [101] 取回并算一致率 + Cohen's κ
scp -P 31179 root@10.55.87.68:~/judge/s2_verdicts_judge.jsonl outputs/audit_10k/
bash tools/data_audit/run_v100.sh judge-agree
```

**一个已经站住的中间结论**（n=13 时）：`en==trans` **30.8%** vs 主力 **92.5%**、
`zh==trans` **30.8%** vs **98.9%**。若裁判和 8B 一样塌缩，这个结果的概率是
`4e-08` / `2e-15` —— 即**裁判确实在独立评估三个维度，halo 是 8B 这个模型的毛病，
不是 rubric 的毛病**。它的 `trans_score` 因此可以拿来裁决决策 1。

**一个预测**：action 分布已透出 T4 会怎么走 —— 主力最大桶是 `retranslate`（41.2%），
裁判至今一条没给、全判 `recaption`。**五类一致率大概率过不了 75%**，
但二值口径（主力 `keep+fix_text` 35.7% vs 裁判 30.8%）可能没问题。
真出现这个结果**别简单判 T4 不达标**——它说明的是主力的 `action` 不可信，
而 `apply_buckets.py` 正是按 action 分流任务队列的。

**Then report back** in `from-server.md`：⑤ 的自查输出（n / 两个同值率 / action 分布）、
错误率、以及 ⑥ `judge-agree` 打印的三个门和各维度 κ。拿到就能把 T4 和决策 1 一起结掉。

完整手册：[`tools/data_audit/SETUP_JUDGE_RELAY.md`](../../tools/data_audit/SETUP_JUDGE_RELAY.md)
（§6 跑 500、§7 回传算分、§9 常见问题含「重复写入」一节）。

---

## 2026-07-22 (9) — 崩溃是 `debug True` 自己造成的;顺带查出 temp 把 loss scale 压死

**卡空出来了,bs=64 装得下,前向也跑完了** —— 这次死在**反向**。报错位置很具体:

```
File "models/criterions.py", line 52, in get_sim
    sim_v2t = vision_proj @ text_proj.T / temp
RuntimeError: Function 'DivBackward0' returned nan values in its 1th output.
```

**「1th output」要读准**:`DivBackward0` 有两个输出,0 是对分子的梯度,**1 是对
分母的梯度** —— 也就是 `self.temp`(可学习参数,初值 0.01)。NaN 恰恰在温度的梯度上。

**这里其实是两件事,得分开说。**

**① 为什么会「崩」—— 是 `debug True` 把预期内的事件变成了致命错误。**

`pretrain.py` 原本只要 `debug` 为真就开 `torch.autograd.set_detect_anomaly(True)`。
它和 fp16 **动态 loss scaling** 根本不兼容:动态缩放的做法就是先给一个偏大的初始
系数,**故意**让前几步溢出,DeepSpeed 检出 inf/nan 后跳过该步、系数减半(就是日志里
那句 `OVERFLOW! ... Skipping step`)。而 anomaly detection 一见反向出现 nan 就立刻
抛异常 —— 赶在 DeepSpeed 检查梯度**之前**,于是自愈机制没机会生效。

**已修**:现在 deepspeed+fp16 时自动跳过 anomaly(并打日志说明);bf16 / fp32 /
不用 deepspeed 时照常开启,那些场景下它是安全且有用的。

**② 为什么会有 NaN —— 可学习的 temp 把可用 loss scale 压到了 54。**

`sim = (v @ tᵀ) / temp`,temp=0.01 让温度的梯度带 `1/temp² = 1e4` 的放大。
我在本机用 fp32 实测(B=512,即 all_gather 后的全局批,随机归一化嵌入):

| | |
|---|---|
| `\|dL/dtemp\|` | **1.2e3** |
| fp16 上限 | 65504 |
| **temp 不溢出所允许的最大 loss scale** | **≈ 54** |
| 塔内嵌入允许的最大 scale(`\|dL/dvision_proj\|` 最大 5.7e-2) | ≈ 1.15e6 |

DeepSpeed 默认 `initial_scale_power: 16`(=65536)因此要**连续减半 11 次**才可能有
一步落地;而 `debug` 模式**只跑 5 步**就 break。就算不崩,冒烟也全耗在退避上、
什么都看不到。

**已修**:`initial_scale_power` 改为从配置读取(默认降到 **8**=256),也能命令行覆盖。
loss scaling 纯粹是数值手段(更新前会反缩放回去),**不改变实验语义**,只决定开头
浪费几步。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main

export MVC_DATA_DIR=/home/vca1/datasets
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 \
    deepspeed.initial_scale_power 5 debug True
```

(`5`=32,刚好在实测可用区 54 以内,开头基本不浪费步数。想更稳可以不带这一项,
用默认的 8,代价是多两三步退避 —— 但只有 5 步预算,所以我建议先用 5。)

**Then report back:** `Train Epoch: [0]` 之后带 `loss_vtc` / `lr` / `temperature`
的那几行。判据:
- loss_vtc 是个**有限数**(初值应在 13–14 附近,和我本机算的 13.93 同量级)就算通了;
- 开头有一两条 `OVERFLOW! Skipping step` 仍属正常;
- 若 5 步全是 OVERFLOW,把日志里的 loss scale 数值贴回来。

**还有一件要你拍板的事(我没动)。** 上面那张表说明:**光是可学习的 `temp` 就把
可用 loss scale 压低了约 2 万倍**(54 vs 1.15e6)。其它所有参数的梯度因此只能按
~32 缩放而不是 ~1e6,小梯度会更靠近 fp16 下溢。这是 Volta+fp16 在这种参数化下的
真实代价。两条标准出路:把温度改成 **log 参数化**(r1 的 `logit_scale` 已经是这么做的,
放大从 `1/temp²` 降到 `1/temp`),或者**冻结 `temp`**。两者都会改变 r0 的定义,
所以我不擅自动手 —— 等正式跑起来后看 loss scale 稳定在多少,低于 ~16 就该处理。

---

## 2026-07-22 (8) — dtype 那条线**全通了**;这次 OOM 是因为卡上还跑着 S2 审核

**先说结论:训练代码这边已经没问题了。** 三个 dtype 修复全部生效,前向真正跑进了
视觉塔的 stage-2/3 做实际计算,报错性质完全变了:`torch.OutOfMemoryError`。

**Diagnosis:别改配置,先看报错的第二、三行。**

```
GPU 4 has a total capacity of 31.73 GiB of which 404.50 MiB is free.
Process 2621830 has 16.93 GiB memory in use.          <- 这不是我们的进程
Including non-PyTorch memory, this process has 14.39 GiB memory in use.
```

把账算一下:

| | |
|---|---|
| 单卡总量 | 31.73 GiB |
| **别人的进程** | **约 16.9–17.0 GiB** |
| 我们的训练进程 | 14.36–14.52 GiB(PyTorch 已分配 12.46 GiB) |
| 死时剩余 | 268–436 MiB(它要 480 MiB) |

17.0 + 14.5 ≈ 31.5 —— **卡是被两个任务塞满的**。空卡上 bs=64 只要约 14.5 GiB,
还剩 17 GiB 富余。**训练配置本身一点问题都没有。**

**占卡的是谁:** PID **2621826–2621833,连续 8 个,一卡一个** —— 这正是
`run_v100.sh s2` 的形状(每张卡起一个 `vlm_audit_hf.py`,`CUDA_VISIBLE_DEVICES=$i`),
而 Qwen3-VL-8B 的 fp16 权重就是约 16 GB,`DATA_ISSUES.md` 里 S2 也标着「运行中」。
**是 S2 审核在跑。** 另外 GPU 0 和 GPU 6 上还挂着约 11 个 ~404 MiB 的残留进程,
值得顺手清掉,但不是这次 OOM 的原因。

**Do this next —— 两条路,建议选第一条:**

**① 让 S2 先跑完(推荐)。** 它是整条审核流水线的阻塞项(后面每一步都吃它的产物),
还剩不到 2–3 h。跑完再用空卡按记录的 bs=64 跑试点。先看进度:

```bash
nvidia-smi
bash tools/data_audit/run_v100.sh status
```

**② 想现在就把训练循环验通**,用小 batch 挤进 S2 剩下的约 14.7 GiB:

```bash
cd ~/mobileclip && git pull origin main

export MVC_DATA_DIR=/home/vca1/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 有 654 MiB 是已保留未分配的碎片
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 \
    batch_size 16 debug True
```

命令行覆盖 `batch_size` **确实会传到 dataloader**:`utils/config.py:get_config`
里 `merge_list(cfg, args.opts)` 在 `eval_dict_leaf(cfg)` **之前**执行,所以
`inputs.batch_size.{image,video}` 那个 `"${batch_size}"` 插值是按覆盖后的值解析的
(我查过源码顺序,不是猜的)。

> ⚠️ **但 batch size 不是自由旋钮。** 它决定批内负例数量,直接改变对比任务本身 ——
> bs=16 和 bs=64 的结果不可比。**小 batch 只能用于冒烟**;真正的 r0/r1 那一对
> 必须同批大小,优先用 64/卡(全局 512)。这条已写进 `experiments/README.md` §2b。

**Then report back:** 若走②,给我 `Train Epoch: [0]` 之后带 `loss_vtc` / `lr` /
`temperature` 的头十几行 —— 那就说明全链路通了,只等空卡跑正式 A/B。判据同前:
开头几条 `OVERFLOW! ... Skipping step` 正常,`nan` 或 loss scale 掉到底才是问题。

---

## 2026-07-22 (7) — 视觉塔通了,死在文本塔的 LayerNorm;同族的雷已一次扫干净

**又往前走了一层。** 上次死在视觉塔第一个卷积,这次那里过了,死在**文本塔的
第一个 LayerNorm** —— 同一个病根(DeepSpeed 关掉外层 autocast),换了个受害者。

**Diagnosis:**

```
File "models/backbones/internvideo2/mobileclip/transformer.py", line 46, in forward
    return super().forward(x.to(torch.float32)).to(inp_dtype)
RuntimeError: expected scalar type Float but found Half
```

`LayerNormFP32` 只把**输入**转 fp32,`self.weight` / `self.bias` 被 `model.half()`
变成了 fp16,而 `F.layer_norm` 要求输入与参数同 dtype。**这个类从来就不自洽**,
只是以前有外层 autocast 替它兜底 —— `layer_norm` 在 autocast 的 fp32 名单里,
而该名单会把**所有**输入(含 weight/bias)一起转 fp32。engine 关掉 autocast,
它就得自己做完整。

**Fix(已合进 main,pull 即可):** 两个副本(`mobileclip/transformer.py` 和
`mobileclip/modules/common/transformer.py`,近似重复文件)都改成参数一并转 fp32。
本地验证:旧写法对 `.half()` 模块复现同类报错,新写法与全 fp32 LayerNorm 再转回
fp16 **逐位相等**,`elementwise_affine=False`(参数为 None)分支也正常。

**这次顺手把同族的雷一次扫完了** —— 起了个子代理把 r0 整条前向路径(视觉塔、
文本塔、VTC 损失、all-gather)全读了一遍,专找「靠 autocast 兜底」的写法:

- **不会再有 dtype 崩溃**。文本塔注意力里的 `softmax(attn.float())` 是**对的**
  (转回后才做矩阵乘,且 softmax 无参数),不用动。
- **但有一处静默的精度回退**:VTC 损失里的 `log_softmax` / `sum` 本来在 autocast
  的 fp32 名单里,现在退化成 fp16。已在 `models/criterions.py` 把相似度矩阵转回
  fp32 再进 `log_softmax`。

  **说清楚它到底值多少**:实测 B=512、temp=0.01 下,损失 fp16 `13.929688`
  vs fp32 `13.929939`,只差 **2.5e-4**。`log_softmax` 内核本来就用 fp32 累加,
  所以**不会**出现我上一条里担心的「阶梯状损失」。留这两行只是把 r0 基线恢复成
  仓库原本的意图,顺便在 loss scale 出问题时少一个嫌疑人 —— 不要指望它改变什么。
- **有一处我改了又撤回**:视觉塔的注意力 softmax 也没有 fp32 提升,看着像同一个
  雷,实测却与直接 fp16 softmax **逐元素差 0.0**(那个内核同样 fp32 累加)。
  纯空操作,撤了,不留在 diff 里当噪音。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main

export MVC_DATA_DIR=/home/vca1/datasets
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

`make_pilot_anno.py` **不用再跑**(你刚跑过,9000 + 1000 已经在位;那次输出里的
「共享字幕 0.0%」和下面的抽样率提示是新加的说明文字,不是异常)。

**这次仍然盯损失。** 判据同上一条:开头刷几条 `OVERFLOW! ... Skipping step`
正常;loss 从第 0 步就 `nan`、或 loss scale 一路掉到 `min_loss_scale` 不动才是
真问题。

**Then report back:** `Train Epoch: [0]` 之后带 `loss_vtc` / `lr` / `temperature`
的头十几行,以及 17 个 batch 跑完有没有干净退出。若又崩,traceback 最后 30 行。

---

## 2026-07-21 (6) — DeepSpeed 起来了、训练循环也起来了;死在第一次 forward 的 dtype

**大进展:** FusedAdam 那关过了 —— `Use deepspeed to initialize model!!!` 之后
显存分配正常(211M),`Start training` / `Epoch: 0` / `MetaLoader has 1
dataloaders, 17 batches in total` 全部打出来了。**优化器、DeepSpeed、数据集、
decord 这四层现在都是通的**,只差最后一步。

**Diagnosis:** 8 个 rank 全部死在第一个卷积:

```
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same
```

关键线索是崩溃前那行 warning:

```
[WARNING] [torch_autocast.py:122:autocast_if_enabled] torch.autocast is enabled
outside DeepSpeed but disabled within the DeepSpeed engine.
```

仓库的混合精度是「**模型整体 `.half()` + 训练循环外套一层 `torch.cuda.amp.autocast`**」:
权重是 fp16,batch 还是 fp32,靠 autocast 在算子层面把输入转过去。而新版 DeepSpeed
的 `engine.forward` 会**显式关掉外层 autocast**(`torch_autocast.py` 的
`autocast_if_enabled`:没配 `torch_autocast` 段时,用 `enabled=False` 覆盖外层
上下文)。于是那层 autocast 在 DeepSpeed 路径上成了空操作,fp32 输入直接撞上
fp16 权重。**这是 DeepSpeed 版本回归,不是机器配置问题,也跟 Volta/sm_70 无关。**

**Fix(已合进 main,pull 即可):** `tasks_clip/pretrain.py` 在 forward 前把 batch
转成**模型参数的真实 dtype**。我在本地用 `nn.Conv2d(...).half()` 复现了同一类
报错并验证了修复,fp32 路径下是空操作。

用「模型参数的 dtype」而不是现成的 `data_type`,是因为顺手查出精度开关在仓库里
**有三处、两个键名、两个默认值**,只有配置把它们都显式写对时才一致:

| 文件 | 键 | 默认 | 决定 |
|---|---|---|---|
| `shared_utils.py:57` | **`bf16`** | `True` | 模型转 bf16 还是 fp16 |
| `pretrain.py` | `use_bf16` | `True` | autocast 的 `data_type` |
| `config_utils.py:130` | `use_bf16` | `False` | ds 配置写 bf16 段还是 fp16 段 |

试点配置三个都显式写了 False,所以这次没出事;但以后新配置只写一个就会踩雷。
直接问模型要 dtype 就绕开了整个问题。(细节全部记进 `TROUBLESHOOTING.md`。)

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main

export MVC_DATA_DIR=/home/vca1/datasets
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

**这次要看的是损失能不能正常打出来。** 心里先有个数:

- 外层 autocast 被 DeepSpeed 关掉后,前向是**纯 fp16**,靠 DeepSpeed 的动态
  loss scaling(`initial_scale_power: 16`)兜底。**开头刷几条
  `OVERFLOW! ... Skipping step` 是正常的**,那是缩放系数在找合适的量级。
- **不正常的是**:loss 从第 0 步就是 `nan`,或者 loss scale 一路掉到
  `min_loss_scale` 就再也不动了。真出现这个,就得把对比损失(`log_softmax`
  那两行)single out 到 fp32 算 —— 但先别改,让数据说话。

**Then report back:** `Train Epoch: [0]` 之后的头十几行(带 `loss_vtc`、`lr`、
`temperature` 的那些),以及跑完 17 个 batch 有没有干净退出。若又崩,把 traceback
最后 30 行贴回来即可。

---

## 2026-07-21 (5) — 更正:那个「共享字幕 0.0%」不能当成「全量重复率低」

**问的是:** 为什么共享字幕会是 0%?

**答:主要是抽样造成的,不是数据的性质。** 上一条笔记里我写的
「多正例要等全量数据才谈得上验证」说对了一半,但结论下得太软 —— 准确的说法是:
**这个 0.0% 对全量几乎没有任何约束力,不能拿它去判 `DATA_ISSUES.md` 里
「重复 caption 率 >10%」那个决策门。**

三条原因,按重要性排:

1. **抽样率只有 0.1%(10k / 10M),这是决定性的。** 一组出现 2 次的重复字幕,
   要**两条都**被抽中才看得见,概率 = 0.001² = **1e-6**。假设全量有 5% 的
   clip(50 万条 = 25 万组)处在重复对里,这 10k 里的期望命中 =
   25 万 × 1e-6 ≈ **0.5 组**。实测 1 组 —— 和「全量 5% 重复」完全相容。
   换句话说,**观测 0.0% 和全量重复 5%,在统计上区分不开**。
   这个抽样只对**全局重复上千次**的模板句有检出力(k=1000 时 P(≥2)≈26%),
   对 k=2…100 的中低频重复完全瞎。
2. **只算精确重复。** 训练侧 `caption_to_idx` 是 `pre_text` 归一化后取 md5,
   "a man is talking to the camera" 和 "a man talks to the camera" 是两个 idx。
   语义近重复一律不计。(顺带修了个小口径 bug:`make_pilot_anno.py` 之前按
   **原始**串统计,比训练口径还严;现已改成先过同一套归一化再计数,
   held-out 评测集的判重也跟着改了。)
3. **InternVid 的构造方式本来就不产生共享字幕。** 每个 clip 由 captioner 独立
   生成描述,不存在 MSR-VTT「一个视频 20 条 caption」那种结构性重复。

**没变的结论:** 试点 A/B 里 `caption_idx` 仍然是空操作(全库就一对同字幕 clip,
落进同一 batch=512 的概率单个 epoch 才 ~5.7%),所以 **r1 - r0 的差异照旧只能
归给 sigmoid 损失 + 注意力池化 + t/b 大学习率**。这点不受上面影响。

**要真判多正例值不值得做,唯一的办法是全量跑一遍:**

```bash
# 只有拿到全量 10M jsonl 时才做(H100 那边),CPU ~20 分钟
python tools/data_audit/profile_jsonl.py --input <全量 jsonl> --out outputs/audit_v0_1/s0_summary.json
python -m json.tool outputs/audit_v0_1/s0_summary.json | grep -A3 duplicate
```

看 `duplicate_zh_caption_rate`。注意口径:它数的是 **zh** 侧,试点训练喂的是
**en** 侧字幕;英文重复经逐条翻译可能被打散,所以它是英文侧重复率的**下界**。

**另外提一个可能更值钱的方向(留给 v0.2,先不动):** InternVid 里同一支源视频
被切成多个 clip,画面高度相似但字幕各不相同 —— 它们在 batch 里互为负例,是
**真的**假负例,而按字幕哈希的 `caption_idx` 一条也抓不到。若全量重复率证明偏
低,比「共享字幕多正例」更值得做的是**按源视频 id 分组**(同 `video_id` 共享
idx,或采样器层面禁止同视频 clip 进同一 batch)。

**这条不需要服务器做任何事** —— 推导已落到 `notes/audit-10k-findings.md`
结论 5、`experiments/README.md` §2b、`tools/data_audit/DATA_ISSUES.md`。
**当前待办仍然是下面 (4) 里那三行命令。**

---

## 2026-07-21 (4) — anno 成了;死在 DeepSpeed 现编 FusedAdam(nvcc 配不上 gcc 11)

**先说好消息:** anno 这步彻底通了 —— `可用 10000 / 缺失 0 / root 外 0`,
9k train + 1k val 都写出来了。冒烟也比上次走远得多:数据集建好(9000 条已序列化)、
`mobileclip_s2.pt` 双塔权重加载、优化器分组(458 + 875 个参数)全过,
说明 sm_70、路径、配置这三层都对了。

**Diagnosis:** 新报错在 `deepspeed.initialize` → `FusedAdam` 这一步。
DeepSpeed 默认的 Adam 是 **FusedAdam**,它要在首次运行时用**系统 `/usr/bin/nvcc`
现编一个 CUDA 扩展**(不是 torch wheel 里自带的那份 CUDA)。101vector 上系统
nvcc 比系统 g++ 老,老 nvcc 解析不了 gcc 11 的 `<functional>`,于是:

```
/usr/include/c++/11/bits/std_function.h:435:145: error: parameter packs not expanded with ‘...’
```

其余 7 张卡刷的 `fused_adam.so: cannot open shared object file` 是**下游噪音** ——
只有一张卡负责编译,其它卡在等那个从没被生成出来的 `.so`。编译修好就一起没了。
跟上次的 Volta 问题**无关**:这条 nvcc 命令行里 `-gencode=…sm_70` 是对的。

**修法(代码里已经改好,你只要 pull):** `utils/config_utils.py` 现在往生成的
`deepspeed_config.json` 里写 `"torch_adam": true`,DeepSpeed 就改用
`torch.optim.AdamW` —— 同一个算法,只是没有融合核,**什么都不用编译**。
(顺带修了两个坑:`adam_w_mode` 原来写在 `optimizer` 顶层,DeepSpeed 只从
`params` 里 pop,一直被静默忽略;还有 `bias_correction` 是 FusedAdam 专有参数,
原样传给 torch AdamW 会直接 `TypeError`,torch 路径下已去掉。)

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main

# 上次失败留下的半成品编译缓存,清掉免得干扰
rm -rf ~/.cache/torch_extensions/py310_cu118/fused_adam

export MVC_DATA_DIR=/home/vca1/datasets
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

anno 那两个 json 已经生成好了,**不用再跑 `make_pilot_anno.py`**。

**顺手记一笔(重要,别忘):** 共享字幕 = **1 组 / 2 条 clip(0.0%)**。
意味着 r1 里的 **`caption_idx` 多正例在这批 10k 上是空操作** —— 批内基本撞不上
同字幕的两条 clip。所以试点 A/B 的差异只能归给 sigmoid 损失 + 注意力池化 +
t/b 大学习率这三项,**不能记到多正例头上**;多正例要等全量数据才谈得上验证。
这条已经写进 `experiments/README.md` §2b。

**Then report back** in from-server.md:冒烟的头几十行 —— 具体看
`Use deepspeed to initialize model!!!` 之后有没有顺利进到训练循环、
损失有没有正常打印。若还卡在编译上,把 `nvcc --version` / `g++ --version` /
`echo $CUDA_HOME` 三条的输出一起贴回来。

---

## 2026-07-21 (3) — `没有可用记录` 已修:片段在 InternVId-*/ 嵌套目录里

**Diagnosis:** 不是你命令打错,是 `make_pilot_anno.py` 的 bug。片段**不是平铺在
`videos/` 下**,而是在嵌套的 `InternVId-*/…` 目录里,jsonl 的 `video` 字段常常只是
裸文件名,真正定位文件的是记录里的 `abs_path`。审查工具一直有这套三候选解析
(`tools/data_audit/score_alignment.py: resolve_video_path()`),但
`make_pilot_anno.py` 只做了朴素的 `os.path.join(clips_root, video)` ——
于是 1 万条全部判为「文件缺失」。而且它在打印计数**之前**就 `sys.exit`,
所以那句话没法区分「字段名不对」还是「路径不对」。两个问题都已修并合进 `main`。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main     # 必须先拉,否则还是旧脚本

python tools/make_pilot_anno.py \
    --input /home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl \
    --clips-root /home/vca1/datasets/internvid_zh_10k/videos \
    --out-train anno_downstream/internvid_zh_10k_pilot.json \
    --out-eval  anno_downstream/internvid_zh_10k_pilot_val.json

ls -l anno_downstream/internvid_zh_10k_pilot*.json

# 成功后接 §2b 第 2–4 步
export MVC_DATA_DIR=/home/vca1/datasets
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

**若还是没记录**:新脚本会打印**分项计数** + 首条记录的**真实字段名** + 试过的路径。
把那一整段贴回来,一轮就能定位。对应处理见 `TROUBLESHOOTING.md`:
- 缺 video/caption 多 → 用新加的 `--video-key` / `--caption-key` 指定真实字段名;
- 文件缺失多 → `--clips-root` 指错了,或记录没有 `abs_path`;
- `root 外` 多 → 文件在 `--clips-root` 之外,anno 写不出相对路径。

⚠️ **不要**因为跑不通就改 `--caption-field zh`。文本塔是 CLIP 英文 BPE,
中文字幕喂进去是**另一个实验**,不是等价替代,会让 r0/r1 对照失效。
英文字幕若真的缺失,先报回来再一起定。

**Then report back** in from-server.md:脚本打印的那两行(可用条数 / 共享字幕比例),
或失败时的完整分项计数。

---

## 2026-07-21 (2) — sm_70 修好了;现在缺的是试点 anno(§2b 第 1 步没跑)

**Diagnosis:** 好消息:**torch 的问题解决了**。这次报错发生在
`setup_dataloaders` → `create_dataset`,说明 CUDA 初始化、模型构建、
`init_distributed_mode` 全都过了 —— 上一条的 sm_70 修复生效了。

现在只是 `anno_downstream/internvid_zh_10k_pilot.json` **还没生成**:它不在仓库里、
也不在审查抽样目录里,要用 `tools/make_pilot_anno.py` 从审查用的 10k jsonl 现做,
也就是 `experiments/README.md` §2b 的**第 1 步**。直接跳到训练命令就会缺这个文件。

**路径本身没问题** —— 已逐项对过 `tools/data_audit/`:

| 项 | 值 | 出处 |
|---|---|---|
| 审查 jsonl | `/home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl` | `run_v100.sh` `$DATA` |
| 视频根目录 | `/home/vca1/datasets/internvid_zh_10k/videos` | `run_v100.sh` `$CLIPS` |
| `data_root` | `$MVC_DATA_DIR/internvid_zh_10k/videos` | `configs/data.py` |
| `anno_path` | `$MVC_ANNO_DIR/internvid_zh_10k_pilot.json` | 同上,`MVC_ANNO_DIR` 默认 `anno_downstream` |

只要 `MVC_DATA_DIR=/home/vca1/datasets`,`data_root` 就正好是审查工具用的那个
`videos/`,和脚本写出的相对 `video` 字段对得上(脚本会拿 `--clips-root` 逐条校验存在性)。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin handoff

# §2b 第 1 步(只需跑一次)
python tools/make_pilot_anno.py \
    --input /home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl \
    --clips-root /home/vca1/datasets/internvid_zh_10k/videos \
    --out-train anno_downstream/internvid_zh_10k_pilot.json \
    --out-eval  anno_downstream/internvid_zh_10k_pilot_val.json

ls -l anno_downstream/internvid_zh_10k_pilot*.json     # ~9k + 1k

# §2b 第 2–4 步
export MVC_DATA_DIR=/home/vca1/datasets
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

⚠️ 看一眼脚本打印的两行再往下走:
- `文件缺失 M` 若很大 → jsonl 的 `video` 字段不是相对 `--clips-root` 的(可能带子目录
  或绝对路径),先看 schema:
  `head -1 /home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl | python -m json.tool`
- `共享字幕: … 组 / 覆盖 … 条 clip (…%)` → 这就是 `caption_idx` 多正例修复在这批数据上的
  影响面上限,**记下来**,后面 r1 结论要用。

**Then report back** in from-server.md:`make_pilot_anno.py` 打印的那两行(可用条数 /
文件缺失 / 共享字幕比例),以及冒烟跑起来后损失有没有正常打印。

---

## 2026-07-21 — pilot_10k_r0 起不来:torch 没有 Volta 内核(**不是 deepspeed 的问题**)

**Diagnosis:** deepspeed 已经装好了(0.19.2,import 正常),这一步已经过去了。
新报错是**另一个更底层的问题**:`mobileviclip` 环境里的 **torch 2.11.0 是 CUDA 13
构建**(依赖里能看到 `cuda-toolkit 13.0.2` / `nvidia-cudnn-cu13` / `nvidia-nccl-cu13`)。
CUDA 13 已经**彻底删掉 Volta(sm_70)**,所以 warning 里列出的架构是
`7.5 / 8.0 / 8.6 / 9.0 / 10.0 / 12.0` —— **没有 7.0**。V100 正是 CC 7.0,
于是**任何一个 CUDA kernel 都跑不了**,第一个碰到的 `torch.distributed.barrier()`
就炸了(`utils/distributed.py:119`),8 张卡全一样。

所以 `deepspeed.enable False` 这条旁路救不了:崩溃发生在 `init_distributed_mode`,
比选不选 deepspeed 更早,跟框架无关。这正是 `tools/data_audit/SETUP_V100.md` §0
早就写过的 V100 三大坑之一 —— 当时只对**审查环境**做了处理,**训练环境
`mobileviclip` 漏了同样的处理**。

**Do this next:**

```bash
# 0) 先拉最新(本条修复 + §2b 的 sm_70 预检都在里面)
cd ~/mobileclip && git pull origin handoff

conda activate mobileviclip

# 1) 卸掉 CUDA-13 构建
pip uninstall -y torch torchvision torchaudio

# 2) 换成带 sm_70 的 cu118 构建(版本与审查环境一致、已在本机验证过;
#    torch>=2.5 才能配 deepspeed 0.19)
pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu118

# 3) decord 0.6 在 numpy 2 下会报 _ARRAY_API,数据加载器是 decord 的
pip install "numpy<2"

# 4) 自检:arch list 里必须出现 sm_70,且 .cuda() 不报错
python -c "import torch, decord; print(torch.__version__, torch.cuda.get_arch_list()); \
  print(torch.zeros(1).cuda(), torch.cuda.get_device_name(0))"
python -c "import deepspeed; print('deepspeed', deepspeed.__version__)"

# 5) sm_70 出现后再重开冒烟
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

**冒烟过了之后**:回 `experiments/README.md` §2b 按原顺序续跑,不用另记命令 ——
§2b 的 4 步已经是完整的(1 建 anno → 2 `export MVC_DATA_DIR` → 3 sm_70 预检 →
4 冒烟 + 等步数 A/B)。两个提醒:

- §2b 第 1 步 `tools/make_pilot_anno.py` **必须先跑过一次**,否则 r0 配置找不到
  `anno_downstream/internvid_zh_10k_pilot.json`;已经跑过就跳过。
- 第 4 步的等步数 A/B 是**成对**的,两条都要跑完才能比:
  `pilot_10k_r0.py 8` 与 `pilot_10k_r1.py 8`。

⚠️ 以后**不要再在这个环境里 `pip install -U torch/torchvision/torchaudio`,
也不要裸 `pip install torchvision`** —— PyPI 默认 wheel 是 CUDA 12.8/13 构建,
一升就把 Volta 支持删掉。101vector 上任何 torch 升级都必须走 cu118 index。

顺带:`FusedMLP of flash_attn is not installed!!!` 这类是**正常回退提示**不是错误
(sm_70 装不了 flash-attn,见 README_zh Q9);V100 无 bf16 已在配置里处理好
(`use_bf16=False` / `bf16=False`,fp16 AMP)。

**Then report back** in from-server.md:第 4 步自检的完整输出(尤其 arch list 里
有没有 `sm_70`),以及第 5 步冒烟跑起来后的头几十行(损失有没有正常打印)。

完整版已写进 `TROUBLESHOOTING.md` 末节;`experiments/README.md` §2b 也加了一条
开跑前的 sm_70 预检命令。

---

## 2026-07-23 · 10k 试点跑完 —— 只差一条 ckpt 核实命令(ROADMAP 1.3 判据二三)

`pilot_10k_r0` / `pilot_10k_r1` 都**正跑完**了(结果 + go/no-go 记在
`experiments/pilot_10k_results.md`)。**r0 全过;r1 判据一(损失 7.07→4.2 单调降)过**。
r1 的 held-out 分数低于 r0 是**预期**(新 attn 池化 init + sigmoid 重标定 + 仅 170 步),
按两级设计 10k 不用来排方案,只看动力学。

只剩判据二三要读 r1 的 ckpt —— **只能你在服务器上跑**,Claude 端无 GPU/ckpt。

⚠️ ckpt 是 **DeepSpeed 目录**不是单文件:`ckpt_09.pth/` 里权重在
`mp_rank_00_model_states.pt` 的 `module` 键下。直接 `torch.load` 目录会 `IsADirectoryError`。

```bash
# 仓库根目录，conda activate mobileviclip
python - <<'PY'
import torch
f = "experiments/runs/pilot_10k_r1/ckpt_09.pth/mp_rank_00_model_states.pt"
obj = torch.load(f, map_location="cpu")
sd = obj.get("module", obj)
ls, lb = sd.get("logit_scale"), sd.get("logit_bias")
w  = sd.get("vision_encoder.temporal_pool.proj.weight")
print("present:", ("logit_scale" in sd), ("logit_bias" in sd),
      ("vision_encoder.temporal_pool.proj.weight" in sd))
print("exp(logit_scale) =", None if ls is None else round(ls.float().exp().item(),4), " (init 10 → ≠10)")
print("logit_bias       =", None if lb is None else round(lb.float().item(),4),       " (init -10 → ≠-10)")
print("temporal_pool.proj.weight L2 =", None if w is None else round(w.float().norm().item(),4),
      " mean|w| =", None if w is None else round(w.float().abs().mean().item(),6), " (非零)")
PY
```

判据:`exp(logit_scale)≠10` 且 `logit_bias≠-10`(**关键** —— 证明 `different_lr @1e-3`
在更新它们)、proj 范数非零。三行都满足 → **1.3 全过**,V100/10k 链路筛完,下一步去
H100 跑全量 A/B(`experiments/SETUP_B100.md`,r1a/r1b 分开跑)。顺带
`pytest tests/test_optimization_components.py`。

**把这三行输出贴回 `from-server.md`**,我据此把 1.3 状态 🟡→✅。
细节见 `experiments/pilot_10k_results.md`。
