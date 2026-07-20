# 跨机裁判部署手册：H100 起 235B 服务，V100 走 API 调用

> 目标：在 8×H100-80G 机上用 vLLM 起 **Qwen3-VL-235B-A22B-Instruct-FP8**
> 的 OpenAI 兼容服务，让 V100 机（101vector，10.235.31.101）的
> `s2-judge` 裁判步骤通过内网 API 调用它——**零云端费用**，且两台机器的
> 裁判判定出自同一个模型，校准结果可互相复用。
>
> 前置条件只有一个：**两台机器内网互通**（§3 有验证方法）。
> 配套手册：[`SETUP_H100.md`](SETUP_H100.md)（H100 全量审查）、
> [`SETUP_V100.md`](SETUP_V100.md)（V100 10k 审查）。

## 总览

```
V100 机 (101vector)                     H100 机
┌─────────────────────┐    内网 HTTP    ┌──────────────────────────┐
│ run_v100.sh s2-judge │ ─────────────> │ vllm serve 235B-FP8      │
│ (vlm_audit.py 抽帧、 │  8帧JPEG/条    │ --port 8001 --host       │
│  base64、发请求)     │ <───────────── │ 0.0.0.0（tp8 或 tp4）    │
└─────────────────────┘   JSON 判定     └──────────────────────────┘
```

每条请求只发 8 帧压缩 JPEG（约 1–2MB），抽帧解码都在 V100 本地做，
内网千兆带宽绰绰有余。

---

## 1.（H100 机）环境与权重

已按 `SETUP_H100.md` §0 装好环境的直接跳到 §2。

### 1.1 Python 环境（~10 分钟）

```bash
# venv 优先（国内 conda 镜像不可靠）；base 的 python ≥3.9 即可
python -m venv ~/audit_env && source ~/audit_env/bin/activate
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

pip install "vllm>=0.11"        # 会自动带上匹配的 torch cu12x
pip install "transformers>=4.57" decord pillow requests
# 若 decord 报 numpy _ARRAY_API 错误：pip install "numpy<2"

# 验收点：应打印 True 和 H100
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

网络不通（pip/HF 被 reset）时，先按 `SETUP_H100.md` 开头的"联网"段配好
代理，或参考 `README.md`「网络受限机器」一节换源/离线安装。

### 1.2 下载 235B-FP8 权重（~235GB，放后台，看带宽 0.5–2h+）

```bash
pip install modelscope
mkdir -p ckpts
df -h .    # 验收点：磁盘剩余 ≥300GB 再开下
nohup modelscope download --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --local_dir ckpts/Qwen3-VL-235B-A22B-FP8 > download_235b.log 2>&1 &
tail -f download_235b.log   # 断了重跑同一命令即可断点续传
```

**验收点**：`du -sh ckpts/Qwen3-VL-235B-A22B-FP8` 约 235G，
且目录里有 `config.json` 和一批 `*.safetensors`。

---

## 2.（H100 机）起服务

两种方案二选一。关键参数是 `--host 0.0.0.0`（监听所有网卡，内网才能
访问）；`--port 8001` 与主力档的 8000 错开。

### 方案 A：8 卡独占 tp8（H100 空闲时，吞吐最高）

```bash
source ~/audit_env/bin/activate
nohup vllm serve ckpts/Qwen3-VL-235B-A22B-FP8 --port 8001 --host 0.0.0.0 \
  --tensor-parallel-size 8 --limit-mm-per-prompt image=8 --max-model-len 32768 \
  > judge_serve.log 2>&1 &
```

### 方案 B：GPU 对半（H100 同时在跑自己的 S2 主力时）

8B 主力 dp=4 占 GPU0-3，235B tp=4 占 GPU4-7。tp4 时每卡 ~59GB 权重，
KV 余量紧，`--max-model-len` 不要超 16384：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup vllm serve ckpts/Qwen3-VL-8B-Instruct \
  --port 8000 --data-parallel-size 4 --limit-mm-per-prompt image=8 \
  --max-model-len 16384 > main_serve.log 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup vllm serve ckpts/Qwen3-VL-235B-A22B-FP8 \
  --port 8001 --host 0.0.0.0 --tensor-parallel-size 4 \
  --limit-mm-per-prompt image=8 --max-model-len 16384 > judge_serve.log 2>&1 &
```

### 2.1 等服务就绪（235B 加载要几分钟）

```bash
tail -f judge_serve.log        # 出现 "Application startup complete" 即就绪
curl -s http://localhost:8001/v1/models | python -m json.tool
```

**验收点**：curl 返回 JSON，`"id"` 字段为 `ckpts/Qwen3-VL-235B-A22B-FP8`
——这个字符串就是 V100 侧要填的 `JUDGE_MODEL`（vLLM 用启动路径作模型名）。

### 2.2 本机冒烟一条请求（可选但推荐）

```bash
curl -s http://localhost:8001/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "ckpts/Qwen3-VL-235B-A22B-FP8",
  "messages": [{"role": "user", "content": "用一句话自我介绍"}],
  "max_tokens": 50}' | python -m json.tool
```

**验收点**：返回里有正常中文回复（纯文本请求即可，图片链路由 V100 侧
`s2-judge` 冒烟验证）。

---

## 3.（V100 机）连通性验证

```bash
# <H100_IP> 换成 H100 机的内网地址
curl -s --connect-timeout 5 http://<H100_IP>:8001/v1/models | python -m json.tool
```

**验收点**：返回与 H100 本机 curl 相同的 JSON。失败排查顺序：

| 现象 | 原因与处理 |
|---|---|
| `Connection timed out` | 防火墙挡了 8001：H100 机 `iptables -I INPUT -p tcp --dport 8001 -j ACCEPT`（或找管理员开安全组）|
| `Connection refused` | 服务没起/没监听外网：确认进程活着、启动参数带 `--host 0.0.0.0`、`ss -tlnp \| grep 8001` 显示 `0.0.0.0:8001` 而非 `127.0.0.1:8001` |
| 能 ping 通但 curl 挂起 | V100 机 shell 里有 `http_proxy` 残留把内网请求带偏：`unset http_proxy https_proxy` 或 `export no_proxy=<H100_IP>` |
| 返回 404/模型名不对 | `JUDGE_MODEL` 要与 `/v1/models` 的 `"id"` 完全一致（含 `ckpts/` 前缀）|

---

## 4.（V100 机）跑裁判

```bash
source ~/audit_env/bin/activate && cd ~/mobileclip
export JUDGE_API_BASE=http://<H100_IP>:8001/v1
export JUDGE_MODEL=ckpts/Qwen3-VL-235B-A22B-FP8   # 以 /v1/models 返回为准
export JUDGE_API_KEY=EMPTY   # vLLM 未设 --api-key 时任意非空值；run_v100.sh 要求非空

# 先 5 条冒烟（验证抽帧→base64→API→JSON 解析全链路）：
JUDGE_N=5 bash tools/data_audit/run_v100.sh s2-judge
cat outputs/audit_10k/s2_verdicts_judge.jsonl | head -2

# 冒烟 OK 后正式跑（tp8 用 500–2000 条；对半 tp4 时建议先 500）：
JUDGE_N=500 bash tools/data_audit/run_v100.sh s2-judge
```

跑完自动打印 **8B vs 裁判 action 一致率**（决策门 ≥75%，不达标先改
rubric 再重跑 S2，见 `SETUP_V100.md` §4）。

**并发说明**：`run_v100.sh` 里裁判并发是 8。H100 侧 tp8 独占时可以更高，
想加速可手动跑 `vlm_audit.py --concurrency 16`（对半 tp4 时保持 8）。

**前提**：`s2-judge` 需要 `outputs/audit_10k/s2_sample.jsonl` 已存在
（即 S1 + `s2-sample` 已完成）；裁判不依赖 S2 主力的结果，可与主力并行跑。

---

## 5. 收尾

```bash
# H100 机：裁判跑完释放显存（下次要用再起）
pkill -f "vllm serve ckpts/Qwen3-VL-235B-A22B-FP8"
```

H100 机自己的全量审查后续也要跑裁判（§4.3），服务可以留着不杀，
两边共用；只是注意别和全量 S2 主力抢卡（用方案 B 的对半布局）。

## 常见问题

- **OOM / 起不来**：确认没有别的进程占卡（`nvidia-smi`）；tp4 方案把
  `--max-model-len` 降到 16384、必要时加 `--gpu-memory-utilization 0.92`。
- **加载极慢**：FP8 权重 235GB 首次从磁盘读入需要几分钟属正常；放在
  NVMe 而非 NFS 挂载上会快很多。
- **一致率异常低（<50%）**：先抽几条 `s2_verdicts_judge.jsonl` 人工看
  ——大概率是 `JUDGE_MODEL` 填错导致 404 后大量空判定，或帧没发出去
  （检查 `s2_judge_sample.jsonl` 的 video 路径能否在 V100 本地解码）。
- **安全**：服务无鉴权暴露在内网。介意的话起服务加 `--api-key <随机串>`，
  V100 侧 `JUDGE_API_KEY` 填同一个值。
