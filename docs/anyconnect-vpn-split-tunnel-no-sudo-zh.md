# AnyConnect VPN 分流指南:无 sudo 权限下同时访问 VPN 内网与本地网络

> 场景:公司 VPN(Cisco AnyConnect)是全局隧道(Tunnel All Networks),连上后 terminal
> 无法访问本地局域网的服务器;你没有 sudo/管理员权限,也联系不到 VPN 管理员。
>
> 目标:terminal 能**同时**访问 VPN 内网服务器和本地网络服务器。
>
> 核心思路:**不碰系统路由表**,把 VPN 变成一个用户态 SOCKS 代理 —— 默认流量走本地网络,
> 只有明确指定走代理的命令才进 VPN。

---

## 目录

- [第 0 步:先确认问题真的存在(1 分钟)](#第-0-步先确认问题真的存在1-分钟)
- [第 1 步:试试 AnyConnect 自带的本地局域网访问开关(零成本)](#第-1-步试试-anyconnect-自带的本地局域网访问开关零成本)
- [第 2 步(推荐):openconnect + ocproxy 用户态分流](#第-2-步推荐openconnect--ocproxy-用户态分流)
  - [2.1 安装(macOS)](#21-安装macos)
  - [2.2 安装(Linux,无 root)](#22-安装linux无-root)
  - [2.3 连接 VPN](#23-连接-vpn)
  - [2.4 日常使用:哪些命令走 VPN,怎么走](#24-日常使用哪些命令走-vpn怎么走)
- [第 3 步(备选):官方客户端 + 公网 VPS 反向隧道](#第-3-步备选官方客户端--公网-vps-反向隧道)
- [常见问题排查](#常见问题排查)
- [方案对比总结](#方案对比总结)

---

## 第 0 步:先确认问题真的存在(1 分钟)

有些名义上"全局"的 VPN 配置其实放行了本地子网。先连上 AnyConnect,然后:

```bash
# 1. 查看你的本地网卡 IP(记下网段,例如 192.168.1.0/24)
#    macOS:
ipconfig getifaddr en0
#    Linux:
ip addr show 2>/dev/null || ifconfig

# 2. 直接尝试访问本地服务器(换成你的本地服务器 IP)
ping -c 3 192.168.1.100
ssh user@192.168.1.100
```

- **能通** → 问题不存在,不用往下看了。
- **不通**(超时/no route)→ 继续第 1 步。

---

## 第 1 步:试试 AnyConnect 自带的本地局域网访问开关(零成本)

这个开关**普通用户就能勾**,不需要 sudo,但只有服务端策略允许时才生效:

1. 打开 AnyConnect 客户端主界面。
2. 点击左下角 **齿轮图标**(Preferences / 首选项)。
3. 勾选 **"Allow local (LAN) access when using VPN (if configured)"**
   (使用 VPN 时允许本地局域网访问)。
4. **断开 VPN,重新连接**(必须重连才生效)。
5. 重复第 0 步的 `ping` / `ssh` 测试。

- **能通** → 完成,不用继续了。
- **不通** → 服务端没放行这个选项,进入第 2 步。

---

## 第 2 步(推荐):openconnect + ocproxy 用户态分流

[openconnect](https://www.infradead.org/openconnect/) 是兼容 AnyConnect 协议的开源客户端;
[ocproxy](https://github.com/cernekee/ocproxy) 让它**不创建 tun 网卡、不修改路由表**,
把整条 VPN 变成本地的一个 SOCKS5 代理端口。整个过程以普通用户身份运行,零 sudo。

连接后的效果:

```
默认所有流量          ──────────────► 本地网络(不变)
指定走 127.0.0.1:11080 的流量 ──────► VPN 内网
```

> ⚠️ 使用前先把官方 AnyConnect **断开**,两者不要同时连。

### 2.1 安装(macOS)

Homebrew 本身安装和使用都不需要 sudo(装在用户目录):

```bash
# 如果还没有 Homebrew(官方脚本,装到 /opt/homebrew,只需要你自己的密码用于创建目录;
# 如果连这也不行,参考 https://docs.brew.sh/Installation 的"untar anywhere"免权限方式)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 openconnect 和 ocproxy
brew install openconnect ocproxy

# 验证
openconnect --version
which ocproxy
```

### 2.2 安装(Linux,无 root)

方式 A —— 用 conda / mamba(最简单,推荐):

```bash
# 如果还没有 conda,先装 Miniforge(纯用户目录安装,不需要 root)
curl -fsSL -o Miniforge3.sh \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/bin/activate"

# 安装 openconnect 和 ocproxy
conda install -c conda-forge openconnect ocproxy
```

方式 B —— 源码编译到 `$HOME/.local`(conda 不可用时):

```bash
mkdir -p ~/.local ~/build && cd ~/build

# 依赖:需要系统已有 gcc/make/pkg-config 和 gnutls/libxml2 的开发头文件。
# 如果缺开发头文件且装不了,请改用方式 A。

# 编译 openconnect
curl -fsSL -o openconnect.tar.gz \
  https://www.infradead.org/openconnect/download/openconnect-9.12.tar.gz
tar xzf openconnect.tar.gz && cd openconnect-9.12
./configure --prefix="$HOME/.local" --with-vpnc-script=/bin/true
make -j4 && make install
cd ..

# 编译 ocproxy(依赖 libevent)
git clone https://github.com/cernekee/ocproxy && cd ocproxy
./autogen.sh && ./configure --prefix="$HOME/.local"
make && make install

# 加入 PATH(写进 ~/.bashrc 或 ~/.zshrc 永久生效)
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
```

### 2.3 连接 VPN

```bash
# 把 vpn.company.com 换成你平时在 AnyConnect 里填的服务器地址,
# 用户名换成你的 VPN 账号
openconnect \
  --script "ocproxy -D 11080" \
  --user=你的用户名 \
  vpn.company.com
```

按提示输入密码(和二次验证,如果有)。看到类似
`Connected as 10.x.x.x` / `Session established` 即成功。

- 这个终端窗口要**保持开着**(它就是 VPN 会话)。想放后台:

  ```bash
  # 方式一:tmux(推荐,断线可恢复交互输密码)
  tmux new -s vpn
  openconnect --script "ocproxy -D 11080" --user=你的用户名 vpn.company.com
  # 按 Ctrl+B 再按 D 挂起;tmux attach -t vpn 找回

  # 方式二:密码从 stdin 传入并后台运行
  echo '你的密码' | openconnect --script "ocproxy -D 11080" \
    --user=你的用户名 --passwd-on-stdin -b vpn.company.com
  ```

- 如果公司用的是分组(Group)登录,加 `--authgroup=组名`。
- 如果证书报错,先看[常见问题排查](#常见问题排查),不要盲目加 `--no-cert-check`。

### 2.4 日常使用:哪些命令走 VPN,怎么走

连接成功后,`127.0.0.1:11080` 就是通往 VPN 内网的 SOCKS5 代理。
**默认什么都不变**——本地服务器照常直连:

```bash
ssh user@192.168.1.100        # 本地服务器,直连,和没连 VPN 一样
```

需要访问 VPN 内网时,给命令加上代理:

```bash
# ssh 到内网服务器(10.0.0.5 换成实际内网 IP / 主机名)
#   macOS / 带 -X 参数的 netcat:
ssh -o ProxyCommand='nc -X 5 -x 127.0.0.1:11080 %h %p' user@10.0.0.5
#   Linux 通用(ncat 来自 nmap 包;conda install -c conda-forge nmap 可装):
ssh -o ProxyCommand='ncat --proxy-type socks5 --proxy 127.0.0.1:11080 %h %p' user@10.0.0.5

# curl / wget
curl --socks5-hostname 127.0.0.1:11080 http://10.0.0.5:8080/api
wget -e use_proxy=yes -e https_proxy=socks5://127.0.0.1:11080 http://10.0.0.5/file

# git(仅对内网 git 服务器生效,写进对应 remote 的配置)
git config --global http.http://git.company.internal/.proxy socks5://127.0.0.1:11080

# 让当前 shell 里所有支持代理的程序都走 VPN(只在需要的窗口 export,别写进 rc 文件)
export ALL_PROXY=socks5://127.0.0.1:11080
```

推荐把常用内网机器写进 `~/.ssh/config`,以后 `ssh 内网别名` 一步到位:

```
# ~/.ssh/config
Host intranet-*
    ProxyCommand nc -X 5 -x 127.0.0.1:11080 %h %p   # Linux 用 ncat 那行替换

Host intranet-dev
    HostName 10.0.0.5
    User your_name
```

对不支持 SOCKS 代理的程序,用 [proxychains-ng](https://github.com/rofl0r/proxychains-ng)
(同样可 conda/brew 安装)强制接管:

```bash
proxychains4 -q psql -h 10.0.0.20 -U dev mydb
```

---

## 第 3 步(备选):官方客户端 + 公网 VPS 反向隧道

如果公司 VPN 有强制的客户端安全检查(HostScan/CSD)导致 openconnect 连不上,
且你恰好有一台公网 VPS,可以反过来让**本地服务器**绕出来:

```bash
# ① 在本地服务器上执行(需要能物理/预先操作它一次),建立反向隧道:
ssh -N -R 2222:localhost:22 vps_user@your-vps.com
#    建议配合 autossh 保活:
autossh -M 0 -f -N -R 2222:localhost:22 vps_user@your-vps.com

# ② 你的电脑(连着全局 VPN)通过 VPS 访问本地服务器:
ssh -J vps_user@your-vps.com -p 2222 user@localhost
```

缺点:多一跳延迟、需要 VPS、需要预先在本地服务器上布置。仅作兜底。

---

## 常见问题排查

| 现象 | 原因与解决 |
|---|---|
| `openconnect: command not found` | PATH 没生效,重新 `source ~/.bashrc` 或确认安装前缀在 PATH 里 |
| 证书校验失败(certificate verify failed) | 先用 `--servercert` 固定服务器证书指纹(openconnect 报错信息里会给出 pin 值,确认无误后加上),比 `--no-cert-check` 安全 |
| 提示需要 CSD/HostScan | 尝试加 `--csd-wrapper` 脚本(openconnect 源码 `trojans/` 目录里有 `csd-post.sh`/`hostscan-bypass`);仍失败则此路不通,考虑第 3 步 |
| 连接成功但代理不通 | 确认 ocproxy 进程在(`pgrep ocproxy`);确认命令确实指定了 `127.0.0.1:11080`;换个端口试试(把 11080 全部替换) |
| `ssh` 报 `nc: invalid option -- X` | 该系统的 netcat 不支持 SOCKS,改用 ncat(见 2.4)或 `connect-proxy` |
| DNS 解析不到内网域名 | 用 `--socks5-hostname`(curl)/ ProxyCommand 里的 `%h`(ssh)让**代理端**解析域名,不要本地先解析;ocproxy 会用 VPN 下发的 DNS |
| 二次验证(OTP/Duo) | openconnect 会交互式提示,照常输入;Duo Push 选项一般输入 `push` |
| 官方 AnyConnect 自动重连抢占 | 在官方客户端里彻底 Disconnect 并退出程序,再启动 openconnect |

---

## 方案对比总结

| 方案 | 需要 sudo | 需要管理员配合 | 稳定性 | 推荐度 |
|---|---|---|---|---|
| 第 1 步:客户端勾选 Local LAN Access | 否 | 是(服务端需允许) | 高 | 先试,零成本 |
| **第 2 步:openconnect + ocproxy** | **否** | **否** | 高 | ⭐ 推荐 |
| 第 3 步:VPS 反向隧道 | 否 | 否(需 VPS) | 中 | 兜底 |
| 手动改路由表 | 是 | — | 会被 AnyConnect 撤销 | ❌ 不可行 |

> 合规提醒:以上方法都使用你自己的合法 VPN 账号建立连接,只是改变了本机的流量分配方式。
> 请确认这不违反你所在组织的 IT 使用政策;有疑问时,恢复联系管理员后补一句报备是最稳妥的。
