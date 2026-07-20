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
  - [2.3.1 如果 VPN 用 SSO 单点登录(浏览器网页认证)](#231-如果-vpn-用-sso-单点登录浏览器网页认证)
  - [2.4 日常使用:哪些命令走 VPN,怎么走](#24-日常使用哪些命令走-vpn怎么走)
  - [2.5 隔离性/安全性:为什么它不会干扰原有网络与 SSH](#25-隔离性安全性为什么它不会干扰原有网络与-ssh)
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

> ⚠️ **重要**:`openconnect` 在 conda-forge 里有,但 **`ocproxy` 不在 conda-forge**,
> 需要自己从源码编译。下面的方式 A 用 conda 装 openconnect + libevent + 编译工具,
> 再编译 ocproxy —— 全程无 root。以下步骤已实测通过。

方式 A —— conda 装依赖 + 编译 ocproxy(推荐):

```bash
# 如果还没有 conda,先装 Miniforge(纯用户目录安装,不需要 root)
curl -fsSL -o Miniforge3.sh \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/bin/activate"

# 1. openconnect + libevent + 编译工具全部从 conda 装(ocproxy 不在 conda,要自己编)
conda install -c conda-forge openconnect libevent autoconf automake libtool pkg-config make

# 2. 编译 ocproxy 到 ~/.local
#    坑:ocproxy 的 configure 不认 PKG_CONFIG_PATH,必须用 CPPFLAGS/LDFLAGS 手动指向 libevent;
#    再用 -Wl,-rpath 把 libevent 路径写进二进制,免得每次都要设 LD_LIBRARY_PATH。
git clone https://github.com/cernekee/ocproxy
cd ocproxy
./autogen.sh
./configure --prefix="$HOME/.local" \
  CPPFLAGS="-I$CONDA_PREFIX/include" \
  LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
make && make install
cd ..

# 3. 加入 PATH(写进 ~/.bashrc 或 ~/.zshrc 永久生效)
export PATH="$HOME/.local/bin:$PATH"

# 4. 验证:打印用法即成功(单独跑会提示 "VPNFD is not set",正常,那是没接 openconnect 时的提示)
ocproxy -h
openconnect --version
```

方式 B —— 完全不用 conda,libevent 也从源码编译(系统连 conda 都没有时):

```bash
mkdir -p ~/.local ~/build && cd ~/build
# 前提:系统已有 gcc / make / git / autoconf / automake / libtool(多数开发机自带)

# 1. 编译 libevent 到 ~/.local
git clone --depth 1 --branch release-2.1.12-stable https://github.com/libevent/libevent
cd libevent && ./autogen.sh
./configure --prefix="$HOME/.local" --disable-openssl
make -j4 && make install
cd ..

# 2. 编译 openconnect 到 ~/.local(需要 gnutls/libxml2 的开发头文件;缺了且装不上就改用方式 A)
curl -fsSL -o openconnect.tar.gz \
  https://www.infradead.org/openconnect/download/openconnect-9.12.tar.gz
tar xzf openconnect.tar.gz && cd openconnect-9.12
./configure --prefix="$HOME/.local" --with-vpnc-script=/bin/true
make -j4 && make install
cd ..

# 3. 编译 ocproxy(指向上面装好的 libevent)
git clone https://github.com/cernekee/ocproxy && cd ocproxy
./autogen.sh
./configure --prefix="$HOME/.local" \
  CPPFLAGS="-I$HOME/.local/include" \
  LDFLAGS="-L$HOME/.local/lib -Wl,-rpath,$HOME/.local/lib"
make && make install
cd ..

# 4. 加入 PATH(写进 ~/.bashrc 或 ~/.zshrc 永久生效)
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
```

### 2.3 连接 VPN

```bash
# 把 vpn.company.com 换成你平时在 AnyConnect 里填的服务器地址,
# 用户名换成你的 VPN 账号
openconnect \
  --script-tun "ocproxy -D 11080" \
  --user=你的用户名 \
  vpn.company.com
```

> ⚠️ **必须是 `--script-tun`,不是 `--script`。** 这是最容易踩的坑:
> - `--script-tun "ocproxy ..."` = 把隧道数据经 socketpair 交给 ocproxy(用户态,**不建 tun 网卡、无需 root**),ocproxy 靠它拿到 `VPNFD`。
> - `--script "ocproxy ..."` = openconnect 仍会去建**真正的 tun 网卡**(需要 root),于是报
>   `VPNFD is not set` + `TUNSETIFF 不允许的操作` + `必须以 root 权限运行`,连接失败。

按提示输入密码(和二次验证,如果有)。看到类似
`Connected as 10.x.x.x` / `Session established` 即成功。

- 这个终端窗口要**保持开着**(它就是 VPN 会话)。想放后台:

  ```bash
  # 方式一:tmux(推荐,断线可恢复交互输密码)
  tmux new -s vpn
  openconnect --script-tun "ocproxy -D 11080" --user=你的用户名 vpn.company.com
  # 按 Ctrl+B 再按 D 挂起;tmux attach -t vpn 找回

  # 方式二:密码从 stdin 传入并后台运行
  echo '你的密码' | openconnect --script-tun "ocproxy -D 11080" \
    --user=你的用户名 --passwd-on-stdin -b vpn.company.com
  ```

- 如果公司用的是分组(Group)登录,加 `--authgroup=组名`。
- 如果证书报错,先看[常见问题排查](#常见问题排查),不要盲目加 `--no-cert-check`。

### 2.3.1 如果 VPN 用 SSO 单点登录(浏览器网页认证)

现在很多高校/机构的 VPN 不再是"账号密码直登",而是**跳转到一个网页做 SSO(单点登录)**——
用官方 AnyConnect 连接时会**弹出一个浏览器登录页**,就是这种。此时上面的账号密码方式会
陷入"输完密码又跳回密码框"的死循环,因为你的账号根本不走密码认证。

openconnect 9.x 支持外部浏览器 SSO,用 `--external-browser`。**能一次成功的完整命令**
(已在 vpn.fu-berlin.de 实测)长这样——注意 `--useragent` 和浏览器**绝对路径**两个都不能少:

```bash
openconnect --useragent='AnyConnect Windows 4.10.06079' \
  --external-browser="$(which xdg-open)" \
  --script-tun "ocproxy -D 11080" \
  vpn.company.com
```

流程:openconnect 自动打开浏览器 → 你在网页里完成 SSO 登录 → 浏览器跳回一个
`http://localhost:<port>/...` 地址 → openconnect 捕获令牌、建立隧道。

> 💡 `--useragent` 是**发给服务器的身份标识,与你本机是什么系统无关**——在 Linux
> (含国产 NewStart / 麒麟等)上照发 `AnyConnect Windows ...` 完全没问题,服务器只认
> 开头的 `AnyConnect`。想名副其实也可用 `AnyConnect Linux_64 4.10.06079`。

**关键坑与排查(按顺序):**

1. **⭐ 头号坑:User-Agent 不对导致服务器返回 404,SSO 通道根本不开。**
   openconnect 默认的 UA 是 `Open AnyConnect VPN Agent v9.12`,**以 `Open` 开头**。
   很多 Cisco ASA 会检查 UA 是否**以 `AnyConnect` 开头**,不匹配就对初始的
   `POST /` 返回 **404**,于是退化成老式账号密码门户,`--external-browser` 永远不弹网页。
   加 `--useragent` 伪装成官方客户端即可:
   ```bash
   openconnect --useragent='AnyConnect Windows 4.10.06079' \
     --external-browser=xdg-open \
     --script-tun "ocproxy -D 11080" \
     vpn.company.com
   ```
   判断是否踩中此坑:用 `--dump-http-traffic` 看初始 `POST /` 是不是返回 `404 Not Found`
   然后才退回 `/+webvpn+/index.html` 密码表单——是的话就是 UA 问题,不是分组问题。
   个别 ASA 还会看版本号,可再试更完整的
   `--useragent='Cisco AnyConnect VPN Agent for Windows 4.10.06079'`。
2. **⭐ 次号坑:`Failed to spawn external browser` / `Spawn browser: 没有那个文件或目录`。**
   这说明 UA 那关已过、服务器**已进入 SSO**(会看到 `Please complete the authentication
   process in the browser window.` 和一个 `.../+CSCOE+/saml/sp/login?...` 地址),只是
   openconnect 启动浏览器时按名字没找到——它 spawn 时的 PATH 查找和你手敲不同。
   **改用绝对路径**即可:
   ```bash
   which xdg-open firefox google-chrome chromium chromium-browser 2>/dev/null
   # 用 $(which xdg-open),或直接指向真正的浏览器二进制:
   openconnect --useragent='AnyConnect Windows 4.10.06079' \
     --external-browser="$(which xdg-open)" \
     --script-tun "ocproxy -D 11080" vpn.company.com
   # xdg-open 仍不行就直接用浏览器,如 --external-browser=/usr/bin/firefox
   ```
3. **确认 openconnect 编进了 SSO 支持**:
   ```bash
   openconnect --version
   openconnect --help 2>&1 | grep -i external-browser   # 有输出才支持
   ```
4. **确认当前 shell 有图形环境**:`xdg-open` 只有在有桌面时才能弹浏览器。
   ```bash
   echo "$DISPLAY"                    # 空 = 当前 shell 没图形环境(比如 SSH 进来的)
   xdg-open https://example.com       # 能弹出浏览器才行
   ```
   `$DISPLAY` 为空时,要么改到图形桌面的终端里跑,要么用下面的"手动令牌"兜底。
5. **确认分组(group-url)**。改对 UA 后若服务器仍只给密码表单,才考虑分组:
   `--external-browser` 只有在**服务器真的发来 SSO 请求**时才弹浏览器。很多机构同时
   存在"密码分组"和"SSO 分组",官方客户端连的是后者。
   - **从官方客户端配置里挖分组**(以下是分组名/地址,非敏感):
     ```bash
     # Cisco Secure Client / AnyConnect 记录的默认主机与分组
     cat ~/.cisco/vpn/preferences 2>/dev/null      # 看 DefaultGroup / DefaultHostAddress
     cat ~/.anyconnect 2>/dev/null
     # 连接 profile 里的 HostAddress / GroupURL / UserGroup
     grep -rhoE '<(HostName|HostAddress|UserGroup|GroupURL)>[^<]+' /opt/cisco 2>/dev/null
     ```
   - 拿到分组名 `XXX`(DefaultGroup / UserGroup,或 GroupURL 末段)后接到地址后面:
     ```bash
     openconnect --useragent='AnyConnect Windows 4.10.06079' \
       --external-browser=xdg-open --script-tun "ocproxy -D 11080" \
       vpn.company.com/XXX
     # 或等价地:--authgroup=XXX
     ```
   - ⚠️ 注意:profile 里若**没有** GroupURL/UserGroup(官方客户端连裸地址就行),
     那 404 十有八九是坑 1 的 UA 问题,**别乱猜分组**——瞎接 `/xxx` 会得到
     `Invalid host entry`。

**兜底:无图形环境 / 挖不到分组时**,用封装脚本
[`openconnect-sso`](https://github.com/vlaci/openconnect-sso)(内置浏览器自动完成 SAML,
`pipx install openconnect-sso` 免 root 安装),或在有浏览器的机器上手动完成 SSO、
把最终 `localhost:<port>` 回调里的令牌用 `--cookie` 传给 openconnect。

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

### 2.5 隔离性/安全性:为什么它不会干扰原有网络与 SSH

用 `--script-tun "ocproxy ..."` 这种用户态模式,openconnect **从设计上就不可能**动到
这台机器原有的网络和 SSH——尤其你**没有 root**,更改不了内核网络:

- **不创建 tun 网卡**(不会出现 `tun0`);
- **不修改内核路由表**(默认路由、本地网段路由原样不动);
- **不碰 iptables、不改 `/etc/resolv.conf`**;
- 全部 VPN 流量只存在于**监听在 `127.0.0.1:11080` 的用户态 SOCKS 代理**里。

关键结论:**默认什么流量都不走 VPN**;只有你主动给某条命令加 `--socks5-hostname
127.0.0.1:11080` / SOCKS ProxyCommand,那一条才进 VPN。你**当前的 SSH 会话、本机服务、
其他用户的连接全部照旧**,openconnect 退出后系统层面不留任何改动。

**眼见为实**——连接前存基线,连接后对比,应完全一致:

```bash
# 连接前
ip route > /tmp/route.before ; ip -br link > /tmp/link.before
# 连上后(另开终端):
ip route > /tmp/route.after  ; ip -br link > /tmp/link.after
diff /tmp/route.before /tmp/route.after   # 无输出 = 路由没变
diff /tmp/link.before  /tmp/link.after    # 无输出 = 没多出 tun 网卡
ss -tlnp | grep 11080                     # 应是 127.0.0.1:11080,未对外暴露
```

**使用习惯**(与"干扰"无关,但值得注意):

- `export ALL_PROXY=...` 只影响你从该 shell 启动的程序,**不影响 sshd、不影响别人**;
  别写进 `~/.bashrc` 全局生效,否则你自己后续所有命令会默认走 VPN。
- 端口 `11080` 被占用就换一个(`-D 11081`,同时改用代理的命令);它只绑本地环回口。
- 退出即干净:在跑 openconnect 的终端按 `Ctrl-C`,隧道和 ocproxy 一起结束,无残留。
- openconnect 是你 SSH 会话的子进程——**你 SSH 一断它就被杀、VPN 断开**(服务器网络完好)。
  想让 VPN 在断开后存活,用 `tmux`(见 [2.3](#23-连接-vpn) 的后台方式)。

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
| conda 报 `PackagesNotFoundError: ocproxy` | 正常,ocproxy 不在 conda-forge,按 [2.2 方式 A](#22-安装linux无-root) 自己编译 |
| 编译 ocproxy 时 `Missing development files for libevent2` | configure 没找到 libevent。**不是** 加 `PKG_CONFIG_PATH`,而是加 `CPPFLAGS="-I<libevent前缀>/include"` 和 `LDFLAGS="-L<libevent前缀>/lib"`(conda 装的用 `$CONDA_PREFIX`) |
| ocproxy 运行报 `libevent-2.1.so.7: cannot open shared object file` | 编译时漏了 `-Wl,-rpath,...`;临时用 `LD_LIBRARY_PATH=$CONDA_PREFIX/lib ocproxy ...`,或按 2.2 重新 configure |
| 证书校验失败(certificate verify failed) | 先用 `--servercert` 固定服务器证书指纹(openconnect 报错信息里会给出 pin 值,确认无误后加上),比 `--no-cert-check` 安全 |
| 提示需要 CSD/HostScan | 尝试加 `--csd-wrapper` 脚本(openconnect 源码 `trojans/` 目录里有 `csd-post.sh`/`hostscan-bypass`);仍失败则此路不通,考虑第 3 步 |
| 认证成功后报 `VPNFD is not set` / `TUNSETIFF 不允许的操作` / `必须以 root 权限运行` | **用错了参数**:ocproxy 必须配 `--script-tun`,不是 `--script`。改成 `--script-tun "ocproxy -D 11080"` 即可(用户态、无需 root) |
| 连接成功但代理不通 | 确认 ocproxy 进程在(`pgrep ocproxy`);确认命令确实指定了 `127.0.0.1:11080`;换个端口试试(把 11080 全部替换) |
| `ssh` 报 `nc: invalid option -- X` | 该系统的 netcat 不支持 SOCKS,改用 ncat(见 2.4)或 `connect-proxy` |
| DNS 解析不到内网域名 | 用 `--socks5-hostname`(curl)/ ProxyCommand 里的 `%h`(ssh)让**代理端**解析域名,不要本地先解析;ocproxy 会用 VPN 下发的 DNS |
| 二次验证(OTP/Duo) | openconnect 会交互式提示,照常输入;Duo Push 选项一般输入 `push` |
| 输完密码又跳回 `Password:` 死循环 | ① 账号名/密码错;② 账号走 **SSO** 而非密码认证——见 [2.3.1](#231-如果-vpn-用-sso-单点登录浏览器网页认证)。用 `--dump-http-traffic` 看返回 XML:无 `<error>`、无 `group_list` 而反复回同一表单 = 认证被拒或分组不对(**dump 含明文密码,贴前务必删除 `password=` 行**) |
| `--external-browser` 不弹网页 / 初始 `POST /` 返回 404 | **头号原因是 User-Agent**:openconnect 默认 UA 以 `Open` 开头,ASA 只认以 `AnyConnect` 开头的,不匹配就 404 并退回密码门户。加 `--useragent='AnyConnect Windows 4.10.06079'`。其次才是分组/`$DISPLAY`,详见 [2.3.1](#231-如果-vpn-用-sso-单点登录浏览器网页认证) |
| SSO 连接报 `Invalid host entry. Please re-enter.` | 你在地址后接了错误的分组路径(如 `/zedat`),而该服务器本无此 group-url。去掉分组用裸地址,先按坑 1 修 UA |
| SSO 已启动但 `Failed to spawn external browser` / `Spawn browser: 没有那个文件或目录` | 已进 SSO,只是启动浏览器失败。`--external-browser` 用**绝对路径**:`--external-browser="$(which xdg-open)"`,或直接 `--external-browser=/usr/bin/firefox` |
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
