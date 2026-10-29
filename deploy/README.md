# 单服务器 dev / main 发布

## 当前工作流

- `dev` → development → `dev.cancercell.otisliu.com`
- `main` → production → `cancercell.otisliu.com`
- 向上述分支推送或提出 PR 会运行 `test`：Python 编译检查、Shell 语法检查、完整 CPU pytest。
- `master` 保留历史并继续运行 CI；不发布。
- 正常开发推送 dev，通过测试后提出 `dev → main` PR；main 合并后发布正式环境。
- Actions 仓库变量 `DEPLOY_ENABLED` 只有设为字符串 `true` 才会发布。服务器和两个 Environment 配置完成前保持未设置，CI 可独立运行。
- 只有测试成功的同一提交会发布；PR（包括外部分支）不会获得部署密钥。
- CI 使用 GitHub 托管 Ubuntu runner，不使用 Docker、Docker Hub 或 GHCR 镜像。

## 隔离

| 项目 | 测试 | 正式 |
|---|---|---|
| 运行账号 | cancer-dev | cancer-prod |
| 发布账号 | cancer-deploy-dev | cancer-deploy-prod |
| systemd | cancer@dev | cancer@prod |
| 本机监听 | 127.0.0.1:8766 | 127.0.0.1:8765 |
| 发布目录 | /srv/cancer-dev | /srv/cancer-prod |
| 病例目录 | /var/lib/cancer-dev/cases | /var/lib/cancer-prod/cases |
| 模型目录 | /var/lib/cancer-dev/models | /var/lib/cancer-prod/models |
| 配置密钥 | /etc/cancer/dev.env | /etc/cancer/prod.env |
| Basic Auth | /etc/cancer/dev.htpasswd | /etc/cancer/prod.htpasswd |

运行账号无登录 Shell，数据目录权限 700，发布账号不能读取病例、配置密钥。
每个环境有独立虚拟环境、进程、缓存、随机代理令牌和 Basic Auth 密码。
代理令牌阻止测试进程直接访问正式环境的本机 HTTP 接口；Nginx 总是覆盖客户端传入的令牌。
保留原应用的修改请求令牌检查、Host 校验和单 worker 后台任务行为。
这是每个环境一个共享工作台，Basic Auth 用户共享该环境病例；不是用户级多租户隔离。
共享一台主机仍共享内核、磁盘和故障域；具备服务器 root 权限的人能访问两个环境。

## Ubuntu 初始化（先查看文件再执行）

支持 Ubuntu 22.04 / 24.04，Python 3.10–3.12。先确认端口未被占用、磁盘与内存足够。
服务器执行 `sudo bash deploy/bootstrap.sh prepare`。
它安装 Python/venv、Nginx、Certbot 等系统软件，创建上述账号和目录。
初始 Nginx 仅开放证书验证路径，其余返回 503；不会以 HTTP 暴露病例。
不会清除现有 Nginx 站点；需要检查服务器现有 server_name 是否冲突。

Python 包默认使用腾讯云 PyPI 索引，CPU PyTorch 使用官方
`https://download.pytorch.org/whl/cpu`，不是容器镜像。
这些是外部下载源；以后可在 `/etc/cancer/build.env` 改为内网包源。
系统 apt 源沿用服务器现有配置。首次安装依赖需要可用外网。

服务默认内存硬上限：测试 25%、正式 50%；CPU 上限：测试 0.5 核、正式 1.5 核。
需要根据实际模型和服务器内存调整各自 systemd limits.conf。
这些是运行服务限制，依赖安装/构建过程不在该服务限制内；两个环境构建可能同时运行。
小内存服务器不保证能同时加载两个真实 ViT；确认资源后才启用推理。

### 部署 SSH 密钥

在可信本机为 development、production 各生成一对专用 Ed25519 密钥。
将各自公钥写入服务器 root 所有的文件：

- `/etc/ssh/authorized_keys/cancer-deploy-dev`
- `/etc/ssh/authorized_keys/cancer-deploy-prod`

文件权限 644、owner root。sshd Match 限制这些账号：只运行固定接收脚本；禁止密码、PTY 和端口转发。
发布账号仅允许 sudo 重启/停止自己的服务。不要把管理员 ubuntu 登录密钥用于 Actions。
通过可信渠道核对服务器 SSH 主机公钥指纹，再保存到 known_hosts；Actions 禁止自动接受新主机密钥。

### 配置 HTTPS 和登录

分别运行 `sudo htpasswd -c /etc/cancer/dev.htpasswd <用户名>` 和
`sudo htpasswd -c /etc/cancer/prod.htpasswd <用户名>` 设置不同密码。
第一次用 `-c` 创建文件；后续添加用户时不要使用 `-c`。
确认两个域名的 A/AAAA 记录均正确，云防火墙开放 80、443，SSH 仅按实际网络放行。
运行：

```bash
sudo CERTBOT_EMAIL=你的证书邮箱 bash deploy/bootstrap.sh tls
```

脚本申请证书后配置 HTTPS、HTTP 跳转与登录保护。Certbot 续期成功后自动校验并 reload Nginx。
未提供 CERTBOT_EMAIL 时使用无邮箱注册，依靠服务器定时续期；不接收证书邮箱通知。
不要在同一时刻对现有站点重复配置相同域名。

### 模型与服务密钥

分别将可信训练模型和可选阈值文件放入每个环境的 models 目录；不要从正式目录软链接给测试使用。
文件必须归对应运行账号所有，不能交给发布账号读取。
分别编辑 `/etc/cancer/{dev,prod}.env`：

```dotenv
CANCER_MODEL_PATH=/var/lib/cancer-dev/models/model.pt
CANCER_DECISION_PATH=
DEEPSEEK_API_KEY=
```

正式环境使用 prod 路径；禁止复制真实正式病例到测试目录。
若未配置模型，网站可以启动并明确报告 model_ready=false，不能进行推理；不生成虚假模型。
若配置模型但加载失败，健康检查返回 503，发布失败并回滚。
DeepSeek Key 为空时不能生成 LLM 病例汇总；分别配置适用的凭据后再验证实际业务。

## GitHub 配置

用有仓库管理和 Actions 权限的账号运行 `gh auth login`，然后执行：

```bash
bash deploy/configure-github.sh
```

此脚本切换默认分支为 main，创建 development/production 环境并限制分支，
要求 main PR 的 test 检查成功，保留管理员紧急绕过能力，不要求单人项目无法满足的自我审批。
需要管理权限；遇到错误会退出，不会偷偷启用发布。

在每个 GitHub Environment 分别设置同名 Secrets（两套私钥不同）：

- SSH_HOST：43.133.91.17
- SSH_PORT：22
- SSH_USER：对应 cancer-deploy-dev 或 cancer-deploy-prod
- SSH_PRIVATE_KEY：对应专用私钥
- SSH_KNOWN_HOSTS：已核验的 OpenSSH known_hosts 行（自定义端口使用 [host]:port）

使用 `gh secret set --env development --repo FazhanLiu/cancer-cell-classification-pipeline ...`，
私钥通过文件重定向提供，不写进命令参数、提交或聊天记录。
服务器、TLS、Secrets 和分支限制均验证后再启用：

```bash
gh variable set DEPLOY_ENABLED --body true --repo FazhanLiu/cancer-cell-classification-pipeline
```

然后从 dev 的 Actions 页面手动运行 workflow_dispatch，验证成功后再发布 main。

## 发布、回滚与备份

Actions 使用只读 GITHUB_TOKEN 检出代码，通过严格主机密钥校验的 SSH 流式传输 git archive。
服务器校验归档路径/类型/大小，独立创建新 release 和 venv；不覆盖运行中的依赖或数据。
通过原子 current 软链接切换到新 release，重启自己的服务并检查健康状态与提交 SHA。
健康检查失败自动恢复旧链接并尝试重启旧版本，流水线报错；首次发布失败则停止该环境。
如果旧版本也无法恢复，需管理员检查 journalctl；回滚不是数据库或病例文件的事务回滚。
正常发布有短暂停机，正在运行的后台任务可能被中断，可在工作台重试。

查看日志：`sudo journalctl -u cancer@dev -n 100 --no-pager`，正式改为 prod。
手动回滚：把对应 current 原子切换到已验证旧 release，再重启对应服务。
成功 release 保留便于回滚；定期清理不再使用且非 current 的 release，避免 venv 占满磁盘。

备份应分别覆盖各自 `/var/lib/cancer-*`、`/etc/cancer/*.env` 和 htpasswd，存储到不同受控目录。
本仓库不会把病例、模型、凭据或备份上传 GitHub；自动异地备份需另配置存储目的地。

## 后续内网连接

GitHub 托管 runner 无法直接访问 VPC 私网。后续可选择可信独立机器上的 self-hosted runner
或明确的 VPN/隧道，再把 SSH_HOST 切换为内网地址；不能仅替换 IP 就宣称内网已连通。
不要在承载正式病例的服务器上运行来自外部 PR 的 self-hosted CI。

参考：[GitHub environments](https://docs.github.com/en/actions/concepts/workflows-and-actions/deployment-environments)、
[systemd 目录保护](https://www.freedesktop.org/software/systemd/man/254/systemd.exec.html)。

## 切换报告 API（不更改本地图像推理接口）

网站报告支持 `LLM_PROVIDER=compatible`，统一读取 `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY`。
兼容 Chat Completions 的服务只需更换变量，不增加模型专用 Python 文件。
`LLM_BASE_URL` 填 HTTPS API 根路径（例如供应商的 `/v1`），也接受完整 `/chat/completions` 地址；
不允许 URL 内嵌账号、Key、查询参数。原来的 DeepSeek 配置继续有效。

管理员可以通过 SSH 把私有 JSON 输入 `/usr/local/libexec/cancer-configure-report.py`，
无需手动编辑环境文件。工具只更新指定环境的报告变量，保留本地图像模型路径、病例目录与代理令牌。
文件原子替换，权限保持 root:对应运行账号 640，重启对应服务；重启命令失败时恢复旧配置。
JSON 格式如下（请保存到 Git 忽略的 `.deploy-secrets/` 中；真实 Key 不提交或放进命令行参数）：

```json
{"provider":"compatible","base_url":"https://供应商地址/v1","model":"模型名称","api_key":"真实Key"}
```

通过已经配置的管理员 SSH 通道运行：

```bash
ssh ubuntu@43.133.91.17 'sudo python3 /usr/local/libexec/cancer-configure-report.py dev' < 私有配置.json
```

正式环境将 dev 改为 prod，分别配置以保持隔离。也可以将本机私有配置文件路径交给部署助手执行。
换 API 地址或模型名会使旧报告缓存失效；不会把同名但不同供应商的结果混用。
未配置本地分类权重时，只支持保存病例，仍不生成分类或区域解释。

Python 包源参考：[腾讯云软件源文档](https://cloud.tencent.com/document/product/213/8623)。
CI 的 Node 24 Actions 用法参考 [checkout](https://github.com/actions/checkout) 和
[setup-python](https://github.com/actions/setup-python)。
