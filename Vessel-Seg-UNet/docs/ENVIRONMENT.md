# 环境搭建说明(Windows + RTX 4060 Laptop)

> 记录本项目在训练机上的实际安装过程与踩坑,便于重装/换机复现。

## 1. 创建虚拟环境

```powershell
# 项目根目录下(Python 3.11;torch 对 3.11 支持最稳)
python -m venv .venv
```

## 2. 安装 PyTorch(CUDA 12.8)

直连 download.pytorch.org 可能被墙;本机可用阿里云镜像(实测有效):

```powershell
$env:NO_PROXY = "*"   # 重要: 防止 pip 读取系统注册表里的 Clash 代理(代理下大文件会卡死)
$env:PIP_NO_INPUT = "1"
.\.venv\Scripts\pip.exe install "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" `
  --index-url https://mirrors.aliyun.com/pypi/simple/ `
  --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/ `
  --no-cache-dir --timeout 120 --retries 5
```

> 注意版本配对: torch 2.10 ↔ torchvision 0.25(2.7↔0.22、2.8↔0.23、2.9↔0.24、2.11↔0.26)。

## 3. 安装其余依赖

```powershell
$env:NO_PROXY = "*"
$env:PYTHONUTF8 = "1"   # requirements 文件里的注释/编码需要 UTF-8 模式
.\.venv\Scripts\pip.exe install -r requirements.txt -r requirements-dev.txt `
  --index-url https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir --timeout 120 --retries 5
```

## 4. 关键约束与补丁

### albumentations 锁定 <1.4.8

- 1.4.8+ 依赖 `albucore`,后者依赖 `stringzilla`(C 扩展,Windows 无 MSVC 时源码构建失败);
- 1.4.7 是最后一个无 albucore 的版本,且支持 `is_check_shapes`;
- **venv 内补丁**:1.4.7 在 `import albumentations` 时联网检查更新,断网环境抛未捕获的 `TimeoutError` 导致导入失败。已对 `.venv/Lib/site-packages/albumentations/__init__.py` 注释掉 `check_for_updates()`(重装后需重新打补丁);
- 1.4.7 的 pydantic 校验要求 `A.PadIfNeeded(border_mode=cv2.BORDER_CONSTANT)` 显式提供 `value=0`,已在 `src/transforms.py` 处理。

### 网络特性(本机)

- 沙箱/受限上下文里 curl/Invoke-WebRequest 的 TLS 会因 `SEC_E_NO_CREDENTIALS` 失败(schannel 缺凭证),但 **Python(OpenSSL)不受影响**;
- pip 会读取 Windows 注册表代理(Clash Verge, 127.0.0.1:7897),大文件下载会卡死 → 必须 `$env:NO_PROXY = "*"`;
- 国内直连:阿里云镜像可用;PyPI/谷歌类站点需走 Clash。

### opencv 中文路径

OpenCV 的 `imread/imwrite` 不支持中文路径;本项目统一使用 `np.fromfile + cv2.imdecode` / `cv2.imencode + open(wb)`。

## 5. 验证

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m pytest -q          # 27 passed
.\.venv\Scripts\python.exe experiments\smoke_test.py   # GPU 全链路冒烟
```
