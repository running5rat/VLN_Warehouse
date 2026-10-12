# 本机 Isaac Sim 环境与启动路径

2026-09-28 在宿主机核对。项目仿真统一使用 `/data/isaacsim`，训练继续使用现有 Docker/Conda 环境。

| 位置 | 实际版本/用途 |
| --- | --- |
| `~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64.zip` | Isaac Sim 5.0.0 安装包，压缩文件约 8.17 GiB；不是正在运行的环境 |
| `/data/isaacsim` | 项目当前使用的 5.0.0，内置 Python 3.11.13；`.env.testing` 已指向这里 |
| `~/isaacsim` | 另一份独立的 5.0.0；此前 `.bashrc` 指向这里 |
| `~/Desktop/isaac_sim` | 6.1.0，内置 Python 3.12；此前桌面图标实际启动这里 |
| Docker `ads_docker_root` | 模型训练环境；挂载 `/data`，但系统库与宿主机不同 |

ZIP 与 `/data/isaacsim` 的 `VERSION`、`isaac-sim.sh`、`python.sh`、
`setup_python_env.sh`、`apps/isaacsim.exp.full.kit` 和 `kit/kit` 的大小和 CRC
逐项一致。两者版本字符串均为
`5.0.0-rc.45+release.23960.184afb15.gl`。这验证了版本与关键文件关系，
不等于对压缩包内所有文件做了完整校验。

## 本次修改

- `.bashrc` 的 `ISAAC_SIM_ROOT` 改为 `/data/isaacsim`，并把这两项 Isaac
  环境变量移到 Conda 自动管理段之外。原文件备份为
  `~/.bashrc.before-isaac-path-20260928-151310`。
- 新增 `~/.local/share/applications/isaac-sim-5.0-ticvla.desktop`，名称为
  **Isaac Sim 5.0 (TIC-VLA)**，固定执行 `/data/isaacsim/isaac-sim.sh`。
  使用独立文件名，因为 5.0 的 `isaacsim.app.setup` 会自动覆盖通用的
  `IsaacSim.desktop`，且生成内容没有 `Exec`。

## 如何启动

在 Ubuntu 宿主机终端运行；若启用了 Conda，先 `conda deactivate`，直到
`CONDA_PREFIX` 不再设置。新终端会读取修正后的路径，也可直接执行：

```bash
/data/isaacsim/isaac-sim.sh
```

应用菜单中也可选择 **Isaac Sim 5.0 (TIC-VLA)**。不需要再次解压 ZIP，
也不需要为此修改模型 pipeline。

DynaNav 默认是无窗口模式，这与 GUI 应用是否能启动是两个问题。
从本仓库根目录运行带窗口的 benchmark：

```bash
source .env.testing
export TICVLA_HEADLESS=0
./DynaNav/run_benchmark.sh DynaNav/configs/benchmark_example.yaml
```

宿主机是 Ubuntu 22.04，glibc 2.35；当前训练容器的 glibc 是 2.31。
项目 `run_benchmark.sh` 对 Isaac Sim 5.0 要求 glibc ≥2.34，因此不能仅因
容器也挂载了 `/data/isaacsim` 就认为它具备相同的仿真运行环境。
继续使用“宿主机运行 Isaac Sim、容器运行模型训练”的现有划分。

## 验证与结论边界

宿主机 RTX 4090、驱动 580.178.04，显示会话 X11 `:0` 可用。
`/data/isaacsim/python.sh` 的 PyTorch 2.7.0+cu128 能访问 CUDA。
实际启动 5.0 GUI 后，窗口标题为 `Isaac Sim Full 5.0.0 - New Stage*`，
约 13 秒记录 `app ready`，首次着色器编译后约 129 秒记录
`Isaac Sim Full App is loaded.`，随后窗口更新脚本输出 `ISAAC_GUI_PROBE_OK`。
首次探针退出被未保存场景提示拦截，最终由测试超时清理；这不是启动失败。
完整探针日志位于 `/tmp/ticvla-isaac-startup-7mjfdgj0/isaac5-gui.log`。
随后仅在测试实例中忽略空场景的未保存退出提示，第二次启动约 9 秒记录
`app ready`，通过 180 次窗口更新检查并正常退出（退出码 0）。日志为
`/tmp/ticvla-isaac-startup-7mjfdgj0/isaac5-gui-warm.log`。独立启动项没有被
程序自动改写，桌面入口格式和 `.bashrc` 语法检查均通过。

2026-09-28 15:07 的失败记录来自 **6.1.0**，在扩展加载约 15.5 秒处结束，
没有记录明确的 Fatal/Error 或退出原因。因此可以确认之前启动的版本与项目
不一致，不能据此断言 6.1.0 的具体失败原因，也没有证据要求重装显卡驱动。
