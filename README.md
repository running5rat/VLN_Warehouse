# Warehouse VLN

## 语义建图演示

[**打开完整演示：视频、语义地图与实验报告 →**](https://running5rat.github.io/VLN_Warehouse/)

首次使用需按下文启用 GitHub Pages；部署成功后上方入口生效。

[![Isaac Sim 原始仓库实时语义建图预览](demo/semantic_mapping/preview.gif)](https://running5rat.github.io/VLN_Warehouse/)

*12 秒演示片段，3 fps GIF。完整页面包含 95 秒实时建图组合视频和 89.6 秒车载相机视频；预览帧率不代表模型完成频率。*

![最终语义地图与地标文字](demo/semantic_mapping/original-20261013/semantic_bev.png)

基于 `7b97f85622de91674327166f6f84d5390059a306` 的建图能力，在 Isaac Sim 5.0 的 `original` 仓库场景实际绕环一周：51.59 米、448 帧 RGB-D 全部进入几何地图；27 个原始地标，后处理显示 17 个，其中包含六个完整编号。物体静态状态仍为 `tentative`，实时延迟门槛未全部达到。

- [本次结果与限制](demo/semantic_mapping/original-20261013/REPORT.md)
- [演示目录、地图加载及复现说明](demo/semantic_mapping/README.md)
- [车载相机视频文件](demo/semantic_mapping/original-20261013/isaac_camera.mp4)
- [实时建图视频文件](demo/semantic_mapping/original-20261013/mapping_live.mp4)
- [HTML 源文件](demo/semantic_mapping/original-20261013/index.html)（仓库文件页显示源码；观看完整界面请使用上方 Pages 入口）

### 启用在线演示

1. 将本次 README、`demo/`、`.gitignore` 和 `.github/workflows/semantic-demo-pages.yml` 推送到 `main`。
2. 在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。
3. 在 **Actions → Publish semantic mapping demo → Run workflow** 运行发布。之后 `main` 上演示文件更新会自动发布。
4. 部署成功后，通过 `https://running5rat.github.io/VLN_Warehouse/` 观看完整页面。

README 中的 GIF 和地图随代码推送即可显示；完整 HTML 页面通过 Pages 提供视频播放器。配置文件已准备好，仓库设置和远端发布尚需执行。发布构建仅复制页面实际引用的静态素材，不启动仿真或模型。

完整地图归档 `map.tar.gz` 约 833 MiB，保留在本地并已加入忽略规则，不能作为普通 Git 文件推送。需要公开下载时，可单独上传到 GitHub Release，再添加下载链接；不影响视频与地图预览。参见 [GitHub Pages 配置说明](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) 和 [GitHub 大文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

## 建图运行

当前建图入口为 [run_realtime.sh](DynaNav/warehouse_loop/run_realtime.sh)，在项目根目录执行：

```bash
bash DynaNav/warehouse_loop/run_realtime.sh \
  --selection-mode instances \
  --landmark-variant original \
  --port 8765 \
  --output "/data/ticvla/outputs/warehouse-loop/instances-original-$(date +%Y%m%d-%H%M%S)" \
  --keep-viewer
```

它会自动启动 Isaac Sim，控制车辆绕环，同时运行：

- RGB-D 采集、几何建图和 YOLO-E 候选分割：目标 5 Hz。
- InternVL：每 2 秒筛选一次源帧候选实例，不输入导航任务。
- OCR：目标 1 Hz，文字绑定到对应实例。
- 地图后处理：过滤普通重复结构，按当前子字符串规则处理 OCR 冲突。

无需手动切换 Python 环境，也无需提前打开 Isaac Sim；脚本使用 `/data/ticvla/envs/warehouse-realtime/bin/python` 运行感知链路。

启动后，在运行机器的浏览器打开 `http://localhost:8765` 查看检测、文字和地图。`--keep-viewer` 会在绕环结束后保留查看服务，按 Ctrl+C 退出。

综合变化场景将 `--landmark-variant original` 改成 `--landmark-variant combined`。上述频率为调度目标，实际完成频率以输出报告为准。

重新录制并生成演示材料：

```bash
bash demo/semantic_mapping/run_demo.sh
```
