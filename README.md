## 1.Mapping 
当前建图使用 [run_realtime.sh](/project/docker/VLN/TIC-VLA/DynaNav/warehouse_loop/run_realtime.sh)，在项目根目录执行：
cd /project/docker/VLN/TIC-VLA

bash DynaNav/warehouse_loop/run_realtime.sh \
  --selection-mode instances \
  --landmark-variant original \
  --port 8765 \
  --output "/data/ticvla/outputs/warehouse-loop/instances-original-$(date +%Y%m%d-%H%M%S)" \
  --keep-viewer
它会自动启动 Isaac Sim，控制车辆绕环，同时运行：
- RGB-D 采集、几何建图和 YOLO-E 候选分割：目标 5 Hz。
- InternVL：每 2 秒筛选一次源帧候选实例，不输入导航任务。
- OCR：目标 1 Hz，文字绑定到对应实例。
- 地图后处理：过滤普通重复结构，按当前子字符串规则处理 OCR 冲突。
无需手动切换 Python 环境，也无需提前打开 Isaac Sim；脚本指定使用 /data/ticvla/envs/warehouse-realtime/bin/python 运行感知链路。
启动后，在运行机器的浏览器打开 http://localhost:8765 查看检测、文字和地图。--keep-viewer 会在绕环结束后保留查看服务，按 Ctrl+C 退出。
综合变化场景将 --landmark-variant original 改成 --landmark-variant combined。上述频率为调度目标，实际完成频率以本次输出报告为准。