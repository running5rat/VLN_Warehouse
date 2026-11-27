# Warehouse NavMesh 实测

在现有 Isaac Sim 5.0 仓库中，实际导出静态三角形几何，使用固定版本 Recast 1.6.0 建网、Detour 查路并保存示意图。工具不改变场景，也不执行车辆控制。

## 运行

在仓库根目录、当前容器中执行：

```bash
bash docs/design/navmesh_demo/run.sh \
  --output /data/ticvla/outputs/navmesh/warehouse-new \
  --publish demo/navmesh/warehouse-new
```

已有几何可跳过 Isaac 导出，仅重新建网、查询和绘图：

```bash
bash docs/design/navmesh_demo/run.sh \
  --skip-export \
  --output /data/ticvla/outputs/navmesh/warehouse-20261127 \
  --publish demo/navmesh/warehouse-20261127
```

脚本自动通过已有宿主桥接运行 `/data/isaacsim/python.sh`；无需切换当前 Python 为 Isaac 环境。Recast 在 `/data/ticvla/tools/recast-v1.6.0` 独立编译，绘图使用 `NAVMESH_PYTHON`（默认 `/root/miniconda3/bin/python`，需 NumPy、Matplotlib，首次下载需 requests）。不升级模型或 Isaac 的依赖。下载归档 SHA256 固定在 `source-lock.json`。

## 实际过程和参数

1. 从原始 `Simple_Warehouse/full_warehouse.usd` 导出可见 Mesh 的世界坐标三角形，包括实例变换；只取演示通道范围 `x=[-18,-3], y=[2,31], z=[-0.1,3] m` 相交的几何。未使用手绘货架盒子或补造地板。
2. 世界 Z-up 转为 Recast Y-up，栅格尺寸 0.10 m，高度分辨率 0.05 m；斜率上限 15°，台阶上限 0.10 m，净空高度 1.60 m。只标记世界高度不超过 0.15 m 的地面三角形为可通行候选，保留其余几何作为障碍。
3. 高度场过滤、紧凑高度场、机器人半径收缩、距离场、Watershed 区域划分、轮廓、多边形及细节三角网格。区域最小 16 cells，合并阈值 64 cells，多边形至多六个顶点，轮廓误差 1 voxel。
4. 半径 0.35 m 向上取整为 4 cells，实际收缩尺度为 0.40 m；另运行半径 0 的点机器人对照。
5. 将 Detour 单 tile 数据保存后重新加载，执行四段有序绕环查询、跨通道目标查询和货架内目标负例。路径由多边形 A* 搜索和 Funnel 提取；本次没有增加通道中线偏好。
6. 路径每 0.05 m 采样，检查其位于输出多边形内，并在收缩前的可通行栅格上复核半径 0.35 m 的圆形 footprint 扫掠区域。

导出几何为 USD 三角形，不等同于 PhysX cooked 碰撞形状；圆形近似不代替矩形 footprint、朝向、动力学和控制器实际执行验证。完整场景几何只用于离线 Oracle，此结果不作为 VLM 已观测语义地图。

## 输出格式

| 文件 | 内容 |
|---|---|
| `scene_world.npz` | 输入三角形、世界坐标顶点和截取范围，保存在 `/data` 实验目录 |
| `scene.bin` | 建网输入：顶点数/三角形数、范围、Y-up 顶点和三角形索引 |
| `walkable-before.csv` / `walkable-after.csv` | 半径收缩前后的可通行栅格中心及区域 ID |
| `navmesh.json` | 世界坐标顶点、多边形索引、区域 ID、逐边邻接关系；`-1` 表示边界 |
| `navmesh.obj` | 可被网格查看器加载的多边形几何；不包含 Detour 查询结构 |
| `navmesh.bin` | Recast/Detour 1.6.0 本机编译的单 tile 二进制；不能当作任意版本通用格式 |
| `queries.json` | 请求点、吸附点、多边形走廊、Funnel 路径和查询状态 |
| `report.json` / `environment-lock.json` | 参数、结果、校验、限制、来源及环境哈希 |
| `index.html` 和五张 PNG | 场景几何、NavMesh、半径对照、跨通道路径及绕环查询 |

## 本次结果

- 174,239 个输入三角形，72 个导航顶点，38 个导航多边形。
- 可通行栅格面积由 359.93 m² 收缩至 278.58 m²。
- 五条指定路线全部查询成功；货架内部目标被拒绝。
- 保存重载、路径网格包含性、圆形 footprint 栅格复核均通过。
- 首次建网、保存重载和六条查询合计约 0.069 s；不包含 Isaac 启动/导出、编译和绘图。每次实测值见报告。
- 未执行车辆移动，不将查路成功记为导航成功。

参考实现：[Recast Navigation](https://recastnav.com/)、[Sample_SoloMesh.cpp](https://github.com/recastnavigation/recastnavigation/blob/v1.6.0/RecastDemo/Source/Sample_SoloMesh.cpp)。
