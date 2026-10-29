# MJWarp 后端

仓库提供一个独立的 `mujoco_cable_warp` 后端。模型仍使用 MJCF，绳索路线、
单边张力、控制状态、摩擦和受力映射由 Warp 内核执行，刚体运动与碰撞由
MJWarp 执行。运行时不需要加载 C++ 绳索动态库。

这是第一版后端，尚未覆盖全部 C++ 功能。建议从 Demo 15、17、19 开始。

## 安装

从仓库根目录执行，使用独立环境。原 C++ 后端使用 MuJoCo 3.4，MJWarp 环境
使用 `requirements-warp.txt` 中固定的 MuJoCo 3.16.0、MJWarp 3.16.0 和
Warp 1.18.0；不要在这个环境中执行原项目的 `pip install -e .`，它会要求
MuJoCo 3.4。

```bash
python3 -m venv .venv-warp
source .venv-warp/bin/activate
python -m pip install -r requirements-warp.txt
```

Linux x86_64 上使用 CUDA 12 驱动（例如本机 RTX 4090 的 R550 驱动）时，
把上面的安装命令改为 `python -m pip install -r requirements-warp-cu12.txt`。
该文件选择同版本的官方 `1.18.0+cu12` wheel；普通 PyPI wheel 要求 CUDA 13
驱动。参见 [Warp 官方安装说明](https://github.com/NVIDIA/warp#installing)。

CUDA 运行需要可用的 NVIDIA 驱动。没有 GPU 时可以用 `--device cpu` 执行相同
Warp 内核，适合开发和数值验证。首次运行会编译内核，耗时明显高于后续运行。
如果默认缓存目录不可写，可设置 `WARP_CACHE_PATH` 指向可写目录。

## 运行

统一入口增加了 `--backend mjwarp`，默认仍为 `cpp`：

```bash
# CUDA 单圆柱包络，交互窗口
./scripts/run_demo.sh 15 --backend mjwarp --device cuda:0 --show-route-debug

# CUDA 批量仿真，启用图捕获
./scripts/run_demo.sh 15 --backend mjwarp --headless --nworld 256 \
  --steps 3500 --graph --output outputs/mjwarp/demo15_cuda.json

# CPU 无窗口验证，四个独立环境
./scripts/run_demo.sh 15 --backend mjwarp --device cpu \
  --headless --nworld 4 --steps 3500 --output outputs/mjwarp/demo15.json

# 三杆九索张拉整体，保持控制量为零
./scripts/run_demo.sh 17 --backend mjwarp --device cuda:0 \
  --headless --nworld 256 --steps 1000 --no-auto-control --graph

# 混合刚度模型
./scripts/run_demo.sh 19 --backend mjwarp --device cpu --no-auto-control

# 指定自己的 XML；也可直接调用 Python 入口
python scripts/run_mjwarp_demo.py path/to/model.xml --device cuda:0
```

启动器优先使用仓库 `.venv-warp` 中的 `python3`（macOS 使用 `mjpython`），
不存在时使用当前环境的解释器；`MUJOCO_PYTHON` 可以显式覆盖这一选择。
`MUJOCABLE_BACKEND=mjwarp` 可设置默认后端。

常用参数：

| 参数 | 含义 |
|---|---|
| `--device cpu` / `cuda:0` | 执行设备；命令行默认使用 `cuda:0`，无 GPU 时须显式选择 CPU |
| `--nworld N` | 相同拓扑的独立环境数量 |
| `--headless --steps N` | 无窗口执行指定步数，结束时检查状态并输出 JSON |
| `--duration 120` | 窗口运行的仿真时长；到时退出，不循环重置 |
| `--world N` | 窗口显示哪个环境 |
| `--auto-actuator NAME` | 自动控制的执行器，默认第一个 |
| `--ctrl-final 0.05` | 自动控制最终值，单位取决于模型的控制模式 |
| `--ctrl-ramp-start 0.5 --ctrl-ramp-end 3` | 自动控制线性上升的时间段 |
| `--no-auto-control` | 无窗口时保持零控制；有窗口时使用控制滑块 |
| `--show-route-debug` | 显示端点、引导点和路线 hint |
| `--graph` | 预热后捕获 CUDA 步进图；仅 CUDA 可用 |
| `--nconmax 64 --njmax 256` | 接触和约束缓冲区容量 |
| `--output PATH` | 保存最终状态与运行统计 JSON |

交互模式按显示帧率取回指定环境的状态，表面绳路另行绘制，松弛时变灰。
手动滑块会把同一组控制量应用到全部环境；不同环境的独立控制使用下面的 API。
JSON 中的 `max_final_tension` 是所有环境在最后一步的最大张力，并非全过程峰值。

## 支持范围

| 功能 | 当前状态 |
|---|---|
| `route_mode=native` | 支持，使用 MJWarp 原生 tendon 几何和传力 |
| 无包络的 surface 路线 | 支持端点和 role-3 guide 的折线路线 |
| 单圆柱 surface 包络 | 支持 `endpoint → hint → endpoint`，包括运动圆柱和轴向变化 |
| 圆柱端面约束 | 暂不支持；接触点超出圆柱侧面时报告无效路线并停用该绳的力 |
| 多圆柱复合包络、mesh 表面路线 | 暂不支持，加载时抛出明确异常 |
| 单边弹性、阻尼、松弛、预张紧、张力上限 | 支持 |
| 张紧迟滞、平滑张紧过渡 | 支持 |
| 收缩量、卷轴角、卷轴线/角速度、真实卷轴关节 | 支持五种控制模式 |
| 控制滤波和最大收绳速率 | 支持，状态每个物理步只推进一次 |
| 有符号预绕余量、卷轴反力矩 | 支持 |
| surface 包络/guide 的 Capstan 摩擦 | 支持正向、反向和速度定向传播 |
| native 路线的插件 Capstan 摩擦 | 暂不支持 |
| `integration_mode=explicit` | 支持 |
| `local_implicit` / `implicit_compliant` | 暂不支持，加载时报错 |
| 刚体积分器 | Euler、implicitfast；不支持 RK4 和 sleeping |
| 路线/复合连接迟滞、显示平滑 | 暂不支持非零设置 |
| 批量状态与按环境重置 | 支持 |

单圆柱求解固定初始化 hint 选出的绕行方向，运行时不把 hint 当作受力点。
侧面展开后解析计算螺旋切线的轴向坐标。`capstan_mu=0` 时自动使用解析快速路径：
长度为两段直线加精确螺旋弧长，长度速度由端点与切点的刚体速度计算，受力只在
两个端点和两个切点映射到广义力，包含运动圆柱的反力和反力矩。它不执行逐弧点
物理采样；其连续弧长与 C++ 的折线近似存在小幅差异。

有摩擦的圆柱仍使用原有分段传播算法，采样间距与上限采用 C++ 实现的 1.5 mm
和 96 点。绳路与受力内核使用 32 线程的 CUDA 线程块，增加小批量绳路的并行块数。
直线 guide 和 native 路线保持原有力学语义。

`implicitfast` 仍负责刚体系统中 MJWarp 已知力项的积分，但**不会自动把自定义
绳索刚度/阻尼加入隐式算子**。高刚度模型需要验证步长；不得将本后端结果解释为
原 C++ `implicit_compliant` 的等价实现。

Demo 32 原文件使用 RK4，因此不能直接启动。若要试验解析孔口，可复制模型并
显式改为 Euler/implicitfast，再验证所选步长；加载器不会自行修改积分器。
Demo 30 的独立有质量绳连续体控制器也不在本后端范围内。

## Python API

在仓库根目录设置 `PYTHONPATH=python`，然后：

```python
import numpy as np
from mujoco_cable_warp import CableSimulation

sim = CableSimulation(
    "cable_plugin_demos/15_cpp_plugin_surface_single_pulley.xml",
    nworld=4,
    device="cuda:0",  # 无 GPU 时改为 cpu
)
sim.set_control(np.array([[0.00], [0.01], [0.02], [0.03]]))
sim.step(100)
sim.check()  # 显式同步：检查数值、无效路线和缓冲区溢出

state = sim.state.values.numpy().copy()  # (nworld, ncable, 12)
sim.reset([1, 3])                       # 只重置指定环境，包括控制和绳索历史
host_data = sim.get_data(world=0)       # 用于 CPU 检查或显示
points, counts = sim.get_route_points(world=0)  # 按需生成一个环境的显示路线
```

`get_route_points()` 返回 `(ncable, capacity, 3)` 的 CPU 点数组和每根绳的有效点数。
解析圆柱在独立显示缓冲区中按点并行采样，只有调用此方法（如 viewer 每帧）时才
执行。无窗口物理步进不分配/生成显示样本。采样不会修改物理状态、控制历史或
CUDA 图；`state.points` 是物理内部数据，解析圆柱仅有四个受力点，不能直接当作
显示折线使用。有摩擦的路线直接复制已有物理采样点。

十二个诊断量依次为：

```text
length, velocity, free_length, contraction, extension, tension,
taut, saturated, route_status, tangent_residual,
surface_residual, solver_iterations
```

原生路线的 MJCF 传感器仍为八维，surface 传感器为十二维。诊断量对应最近一次
求力状态；`step()` 后的积分位置已经前进一个时间步，和 MuJoCo 的常规数据语义
一致。需要重新求值时调用 `forward()`，它不会推进卷绳量或仿真时间。

`sim.data` 是 MJWarp 的设备数据，`sim.state` 是额外的绳索设备状态。
训练程序可以直接操作 `sim.data.ctrl`，避免逐步 CPU 上传/下载。保存和恢复
完整状态时还必须保存绳索的积分控制量、滤波量、张紧 latch、参考长度和绕行
分支；仅保存 MJWarp 的 qpos/qvel 不足以恢复绳索历史。

CUDA 图捕获前先执行一次预热步进，再 reset，随后调用 `sim.capture_step()`。
捕获后更新数组内容，不替换数组对象。图捕获不包含外部控制策略。

## 代码位置

- `python/mujoco_cable_warp/model.py`：解析配置、转换 MJCF、检查支持范围。
- `python/mujoco_cable_warp/kernels.py`：绳路、张力、摩擦、力映射和控制状态内核。
- `python/mujoco_cable_warp/simulation.py`：分配批量状态，连接 MJWarp 回调，提供步进与重置。
- `python/mujoco_cable_warp/cli.py`：无窗口运行、交互显示和结果导出。
- `scripts/run_mjwarp_demo.py`：源码树入口。

静态拓扑在 CPU 加载一次；运行时计算留在设备。张力通过 actuator gain 回调
提供给原生 tendon，surface 力与卷轴反力矩通过 passive 回调加入广义力。
模型中的插件传感器转换为 user sensor，名称、顺序和维度保留。

## 验证

在 MJWarp 环境中：

```bash
python -m pytest tests/test_mjwarp_backend.py tests/test_portable_paths.py -q

# 在有 NVIDIA 驱动的机器上运行相同测试，包括 CUDA 图测试
CABLE_WARP_DEVICE=cuda:0 python -m pytest tests/test_mjwarp_backend.py -q
```

测试覆盖单边本构、控制限幅、传感器布局、独立 CPU 力学轨迹、卷轴反力矩、
孔口张力比、圆柱表面接触与虚功、无效路线停用、批量独立性及局部重置。

与 C++ 插件比较时使用两个独立解释器，避免混用 MuJoCo ABI。先在原 MuJoCo
3.4 环境中构建插件，再在 MJWarp 环境执行：

```bash
python scripts/compare_mjwarp_reference.py \
  --reference-python /path/to/cpp-env/bin/python \
  --plugin build/plugin/libcable_unilateral.so \
  --model cable_plugin_demos/15_cpp_plugin_surface_single_pulley.xml \
  --steps 200 --device cpu --output outputs/mjwarp/compare15.json
```

比较脚本检查全程 qpos、qvel、绳长和张力，而非只比较最终位置。允许 float32
误差；报告中包含实际最大误差和验收阈值。程序返回非零退出码表示比较失败。

对包含接触的机构，还可以使用 `--mode forces`，在相同参考姿态和速度下检查
绳长、张力与广义力，分离绳索计算误差和轨迹累积偏差。例如将上面的模型改为
Demo 17 并添加 `--mode forces`。

早期 Warp CPU 路径的 200 步对照中，Demo 15 的完整轨迹通过，最大绳长误差约 `1.1e-7 m`，最大
张力误差约 `1.2e-4 N`；Demo 17 在相同姿态/速度下的受力比较通过，最大张力
误差约 `2.4e-4 N`。该 CPU 路径的 Demo 17 自由接触轨迹未通过同一套严格阈值（最大 qvel
分量差约 `7.0e-3`），因此尚不能把它与 MuJoCo 3.4 的轨迹视为逐步一致。

本机也已在 RTX 4090（驱动 550.163.01，Warp `1.18.0+cu12`）上验证 CUDA：

- CUDA 后端与路径检查合计 16 项测试通过，另有 2 项子测试通过；包含图捕获、
  更新控制量、卷绳历史和图捕获后的局部重置。
- 200 步 C++ 对照：Demo 15 轨迹通过，最大绳长误差 `1.05e-7 m`、
  最大张力误差 `9.86e-5 N`；Demo 17 同状态受力通过，最大张力误差
  `2.16e-4 N`、最大广义力分量误差 `1.61e-4`。
- Demo 15 在 `cuda:0` 上使用 CUDA 图运行 256 个环境、每环境 3500 步，
  预热后实测约 1.52 秒，合计约 58.9 万环境步/秒。负载上升约 5 cm，
  最终张力约 `1.97088 N`。此数据为单卡单次运行，不含加载/JIT 时间，
  不是与 C++ 后端的公平加速比测试。

结果保存在 `outputs/mjwarp/demo15_cuda.json`、`compare15_cuda.json` 和
`forces17_cuda.json`（输出目录不纳入版本管理）。GPU 图形窗口尚未实测。

## 2048 环境的绳索计算开销

使用 `scripts/benchmark_mjwarp_overhead.py` 可隔离在线绳索计算的成本：

```bash
.venv-warp/bin/python scripts/benchmark_mjwarp_overhead.py 15 \
  --nworld 2048 --control 0.01 --output outputs/mjwarp/overhead15_2048.json
.venv-warp/bin/python scripts/benchmark_mjwarp_overhead.py 17 \
  --nworld 2048 --output outputs/mjwarp/overhead17_2048.json
```

基准在预热/运动 200 步后的同一输入状态上重复单步。对照组使用预计算的绳索
力和传感器值，实验组每步执行完整绳索内核及控制状态更新；两组均使用 CUDA 图，
保留相同 tendon 拓扑、设备分配、姿态恢复拷贝和物理求解。脚本检查两组的位置、
速度、受力及传感器数值。GPU 接触求解的加速度允许同图重复执行测得的舍入噪声，
并把原始差值写入报告。这样避免直接关闭绳索后接触/运动改变而污染计时。

这测量的是**相对于预计算力回放的在线绳索计算增量**，不是删去所有 tendon 后
与纯关节电机的比较，也不是完整 RL 训练测速。两组姿态/控制均固定，各环境没有
随机化，不能覆盖抓取过程中各种接触状态。

RTX 4090、MuJoCo/MJWarp 3.16.0、Warp 1.18.0+cu12，2048 环境，
每组 1000 步、交替执行 5 次取中位数，优化前实测：

| 模型 | 预计算力回放基线 | 在线绳索 | 额外耗时 | 单步耗时增加 | 吞吐下降 |
|---|---:|---:|---:|---:|---:|
| Demo 17：每环境 9 根 native 绳，18 自由度 | 0.319 ms | 0.336 ms | 0.017 ms | 5.4% | 5.1% |
| Demo 15：每环境 1 根圆柱 surface 绳，6 自由度 | 0.237 ms | 0.454 ms | 0.217 ms | 91.5% | 47.8% |

这里一“步”是全部 2048 个环境各前进一个物理子步。差异包含两个模型本身的
物理复杂度，不能据此推算每增加一根手指绳索的固定成本。

加入无摩擦圆柱解析路径、显示采样分离及 32 线程块后，用同一套基准重测：

| 模型 | 预计算力回放基线 | 在线绳索 | 额外耗时 | 单步耗时增加 | 吞吐下降 |
|---|---:|---:|---:|---:|---:|
| Demo 17：每环境 9 根 native 绳 | 0.320 ms | 0.334 ms | 0.014 ms | 4.5% | 4.3% |
| Demo 15：每环境 1 根无摩擦圆柱绳 | 0.236 ms | 0.253 ms | 0.017 ms | 7.0% | 6.6% |

Demo 15 的整个批量物理步耗时降低约 44%，单独的在线绳索增量耗时降低约 92%。
这些结果不代表有摩擦包络也有相同收益：有摩擦路径继续执行逐段物理计算。
报告为 `overhead15_2048_analytic.json` 和 `overhead17_2048_analytic.json`。

另用统一入口执行真实连续 rollout（相同卡、CUDA 图、不含 JIT 和显示）：

| 模型 | 每环境步数 | 墙钟时间 | 合计环境步/秒 |
|---|---:|---:|---:|
| Demo 15，自动收绳 | 3500 | 1.590 s | 451 万 |
| Demo 17，零控制 | 1000 | 0.342 s | 599 万 |

结果分别保存在 `outputs/mjwarp/demo15_2048_cuda.json` 和
`outputs/mjwarp/demo17_2048_cuda.json`，为优化前记录。
优化后 Demo 15 的同样连续 rollout 耗时为 **0.893 s**，约 **803 万环境步/秒**，
吞吐提升约 **1.78 倍**，最终张力仍约 `1.97088 N`；报告为
`outputs/mjwarp/demo15_2048_cuda_analytic.json`。

优化后 CUDA 后端与路径检查合计 18 项测试通过（另有 2 项子测试），CPU 后端
15 项通过、2 项 CUDA 专用测试跳过。新增测试包含倾斜运动圆柱及活动端点的
全自由度长度梯度/虚功、长度速度与功率一致性、精确螺旋弧长、显示采样不修改
物理状态，以及显示/局部重置后的 CUDA 图一致性。
Demo 15 的 200 步 C++ 轨迹对照继续通过原有阈值，最大绳长差 `4.18e-6 m`、
最大张力差 `1.18e-4 N`（`compare15_cuda_analytic.json`）；长度差包含解析弧长
与原有折线长度的差异，不能解释为两个算法逐位一致。

进一步对当前 CUDA 版本执行 **2000 步完整轨迹对照**，两个模型均通过同一套
原有阈值（不调整验收阈值）。参考端为 MuJoCo 3.4 + C++ 绳索插件，测试端为
MuJoCo/MJWarp 3.16.0 + 当前 Warp 绳索实现；控制为第一个执行器恒定 `0.01`：

| 模型 | 仿真时长 | 最大 qpos 分量差 | 最大 qvel 分量差 | 最大绳长差 | 最大张力差 |
|---|---:|---:|---:|---:|---:|
| Demo 15，无摩擦单圆柱 | 2 s | 7.86e-8 | 2.67e-6 | 4.18e-6 m | 1.18e-4 N |
| Demo 17，三杆九索 | 1 s | 1.01e-6 | 3.89e-5 | 1.79e-7 m | 2.80e-4 N |

报告为 `compare15_cuda_2000.json`、`compare17_cuda_2000.json`。
当前 CUDA Demo 17 的 200 步同状态受力对照也通过，最大张力差 `2.16e-4 N`、
最大广义力分量差 `1.61e-4`（`forces17_cuda_analytic.json`）。
上述结果表明已测配置下的数值近似一致，不能推导出任意模型、控制轨迹、刚度、
步长或长期接触过程都等价；尤其不覆盖尚未移植的复合包络和隐式绳索积分。

此处没有接入 ParaHand-Sim2Real 的
真实手部绳路、物体分布、策略、奖励、重置或 PPO 更新，因此不能将上述百分比
当作该项目训练时间的增幅。绳索若要求更小物理步长，还需单独计入增加的子步数。
