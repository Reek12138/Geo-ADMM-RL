# Geo-ADMM RL 进展

## 2026-11-23

### 单卡节点与初始模型

- Fuyao `50212646` 提供一张 PPU-ZW810E。SOFY 作业名为 `bifrost-2026112320344601-zhuy25`。本实验在同一节点顺序运行。
- 同一 30 机交叉拓扑族的 512 个场景通过解析间距复核。单场景求解耗时中位数 7.89 ms，P95 10.69 ms。测试使用 50 m 水平和 91.44 m 垂直门槛。此数据来自简化几何求解，BlueSky 的仿真吞吐另行测量。[PPU 几何结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/milestone1/ppu.GTAiIE/dense512.json)
- 调速行为克隆共有 768 个训练场景尝试，642 个得到可验证专家标签。验证动作 MSE 0.0343。101 个标称有冲突且专家可解的验证场景，K=2 热启动解 34 个，K=10 冷启动解 32 个。[首轮训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/stage0/pilot.Hmh0CY/model/training.json)
- 调速和高度联合行为克隆尝试了 2,048 个训练场景，1,418 个得到可验证标签，其中 442 个依赖高度。验证动作 MSE 0.0455。K=2 热启动在验证集解决 160/212 个标称有冲突且专家可解场景，K=10 冷启动为 148/212。留出的近对头集合分别为 42/69 和 57/69。全量验证和近对头集合分别为 214/384 和 74/384。双层策略是当前 Geo-ADMM 的初始化候选，近对头泛化还未达到阶段门槛。[第二轮训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/stage0/tiered.v5cUvo/model/training.json) · [初始化权重](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/stage0/tiered.v5cUvo/model/actor.pt)

### 两个 BlueSky 加速分支

两个分支独立接入 Geo-ADMM RL。方案 A 用于原生 BlueSky 采样和微调，方案 B 用于 PPU 预训练。正式配对评测统一用原生 BlueSky 0.05 s，配对基准由 Geo-ADMM 实验的权重与冻结流量确定。

| 验收 | 证据 |
| --- | --- |
| 方案 A 单元测试和多空域实测 | 通过 2 项上游测试。修复航点 API 和仿真时钟后，原生为 263.8 飞机·控制步/秒，打包 1/8/16/32 为 261.3/1,975.5/3,095.9/5,266.3，同一容器内对应 1.0×/7.5×/11.7×/20.0×。[测速](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate1/a-fixed.iUkCRL/results/benchmark-a.log) |
| 方案 A 原生逐帧对照 | 12 架、600 帧、0.05 s，含调速、升降和航向命令。fast 单空域、fast 打包 8 空域与原生的经纬度、高度、真空速、航向和 LNAV 最大误差均为 0。包含 0.2° 输入和 0.25° 死区。当前环境使用 BlueSky legacy 气动模型。[逐帧结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate1/a-fixed.iUkCRL/results/parity-a.log) · [兼容修补](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate1/a-fixed.iUkCRL/compatibility.patch) |
| 方案 B 上游单元测试与原生逐帧对照 | 通过 2 项上游测试。0.05 s、120 帧、短程单机机动、CPU float64，使用原生 legacy 模型的实测性能参数，最大经纬度与航向误差为 0，高度 1.27e-11 m，真空速 3.13e-13 m/s。[逐帧结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate1/b.RjPRP4/results/parity.log) · [测试](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate1/b.RjPRP4/results/pytest-b.log) |
| 方案 B 单卡 PPU 仿真微基准 | 每空域 16 架，0.25 s、50 控制步。256/512/1024 空域分别为 182,929.7/386,712.9/768,870.9 飞机·控制步/秒，Torch 峰值显存 4.9/9.7/19.4 MiB。[测速](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate1/b.RjPRP4/results/benchmark-b.log) |
| 0.05 s 原生配对评测 | 18 组完整。冻结基线与锚定候选均使用同一逐帧口径和五个流量种子，并含候选关闭投影的消融。 |

上游引用分别为 `406a22fbf31c4e141442e1fc745b6b3c1c35fbd2` 和 `f38a7d9b200750b226c49af149b2d0b49b0e4772`。[上游仓库](https://github.com/Reek12138/bluesky-for-rl)

### 首轮端到端训练吞吐

训练使用同一个 Geo-ADMM GraphFormer/Twin Critic 和设备回放。方案 B 的权重已加载到方案 A 的高保真微调。以下三组各运行 440 个决策周期。A 的配对使用相同起始权重、worker 种子、物理步长、更新预算与控制周期。PPU 利用率为完整进程时段每秒 PPU-SMI 样本的算术平均，包含启动和收尾。

| 配置 | 完整轮数 | 轮/h | 更新/s | 空域·控制步/s | PPU 平均 | Torch 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原生 0.05 s，9×1 | 9 | 439 | 11.78 | 53.6 | 19.2% | 771 MiB |
| A fast 0.05 s，9×8 | 73 | 3,464 | 11.52 | 417.5 | 19.6% | 772 MiB |
| B 0.25 s，512 空域 | 520 | 29,274 | 13.67 | 3,522.8 | 41.4% | 756 MiB |

方案 A 的样本吞吐为 7.79×，轮/h 为 7.89×，更新速度基本持平。Learner 每个决策周期最多执行两步更新。方案 B 已验证动作、物理、观测、回放和优化均走 PPU，日志汇总每 20 步读取设备计数。三组为吞吐诊断运行，使用上游流量且未启用 ADMM 投影。当前代码中的可选 K=2 投影、有向对头航路、两端的实际梯度更新已通过小规模 CPU 烟测，后续要在 PPU 测量带投影的收益及原生留出质量。[B 训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-512.pxlfBS/run/metrics.jsonl) · [B PPU 日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-512.pxlfBS/ppu-utilization.log) · [原生训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/a-native-paired.3uJP81/native/metrics.jsonl) · [A 训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/a-native-paired.3uJP81/a/metrics.jsonl)

### 启用 Geo-ADMM 后的训练

- 扩展方案 B parity 在 PPU 上执行 float64 物理，0.05 s 共 600 帧。最大误差为经纬度 0°、高度 1.40e-10 m、航向 2.84e-14°、真空速 1.48e-12 m/s。单机机动覆盖不到多机转移误差，正式评测仍采用原生 BlueSky。[扩展 parity](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-512-admm.Z0xnf0/ppu-parity.log)
- 加载原生 B737 巡航标定的可选入口为 `--performance-calibration`。12 个原生巡航点测得加速度 2.0 m/s²、坡度 25°、默认垂直速度 7.62 m/s。此前 B 的训练吞吐试验采用上游默认巡航参数，逐帧 parity 使用原生标定参数。[测量](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate1/b737-cruise.odlLuv/perf.json) · [配置](../configs/b737_legacy_cruise.json)
- 方案 B 改进运行使用 512 个空域、随机 2–8 航路、float64 物理、K=2 投影、128 样本梯度批和每决策 4 次更新。152.8 s 完成 1,139 轮和 3,028 次更新，即 26,842 轮/h 和 19.82 更新/s。PPU-SMI 的 162 条每秒样本平均 79.3%，中位数 86%，峰值 98%；Torch 峰值 1,481 MiB。累计碰撞记录 9,180 架次，训练质量仍待改善。[训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-512-admm.Z0xnf0/run/metrics.jsonl) · [PPU 利用率](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-512-admm.Z0xnf0/ppu-utilization.log)
- 两个开发种子使用原生 0.05 s。行为克隆基线与首轮 A 微调在四航路交叉分别到达 15/24 和 12/24，近对头分别到达 17/24 和 9/24。关闭投影的 A 首轮权重仅到达 8/24 和 3/24。两套权重在 ±15° 各 24 架的诊断中均没有到达。首轮 A 权重不提升为基线。[配对诊断](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/development.dg29rA/run/comparison.json)
- 下一轮 A 加入 CPU 采样与 PPU 更新重叠、航向恢复监督、通过投影的动作蒸馏、同高度图着色与 K=10 冷启动回退。已有两 worker 烟测并提交到同一单卡节点。新权重沿用独立开发流量作为候选门槛。[A 改进运行](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/a-overlap-admm.dx7HZY/run.sh)

### 方案 A 重叠采样与正式评测

- 9 个 worker、每 worker 8 空域、0.05 s，启用 ADMM 及回退。152.2 s 完成 204 轮和 3,188 次更新，即 4,827 轮/h 和 20.95 更新/s。169 条每秒 PPU-SMI 样本平均 73.2%，中位数 84%，峰值 98%。Torch 峰值为 1,489 MiB。相对首轮方案 A 改变了航路、学习批和每轮更新预算，增益反映整个配置。[微调记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/a-overlap-admm.dx7HZY/run/metrics.jsonl) · [利用率](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/a-overlap-admm.dx7HZY/ppu-utilization.log)
- 三个开发种子近对头为 34/36 基线到达和 20/36 微调到达。+15° 为 0/36 和 9/36。新权重仅作为在研候选，保留 Geo-ADMM 基线。[开发评测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/a-overlap-admm.dx7HZY/development-eval/comparison.json)
- 额外保留气动预更新的方案 A 兼容版通过 600 帧复验，质量与性能阶段最大误差也为 0。打包 1/8/16/32 相对同容器原生步进的吞吐为 1.0×、7.5×、12.2×、20.8×。气动预更新修补经独立快照验真，以上训练吞吐对应训练时的兼容修补版本。[气动 parity](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate1/a-aerodynamics.XVdwU7/results/parity-a.log) · [复测速](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate1/a-aerodynamics.XVdwU7/results/benchmark-a.log)
- 早期配对扫描使用原生 0.05 s 积分，每个 6 s 控制帧末检查事件，用于诊断。逐帧评测入口已补齐。6 s 内短暂交会测试确认两架飞机在控制窗口内被正确判为碰撞。[控制帧诊断](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/holdout.1pMj3p/run/comparison.json) · [瞬时冲突回归](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/frame-audit-smoke.56b1ck/short-encounter.log)
- 后续微调以冻结的行为克隆基线为锚，优先保留已验证的冲突行为，继续训练无风险时的航向恢复。[锚定微调](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/a-anchor.JHbi6d/run.sh)
- 正式配对固定 12001–12005 五个流量种子，使用独立 Geo-ADMM 三/四/六航路、近对头和 ±15° 场景。两个权重均按原生 0.05 s 逐帧检查，使用同一投影循环和回退，并为候选做关闭投影消融。碰撞门槛为水平小于 5 km 且垂直小于 300 ft，到达门槛为目的地 5 km，航路横向越界为 20 km。成本为逐机累计绝对归一化指令的线性加权，调速/升降/航向权重分别为 1/2/4。[逐帧配对运行](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/frame-audited-holdout.moQMNS/run.sh)

### 锚定微调与航向投影诊断

- 锚定训练用时 125.2 s，完成 155 轮、2,548 次更新，实测 4,457 轮/h、20.35 更新/s，平均 PPU 利用率 72.5%。同一组开发流量近对头到达由上一轮 20/36 回升至 31/36，冻结基线为 34/36。正负 15° 到达分别为 10/36 和 8/36，冻结基线均为 0/36。开发评测每 6 s 检查事件，与逐帧正式口径分开报告。[训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/a-anchor.JHbi6d/run/metrics.jsonl) · [开发评测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/a-anchor.JHbi6d/development-eval/comparison.json)
- 两个偏航探针中，旧保护规则首步都清零了所有实际航向指令，各有 6 架飞机未在冲突图中。工作区现使用逐机的转向可达横向偏差上界和预测间距限制航向，隔离飞机保留可执行转向。新增回归后，10 项本地测试通过。逐帧正式评测沿用冻结的上一版控制器，以保持同种子协议；新版控制器单独评测。[动作探针](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/nav-probe.DlNvkT) · [新版开发评测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/component-heading.1vjJL7/evaluation.log)
- 新版航向保护使用两个开发种子、原生逐帧口径。基线与锚定模型同用局部保护和 K=2/K=10。+15° 候选到达 15/24、碰撞 4/24、越界 5/24。−15° 候选到达 12/24、碰撞 6/24、越界 6/24。两个偏航方向的冻结模型均 0/24 到达、24 架越界。对头候选为 20/24 到达、4/24 碰撞，基线为 22/24 到达、2/24 碰撞。回航进展与对头差距均需更大样本确认。[逐帧开发对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/component-heading.1vjJL7/run/comparison.json)

### 冻结留出逐帧结果

使用原生 BlueSky legacy 气动模型，0.05 s 物理和事件采样，6 s 控制周期。每个场景类别有 5 个配对流量种子，共 60 架飞机。每格为到达、碰撞、越界数，三组均无超时。冻结锚定候选与基线使用上一版航向保护，逐机新版结果详见开发对照。

| 场景 | 冻结基线 | 锚定候选 | 锚定候选关闭投影 |
| --- | ---: | ---: | ---: |
| 三航路交叉 | 48 / 12 / 0 | 46 / 14 / 0 | 28 / 32 / 0 |
| 四航路交叉 | 36 / 24 / 0 | 30 / 30 / 0 | 24 / 36 / 0 |
| 六航路交叉 | 32 / 28 / 0 | 28 / 32 / 0 | 24 / 36 / 0 |
| 近对头 | 58 / 2 / 0 | 54 / 6 / 0 | 22 / 38 / 0 |
| +15° | 0 / 0 / 60 | 7 / 0 / 53 | 32 / 12 / 16 |
| −15° | 0 / 0 / 60 | 6 / 0 / 54 | 28 / 8 / 24 |
| 合计 | 174 / 66 / 120 | 171 / 82 / 107 | 158 / 162 / 40 |

锚定候选的总体到达率为 47.5%，基线为 48.3%。总体碰撞率为 22.8%，基线为 18.3%。平均每架成本代理为 2.599 对 1.954。关闭投影后碰撞增加至 162/360，偏航到达增至 60/120，说明旧投影存在实际回航与安全的取舍。按到达率和碰撞率保留冻结基线，质量泛化门槛仍未通过。[完整配对记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/frame-audited-holdout.moQMNS/run/comparison.json) · [权重校验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/frame-audited-holdout.moQMNS/evaluation-checkpoints-sha256.txt)

同一单卡继续训练逐机航向保护和均衡回航监督。后续开发阶段使用 91206–91208 三个新流量种子。[新一轮运行](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/a-local-protection.laeWHx/run.sh)

### 逐机航向保护训练与新留出结果

- 任务 11 完成 3,188 次更新，157.7 s 完成 188 轮，平均 PPU 利用率 74.0%。三个开发种子的对头为候选 36/36 到达、零碰撞，基线 34/36 到达、2 架碰撞。候选正负 15° 到达分别为 17/36、14/36。[训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/a-local-protection.laeWHx/run/metrics.jsonl) · [开发集逐帧结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/a-local-protection.laeWHx/development-eval/comparison.json)
- 任务 12 冻结该候选及逐机航向保护，在 13001–13005 五个独立流量种子上，以原生 0.05 s 逐帧审核。每格为到达、碰撞、越界计数，均无未关闭事件。

| 场景 | 冻结基线 | 逐机航向候选 | 候选关闭投影 |
| --- | ---: | ---: | ---: |
| 三航路交叉 | 48 / 12 / 0 | 46 / 14 / 0 | 34 / 26 / 0 |
| 四航路交叉 | 40 / 20 / 0 | 40 / 20 / 0 | 34 / 26 / 0 |
| 六航路交叉 | 30 / 30 / 0 | 28 / 32 / 0 | 26 / 34 / 0 |
| 近对头 | 58 / 2 / 0 | 58 / 2 / 0 | 18 / 42 / 0 |
| +15° | 0 / 0 / 60 | 30 / 12 / 18 | 29 / 14 / 17 |
| −15° | 0 / 0 / 60 | 31 / 16 / 13 | 30 / 18 / 12 |
| 合计 | 176 / 64 / 120 | 233 / 96 / 31 | 171 / 160 / 29 |

到达率由 48.9% 提升至 64.7%，碰撞率由 17.8% 升至 26.7%。候选、基线和关闭投影的每架成本代理分别为 5.836、1.754、5.008。对头质量持平，偏航回航增加的同时产生了额外碰撞，暂不更换基线。[新留出完整逐帧评测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/local-policy-holdout.CYrqY2/run/comparison.json)

### 分量投影、机群规模和附加指标

- 首轮联合扩大约束边并按分量执行，同权重配对中头对头和六航路均多出 2 架碰撞。固定原求解器约束边、单独切换分量判定后，对头场景均为 24/24 到达，六航路均为 16/24 到达。分量执行仍是可选项，有可解与不可解分量并存的回归验证。[联合改动](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/component-projection.28zbbq/comparison.json) · [独立对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/isolated-projection.lpo8Cs/comparison.json)
- 任务 13 已在同一张 PPU 上运行，使用实测 B737 巡航标定和 512 个空域。活动交通量每轮在 6–20 架间采样，航路可在 2–12 条间采样，空间发机间隔随机化。训练后用两类 30 架原生逐帧流量验证规模外推。[任务 13](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-scale-curriculum.TELqoL/run.sh)
- 原生逐帧评测已加入同空域每架水平与三维最小间距、控制器改写动作计数、策略加投影的每批推理时间。终止前的间距仍被统计。控制帧仅作为诊断口径，关闭投影回归的介入计数为零。[新增指标烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/metrics-smoke.vPrikm/run/comparison.json)

### 全分布长训与新阶段门槛

- 上一轮 30 架外推已结束。任务 13 与任务 11 均在对头开发流量达到 90/90、零碰撞。六航路由 38/90 降至 28/90 到达，未晋升该权重。任务 14 从任务 11 权重起步。[规模对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-scale-curriculum.TELqoL/scale-eval/comparison.json)
- 单卡任务 14 在 512 个设备常驻空域上同时以 20,000 轮和 100,000 次更新为最低停止条件，每 10,000 次更新保存策略、Critic、目标网络和优化器。混合类别按 40%/30%/30% 随机抽样，分别对应 3/4/6/8 航路交叉、近对头和初始随机 ±20° 偏航。发机间隔在本版用随机初始纵向间距近似，所有飞机在回合初始化时已生成。第 10,764 次更新累计完成 1,632 轮，三个类别累计抽样为 865/655/624，实测 18.1 次更新/s，PPU 平均 89.9%，首个周期权重已落盘。[启动与配置](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-stage1-full.TFCVbK/run.sh) · [训练指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-stage1-full.TFCVbK/run/metrics.jsonl) · [首个周期权重](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-stage1-full.TFCVbK/run/checkpoint-0010764.pt)
- 在 60 s 和 120 s 预测窗口对水平和垂直间距裕度施加平滑惩罚。相同几何代价另用于原始策略动作，避免训练只依赖动作投影。逐机航向保护直接比较同一邻居约束下的转向前后 CPA。图上度为零的飞机直接执行航向，图中飞机限制会缩小间距并进入风险带的转向。Tier-2 对所有水平重叠对检查其原高度和计划高度，在含三角形的约束分量使用 4 轮，其它分量使用 2 轮。16 项回归与全混合训练烟测通过。
- 四航路和六航路各两个开发种子出现 9 对原生逐帧碰撞。8 对在最后一个控制窗口的可用调速幅度上界下仍无法避免预计进入间距带。9 对的 Tier-2 预测在自适应加轮后均仍有冲突。该样本未观察到从安全层高连续下降进入垂向风险带的事件。[逐对碰撞轨迹](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/collision-trajectory.9MfApT)
- 对其中 5 对碰撞回看历史计划。在碰撞前约 60 s 的状态，5 对均无法通过单次可用调速幅度消除预测冲突，Tier-2 由 2 轮增至 4 轮可消除其中 2 对的预测冲突。在约 30 s 的历史状态，增加轮数可消除 3 对。以上是冻结历史状态的反事实几何结果，原生闭环仍以正式评测为准。[历史计划验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/collision-lookback.W6B3Xh)
- 任务 14 完成时先跑方案 B 六类开发流量，每 0.25 s 检查一个物理步内的间距。随后固定 12001–12005，输出原生 BlueSky 0.05 s 逐帧正式对比，包含冻结基线、本阶段与上一阶段策略的投影消融。任务 15 在同一卡上等待，只有六类总体到达率 >90% 且碰撞率 <3% 才加载 Stage 1 开始 2,000 轮方案 A。9×8 worker 使用同一混合分布、余弦学习率，每 500 轮用独立开发流量按到达率、碰撞率、成本排序，并要求近对头到达至少 58/60 且保持入场水平。烟测已验证超过 90% 到达但碰撞率超门槛时拒绝 Stage 2。[阶段门槛任务](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/a-stage2-gated.zFESRN/run.sh) · [方案 A 逐帧烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage2-audit-smoke.YV9BCo/training.log)

### 长训过程检查与发机调度

- 第一个周期权重在六类各 24 架的独立方案 B 探针中到达 102/144（70.8%）、碰撞 24/144（16.7%）、越界 18/144。对头 24/24 到达、零碰撞，六航路 14/24 到达、10 架碰撞。正负 15° 共到达 26/48。探针样本量小于正式开发集门槛。[首个周期探针](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-early-check.Mmw3H0/development.json)
- 第二个周期权重在同一小样本中开启投影为到达 95/144（66.0%）、碰撞 32/144（22.2%），关闭投影为到达 87/144（60.4%）、碰撞 40/144（27.8%）。开启投影时对头保持 24/24。四航路到达由 18/24 降至 14/24，六航路由 14/24 降至 8/24。原长训继续，同种子探针用于定位早期训练问题。[开启投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-20k-probe.spRHn6/projection.json) · [关闭投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-20k-probe.spRHn6/raw.json)
- 新增按仿真时间实际入场的发机调度。同一航路按独立的 48–80 秒随机间隔发机，等待时从物理活动、观测和冲突图移除。方案 A 在 0.05 s 物理帧释放飞机，方案 B 常规训练在 6 s 控制步边界释放。A/B 烟测均验证了 42 s 前维持初始机群、90 s 前出现新飞机。已更新尚在排队的任务 15 快照，任务 14 的原训练版本固定。[调度烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/timed-departures-smoke.6rpviW)
- 新增从阶段 checkpoint 恢复 Actor、Critic、目标网络、优化器和累计轮数/更新数的入口。续训重新采样流量并预热空回放。烟测从 8 轮、4 次更新续至累计 16 轮、8 次更新，原始 checkpoint SHA-256 校验通过。任务 14 的早期 checkpoint 保存了累计轮数和更新数，完整终局计数只在更新版 checkpoint 中保存。[续训烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/resume-smoke.UZeOx0/resumed/metrics.jsonl)

### 第三个周期权重与条件续训

- 第 31,164 次更新的同种子方案 B 开发探针覆盖 144 架飞机。开启投影到达 96/144（66.7%）、碰撞 34/144（23.6%）、越界 14/144。关闭投影到达 95/144（66.0%）、碰撞 36/144（25.0%）、越界 13/144。开启投影时对头为 24/24、六航路为 10/24。六航路关闭投影后为 14/24。该探针仍低于 Stage 2 门槛，冻结种子 12001–12005 仍按正式流程评估。[开启投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-30k-probe.98hkE1/projection.json) · [关闭投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-30k-probe.98hkE1/raw.json)
- 单卡原长训累计 7,072 轮、40,764 次更新，实测 11,346 轮/h、18.17 次更新/s。229 个 PPU-SMI 样本平均 90.61%，Torch 峰值分配 2,349 MiB。全混合延长回合并增加每控制步的更新数，因此实际端到端轮/h 与短基准不同。[长训指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-stage1-full.TFCVbK/run/metrics.jsonl) · [利用率](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-stage1-full.TFCVbK/ppu-utilization.log)
- 首版多邻居风险在一组六航路轨迹上的梯度覆盖率由 44.2% 降至 28.2%，未采用。新版保留原最危险邻居的 60/120 s 代价，并加入最多三个邻居的平滑水平侵入项和零高度指令下的梯度。在另一组同状态、同动作配对的六航路轨迹中，3,330 个冲突图内飞机观测的范数超过 1e-5 比例从 55.9% 升至 65.9%，平均范数从 0.072 增至 0.987。该结果衡量梯度，闭环安全效果由后续配对试验测量。[首版梯度](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/dense-risk-probe.sxqAm3/gradient.json) · [新版梯度](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/combined-risk-probe.dmARGH/gradient.json)
- 新增按场景分层设备回放。相同 4 批各 10 个样本的 A/B 真后端烟测均从不同实际流量比例采得交叉/对头/偏航 16/12/12。对头限定锚定和多邻居风险的联合训练烟测完成 4 次更新。18 项回归通过。[方案 A](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stratified-mixture-check.iqmpSD/a/metrics.jsonl) · [方案 B](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stratified-mixture-check.iqmpSD/b/metrics.jsonl) · [联合训练烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/head-on-anchor-check.muoryc/run/metrics.jsonl)
- 同一张 PPU 的原训练运行时完成 8,192 项回放的真实设备抽样，64 项训练批次的场景计数为 26/19/19。任务 16 已入同一张卡的队列，优先级低于已有任务 15。[PPU 回放验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage1-paired-followup.2QyXxG/ppu-replay-verified.log)
- 同一单卡排队任务 16 仅在原 Stage 1 达不到开发门槛时运行。两组各从同一个最终 checkpoint 恢复，使用相同 600 个决策周期、512 空域、40/30/30 分层回放和实际延迟发机。处理组把锚定限定在对头流量，加入多邻居惩罚，并根据梯度实测将原动作风险权重由 0.75 调至 0.15。独立开发集配对包含开启和关闭投影，候选需保住近对头且相对入场权重同时达到到达不降、碰撞不升，再接受第二批独立开发流量的 90%/3% 门槛。达标后执行原生 0.05 s 正式留出和方案 A 高保真微调。任务 14 的快照、任务 15 的入场条件与所有原始权重保持固定。[条件续训](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage1-paired-followup.2QyXxG/run.sh)

### 第四个周期权重和回航诊断

- 任务 14 已完成 8,901 轮、50,364 次更新，累计 11,563 轮/h、18.17 次更新/s，283 个 PPU-SMI 样本平均 90.77%。第 50,364 次更新的权重已保存。任务 15 与任务 16 均继续等待同一张卡。[长训指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-stage1-full.TFCVbK/run/metrics.jsonl) · [周期权重](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-stage1-full.TFCVbK/run/checkpoint-0050364.pt)
- 第 40,764 次更新的同种子开发探针开启投影后为到达 92/144（63.9%）、碰撞 42/144（29.2%），对头仍到达 24/24。关闭投影后为到达 83/144（57.6%）、碰撞 54/144（37.5%），对头到达 20/24。该检查点低于开发门槛。[开启投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-40k-probe.GLX6y8/projection.json) · [关闭投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-40k-probe.GLX6y8/raw.json)
- 对第 31,164 次更新的权重，在独立种子 41707–41708 使用原生 BlueSky 0.05 s 逐帧验证四、六航路。四航路在全空域投影、分量投影及关闭投影下均为到达 14/24、碰撞 10/24。六航路两种投影均为到达 16/24、碰撞 8/24，关闭投影为到达 10/24、碰撞 14/24。结合早先方案 B 探针，当前投影的闭环收益随流量变化，未切换分量投影。[原生逐帧配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-crossing-projection-probe.FwMOQM/pair.json)
- 对独立的正负 15° 各 48 架追踪需要转向超过 2° 且 LNAV 关闭的控制时刻。孤立飞机 258/258 次向目标方向转向。连通飞机中，在固定该时刻的执行速度和高度命令时有 1,493 次目标转向通过局部安全检查，其中 1 次实际执行相同方向的转向，另有 10 次受局部保护限制。计数按飞机控制时刻统计。两类偏航分别到达 28/48、29/48，均有 12 架碰撞。[回航机会分解](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/yaw-opportunity-probe.ZnDbXW/result.json)
- 对照组保留原回航监督。任务 16 处理组增加图内安全方向的回航监督，相对导航损失权重 0.35。训练标签在保持速度和当前所选高度的条件下通过同一个逐机 CPA 保护器，危险方向不参与监督。回归确认允许方向有梯度且缩小冲突间距方向的新增梯度为零，组合训练完成 12 次更新。任务 16 排队期间已封存旧快照并校验新版本。[训练烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/safe-turn-training-smoke.kgAfM8/run/metrics.jsonl) · [快照变更](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage1-paired-followup.2QyXxG/before-safe-navigation)

### 五万次更新与提前观测诊断

- 第 50,364 次更新的同种子方案 B 开发探针开启投影后到达 96/144（66.7%）、碰撞 40/144（27.8%），关闭投影后到达 94/144（65.3%）、碰撞 42/144（29.2%）。开启投影的四、六航路分别到达 16/24 和 12/24，对头两组均到达 24/24。[开启投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-50k-probe.wAwVNR/projection.json) · [关闭投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-50k-probe.wAwVNR/raw.json)
- 使用该权重，在种子 49311–49312 复跑四、六航路原生 0.05 s 接触轨迹，记录 9 对碰撞，其中 4 对从安全高度间距进入相撞带且带下降速度。最后控制窗口有 8/9 对在固定水平计划下无法通过持层或单次 ±300 ft 的全分量选层避免预测冲突。考虑原生爬升加速度后结果仍为 8/9。当前 K=4 标记了 5/9 对冲突，两种分层搜索均未在该 5 对上找到可行计划。碰撞前约 60 s，20 km 冲突观测看到 2/9 对，35 km 看到 7/9 对。约 90 s 前有 2 对在固定水平计划下存在离散分层方案，20/35/50 km 分别看到 0/1/2 对。离散诊断固定了水平运动，并按分量约束回放高度，结论限于该历史状态和单步动作集合。[原生接触和历史状态汇总](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-native-inertia-oracle.cAUInz/aggregate.json)
- 方案 B 独立配对开发流量共 288 架。对同一冻结权重，观测与投影图一同从 20 km 增至 35 km 时，六航路碰撞从 28/48 降至 20/48，三航路从 4/48 增至 10/48，近对头从 4/48 降至 2/48。总体碰撞同为 68/288，到达从 192/288 降至 189/288。保持 20 km 作为当前训练与验收配置，继续做原生对照和仅改变策略观测的消融。[20 km](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-observation-range.U53FGg/20km.json) · [35 km](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-observation-range.U53FGg/35km.json)
- 原生 0.05 s 同种子下，35 km 对六航路把碰撞由 14/24 降到 10/24，对四航路由 4/24 升到 8/24，近对头两组均 24/24 到达且无碰撞。方案 B 的固定投影图和固定策略观测两种消融，任一侧单独换到 35 km 时，六航路均为 26/48 架碰撞；双侧同时扩大为 20/48，20 km 原设置为 28/48。关闭投影时 20/35 km 的六航路均为 26/48，整个六类的碰撞由 76/288 降到 68/288。该结果用于区分策略与投影的影响，尚不足以全局替换当前设置。[原生逐帧对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-observation-range.umdhmD/comparison.json) · [方案 B 消融](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/range-projection-ablation.AoHRPY)
- 第 61,164 次更新开启投影后到达 110/144（76.4%）、碰撞 28/144（19.4%）；关闭投影后到达 108/144（75.0%）、碰撞 30/144（20.8%）。开启投影三航路 24/24、四航路 20/24、六航路 10/24、近对头 24/24，正负偏航各到达 16/24。单卡已累积 12,914 轮、70,764 次更新，392 个 PPU 样本平均 91.08%。上游阶段仍在训练，后续条件队列状态已核验。[开启投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-60k-probe.Nu9VTX/projection.json) · [关闭投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-60k-probe.Nu9VTX/raw.json) · [实时日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-stage1-full.TFCVbK/run/metrics.jsonl)
- 在相同 288 架开发流量上比较 6 个周期权重，31,164 次更新的权重到达 211/288（73.3%）、碰撞 44/288（15.3%），61,164 次更新的权重到达 209/288（72.6%）、碰撞 56/288（19.4%）。两者对头均 48/48、六航路均 26/48；前者正负偏航为 32/48、31/48，后者为 35/48、34/48。到达、碰撞、对头和正负偏航的帕累托前沿另含 40,764 次更新权重。[开发集周期权重面板](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-checkpoint-panel.V8TooL/comparison.json)
- 为任务 16 增加独立的续训入场权重比较。若原 Stage 1 未达标，任务在种子 45613 的新开发流量上比较原最终权重与前沿前两名，只有到达与碰撞均不退步、对头和两类偏航均不退步时，才从同一候选对照续训两组。原阶段评测来源和校验保留。真实权重烟测验证配对评估和并列时保留原权重，当前 20 项项目回归已通过。[入场选择烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage1-resume-selection-smoke.A2nOgm/smoke/selection.json) · [待执行版本](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage1-paired-followup.2QyXxG/run.sh)

### 七万次更新与垂向时序奖励

- 第 70,764 次更新在同种子开发小样本上开启投影到达 102/144（70.8%）、碰撞 40/144（27.8%），关闭投影到达 88/144（61.1%）、碰撞 54/144（37.5%）。开启投影时三、四、六航路到达分别为 20/24、12/24、12/24，近对头两组均到达 24/24。原长训达到 15,521 轮、83,964 次更新时持续约 12,099 轮/h、18.18 次更新/s。[开启投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-70k-probe.uNNRzY/projection.json) · [关闭投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-70k-probe.uNNRzY/raw.json) · [实时日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-stage1-full.TFCVbK/run/metrics.jsonl)
- 原生 0.05 s 的四、六航路真实接触轨迹包含 9 对碰撞。固定当时速度、航向与高度指令，用 60/120 s 窗口检查最后一次控制决策，原水平 CPA 风险判据识别 5/9，对整个水平重叠区间求最小垂直间隔后识别 7/9。碰撞前约 30 s 从 3/9 提高到 4/9，约 60 s 时两者覆盖相同。以上是固定历史状态的几何诊断。[风险覆盖](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/temporal-overlap-risk-audit.pW6Kpn/coverage.json)
- 新增可选的时序风险奖励。水平重叠区间内分别检查进入、退出和两个目标高度捕获时刻，使垂直间距暂时侵入也能影响 60/120 s 损失。方案 B 实际烟测完成 12 次训练更新。PPU 反向试验测得典型漏判输入的风险由 0.101 提高到 0.975，动作梯度有限且非零。项目 21 项回归通过。[训练烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/temporal-overlap-risk-smoke.F9oKil/run/metrics.jsonl) · [PPU 前向与梯度](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/temporal-overlap-risk-smoke.F9oKil/ppu-overlap-smoke.log)
- 同卡任务 16 增加第三组，使用相同入场权重、种子、600 决策周期与混合抽样，在组合处理组基础上单独启用时序风险。对照、组合、时序三组均用同一开发流量做投影开关对比。若候选达到独立开发门槛，方案 A 将继承选中组的前瞻与回航训练损失。新增交叉、对头、偏航三类训练终局计数，A/B 烟测均与总计逐项对齐。原长训继续使用提交时的配置。[条件续训](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage1-paired-followup.2QyXxG/run.sh) · [分类计数烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/mixed-outcomes-check.md6yYr)

### 对头遗忘复核与续训锚点

- 第 80,364 次更新在同种子探针中开启投影总到达 97/144（67.4%）、碰撞 44/144（30.6%）。对头开启投影到达 20/24，关闭后到达 0/24。第 91,164 次更新在相同流量中开启投影到达 111/144（77.1%）、碰撞 30/144（20.8%），六航路到达 16/24；关闭投影到达 102/144、碰撞 40/144。最新两种模式的对头均恢复 24/24。单卡已达 17,905 轮、95,964 次更新。[80k 权重探针](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-80k-probe.rQ5f7Z) · [91k 权重探针](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-90k-probe.tVl2KC) · [实时训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-stage1-full.TFCVbK/run/metrics.jsonl)
- 在独立的 192 架近对头方案 B 流量上，61,164 次更新的权重关闭投影 192/192 到达，开启 190/192；80,364 次更新的权重分别为 20/192、168/192；冻结锚点为 44/192、180/192。两周期权重在冲突图内的原动作调速比例从 5.05% 升至 25.12%，高度动作比例从 1.13% 降到 0.18%。比例为各自闭环轨迹的飞机控制时刻统计。[独立对头流量](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/head-on-retention-panel.j4rDV6)
- 原生 0.05 s 独立种子 61931–61935 上，61,164 次更新的权重关闭投影到达 56/60、碰撞 4/60，开启后到达 60/60、零碰撞。原冻结锚点对应到达为 14/60、56/60。已把 61,164 次更新的权重作为任务 16 两个处理组的仅对头锚点。对照组与正式评测基线继续使用原冻结权重，输入与验证证据均做了 SHA-256 校验。[原生锚点配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-head-on-anchor.OWyeHh/comparison.json) · [条件续训源](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage1-paired-followup.2QyXxG/run.sh)
- 扩展至八个权重的 288 架开发面板中，91,164 次更新权重到达 226/288（78.5%）、碰撞 56/288（19.4%）、对头到达 44/48。31,164 次更新权重仍是碰撞最低者，为 44/288（15.3%），对头到达 48/48。任务 16 的独立复筛改为同时保留这两个方向的代表候选。真实权重烟测确认两者均进入复筛，未改变开发门槛。[扩展权重面板](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage1-late-checkpoint-panel.eXCpmP/comparison.json) · [复筛烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/late-resume-selection-smoke.5M4VJp/run/selection.json)
- 原长训达到 100,764 次更新与 18,866 个完成回合，实测 12,255 轮/h、18.18 次更新/s，562 个 PPU 样本平均 91.05%。当前等待达到 20,000 轮的同一训练任务仍在运行，正式评测和后续条件任务仍按原门槛执行。[长训实时数据](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-stage1-full.TFCVbK/run/metrics.jsonl)

### Stage 1 训练完成与完整开发集

原长训完成 20,047 轮、106,764 次更新，训练用时 5,872 s。端到端吞吐为 12,290 轮/h、18.18 次更新/s，Torch 峰值分配 2,349 MiB。流量抽样交叉 8,301、对头 6,085、偏航 6,173。最终 checkpoint 已落盘并进入评测。[训练指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-stage1-full.TFCVbK/run/metrics.jsonl) · [最终权重](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-stage1-full.TFCVbK/run/checkpoint.pt)

方案 B 开发集固定种子基值 43209，每类 32 空域、384 架，按 0.25 s 逐物理帧检查。

| 场景 | 到达 | 碰撞 | 越界 |
| --- | ---: | ---: | ---: |
| 三航路 | 322 | 62 | 0 |
| 四航路 | 280 | 104 | 0 |
| 六航路 | 237 | 147 | 0 |
| 对头 | 374 | 10 | 0 |
| +15° | 302 | 78 | 4 |
| −15° | 296 | 87 | 1 |
| 合计 | 1,811 | 488 | 5 |

总体到达率 78.60%、碰撞率 21.18%，无超时，Stage 2 开发门槛未通过。原生 0.05 s 的 12001–12005 固定留出按原协议继续，包括冻结基线、本阶段权重及投影消融。[开发门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-stage1-full.TFCVbK/development-gate.json) · [正式留出](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-stage1-full.TFCVbK/formal-stage1/comparison.json)

### 提前错峰与分层示教可行性

- 对 12 个独立四航路、六航路与对头空域做集中式轨迹搜索。每架机有 9 个速度和高度候选，以合法的每控制步幅度逐步跟踪目标。先用校准 B737 积分器生成 1 s 轨迹弦，再在方案 B 的 0.25 s 每物理帧闭环复验。对照到达 48/144、碰撞 96/144，提前规划到达 144/144、零碰撞。同一初始位置、目标点和计划移植到原生 0.05 s，仍到达 144/144、零碰撞。该试验使用整组已知航线作集中搜索，测量可行性并提供示教数据来源。[方案 B 规划](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/early-plan-feasibility.7igWjM) · [原生逐帧同布局复跑](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/early-plan-native-replay.0xq0HC/aggregate.json)
- 将速度和高度指令推迟到各机标称水平重叠前约 120 s，在六类各 48 架的方案 B 流量中到达 254/288、碰撞 34/288。六航路 34/48、负偏航 28/48，另外四类均 48/48。142 个首次非零调速或分层指令中，53 个在 20 km 冲突图里仍是度零，说明现有输入对部分提前动作缺少邻机信息。可执行单轴加宽调速候选使同一负偏航从 28/48 改善到 48/48，六航路仍为 34/48。[120 s 对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/early-plan-120s-feasibility.rvRQLi) · [单轴试验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/early-plan-single-axis.7m4Nyi)

### Stage 1 原生正式留出

固定种子 12001–12005，每类 60 架，原生 BlueSky 每 0.05 s 逐帧检查。每格为到达、碰撞、越界，各组均无未关闭终局。开发集与正式评测的 Stage 1 权重 SHA-256 一致。原长训 589 个 PPU 利用率样本的均值为 90.90%。

| 场景 | 冻结基线 | Stage 1 开投影 | Stage 1 关投影 | 上一候选关投影 |
| --- | ---: | ---: | ---: | ---: |
| 三航路 | 42 / 18 / 0 | 46 / 14 / 0 | 50 / 10 / 0 | 28 / 32 / 0 |
| 四航路 | 28 / 32 / 0 | 42 / 18 / 0 | 38 / 22 / 0 | 26 / 34 / 0 |
| 六航路 | 26 / 34 / 0 | 34 / 26 / 0 | 28 / 32 / 0 | 26 / 34 / 0 |
| 对头 | 56 / 4 / 0 | 56 / 4 / 0 | 48 / 12 / 0 | 22 / 38 / 0 |
| +15° | 0 / 0 / 60 | 51 / 8 / 1 | 45 / 14 / 1 | 27 / 10 / 23 |
| −15° | 0 / 0 / 60 | 48 / 12 / 0 | 46 / 14 / 0 | 26 / 10 / 24 |
| 合计 | 152 / 88 / 120 | 277 / 82 / 1 | 255 / 104 / 1 | 155 / 158 / 47 |

Stage 1 开投影达到 76.94% 到达、22.78% 碰撞。投影减少 22 架碰撞并改善两类偏航到达。关投影相对上一候选关投影增加 100 架到达、减少 54 架碰撞，五种子单侧精确配对检验 p=0.03125。冻结基线、Stage 1 开投影、Stage 1 关投影的平均每机动作代价为 1.80、16.10、16.10。总体到达、碰撞、两类偏航、对头均未满足用户门槛。任务 15 记录开发门槛未通过，任务 16 已开始在同一张 PPU 上做入场权重的独立复筛。[正式逐帧数据](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b-stage1-full.TFCVbK/formal-stage1/comparison.json) · [验收判定](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/a-stage2-gated.zFESRN/stage1-acceptance.json) · [续训复筛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage1-paired-followup.2QyXxG/resume-selection.log)

### 单轴示教的原生筛选与混合训练

- 扩大允许分步到达的目标层后，120 s 单轴方案在方案 B 的六航路和负偏航各达到 48/48、零碰撞，分别产生 77、269 个空域控制决策样本。原生 0.05 s 同布局复验为六航路 48/48、零碰撞，负偏航 36/48、12 架碰撞。按完整空域终局筛选后保留六航路 4 个空域 77 条、负偏航 1 个空域 57 条，另存 SHA-256 清单。[方案 B 示教采集](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/early-plan-teacher-smoke.zJh3lI) · [原生示教复验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/early-teacher-native-replay.D7tG5I/aggregate.json) · [筛选清单](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-verified-teacher.N7FYXP/manifest.json)
- 单独使用两类方案 B 师范数据对最终 Stage 1 权重更新 64 次，训练样本损失从 3.408 降至 1.905。独立六类 288 架流量开启投影时到达率 66.0%、碰撞率 34.0%，同种子入场权重为 77.8%、22.2%。关投影为 59.0%、41.0%，入场权重为 72.2%、27.8%。该示教权重未晋升。[示教训练损失](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/early-teacher-policy-smoke.eVcxsv/model/training.json) · [独立配对结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/early-teacher-policy-smoke.eVcxsv)
- 新增原生筛选示教作为 TD3 的可选辅助损失，每个示教文件等量抽样。调速和分层标签只在冲突图连通时使用，自主回航标签允许在孤立时使用，主训练仍使用全场景混合回放。真实混合仿真烟测以 0.05 辅助权重完成 12 次更新，22 项项目回归通过。当前任务 16 的三个已冻结对照组继续运行，辅助损失的闭环增益待独立对照。[TD3 辅助损失烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/auxiliary-teacher-rl-smoke.5KeskP/run/metrics.jsonl)

### 测地线回航对齐与续训复筛

- 原生示教复跑原先用平面图特征还原到目标点的航向，方案 B 示教器用测地线方位。统一测地线命令后，原生 0.05 s 的同一组负偏航从到达 36/48、碰撞 12/48 恢复为 48/48、零碰撞。方案 B 用相同动作规则在 0.25 s 与 0.05 s 均为 48/48、零碰撞。修复应用于示教复跑工具，原 Stage 1 正式数据和已运行的续训保持原记录。[原生同布局复跑](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-geodesic-teacher.4sFoFN/aggregate.json) · [方案 B 步长配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/torch-teacher-dt-pair.iqoioe)
- 任务 16 在独立种子 45613 共 1,152 架配对开发流量复筛。最终 Stage 1 权重到达 78.21%、碰撞 21.70%、对头 192/192，其两个周期候选分别为 75.78% / 23.09%、67.10% / 23.18%。以最终权重开始三组续训。对照组已完成 600 决策、7,164 次新增更新，实测 16.46 更新/s。[复筛判定](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage1-paired-followup.2QyXxG/resume-selection/selection.json) · [对照训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage1-paired-followup.2QyXxG/control/metrics.jsonl)
- 示教分支使用独立采样随机流与确定性意图前向，使辅助学习保持梯度且不消耗在线探索随机数。23 项回归通过。正在用未用过的六类种子扩展原生复验素材，每个空域的原生逐帧报告与方案 B 计划按内容校验，筛除带碰撞、越界或超时的空域。[六类原生筛选](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/six-family-verified-teacher.43HO5n)
- 新六类示教完成原生 0.05 s 逐帧复验。24 个空域 288/288 到达、零碰撞，保留 813 条空域决策。三航路、四航路、六航路、对头、正负偏航分别为 55、64、60、88、278、268 条。实际调速与分层标签在当前冲突图连通时分别有 172 和 324 次，769 次航向标签保留孤立时的自主回航。改变计划文件一个字节后筛选器拒绝生成资料。[六类素材清单](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/six-family-verified-teacher.43HO5n/verified/manifest.json) · [哈希拒绝检查](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/teacher-integrity-check.mLB6MA/rejection.log)
- 短程同源权重对照运行 120 决策、117 次新增更新，处理组仅加入 0.05 辅助示教。第一组 144 架独立开发流量关投影到达由 90 升至 98、碰撞由 54 降至 46，开启投影到达由 102 降至 98、碰撞由 42 升至 46。第二组 288 架独立流量关投影到达由 188 升至 198、碰撞由 100 降至 90，开启投影到达由 199 升至 210、碰撞由 89 降至 78。第二组开启投影的平均每机动作代价从 23.75 增至 26.24，显示安全收益与成本仍需共同检查。[首组配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/verified-teacher-paired-pilot.SpeKsO) · [第二组独立确认](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/teacher-pilot-confirmation.pDXxUZ)
- 任务 17 已排进任务 16 所在的同一张单卡，仅在任务 16 完整跑完且仍未通过开发门槛时启动。对照组与示教组从任务 16 独立开发筛中的同一权重恢复，各运行 450 决策，保留混合放量和原生筛选样本，使用新的两组开发种子。入选顺序为对头保持、到达、碰撞和动作代价，Stage 2 的 90%/3% 入场门槛不变。任务 16 已完成对照和组合两组各 7,164 次更新，时序组正在运行。[同卡任务 17](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/verified-teacher-stage1-followup.sZ7kw5/submit.log) · [任务 16 日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage1-paired-followup.2QyXxG/temporal/metrics.jsonl)

### 随权重保存观测配置与扩大设备回放

- 在同一 Stage 1 权重、288 架独立开发配对流量上，仅改目标方位节点特征为测地线后，开启投影到达由 222 增至 227、碰撞由 66 降至 60。关闭投影则从 218 到达、70 碰撞变为 216 到达、72 碰撞。该单因素结果用于排查回航表现，当前队列沿用提交时的观测配置。[目标航向特征配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/policy-geodesic-observation.Td7VBH)
- 在完全相同的 813 条原生已验证师范决策上，调速动作可见邻机次数从 20 km 的 172/249 增至 35 km 的 243/249 和 50 km 的 248/249，分层从 324/422 增至 393/422 和 419/422。方案 B 的指令、决策编号和终局均逐条配对。已把 35 km 输入另存为同一师范动作的观测对照素材。[观测范围对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/teacher-range-observability.d1rsJ4) · [35 km 师范清单](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-teacher-35km.TBQP5v/manifest.json)
- 增加权重观测配置，恢复、方案 B 开发和原生 0.05 s 评测默认读取训练时的观测距离与目标方位方式。原权重默认 20 km 平面方位。更改评测配置会标成诊断，训练入口拒绝不匹配的师范图。实际保存、继承、原生加载与恢复烟测通过。[观测配置闭环](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/observation-profile-roundtrip.ShAftS)
- 在 32 空域 600 步的独立混合策略轨迹上，16 控制步回放窗口缺少到达样本的窗口比例，交叉、对头、偏航分别为 46.6%、40.1%、48.9%。128 步窗口分别为 1.0%、0.0%、2.0%。交叉和偏航的碰撞样本缺失分别由 66.1% 和 63.8% 降至 19.2% 和 10.7%。这是按场景配额估计的采样覆盖，用于评估记忆窗口。[回放窗口诊断](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/replay-window-audit.juCDf5/report.json)
- 使用实际 512 空域、20 架张量形状在同卡短测。8,192 与 65,536 项完整图回放分别占 503 和 4,005 MiB，64 样本分层抽样均维持 26/19/19，短测分别为 0.383 ms 和 0.355 ms/批。任务 17 在排队状态更新为两组共用 65,536 项回放，保持唯一对照变量为辅助示教。实际端到端吞吐由任务进入训练后记录。[单卡回放测速](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/ppu-replay-window-cost.LKyQEb/result.log) · [任务 17 快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/verified-teacher-stage1-followup.sZ7kw5/run.sh)
- 配对续训从同一个 Stage 1 权重恢复，各组运行 120 个决策、117 次新增更新，分别采用 20 和 35 km 的一致在线观测与对应师范图。独立 288 架开发流量开启投影时，35 km 到达 230、碰撞 58、平均动作代价 33.18，20 km 为 196、92、23.95。关闭投影时 35 km 到达 189、碰撞 98，20 km 为 196、92。开启投影的对头从 48/48 降为 46/48。已启动交换投影范围与原生迁移对照。[观测配置短程配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/teacher-observation-profile-pair.iCZij5)
- 在上述相同流量交叉交换投影图，20 km 策略搭配 35 km 投影到达 204、碰撞 84，35 km 策略搭配 20 km 投影到达 216、碰撞 70。两侧扩大均带来部分保护收益。原生 0.05 s 独立种子 78701–78703 使用各自训练配置，负偏航碰撞从 18/36 降至 8/36，近对头均到达 36/36，六航路碰撞从 14/36 增至 16/36。扩大观测保留为诊断方向，现有续训使用原观测配置。[交换投影范围](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/trained-profile-shield-cross.HIgV3W) · [原生配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/trained-profile-native-pair.Wvn2RE/run/comparison.json)

### 任务 16 混合续训的配对复筛

入场、对照、组合和时序四份权重在同一 864 架开发流量上检查。

| 权重 | 开投影到达 | 开投影碰撞 | 对头到达 | 关投影到达 | 关投影碰撞 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 入场 | 672 | 191 | 142/144 | 634 | 226 |
| 对照 | 627 | 236 | 134/144 | 578 | 286 |
| 组合 | 618 | 246 | 142/144 | 565 | 299 |
| 时序 | 594 | 270 | 136/144 | 583 | 281 |

每组完成 600 决策、7,164 次新增更新。对照、组合和时序的实测更新速度为 16.46、16.21、15.70 次/s。筛选保留入场权重，送入另一批完整开发门槛复核。[全部投影配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage1-paired-followup.2QyXxG) · [筛选判定](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage1-paired-followup.2QyXxG/development-selection.json)

完整开发门槛 2,304 架中到达 1,778（77.17%）、碰撞 521（22.61%）、越界 4、超时 1。六航路到达 222/384，近对头 376/384。Stage 2 继续等待开发达标。任务 16 在训练和评测合计区间的 228 个 PPU 样本均值为 71.68%。[独立开发门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage1-paired-followup.2QyXxG/development-gate.json)

任务 17 已在同一单卡接力，实际配置为 512 并行空域和 65,536 项设备回放。对照组前 100 个决策已新增 1,164 次更新，实测 16.14 次更新/s、Torch 峰值分配 5,889 MiB。后续同源辅助示教组接续，闭环开发门槛仍为到达率 >90%、碰撞率 <3%。[任务 17 配置](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/verified-teacher-stage1-followup.sZ7kw5/control/config.json) · [实时训练指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/verified-teacher-stage1-followup.sZ7kw5/control/metrics.jsonl)

任务 17 对照组完成 450 个决策、5,364 次新增更新，耗时 326.7s，实测 16.42 次更新/s 和 9,807 轮/h。示教组启动时因输出目录与示教输入目录同名中止。已修正运行器，并对完整对照组的权重、配置、日志分别校验 SHA-256。在同一张 PPU 上启动任务 18，接续原配置的示教组及同种子开发配对评测。[对照组终点](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/verified-teacher-stage1-followup.sZ7kw5/control/metrics.jsonl) · [失败日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/verified-teacher-stage1-followup.sZ7kw5/teacher.log) · [恢复提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/verified-teacher-stage1-followup.sZ7kw5/recovery-submit.log)

任务 18 的示教组也完成 450 个决策、5,364 次新增更新，耗时 467.2s，实测 11.48 次更新/s 和 6,595 轮/h。配置逐项核对后，48 项训练参数与对照组相同，只有示教数据、示教辅助权重和输出路径有差异。进入相同开发种子的投影开关配对。[示教组实际训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/verified-teacher-stage1-followup.sZ7kw5/verified-teacher/metrics.jsonl)

在相同的 864 架开发流量上，选中使用 65,536 项回放的常规续训对照组。该组保留原有奖励、锚定和投影配置。开投影时六类到达数均高于入场权重，推进到独立开发门槛。示教辅助权重 0.05 的配对运行落后，主要损失出现在偏航。对照与示教训练期 PPU 利用率分别为 83.15%（33 个采样）和 66.51%（47 个采样）。[配对筛选](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/verified-teacher-stage1-followup.sZ7kw5/development-selection.json) · [对照利用率](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/verified-teacher-stage1-followup.sZ7kw5/ppu-utilization.log) · [示教利用率](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/verified-teacher-stage1-followup.sZ7kw5/ppu-recovery-utilization.log)

| 权重 | 开投影到达 | 开投影碰撞 | 对头到达 | 关投影到达 | 关投影碰撞 | 开投影动作代价 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 入场 | 655 | 204 | 142/144 | 627 | 234 | 17.03 |
| 对照 | 685 | 179 | 144/144 | 660 | 204 | 20.06 |
| 示教 | 614 | 250 | 138/144 | 576 | 288 | 15.91 |

独立开发种子 56377 共 2,304 架，对照权重到达 1,790（77.69%）、碰撞 514（22.31%），无越界或超时。六类到达/碰撞依次为三航路 316/68、四航路 288/96、六航路 226/158、近对头 384/0、正偏航 296/88、负偏航 280/104，每类 384 架。Stage 2 仍未过门槛。原生正式留出结果继续使用已完成的原 Stage 1 配对记录。[独立门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/verified-teacher-stage1-followup.sZ7kw5/development-gate.json)

### 终局转移回放验证

增加可选的 `--terminal-replay`，在每个场景组的评论家批次中分别为到达、碰撞预留约八分之一。缺例时保留常规抽样，策略梯度仍用相同种子和原有 40/30/30 分层回放。同一转移同时出现到达和碰撞时按碰撞登记。训练日志分别统计两者实际采样的事件数。

同一张 PPU 上用 65,536 项合成回放测试每组 100 个批次。评论家碰撞样本从 38/6,400 增至 725/6,400，到达从 296/6,400 增至 933/6,400。两组策略批次的到达、碰撞计数均为 147/3,200 和 20/3,200，场景配额均为 26/19/19。两次采样和计数分别耗时 1.47ms/批和 1.11ms/批。该并行短测用于验证采样行为，泛化和安全率由后续闭环开发评测判定。Geo-ADMM 单元测试 25 项通过。[PPU 采样验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/terminal-replay-smoke.d4YDOg/sampler-ppu.log)

实际 Plan B 全混合流量的小模型短测每组完成 10 次更新。该种子未出现到达和碰撞，两组的损失、样本配额、流量和终局计数一致，验证缺例时按原批次学习。单卡任务 19 以独立源码快照排在任务 18 后。前序独立开发门槛若未通过，两组从同一入选权重、种子和 65,536 项回放出发，各续训 450 个决策。两组仅切换终局回放，在另一组开发种子上比较投影开关，在独立的六类开发集上复核 Stage 2 门槛。[常规短测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/terminal-replay-smoke.d4YDOg/small-control/metrics.jsonl) · [终局回放短测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/terminal-replay-smoke.d4YDOg/small-terminal/metrics.jsonl) · [任务 19 快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/terminal-replay-stage1.qQgjq0/run.sh) · [提交状态](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/terminal-replay-stage1.qQgjq0/submit.log)

在新种子 97547 上用冻结策略跑 32 个真实混合空域、600 个决策。将 16 个决策的回放窗口扩至 128 个决策后，可找到碰撞样本的窗口比例在交叉场景从 40.4% 升至 87.8%，在偏航从 49.4% 升至 85.5%。128 窗口内，交叉与偏航批次合计平均预留 4.34 个碰撞样本，含约 3.62 条独立转移，常规抽样平均 0.31 个。该流量中的近对头碰撞为零，对头碰撞配额回落到常规抽样。此诊断未测量策略更新后的安全率。[实际轨迹窗口统计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/terminal-replay-smoke.d4YDOg/real-events.json)

任务 19 已从任务 18 入选的对照权重实跑，起点为 112,128 次更新。前 50 个决策达到 112,692 次更新，实测 16.03 次更新/s。[单卡常规组实时日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/terminal-replay-stage1.qQgjq0/control/metrics.jsonl)

同种子各跑 100 个决策后，评论家碰撞回放从常规组的 227/74,496 增至终局组的 2,834/74,496。两组场景份额均为 30,264/22,116/22,116，实际更新速度分别为 16.31 与 16.10 次/s。此时尚未发生到达。此项确认真实单卡训练采到了预期的稀有事件，闭环结果需等完整配对。[终局组实时日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/terminal-replay-stage1.qQgjq0/terminal/metrics.jsonl)

两组各完成 450 个决策、5,364 次新增更新。常规/终局组分别达到 9,490/9,558 轮/h、16.48/16.31 次更新/s。评论家批次累计 343,296 项，碰撞样本从 2,429 增至 28,408，到达样本从 10,164 增至 33,583。策略批次分别采到 1,223 和 1,201 次含碰撞转移。六场景闭环筛选已开始。[常规组实测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/terminal-replay-stage1.qQgjq0/control/metrics.jsonl) · [终局组实测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/terminal-replay-stage1.qQgjq0/terminal/metrics.jsonl)

任务 19 的 864 架配对筛选继续保留入场权重。常规组和终局组训练期间 PPU 采样均值分别为 83.64% 和 80.29%，硬件显存峰值均为 6,314 MiB。[筛选结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/terminal-replay-stage1.qQgjq0/development-selection.json) · [实测利用率](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/terminal-replay-stage1.qQgjq0/ppu-utilization.log)

| 权重 | 开投影到达 | 开投影碰撞 | 对头到达 | 关投影到达 | 关投影碰撞 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 入场 | 676 | 188 | 144/144 | 643 | 220 |
| 常规 | 674 | 190 | 144/144 | 638 | 226 |
| 终局回放 | 670 | 190 | 140/144 | 631 | 230 |

独立开发种子 57601 的 2,304 架中，入场权重到达 1,843（79.99%）、碰撞 461（20.01%），近对头 384/384。六航路 243/384 到达、141 碰撞，正负偏航分别 308/384 和 298/384 到达。Stage 2 未达到入场门槛。[独立开发评测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/terminal-replay-stage1.qQgjq0/development-gate.json)

### 同向偏航的全分布配对

原训练偏航逐机独立随机取正负号，六场景中的 ±15° 工况为整个空域同向偏航。新增可选的 `--correlated-yaw`，在偏航子类约一半的空域用共享符号，保留原幅度和其他混合流量。4,096 个同种子场景验证其余所有路由、活跃位和发机时间逐项一致。Stage 1 单卡任务 20 从同一入场权重跑两组各 450 个决策，闭环筛选和放行使用两组新开发种子。若偏航组入选且通过开发门槛，Stage 2 继承该分布。当前 28 项回归通过。[独立快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/correlated-yaw-stage1.d8ZdZa/run.sh) · [任务 20 提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/correlated-yaw-stage1.d8ZdZa/submit.log)

任务 20 已在原单卡开始对照组，前 150 个决策新增 1,764 次更新，实测 16.30 次更新/s。原生方案 A 小跑启用同向偏航抽样，覆盖一个偏航空域并完成 4 次更新。[实时对照组](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/correlated-yaw-stage1.d8ZdZa/control/metrics.jsonl) · [原生兼容性短测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/correlated-yaw-smoke.Ssljo2/run-yaw/metrics.jsonl)

12 架匹配条件下，对恒定初始方向和速度的标称轨迹做圆柱间隔审计。六航路混合训练、固定开发布局平均每空域分别有 7.67 和 7.47 对标称冲突，最大邻机数达到 3 的场景分别为 55.6% 和 61.8%。偏航训练的标称冲突对均值从逐机符号的 2.56 降到混合共享符号的 1.68，正负 15° 固定开发场景分别为 0.24 和 0.27。该审计使用直线标称轨迹，只描述初始化和发机时序的差异。[匹配规模的分布审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/nominal-distribution-audit.f4j5dp/report.json)

固定航路、发机和机速后，全部偏航空域共享符号、共享随机偏航角和统一 ±15° 的标称冲突对均值分别为 1.01、1.43 和 0.40。当前控制变量仅在 50% 偏航空域共享符号，以保留原始幅度覆盖。[偏航角反事实](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/nominal-distribution-audit.f4j5dp/yaw-angle-counterfactual.json)

两组各完成 450 个决策、5,364 次新增更新，常规/同向偏航组实测 9,757/9,790 轮/h、16.43/16.48 次更新/s。两组配置除实验输出与偏航符号开关外逐项相同。[常规组](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/correlated-yaw-stage1.d8ZdZa/control/metrics.jsonl) · [同向偏航组](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/correlated-yaw-stage1.d8ZdZa/correlated-yaw/metrics.jsonl)

配对开发集 864 架，入场/常规/同向偏航开投影分别到达 675/693/672，碰撞 189/170/182，近对头 144/142/142。同向偏航关投影到达 646、碰撞 206，入场关投影为 636/228。因两个新权重的近对头都低于入场权重，保留入场权重。独立六类复核中到达 1,841/2,304（79.90%）、碰撞 463/2,304（20.10%），近对头 384/384，Stage 2 仍未放行。[配对筛选](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/correlated-yaw-stage1.d8ZdZa/development-selection.json) · [独立门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/correlated-yaw-stage1.d8ZdZa/development-gate.json)

### 入选权重的原生接触诊断

在原生 BlueSky 0.05 s，复用四、六航路的两个开发种子 49311–49312，112,128 次更新的权重在 48 架中到达 32、碰撞 16。早先同种子权重为 30 到达、18 碰撞。分场景计数从四航路 20/24 到达降至 18/24，六航路从 10/24 升至 14/24。首次同时开启垂向惯性与重叠风险诊断时触发变量遮蔽，已加回归测试并在独立目录复跑四个空域。当前 26 项测试通过。[四个空域及旧权重对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/replay-control-native-contacts.slLa5O/aggregate.json)

8 对真实接触中，4 对在当前控制周期从安全高度间隔下降进入分离带。最后一次控制决策有 3 对达到纯调速不可行的保守上界条件，固定水平计划后的单步分层有 6 对不可行。约 60 s 前，自适应增加 K 后对 2 对解除求解器的冲突标记。约 90 s 前，考虑垂向惯性后有 7 对具备离散单步分层方案，其中 20/35/50 km 观测分别覆盖 0/2/6 对。上述可行性固定历史水平计划和单步动作集合，观测覆盖用于定位预警时机。[分离量和早期预警聚合](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/replay-control-native-contacts.slLa5O/aggregate.json)

约 90 s 时，50 km 范围中仅扩展标称四维冲突可覆盖 7 对可分层接触中的 2 对，另外 4 对属于潜在边。以 120 s 内标称水平重叠为筛选条件时可覆盖 7 对。该轨迹统计用于选择后续独立闭环对照，普通 20 km 训练与正式协议沿用原冻结配置。[边分类统计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/replay-control-native-contacts.slLa5O/forecast-classification/summary.json)

每架机只增加一个在 20–50 km 内、120 s 有标称水平重叠且最近接时刻最早的候选，取双向并集。约 90 s 时仍覆盖 7 对可分层接触中的 6 对。在另一真实混合种子 98317 的 600 个决策中，交叉场景的平均节点度从 0.78 增至 0.94，P95 保持 4，无筛选的 50 km 四维冲突图分别为 1.30 和 6。已加入开发用的策略注意力诊断，投影继续接收原始 20 km 图。诊断结果无法通过 Stage 2 入场校验。当前回归通过 27 项。[真实混合图密度](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/selective-far-edge-audit.hnKrqe/report.json) · [原生接触覆盖](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/selective-far-edge-audit.hnKrqe/native-contact-coverage.json) · [闭环诊断目录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/far-horizontal-policy-diagnostics.zBqWv5)

独立种子 58909 的 Plan B 闭环对照共 288 架。原注意力开投影到达 240、碰撞 48，增加远距候选到达 243、碰撞 44。关投影则从 226/62 到 230/58。近对头均 48/48 到达。三航路开投影碰撞从 8/48 降至 2/48，六航路从 10/48 增至 16/48，关投影六航路也从 14/48 增至 20/48。该诊断用冻住的旧权重，当前观察改动保留为诊断选项。[投影对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/far-horizontal-policy-diagnostics.zBqWv5/baseline-projection.json) · [远距投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/far-horizontal-policy-diagnostics.zBqWv5/far-projection.json) · [远距关投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/far-horizontal-policy-diagnostics.zBqWv5/far-raw.json)

### 离散分层兜底候选

当默认投影未找到可验证动作时，用已下发的航向与速度预估水平重叠区间。对局部分量，尝试保留已有指令、保持已选目标层、下降 300 ft 和上升 300 ft，在预估水平重叠全区间下用原生垂向惯性筛选候选，并检查分量内每对飞机。四、六航路开发种子 49311–49312 的原生 0.05 s 逐帧小样本从 32/48 到达、16 碰撞提升到 42/48 到达、6 碰撞，四航路从 18/24 到 22/24，六航路从 14/24 到 20/24。六类独立种子 59231–59233 正在做固定权重的闭环配对，统计动作代价与其他场景是否回退。[定位小样本](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-layer-fallback.C2sOAf) · [独立开发对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/independent-native-layer-fallback.G1bs3q)

独立原生逐帧对照中，216 架到达从 166 到 194（76.85%→89.81%），碰撞从 50 降至 22（23.15%→10.19%）。三、四、六航路到达各增加 4 架，正偏航从 22/36 到 32/36，负偏航从 30/36 到 36/36，近对头保持 36/36。平均每机动作代价 18.39→19.14，CPU 三空域决策平均 20.44→29.91 ms。方案 B 同种子 58909 的 288 架短测到达 240→262、碰撞 48→26，近对头均 48/48。方案 B 碰撞率 9.03%，仍高于 Stage 2 门槛。该兜底保留为独立诊断实现，CPU 搜索尚无设备常驻实现。[原生配对全表](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/independent-native-layer-fallback.G1bs3q/aggregate.json) · [方案 B 对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/far-horizontal-policy-diagnostics.zBqWv5/baseline-projection.json) · [方案 B 兜底](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/independent-native-layer-fallback.G1bs3q/plan-b-fallback.json)

四、六航路定位小样本另做了触发条件消融，ADMM 未证实时触发与额外在升降过程中触发，两者均为 42/48 到达、6 架碰撞。保留较窄的未证实触发条件作为当前候选。[惯性校验触发对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/active-vertical-layer-fallback.rFT4QC)

同权重的完整方案 B 对照复用独立种子 57601，2,304 架开投影到达从 1,843 提至 2,081（79.99%→90.32%），碰撞从 461 降至 222（20.01%→9.64%）。六航路从 243/384 到 304/384 到达，近对头保持 384/384，三、四航路和正负偏航到达均提高。碰撞率仍高于 3%。正在验证固定水平计划的分层仍无解时，额外的宽域有界调速候选能否降低残余冲突。[同权重完整对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/plan-b-complete-layer-fallback.hpBLY6/aggregate.json) · [联合候选定位试验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-speed-layer-fallback.6sB4v5)

四、六航路小样本中，联合宽域调速、预览两层并每步最多 300 ft 分别仍为 42/48 到达、6 碰撞。额外增加 250、500 或 1,000 m 的水平预报裕度，三组均为 40/48 到达、8 碰撞。继续使用四项单层候选做下一轮配对，并将任务 20 的三个权重置于同一套离散分层兜底下复核六类场景。[水平裕度](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/layer-horizontal-margin.PMXai6) · [两层预览](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/two-layer-preview-fallback.6UOzx4) · [权重对照目录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/layer-fallback-checkpoint-panel.TPXqGn)

在同一套兜底和 864 架开发流量下，入场/常规/同向偏航权重分别到达 765/760/773，碰撞 97/102/78，近对头均恢复为 144/144。同向偏航权重越界 13 架，其他两组各 2 架。无全解时只减少预测危险对的局部分层，在四、六航路定位样本从 42/48 到达、6 碰撞变为 44/48 到达、4 碰撞，其中六航路的一个种子从 12 到达降至 10。正在完整的 2,304 架流量上做同权重验证。[策略与兜底组合](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/layer-fallback-checkpoint-panel.TPXqGn/aggregate.json) · [局部解四空域追踪](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/layer-partial-resolution.qHQk2W) · [完整局部解验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/plan-b-layer-partial.xoQsmA)

完整 Plan B 2,304 架上，局部分层到达 2,137（92.75%）、碰撞 166（7.20%）、越界 1。相同权重和流量的精确分层全解版本到达 2,081、碰撞 222。局部解的六航路到达 316/384、碰撞 68，正负偏航分别到达 356/384 和 357/384，近对头 384/384。原生 0.05 s 的六类独立配对正在复核，当前碰撞率仍未达到 <3%。[完整方案 B 三组对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/plan-b-layer-partial.xoQsmA/aggregate.json) · [原生独立复核](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/independent-native-layer-partial.Rj0K3R)

独立原生 0.05 s 种子 59231–59233 上，局部分层到达 200/216（92.59%）、碰撞 16/216（7.41%），原投影为 166/216 和 50/216。近对头 36/36，正负偏航 32/36 和 36/36，平均动作代价 18.39→18.87。现有原生与 Plan B 两个开发入口均支持 CPU `--layer-fallback`，正式验收与 Stage 2 入场会拒绝该诊断标记。实际入口复跑原生六航路重现 32/36，Plan B 四空域六类小跑到达 274/288、碰撞 14/288。半层选项在定位样本与原局部解同为 44/48 到达、4 碰撞。当前 32 项回归通过。[原生独立三组结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/independent-native-layer-partial.Rj0K3R/aggregate.json) · [入口复跑](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/discrete-layer-entrypoints.v8q74N) · [同向偏航权重组合复核](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/partial-layer-yaw-checkpoint.m2bjCd)

同向偏航权重组合在完整 2,304 架上到达 2,120、碰撞 158、越界 24、超时 2。现有入场权重组合为 2,137/166/1/0，且六航路到达高 8 架，负偏航到达高 20 架。另一组诊断在不同种子对比球面回航观测，旧权重的局部分层到达 275/288、碰撞 12，切换球面角后 261/288 到达、26 碰撞，两组对头均 48/48。当前选用入场权重和原观测配置作为局部分层对照。 [完整权重组合](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/partial-layer-yaw-checkpoint.m2bjCd/aggregate.json) · [原观测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/geodesic-with-partial-layer.wnNmtY/default.json) · [球面观测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/geodesic-with-partial-layer.wnNmtY/geodesic.json)

### 分层纠正动作示教

用新原生种子 61431–61435 在六类共 30 个空域采集兜底动作。仅保留整空域全员到达的样本，并仅标记实际改动、当时仍在冲突图内的飞机。获得 121 个决策、171 条逐机标签，来自 22 个有标注的成功空域。逐机掩码接入 TD3 与独立监督训练，旧示教数据沿用原资格判定。回归 29 项通过，正用独立种子 60917 做开关投影的短程预试。[示教清单](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/on-policy-layer-teacher.xmlta8/manifest.json) · [监督预试目录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/masked-layer-teacher-pilot.jmM5Z4)

首轮加载发现 3 个标签仅因浮点换算略超动作边界。保留原始资料，生成校验后的归一副本，所有文件通过加载与掩码校验。24 步监督更新后示教训练样本损失由 5.82 降到 3.96。独立 288 架闭环开投影到达/碰撞从 232/56 变为 228/60，关投影从 218/70 变为 216/72，开投影的近对头从 48/48 变为 46/48。该权重未入选。[归一数据清单](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/on-policy-layer-teacher.xmlta8/normalized/manifest.json) · [完整监督预试](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/masked-layer-teacher-pilot.B2rPSW)

### 路由探索的单变量续训

原硬路由的 Gumbel-Softmax 温度只影响梯度松弛。训练动作采样新增独立的路由分布温度，以便降低执行动作与确定性策略的偏差。固定权重的开发诊断中，互相冲突的近对头飞机，常规/低温的期望模式偏离率约 36.6%/24.1%。288 架开投影流量保持相同连续探索噪声时，常规随机路由到达 206、碰撞 82，低温路由到达 220、碰撞 68，两个近对头均为 48/48。确定性参考同为 220 到达、68 碰撞。单卡任务 21 两组从同一权重各续训 450 决策，仅将候选组的执行路由温度从 1.0 改为 0.5。正式确定性评测和学习器内的策略更新公式不变。当前 30 项回归通过。[开发行为诊断](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/router-rollout-exploration-audit.XHmXnj) · [任务 21 快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/router-rollout-stage1.CvTMA7/run.sh) · [任务 21 提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/router-rollout-stage1.CvTMA7/submit.log)

任务 21 在原单卡实际执行，常规组前 400 个决策新增 4,764 次更新，实测 16.43 次更新/s。[实时常规组](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/router-rollout-stage1.CvTMA7/control/metrics.jsonl)

任务 21 两组已分别完成 450 个决策及 5,364 次新增更新。常规/低温更新速度为 16.44/16.43 次/s，实测 9,822/9,099 轮/h，进入确定性策略的六类闭环配对。[常规终点](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/router-rollout-stage1.CvTMA7/control/metrics.jsonl) · [低温终点](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/router-rollout-stage1.CvTMA7/rollout-router/metrics.jsonl)

任务 21 的 864 架配对中，入场、常规、低温组开投影到达 678/666/645、碰撞 185/196/184，近对头均 144/144。低温组负偏航开投影到达 89/144、碰撞 20、越界 34、超时 1，入场对应为 106/37/0/1。关投影低温组 623 到达、208 碰撞、33 越界，原组为 628/236/0。筛选保留入场权重，独立复核 1,799/2,304 到达（78.08%）、504/2,304 碰撞（21.88%），Stage 2 待门槛。[配对筛选](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/router-rollout-stage1.CvTMA7/development-selection.json) · [低温负偏航明细](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/router-rollout-stage1.CvTMA7/rollout-router-projection.json) · [独立门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/router-rollout-stage1.CvTMA7/development-gate.json)

### 航路横向偏差

针对负偏航中从碰撞转为越界的失败，加入可选的有符号大圆航路横向偏差，使用原图的空特征位，尺度为 20 km。训练奖励仅对偏航子类中前后均无冲突、LNAV 关闭且未终止的飞机，在偏离超过 3 km 后加平滑二次代价。观测配置随 checkpoint 保存，两个后端读取同一配置，Stage 2 的周期开发评估也使用当前模型配置。原生和 Torch 实际短训练均完成 5 次更新，两边的评估读取与输出权重一致，33 项回归通过。[A/B 配置往返](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/cross-track-roundtrip.a7b9a7)

任务 22 在原单卡比较入场权重和同样 40/30/30 交通、相同种子的两组续训，只在候选组开启横向偏差观测和权重 1.0 的回航代价。正式评估保留原生 0.05 s 和五个冻结种子，Stage 2 条件继续检查独立方案 B 开发集。[任务 22 源码快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/cross-track-stage1.LMDtIK/run.sh) · [任务 22 提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/cross-track-stage1.LMDtIK/submit.log)

任务 22 常规组已实跑 150 个决策、1,764 次新增更新，实测 16.37 次更新/s。同期在六类原生开发空域独立追踪局部分层后仍发生的接触，记录下降、固定水平计划下的可行性以及 30/60/90/120 s 风险窗口。[单卡训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/cross-track-stage1.LMDtIK/control/metrics.jsonl) · [残余接触追踪](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/partial-layer-residual-contacts.2Q7pc3)

原生六类独立开发空域共 8 对残余接触。约 90 s 前有 6 对具备单步分层方案，20 km 图覆盖其中 2 对，50 km 图覆盖其中 3 对。最后一次决策有 3 对仍具备固定水平计划下的单步分层方案。额外主动触发与优先保留最后目标层在定位小样本同为 44/48 到达、4 碰撞。另独立测试把主动预报提前至 150/180 s。[接触时序及可行性](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/partial-layer-residual-contacts.2Q7pc3/aggregate.json) · [主动触发定位](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/proactive-layer-forecast.Me8DlE) · [目标层保持定位](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/retain-layer-target-pilot.dbKn0S) · [提前窗口定位](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/early-layer-horizon.0kz2K7)

主动预报 150 s 与 180 s 两组均为 44/48 到达、4 架碰撞。任务 22 常规组已完成 450 个决策和 5,364 次新增更新，横向偏差组已完成 400 个决策和 4,764 次新增更新。候选配置确认开启横向偏差观测和 1.0 回航代价，实测 16.44 次更新/s。[预报时长消融](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/early-layer-horizon.0kz2K7) · [横向偏差组实时指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/cross-track-stage1.LMDtIK/cross-track/metrics.jsonl)

任务 22 候选最终完成 5,364 次新增更新，实测 16.45 次/s。同种子 864 架开发流量上，入场/常规续训/横向偏差组开投影分别到达 646/656/664，碰撞 218/208/200，近对头 144/144/140，负偏航 100/86/104。三个组开关投影均无越界。横向偏差组关投影 610 到达、254 碰撞，入场为 616/248。筛选保留常规续训，独立六类复核到达 1,758/2,304（76.30%）、碰撞 546/2,304（23.70%），未通过 Stage 2。[配对筛选](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/cross-track-stage1.LMDtIK/development-selection.json) · [独立门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/cross-track-stage1.LMDtIK/development-gate.json)

### 设备常驻的双机分层

把额外分层改成批量张量运算。在 ADMM 未解决的空域中，以带反向过渡和目标层捕获的分段近似检查水平重叠窗口。每轮可以同时调整一对飞机，并要求预测危险对集合严格缩小。训练时保留原 ADMM 的证实标记，避免把近似投影当作已认证示教。

同一权重和种子 60647 的 288 架闭环流量，常规投影 225 到达、62 碰撞，单机张量搜索 249 到达、38 碰撞，双机张量搜索 267 到达、20 碰撞，CPU 局部分层 275 到达、12 碰撞。各组近对头均为 48/48。双机张量方案六航路由常规 28/48 到达提升到 42/48。512 空域 × 20 架的单卡合成密集微基准上，额外搜索约 4.10 ms/决策，独立测量的原 ADMM 为 7.78 ms/决策。两段各自的峰值分配为 126.7 MiB 和 53.1 MiB，整个训练的开销以后续实跑为准。[常规对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/batched-pair-layer-pilot.8H6J4H/standard.json) · [单机搜索](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/batched-layer-pilot.N4TsYt/plan-b.json) · [双机搜索](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/batched-pair-layer-pilot.8H6J4H/plan-b.json) · [PPU 微基准](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/batched-pair-layer-pilot.8H6J4H/ppu-microbenchmark.log)

训练入口实际完成 7 次更新，34 项回归通过。任务 23 的两组只切换训练采样中的双机分层，使用独立新种子和常规投影检查策略迁移。完整 2,304 架的同权重机制复验另行运行，诊断保护输出不直接用于正式验收。[训练短测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/batched-layer-smoke.4MrOsa/run/metrics.jsonl) · [单卡训练快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/batched-layer-stage1.TQZC8G/run.sh) · [完整机制复验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/batched-pair-layer-full.7IpFhM)

完整同权重开发集种子 57601 上，常规投影 1,843/2,304 到达、461 碰撞，张量双机分层 2,120（92.01%）到达、182（7.90%）碰撞，CPU 局部解 2,137/166。双机分层近对头 384/384，六航路 316/384，正负偏航分别为 343/384 和 357/384。真实原生 0.05 s 独立开发流量亦开始复验。[完整三组对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/batched-pair-layer-full.7IpFhM/aggregate.json) · [原生张量分层诊断](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-batched-pair-layer.f0RTed)

任务 22 完整结束后，任务 23 在同一张 PPU 启动。常规组首 100 个决策新增 1,164 次更新，实测 16.28 次更新/s。[单卡接力](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/batched-layer-stage1.TQZC8G/submit.log) · [实时常规组](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/batched-layer-stage1.TQZC8G/control/metrics.jsonl)

原生 0.05 s 开发种子 59231–59233 的 216 架同权重配对，常规/CPU 局部分层/张量双机分层分别为 166/200/196 到达、50/16/20 碰撞，无越界与超时。张量分层近对头 36/36，三、四、六航路分别到达 34/36、30/36、30/36，正负偏航为 32/36、34/36。两个训练后端的评估入口均可显式启用诊断保护，所有诊断输出都保留正式门槛隔离标记。[原生三组对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-batched-pair-layer.f0RTed/aggregate.json)

任务 23 常规组完成 5,364 次新增更新，16.48 次/s。双机分层组已用真实 512×20 槽位完成首 150 个决策、1,764 次新增更新，实测 16.24 次/s，日志确认开关启用。两组 Torch 峰值均约 5,892 MiB。[常规终点](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/batched-layer-stage1.TQZC8G/control/metrics.jsonl) · [双机分层实跑](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/batched-layer-stage1.TQZC8G/batched-layer/metrics.jsonl)

### 执行动作评论家

原回放中的评论家动作是投影前的提议，环境转换来自下发指令。新增可选的执行动作编码，把修正后的多轴控制输入评论家，同时保留未修正动作的模式、现有奖励和策略更新。方案 B 短测确认实际 191 次干预情况下完成 40 次更新，评论家与策略损失均有限。35 项回归通过。任务 24 已安排到原单卡队列，待任务 23 结束且仍未过独立开发门槛时，用入选权重做配对。若任务 23 的双机分层组入选，两组共同保留该采样保护。[执行动作短测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/executed-critic-smoke.hTzXSA/run/metrics.jsonl) · [任务 24 配置](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/executed-critic-stage1.s27Hjd/run.sh) · [提交记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/executed-critic-stage1.s27Hjd/submit.log)

### 逐机碰撞风险评论家

方案 B 与原生逐帧回放提供每架机的碰撞布尔标签。双评论家共享图特征，各自增加一个折扣碰撞风险头。碰撞终局目标为 1，无碰撞终局目标为 0，其余使用两目标头较大的折扣风险。保持原到达、越界和碰撞奖励，独立风险头先热身 600 次更新，随后在策略目标以权重 40 惩罚预测碰撞风险。新头单独保存与恢复，旧权重按原格式加载。配对的两个组都使用实际下发动作训练评论家，避免把修正后的转换归因给原提议。续训从选中权重所在目录的配置继承实际采样保护及执行动作设置。

原生 0.05 s 对头轨迹记录 8 架实际碰撞，每个控制周期逐机标签之和与总计相符。方案 B CPU 短训完成 68 次更新、实际 2 架碰撞，恢复到 75 次。原单卡 PPU 完成 80 次更新、实际 4 架碰撞，借助短测专用终局采样从回放抽到 102 个逐机碰撞正标签，包含重复采样，风险损失 0.4253。长训配对沿用原采样口径。36 项回归通过。任务 25 使用同一张 PPU，待任务 24 结束且仍未过独立门槛时接力。[原生标签](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/collision-critic-smoke.vo8lm9/native-labels.log) · [PPU 正样本短测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/collision-critic-smoke.vo8lm9/ppu-positive/metrics.jsonl) · [CPU 恢复](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/collision-critic-smoke.vo8lm9/resumed/metrics.jsonl) · [任务 25 提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/collision-critic-stage1.Ay7hcY/submit.log)

任务 23 双机分层训练与常规组均结束，每组新增 5,364 次更新。两组实测 16.39/16.48 次更新每秒，双机分层组的相对吞吐差约 0.5%。独立闭环配对仍在执行，任务 24 等待同一张 PPU。[双机分层训练终点](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/batched-layer-stage1.TQZC8G/batched-layer/metrics.jsonl) · [常规组终点](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/batched-layer-stage1.TQZC8G/control/metrics.jsonl) · [后续执行动作试验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/executed-critic-stage1.s27Hjd/submit.log)

任务 23 同种子 864 架开投影配对中，入场/常规续训/双机分层续训的到达分别为 656/672/677，碰撞 208/192/185，近对头 144/142/144。双机分层组入选。关投影时入场/常规/双机分层的到达为 618/613/613，碰撞为 246/250/246。当前策略本体没有复现保护收益。独立的 2,304 架标准协议复核到达 1,834（79.60%）、碰撞 465（20.18%），近对头 382/384，Stage 2 继续待门槛。额外用新种子 65963 检验入场与训练后权重在同一张量保护下的迁移。[筛选](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/batched-layer-stage1.TQZC8G/development-selection.json) · [独立门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/batched-layer-stage1.TQZC8G/development-gate.json) · [原始策略](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/batched-layer-stage1.TQZC8G/batched-layer-raw.json) · [新种子保护对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage23-layer-policy-transfer.4VRDRO)

用无风险头的既有 TD3 权重恢复，原评论家参数键保持一致，续训更新数 40→46，新风险头计数 6，常规评估可直接读取输出 Actor。[旧权重恢复验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/collision-critic-smoke.vo8lm9/legacy-resumed/metrics.jsonl)

任务 23 的常规/双机分层训练分别为 9,720/9,561 轮/h、16.48/16.39 次更新/s、PPU 利用率均值 85.03%/81.27%（各 33 个定时采样）、PPU 显存峰值 6,314/6,322 MiB。按每架机在一个决策中的动作改动计，累计干预分别为 732,196/942,008。任务 24 已从任务 23 入选权重在原单卡接力，常规组配置确认继续启用双机分层，暂以原提议供评论家训练。另一新增原生 0.05 s 开发种子 66313–66315 的同保护对照单独运行。[单卡完整吞吐](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/batched-layer-stage1.TQZC8G/throughput-summary.json) · [任务 24 实际配置](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/executed-critic-stage1.s27Hjd/control/config.json) · [新种子原生对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-stage23-layer-transfer.PQfPWk)

任务 23 新种子同保护对照完成。方案 B 2,304 架从 2,126/176 到 2,160/132（到达 93.75%、碰撞 5.73%），越界从 0 增到 11，均出现在负偏航。原生逐帧 216 架从 198/18 到 197/18，六航路到达 28→34，正偏航 34→28，近对头两个权重均 36/36。这两个诊断说明当前训练收益集中在方案 B，Stage 2 与正式原生验收条件仍未满足。[新种子方案 B](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage23-layer-policy-transfer.4VRDRO/aggregate.json) · [同权重配对的原生结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-stage23-layer-transfer.PQfPWk/aggregate.json)

任务 24 两组配置确认采样保护均为双机分层，候选实际启用执行动作评论家。两组各完成新增 5,364 次更新，实测 16.33/16.34 次/s，Torch 峰值约 5,892 MiB，进入同种子开关投影闭环评测。[常规训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/executed-critic-stage1.s27Hjd/control/metrics.jsonl) · [执行动作训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/executed-critic-stage1.s27Hjd/executed-critic/metrics.jsonl)

### 执行动作评论家与原生物理复播

任务 24 完成。864 架同开发种子下，入场、常规续训、执行动作续训三组开投影分别到达 695/652/669，碰撞 166/212/193，近对头到达 144/144/140。关投影分别到达 631/606/623，碰撞 230/258/238，近对头到达 144/144/134。保留入场权重。独立标准开发集 2,304 架到达 1,823（79.12%）、碰撞 466（20.23%）。任务 25 随后在同一单卡运行，用实际逐机碰撞事件训练风险评论家。[开发选择](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/executed-critic-stage1.s27Hjd/development-selection.json) · [独立开发门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/executed-critic-stage1.s27Hjd/development-gate.json) · [任务 25 状态](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/collision-critic-stage1.Ay7hcY/status.txt)

从原生 0.05 s 环境记录六航路和正偏航各三条轨迹，保留逐决策的起点、实际控制命令和逐帧接触。六条轨迹共 10 架原生碰撞。从每次决策的同一起点复播时，张量 0.05 s 容差对齐前后检出 6→10 架，张量 0.25 s 检出 4→8 架。六航路种子 66313 的第一处判定差来自 91.439998 m 与 300 ft 的边界，旧张量路径固定扣除 0.0001 m。对齐后，张量 0.05 s 逐步事件审计的整程差异从 2/6 降为 0/6，0.25 s 仍差 4/6。训练使用 6 s 组端点检测，张量 0.05 s 在 5/6 条整程轨迹上一致，正偏航种子 66315 在 1080 s 少检出两架。相同初始状态闭环时，三条六航路在所测张量配置下均为 32 到达、4 碰撞。三条正偏航在 0.25 s 默认、0.05 s 对齐、0.05 s 对齐加 2 m 余量下分别为 28/8、26/10、30/6。[同命令复播汇总](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/matched-native-command-replay.XHr6BI/summary.json) · [训练采样方式的漏检记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/matched-native-command-replay.XHr6BI/yaw_plus15-66315-fine-chord.json) · [同起点正偏航闭环](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/matched-native-command-replay.XHr6BI/yaw_plus15-matched-closed-loop.json)

原生 0.05 s 小样本使用同一权重、同一批开发种子 66313–66315，对照双机分层是否要求额外 2 m 间隔。每场景 36 架。合计到达 197→203、碰撞 18→10、越界 1→3。六类按到达/碰撞排列，三航路 34/2→36/0，四航路 30/6→30/6，六航路 34/2→34/2，近对头 36/0→36/0，正偏航 28/8→34/2，负偏航 35/0→33/0。负偏航损失来自越界，2 m 余量保留为诊断选项。[基础对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-stage23-layer-transfer.PQfPWk/aggregate.json) · [原生六类结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-vertical-margin.rx1jEB)

512 空域 × 20 架 float64 单卡采样微基准中，0.25 s 和 0.05 s 积分分别为 34.18 和 117.95 ms/决策，使用每个 6 s 周期一次事件检查。该测量含采样与观测，未计学习更新。CPU 小规模入口验证完成 38 次更新。[单卡微基准](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/matched-native-command-replay.XHr6BI/benchmark-ppu.log) · [入口验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/matched-native-command-replay.XHr6BI/training-smoke/metrics.jsonl)

任务 25 两组均已完成新增 5,364 次更新。常规组和碰撞评论家组分别为 9,458/9,415 轮/h、16.37/16.22 次更新/s，单卡利用率均值 82.24%/83.94%（各 33 个训练区间采样），PPU 显存峰值 6,498/6,348 MiB。候选记录 5,364 次风险头更新，评论家抽到 1,027 条碰撞事件转移，累计 2,116 个逐机碰撞标签，进入同种子开关投影评测。[单卡吞吐](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/collision-critic-stage1.Ay7hcY/throughput-summary.json) · [碰撞评论家组](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/collision-critic-stage1.Ay7hcY/collision-critic/metrics.jsonl)

任务 25 的 864 架同种子筛选中，入场、常规续训、碰撞评论家三组开投影分别到达 673/623/646、碰撞 184/241/218、近对头到达 144/142/144。关投影分别到达 621/547/581、碰撞 238/317/283、近对头到达 144/122/142。保留入场权重。独立标准开发集 2,304 架到达 1,802（78.21%）、碰撞 497（21.57%）、越界 5。近对头 384/384。Stage 2 继续待门槛。[配对选择](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/collision-critic-stage1.Ay7hcY/development-selection.json) · [独立门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/collision-critic-stage1.Ay7hcY/development-gate.json)

进一步把 6 s 控制周期分为三个 2 s 事件检测区间。实际训练适配器重跑同命令的六条轨迹，0.05 s 积分检出全部 10 架且 6/6 条整程一致，0.25 s 检出 8 架、2/6 条整程一致。方案 B 训练新增可选事件检查次数，定时发机保持在决策边界。入口烟测完成 38 次更新，现有回归测试 38 项通过。[六条实际适配器复播](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/matched-native-command-replay.XHr6BI/event-window/verified-summary.json) · [新选项烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/matched-native-command-replay.XHr6BI/event-window/training-smoke/metrics.jsonl)

用任务 27 替换尚未启动的任务 26。任务 27 已在原单卡启动，校验确认加载的是任务 25 的入选权重。两组都使用 2 s 接触检查和对齐的 300 ft 容差，变量是 0.25 s 或 0.05 s 的训练积分步长。各 512 空域、450 次决策、预计新增 5,364 次更新。控制组实际配置已核对。使用标准方案 B 开发门槛，并在原生 0.05 s 的全新开发种子 67229–67231 做六场景开关投影对照。[任务 26 取消记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/fine-physics-stage1.Q00q8j/cancel.log) · [任务 27 控制组配置](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/fine-physics-window-stage1.SWLVMp/control/config.json)

额外四航路原生直连诊断中，种子 66314 的接触对 5/7、1/6 在接触前处于同一高度层或近同层，当前决策起点距对方约 5.78 和 5.29 km。前 6 次及 10 次决策分别已连边并判为未解。基础 ADMM 轮数 2→4 且保留三角形额外两轮时，仍为 8/12 到达、4 碰撞。该种子额外 2 m 同为 4 碰撞，另一种子 66313 的直连诊断从 0 增到 2。固定方向、即时执行完整单周期速度命令的几何上界显示，两对在接触前 20 个决策的 20/20 和 19/20 个中均无法仅靠当次调速满足 5 km。按已执行的水平计划，枚举至多三架的离散层变更，在这些决策中未找到既保护该对又保持其余原安全对的方案。证据支持检查进冲突区前的跨决策预防动作。[逐帧接触](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-crossing4-admm-trace.Nascry/summary.json) · [调速上界](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-crossing4-admm-trace.Nascry/speed-only-bounds.json) · [离散分层枚举](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-crossing4-admm-trace.Nascry/layer-options.json)

两对接触前约 116 s 时，现有 60/120 s 前瞻状态风险已约为 0.38 和 0.30。实际原提议的垂向动作均为零，标准风险在该分支的垂向梯度也为零，同一状态的速度与航向梯度均有限且非零。该定位结果说明增加标量奖励权重可以放大已有信号，垂向模式是否学到前瞻动作需要单独检查。[真实状态风险与动作梯度](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-crossing4-admm-trace.Nascry/lookahead-signals.json)

任务 27 控制组首个训练区间完成 50 次决策和 564 次新增更新，实测 15.74 次更新/s。[实时训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/fine-physics-window-stage1.SWLVMp/control/metrics.jsonl)

任务 27 的 0.25 s 对照组已完成 450 次决策、5,364 次新增更新，实测 9,555 轮/h、16.29 次更新/s。0.05 s 候选组实际启动且已完成首批 50 次决策、564 次更新，实测 14.10 次更新/s。两组配置确认均为 512 空域和每个 6 s 决策三次接触检查。[对照组指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/fine-physics-window-stage1.SWLVMp/control/metrics.jsonl) · [0.05 s 候选指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/fine-physics-window-stage1.SWLVMp/fine-physics/metrics.jsonl)

### 让前瞻损失评估未选中的动作模式

TD3 增加可选的原始策略前瞻损失。沿用原 60/120 s 水平和垂直安全裕度，对每对飞机按两机各自的保持、调速、分层、转向概率求期望风险。训练时每种模式的输出和选择概率都能收到安全信号，实际动作执行仍由原策略和保护层决定。固定各机为单一模式时，与原风险公式一致。

同一权重与原生逐帧轨迹上，四航路 917 个冲突图内未发出高度指令的决策中，标准风险有 0 个实际得到高度梯度，新损失有 128 个。六航路对应 911 个决策，0→293 个。梯度门槛为绝对值大于 1e-5。真实方案 B 入口烟测完成 38 次更新，39 项回归通过。梯度覆盖率用于验证损失机制，闭环结果由后续配对独立测量。[四航路](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/expected-mode-risk.bgQ4Ly/crossing4.json) · [六航路](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/expected-mode-risk.bgQ4Ly/crossing6.json) · [训练入口烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/expected-mode-risk.bgQ4Ly/smoke/metrics.jsonl)

任务 28 已排在任务 27 之后，复用原单卡和通过校验的入场权重。两组保持相同 512 空域、三类混合流量、积分步长、2 s 接触采样和前瞻损失权重，候选组仅将原始策略的单模式风险改为期望风险。开发筛选同时要求关投影到达、碰撞及近对头不劣于入场。外部复核使用标准方案 B 门槛和原生 0.05 s 新开发种子 67943–67945 的六场景开关投影评测。[任务 28 提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/expected-mode-risk-stage1.uez1Jl/submit.log)

任务 27 两组均已完成 450 次决策、5,364 次新增更新。0.25 s 和 0.05 s 分别实测 9,555/8,748 轮/h、16.29/14.68 次更新/s。PPU 使用率均值为 84.67%/83.30%（33/37 个训练间隔采样），显存峰值均为 6,322 MiB。六场景配对进行中，入场与 0.25 s 对照的 864 架开投影分别到达 692/648、碰撞 170/216、近对头 144/128，关投影分别到达 643/594、碰撞 220/270、近对头 144/140。[端到端单卡吞吐](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/fine-physics-window-stage1.SWLVMp/throughput-summary.json) · [0.25 s 对照开投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/fine-physics-window-stage1.SWLVMp/control-projection.json) · [0.25 s 对照关投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/fine-physics-window-stage1.SWLVMp/control-raw.json)

### 一致轨迹前缀的提前干预

用两个已记录的开发种子做反事实诊断。规划分支从同一原生状态出发，逐机重放原命令，按每 6 s 的合法限幅试验速度或目标高度的分步变化。只在改动决策之前完全复现原轨迹和逐帧结果时，将干预交给原生 BlueSky 0.05 s 核验。规划使用留存的命令后缀，结果用于验证可避免性和时序，不能当作策略闭环成绩。

六航路 66313 原为 10/12 到达、2 碰撞。分别提前 30.45、60.45 和 120.45 s 的方案在原生均恢复 12/12 到达、零碰撞。约 30 s 的方案改变三架，约 60 s 和 120 s 的方案各改变两架。[30 s](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-intervention-lead.aX4FeE/crossing6-30-native.json) · [60 s](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-intervention-lead.aX4FeE/crossing6-60-native.json) · [120 s](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-intervention-lead.aX4FeE/crossing6-120-native.json)

四航路 66314 原为 8/12 到达、4 碰撞。在提前约 30/60/120 s 的三组固定单轴目标中，整数指派均判定有限候选集无解。加入先改目标高度再调速的序列后，约 122 s 的方案调整三架飞机，原生复验为 12/12 到达、零碰撞。约 62 s 的同类候选仍无解。[候选可行性](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-intervention-lead.aX4FeE/crossing4-60-sequence.json) · [原生成功轨迹](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-intervention-lead.aX4FeE/crossing4-120-native.json)

在四条原生验证成功的方案中，现有 60/120 s 风险对第一个动作均给出更高的全空域风险。四航路约 122 s 的方案由 0.735 升至 1.264，六航路约 60 s 的方案由 3.070 升至 3.530。多步原生状态复算中，四航路到干预后第 12 s 才出现更低的风险；六航路 60 s 方案直到原分支接触前，现有风险仍比原分支高。六航路提前 120 s 时，有一架需调整的飞机当前图度为 0，而已有的远距候选机制可观察到一个邻机；四航路有一架需调整的飞机当前预测风险低于 0.0001。该样本支持用真实碰撞终局检查前瞻学习，单纯提高现有风险权重会把这几条有效方案的首步排到后面。[原始第一步动作与风险](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-intervention-lead.aX4FeE/first-action-risk.json) · [逐决策风险](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-intervention-lead.aX4FeE/risk-over-time.json) · [策略观测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-intervention-lead.aX4FeE/visibility.json)

任务 27 的 864 架开发筛选中，入场、0.25 s、0.05 s 开投影到达分别为 692/648/664，碰撞 170/216/200，近对头 144/128/138。关投影到达为 643/594/591，碰撞 220/270/273，近对头 144/140/138。保留入场权重。独立标准方案 B 的 2,304 架到达 1,828（79.34%）、碰撞 470（20.40%）、越界 6，近对头 384/384。原生开发评估继续运行。[筛选](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/fine-physics-window-stage1.SWLVMp/development-selection.json) · [独立门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/fine-physics-window-stage1.SWLVMp/development-gate.json)

任务 27 原生开发配对已完成。三个开发种子 67229–67231，共 216 架，事件按原生 0.05 s 逐帧检查。开投影的三组均使用同一双机分层设置。表格为到达/碰撞，每场景 36 架。入场另有 1 架负偏航越界，0.05 s 组开投影有 1 架越界与 1 架未到达，均为负偏航。0.05 s 组关投影在这批原生种子比 0.25 s 组多到达 10 架、少碰撞 10 架，近对头少到达 2 架。保留入场权重并按原门槛继续训练。[原生六场景配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/fine-physics-window-stage1.SWLVMp/native-development/comparison.json)

| 场景 | 入场开投影 | 0.25 s 开投影 | 0.05 s 开投影 | 0.25 s 关投影 | 0.05 s 关投影 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 三航路 | 36/0 | 36/0 | 34/2 | 24/12 | 26/10 |
| 四航路 | 34/2 | 34/2 | 30/6 | 24/12 | 26/10 |
| 六航路 | 34/2 | 30/6 | 30/6 | 20/16 | 26/10 |
| 近对头 | 36/0 | 36/0 | 36/0 | 36/0 | 34/2 |
| 正偏航 | 30/6 | 34/2 | 32/4 | 20/16 | 26/10 |
| 负偏航 | 33/2 | 34/2 | 34/0 | 28/8 | 24/12 |
| 总计 | 203/12 | 204/12 | 196/18 | 152/64 | 162/54 |

### 用实际终局监督前 120 秒的碰撞风险

增加可选的逐机有限窗口碰撞标签。每 6 s 保存一批待判定的转移，确认其 20 个决策的真实接触后加入设备回放。飞机终止或场景重置时，标签传播对该飞机截止。窗口末端的碰撞纳入标签，未观察完整窗口的末尾转移保留为未成熟样本。常规 TD3 价值目标仍按单步下一个状态计算。

用原生四、六航路接触逐帧记录核对，三对碰撞分别在接触前 116.45、116.15 与 114.45 s 首次进入标签，各覆盖 20 个决策。随机复用机位测试逐项对齐参考实现。方案 B 两个目标的实跑入口均完成 201 次更新，实测采到 40/28 个窗口正标签。原生方案 A 异步采样入口完成 100 次更新，该短测无碰撞。40 项回归通过。[逐帧标签回放](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/observed-horizon-smoke.urWcRh/native-capture-labels.json) · [TD 目标烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/observed-horizon-smoke.urWcRh/td/metrics.jsonl) · [直接观测目标烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/observed-horizon-smoke.urWcRh/observed/metrics.jsonl) · [原生入口](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/observed-horizon-smoke.urWcRh/native-a/metrics.jsonl)

任务 29 已排在任务 28 之后，仍使用同一张 PPU。两组从同一已校验 checkpoint 恢复，共用 512 空域、混合交通、执行动作回放、20 决策标签延迟及风险权重。对照保留自举风险目标，候选使用 120 s 内实测逐机碰撞标签，各 466 次决策，预期 5,364 次新增更新。方案 B 筛选同时检查关闭投影的到达、碰撞和近对头，再做独立开发门槛与原生 0.05 s 的开发种子 68729–68731 配对。[任务 29 提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/observed-collision-stage1.xKVws2/submit.log) · [固定启动配置](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/observed-collision-stage1.xKVws2/run.sh)

### 远距冲突的提前观测

对七条原生开发轨迹、七对真实接触回看接触前 120 s。比较原 20 km 冲突图、最多一个远距水平最近接候选、增加目标高度可达筛选的远距候选。提前 60–90 s 覆盖真实接触对的数量分别为 7/20/26（共 35 个决策时刻），提前 90–120 s 为 5/15/21。仅四、六航路三对接触时两个区间分别为 2/4/9 和 0/0/4（各 15 个时刻）。远距覆盖按至少一架飞机选中该邻机计。此诊断用于衡量物理信息是否可用。[逐对观测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/collision-observability.GetX7T/visibility.json)

方案 A、B 均支持可选的远距候选节点特征。每架机从 20 km 观测外至 50 km 的可达水平交会中，筛掉预测高度间隔达两层的邻机，选择最早最近接者。六个输入编码在机头坐标系的相对位置、相对速度、预计最近接时间和高度差。保护层继续使用原局部冲突图。旧权重首次开启新特征时，相关输入列和优化器动量清零，回归确认开启前后策略初始输出相等，旧锚点行为也不受新列影响。方案 B 从实际入场 checkpoint 续训 87 次更新，新输入权重获得非零值，方案 A 从输出权重继承配置并完成 9 次更新。41 项回归通过。[A/B 迁移与续训](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/far-conflict-smoke.q37sTz/verification.json)

任务 30 已排在任务 29 之后，沿用原单卡。从任务 29 入选权重开始同种子配对。两组保留相同的积分步长、风险目标、回放和投影，候选组启用远距输入。使用标准方案 B 独立门槛与原生 0.05 s 开发种子 69473–69475 的开关投影对照。[任务 30 提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/far-conflict-stage1.OxAdpD/submit.log) · [固定启动配置](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/far-conflict-stage1.OxAdpD/run.sh)

使用同一远距观测权重通过方案 B 六类评测入口和原生 0.05 s 配对入口的短程运行，两个报告均记录了权重内的观测配置。任务 27 已结束，任务 28 在同一原单卡启动，控制组完成首批 150 次决策、1,764 次更新，实测 16.08 次更新/s。[方案 B 入口](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/far-conflict-smoke.q37sTz/plan-b-eval.json) · [原生评测入口](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/far-conflict-smoke.q37sTz/native-eval/comparison.json) · [任务 28 实时指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/expected-mode-risk-stage1.uez1Jl/control/metrics.jsonl)

任务 28 常规与期望风险两组各完成 5,364 次新增更新，实测 9,677/9,607 轮/h、16.24/16.16 次更新/s。PPU 使用率均值 81.76%/82.88%（33/34 个训练区间采样），峰值均为 6,322 MiB。进入独立闭环配对。[单卡吞吐](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/expected-mode-risk-stage1.uez1Jl/throughput-summary.json)

### 分层预报与保护介入后的接触

用任务 27 的 0.05 s 训练权重，在相同原生开发种子 67229 的四航路重现关投影、仅 ADMM、ADMM 加双机分层三种闭环，分别为 10/8/6 架到达、2/4/6 架碰撞。六航路种子 67231 的三种结果为 6/6/8 架到达、6/6/4 架碰撞。保存每步提议、投影、局部图、目标高度和逐帧碰撞身份。[两场景逐步归因](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-projection-regression.6v9l2A/attribution.json)

四航路第 81 个决策中，一对飞机起始高度差为 97.15 m，原折线预报未来 6 s 的最小高差为 93.92 m，判为安全；原生在第 17 帧发生接触，记录高差为 91.26 m。保留分段加速、巡航和捕获的预报将预测最小高差降为 84.70 m，对这一例能报出风险。使用 1,280 个候选对的独立时间网格验证解析实现，4,001 采样点的参考值最大离散差为 0.051 m。单独切换此预报，两条闭环的实际碰撞结果不变。[实际接触差异](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-projection-regression.6v9l2A/contact-clearance-check.json) · [数值验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-projection-regression.6v9l2A/continuous-profile-check.json)

另一对在接触约 18 s 前，原始上升指令的预测最小高差为 91.439 m。两种候选均被旧规则计作一次风险，双机分层因此可以清除这次上升并把该对预测高差降到 0。相同原生状态下仅恢复上升或仅取消邻机下降，会改变随后接触的身份，三种单步回放均保持 6 架碰撞。新增可选的 30 s 临近保护。对于已有风险对，候选分层最多降低现有预测高差 1 m。闭环启用后，四航路碰撞从 6→4、六航路仍为 4，43 个回归测试通过。已在原生 0.05 s 新开发种子 69623–69625 用入场权重做六场景同种子配对。[恢复上升](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-projection-regression.6v9l2A/counterfactual-climb.json) · [同时改两处](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-projection-regression.6v9l2A/counterfactual-both.json) · [新种子复核目录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-preserve-clearance.yzg6Ko)

独立原生六场景各 36 架，共 216 架。入场权重上，原双机分层与 30 s 临近保护两组均为 210 到达（97.22%）、6 碰撞（2.78%），6 架均为正偏航；两组无越界和未到达。每架动作成本 22.228→22.236。保留两项功能作为可选诊断，不变更标准 Stage 2 准入协议。方案 B 入口同时启用两项配置的烟测完整记录了六类，并标记为诊断评估。任务 28 的入场权重筛选开投影为 681/864 到达、180 碰撞，关投影为 635/864 到达、226 碰撞，候选组配对继续运行。[独立原生汇总](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-preserve-clearance.yzg6Ko/aggregate.json) · [方案 B 诊断入口](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-projection-regression.6v9l2A/plan-b-diagnostic-smoke.json) · [任务 28 入场权重](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/expected-mode-risk-stage1.uez1Jl/incoming-projection.json)

任务 28 的 864 架方案 B 配对完成。入场、常规续训、多模式期望风险组开投影到达 681/658/676、碰撞 180/206/188、对头到达 144/138/138。关闭投影到达 635/592/582、碰撞 226/272/282、对头到达 142/138/120。阶段筛选保留入场权重。独立标准开发集到达 1,813/2,304（78.69%）、碰撞 484/2,304（21.01%）、越界 7，近对头 384/384。原生开发配对继续运行，Stage 2 等待准入。[任务 28 配对选择](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/expected-mode-risk-stage1.uez1Jl/development-selection.json) · [独立开发评测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/expected-mode-risk-stage1.uez1Jl/development-gate.json)

### 在原三类配额内加强高航路数场景

用入场权重在实际方案 B 定时发机全混合训练布局上审计两个开发种子，每组 48 空域 × 380 个决策，共 36,480 个空域转移。交叉组中六、八航路占 7,482/15,790（47.4%）的转移、140/170（82.4%）的碰撞飞机。偏航组中八航路占 2,532/7,982（31.7%）的转移、70/92（76.1%）的碰撞飞机。三、四航路的高局部度转移占 55.8%/59.4%，六、八航路只占 31.0%/40.5%，单按邻机数加抽会偏向容易的拓扑。审计中的发机次数未统计，碰撞和样本统计来自实际逐步闭环。[两种子拓扑审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/replay-coverage.zemvmd/aggregate.json)

方案 A/B 现随转移报告该回合的航路数，跨重置仍使用旧回合标签。航路数仅用于回放和统计。候选按原 40/30/30 的三类配额抽样，交叉配额中约四分之一预留给六、八航路，偏航配额中约四分之一预留给八航路。缺少该子类时回退到原抽样；使用独立的随机流，近对头抽样逐批与对照一致。对同一组 36,480 个实测类别标签各抽 160 批，交叉六、八航路覆盖 1,922/4,160→2,421/4,160（46.2%→58.2%），八航路偏航 963/3,040→1,390/3,040（31.7%→45.7%），近对头均为同一批次的 3,040 条。方案 B/A 真实训练入口分别完成 93/21 次更新，覆盖场景重置。44 项回归通过，含航路标签和 120 s 碰撞标签混合回放。[同数据抽样核对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/replay-coverage.zemvmd/replay-draws.json) · [A/B 入口验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/topology-replay-smoke.XsXnCn/verification.json)

任务 31 在任务 30 后排队，复用原单卡。两组沿用同样入场权重、随机种子、原投影、三类交通配额和回放预算，仅候选组应用航路子类分层。按标准独立方案 B 开发门槛筛选后，在原生 0.05 s 开发种子 70111–70113 配对验证。[任务 31 提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/topology-replay-stage1.EjPWYz/submit.log) · [固定启动配置](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/topology-replay-stage1.EjPWYz/run.sh)

### 任务 28 原生配对与评测耗时

任务 28 原生 0.05 s 的开发种子 67943–67945 共 216 架，以下为到达/碰撞。三组开投影共用同一双机分层。入场另有 1 架负偏航越界，两个续训组均无越界和未到达。期望风险组开投影改善四航路和偏航，同时六航路碰撞由 2 增至 8。关投影近对头比常规组少到达 2 架。标准方案 B 筛选保留入场权重，任务 29 已从该权重接力。[原生完整配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/expected-mode-risk-stage1.uez1Jl/native-development/comparison.json) · [标准开发门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/expected-mode-risk-stage1.uez1Jl/development-gate.json)

| 场景 | 入场开投影 | 常规开投影 | 期望风险开投影 | 常规关投影 | 期望风险关投影 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 三航路 | 36/0 | 34/2 | 34/2 | 26/10 | 28/8 |
| 四航路 | 32/4 | 34/2 | 36/0 | 26/10 | 26/10 |
| 六航路 | 34/2 | 29/7 | 28/8 | 18/18 | 18/18 |
| 近对头 | 36/0 | 36/0 | 36/0 | 36/0 | 34/2 |
| 正偏航 | 32/4 | 32/4 | 36/0 | 30/6 | 28/8 |
| 负偏航 | 33/2 | 26/10 | 34/2 | 24/12 | 24/12 |
| 总计 | 203/12 | 191/25 | 204/12 | 160/56 | 158/58 |

任务 27 的方案 B 七次闭环评测共约 651 s，原生 30 个模式与场景批次计约 611 s。工作区的原生评测支持重叠运行两组独立原生采样进程，策略前向在同一设备上依次下发。同一批真实 checkpoint 在六航路新种子 70421–70423，三组开投影、两组关投影共 180 架评测，从 129.16 s 降至 109.11 s。逐种子的结果、动作成本、保护次数及逐帧最小间距逐字段一致。原 PPU 节点可见 12 个 CPU，开发评估两组各占三名原生进程。正式评估继续使用原冻结流程。[原生评测性能与等价核验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-parallel-benchmark.3MQ0K2/verification.json)

方案 B 在一个控制决策内复用固定入场航向。关投影、六场景 216 架的完整逐字段对照一致。32 空域事件检查的交替微基准中位数由 0.646 ms 降至 0.607 ms；这轮完整 CPU 评测在有其他本地负载时为 40.35→51.10 s，尚未证实端到端提速。此改动未并入已提交的候选任务。[完整逐字段验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/tensor-eval-route-cache.n4RGaE/verification.json) · [事件检查微基准](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/tensor-eval-route-cache.n4RGaE/event-microbenchmark.json)

待运行的任务 30、31 分别由 32、33 接替，两组训练的源码仅回移原生评测调度，训练参数和开发种子逐项沿用。各新快照经过原生 0.05 s 启动实测，输出行一致。确认旧任务未启动后取消，旧快照和审计文件保留。任务 29 在同一卡持续运行，控制组前 350 个决策累计 3,972 次新增更新，15.96 次/s，评论家批次共包含 34,698 个真实窗口碰撞标签。[替换前兼容性验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-parallel-benchmark.3MQ0K2/snapshot-smoke.txt) · [任务 32 提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/far-conflict-native-overlap.85hrGN/submit.log) · [任务 33 提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/topology-replay-native-overlap.YWILJI/submit.log) · [任务 30 取消](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/far-conflict-stage1.OxAdpD/cancel.log) · [任务 31 取消](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/topology-replay-stage1.EjPWYz/cancel.log) · [任务 29 实际指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/observed-collision-stage1.xKVws2/control/metrics.jsonl)

### 混合保护与原策略轨迹

四、六航路三条原生逐帧重放中，回收的 5,199 个逐机决策有 1,453 个被投影改写；冲突图内是 1,216/2,100（57.9%），有 44 个下发命令涉及多个控制轴。结果说明常规受保护回放不能直接覆盖这部分原动作执行后的状态。用独立随机流在仿真回合初始化和重置时分配投影模式，候选约 25% 回合全程执行原策略，由执行动作评论家学习同时存在的两类轨迹。原策略回合不产生投影示教。三类交通及其回放配额沿用原配置，日志报告原策略回合的转移和逐机终局。[原生动作覆盖](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-projection-regression.6v9l2A/rollout-action-support.json)

实际方案 B 混合短训完成 237 次更新，其中 1,356 个空域转移来自原策略回合，记录 34 架碰撞。方案 A 完成 21 次更新，36 个原策略转移跨回合重置。比例 100% 时，与直接关投影同种子完整训练 33 次更新，Actor、Critic、目标网络、优化器和环境计数逐项相同。45 项回归通过。[双引擎入口验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/mixed-projection-smoke.EJvS6j/verification.json) · [全关等价检查](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/mixed-projection-smoke.EJvS6j/unprotected-equivalence.json)

任务 34 排在任务 33 后，复用原单卡。对照和候选都使用执行动作评论家，沿用入场权重选定的观测、风险和拓扑回放。候选只切换约 25% 原策略回合。标准方案 B 在同种子开关投影筛选后复核，原生使用新开发种子 71009–71011 做 0.05 s 逐帧配对。[任务 34 提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/mixed-projection-stage1.x24Ft3/submit.log) · [固定启动配置](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/mixed-projection-stage1.x24Ft3/run.sh)

任务 29 两组单卡训练各完成 5,364 次新增更新。自举目标与观测目标实测 10,024/10,110 轮/h、16.03/16.06 次更新/s，均值 PPU 利用率为 82.59%/83.09%（34/33 个时段采样），显存峰值均 6,948 MiB。评论家抽样中实测窗口碰撞正标签累计 51,793/51,677。两组进入标准开关投影闭环筛选，原生开发配对随后执行。[完整单卡运行指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/observed-collision-stage1.xKVws2/throughput-summary.json)

### 多步价值更新

原生 0.05 s 反事实重放中，两条成功避让轨迹的第一个 6 s 控制步因动作成本得分更低。四航路首次接触距干预约 122.45 s，取 120 s 时两架的折扣回报仍偏向原碰撞轨迹（−0.258 对 −0.766），计入 126 s 内真实碰撞后翻转（−82.05 对 −0.78）。六航路首次接触距干预约 60.45 s，计入 66 s 内真实碰撞后同样翻转（−90.79 对 −0.78）。这里按两架飞机的原生命令成本与实际碰撞结算，折扣为 0.99，比较结束于首次接触所在控制步。[原生回报重放](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-intervention-lead.aX4FeE/return-horizon-audit.json)

任务 35 在既有单卡队列接续任务 34。方案 B 两组使用相同的 512 空域、入场权重、混合交通、回放成熟等待和更新预算。候选将价值目标展开至 24 个控制步（144 s），在逐机终止处截断并限制自举，对照保留单步目标。方案 B CPU 短训两组各完成 188 次更新，均在第 43 个决策步产生首批成熟回放。方案 A 原生重叠入口完成 18 次更新。几何和回放测试共 45 项通过、1 项跳过。原生开发配对使用 71429–71431，带投影和无投影均评估。[训练入口验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/multi-step-returns-smoke.Q4JSsg/verification.json) · [任务提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/multi-step-return-stage1.uQh9cA/submit.log) · [启动快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/multi-step-return-stage1.uQh9cA/run.sh)

任务 29 完成标准方案 B 开关投影闭环筛选，保留入场权重。864 架中，开投影时入场权重 669 到达、190 碰撞，单步风险对照 646 到达、218 碰撞，120 s 实际碰撞标签组 672 到达、191 碰撞。标签组的近对头到达从入场权重的 144/144 降到 138/144。关投影时标签组为 620 到达、243 碰撞，入场权重为 650 到达、210 碰撞。截至记录时，独立开发门槛和原生开发配对仍在评估。[选模及各组汇总](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/observed-collision-stage1.xKVws2/development-selection.json) · [标签组关投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/observed-collision-stage1.xKVws2/observed-collision-raw.json)

### 回放预填对照

任务 29 独立开发集完成。入场权重在 2,304 架中到达 1,778（77.17%）、碰撞 516（22.40%）、越界 9、未到达 1，近对头 384/384。三、四、六航路分别有 78、110、132 架碰撞。Stage 2 准入未通过。原生 0.05 s 的开发配对仍在运行。[独立开发门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/observed-collision-stage1.xKVws2/development-gate.json)

两组从同一入场权重恢复时重建回放池。实际训练日志显示，到第 150 个控制决策时，各组已进行 1,572 次更新，评论家各抽取 100,608 条经验，抽样中到达终局为 0，碰撞终局仅 261 和 256 条。到第 200 步才在批次中观测到到达终局。启动方案 B 控制环境短训，比较当回放成熟就更新和推迟至第 171 个决策再更新。两组同权重、同流量与相同的 300 次更新预算，后者额外采集预填经验。使用新开发种子 72119–72121、72329–72331，各评估六场景、开关投影。短训仅用于判断回放初始化的影响。[等更新对照方案](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/replay-prefill-pilot.r3vbJ3/protocol.txt) · [单卡对照训练日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/observed-collision-stage1.xKVws2/control/metrics.jsonl)

预填组前 80 次更新的 2,560 个评论家样本中出现 56 条到达终局、21 条碰撞终局。常规组前 100 次更新的 3,200 个样本均为普通转移。这确认两组在更新开始时取得了不同的终局信号。完整开关投影结果仍在等待训练结束。[常规组日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/replay-prefill-pilot.r3vbJ3/control/metrics.jsonl) · [预填组日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/replay-prefill-pilot.r3vbJ3/prefill/metrics.jsonl)

另用原生状态重算双机分层之后的逐机航向保护，六条四/六航路重放与正负 15° 开发种子 72461 共 14,742 个在场逐机决策，166 个超过航向死区的命令。分层后重新计算没有改变这些航向的许可结果。该样本下暂不以调整航向检查顺序作为改进项。[交叉场景同状态复核](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/post-layer-heading-audit.74itKw/summary.json) · [偏航同状态复核](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/post-layer-heading-audit.74itKw/yaw-summary.json)

### 策略漂移与原生无投影入场对照

在保存的原生四、六航路状态上复算任务 29 中途和最终三组权重。两类共 1,114 个冲突图内逐机决策，入场权重选择调速 739 次。第 372 次新增更新时，单步组和实际碰撞标签组分别选择调速 423 和 426 次，保持由 309 增至 606 和 605。同期训练日志的两组评论家均未抽到实际到达、实际碰撞或 120 s 碰撞标签。复算另含正负偏航，按捕获时入场权重逐机核对命令。结果说明动作变化在终局监督进入前已发生，需要独立闭环评估判断影响。[同状态动作统计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/task29-action-drift.NfdBk0/summary.json) · [运行日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/observed-collision-stage1.xKVws2/control/metrics.jsonl)

原生开发评估新增可选的入场权重关投影项，并支持单权重评估。同一 CPU、种子及配置下，四模式短测中的入场关投影与单独关投影结果一致，45 项回归通过、1 项跳过。任务 29 在原开发种子 68729–68731 补跑三套权重的原生关投影，其中两套可与单卡结果校验。另用独立方案 B 种子 72931–72934，给入场、两组第 3,372 次新增更新与最终权重做六场景开关投影对照。[原生入口验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/task29-native-raw.Y83zxs/smoke-verified.json) · [无投影补跑](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/task29-native-raw.Y83zxs/raw) · [中途权重配置和产物](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/task29-anytime-selection.jzx53j)

任务 29 原生 0.05 s 开发配对完成，种子 68729–68731，共 216 架。三组开投影均启用同一双机分层。下表为到达和碰撞。入场权重有 3 架负偏航越界，标签组开投影有 1 架负偏航越界。两组关闭投影都无越界或未达。实际标签组相对单步续训开投影多到达 3 架、少碰撞 4 架，距入场权重还有 6 架到达和 8 架碰撞差距。独立阶段门槛已结束，保留入场权重。[原生配对逐种子结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/observed-collision-stage1.xKVws2/native-development/comparison.json)

| 场景 | 入场开投影 | 单步开投影 | 标签开投影 | 单步关投影 | 标签关投影 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 三航路 | 36 / 0 | 34 / 2 | 36 / 0 | 22 / 14 | 28 / 8 |
| 四航路 | 32 / 4 | 32 / 4 | 30 / 6 | 24 / 12 | 26 / 10 |
| 六航路 | 34 / 2 | 26 / 10 | 30 / 6 | 12 / 24 | 18 / 18 |
| 近对头 | 36 / 0 | 36 / 0 | 36 / 0 | 30 / 6 | 36 / 0 |
| 正偏航 | 34 / 2 | 30 / 6 | 32 / 4 | 24 / 12 | 24 / 12 |
| 负偏航 | 29 / 4 | 34 / 2 | 31 / 4 | 24 / 12 | 26 / 10 |
| 总计 | 201 / 12 | 192 / 24 | 195 / 20 | 136 / 80 | 158 / 58 |

在同一批四、六航路原生状态上，任务 29 训练使用的冻结行为克隆锚点也仅选择调速 423 次。已启动等预算锚点隔离测试，两组均继续从当前入场权重恢复，分别使用冻结初始策略与入场策略作锚。学习率、回放、流量、分层和风险目标完全配对，用相同状态复测动作变化。[冻结锚点同状态统计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/task29-action-drift.NfdBk0/frozen-anchor.json) · [锚点隔离试验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/resume-anchor-pilot.Bv5Beb)

锚点隔离测试两组各完成 38 次更新，配置除锚点文件和输出目录外相同。相对入场权重，四、六航路 1,114 个冲突图内决策的模式变化由冻结锚点组的 311 个降到入场锚点组的 120 个，调速分别为 430 和 825 次，入场为 739 次。负偏航冲突图内模式变化由 73 个升至 209 个。更换锚点已改变训练早期漂移，六场景开关投影的实际效果正在同种子复测。[锚点对照结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/resume-anchor-pilot.Bv5Beb/summary.json)

SOFY 任务 29 成功结束，原单卡已接力运行任务 32 的远距冲突观测对照。后续训练的入场权重仍由固定开发门槛选定。[任务 32 实际训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/far-conflict-native-overlap.85hrGN/control.log)

### 锚点与中途权重的闭环筛选

新方案 B 开发种子 72931–72934，各配置 288 架。冻结初始策略作锚时，开/关投影分别为 220/68 和 212/76 到达/碰撞。当前入场策略作锚时为 224/64 和 210/78。当前入场权重本身为 234/54 和 210/78。换锚后开投影近对头从 44/48 恢复至 48/48，负偏航由 36/48 降至 28/48。两种全局锚点配置均未晋升。[冻结锚开投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/resume-anchor-pilot.Bv5Beb/frozen-projected.json) · [入场锚开投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/resume-anchor-pilot.Bv5Beb/incumbent-projected.json) · [入场锚关投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/resume-anchor-pilot.Bv5Beb/incumbent-raw.json)

同开发种子对任务 29 的中途权重复筛。实际碰撞标签组在 3,372 次新增更新时较最终权重更好，开投影为 226/62 对 216/72，关投影为 214/74 对 198/90。中途权重关投影优于入场的 210/78，开投影仍少到达 8 架，且近对头 46/48。单步对照组两个检查点开投影均为 220/68。保留入场权重。该对照表明末次更新的权重会掩盖训练中途的局部收益。[中途标签组开投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/task29-anytime-selection.jzx53j/observed-mid-projected.json) · [中途标签组关投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/task29-anytime-selection.jzx53j/observed-mid-raw.json) · [终点标签组关投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/task29-anytime-selection.jzx53j/observed-final-raw.json)

### 三机联合分层定位

按原生留存状态重建双机分层候选矩阵，逐项核对双机输出。四、六航路和正负偏航共 802 个控制决策，79 个在双机搜索后仍有预测冲突。在四航路 3 个、六航路 7 个决策中，三机同时改目标层可进一步减少预测危险对，且没有新增原本安全的配对。候选仅改变三架飞机的目标高度，沿用相同速度和航向指令。原生逐帧闭环正在对预先选定的两条留存轨迹和新种子 73117 做验证。[相同状态对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-altitude-audit.Ducgy2/summary.json) · [原生原型入口](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-altitude-audit.Ducgy2/run_native.py)

原生关投影补跑完成。入场、单步续训、标签续训的到达/碰撞分别为 161/52、136/80、158/58，入场另有 3 架越界。两个续训权重的 CPU 原生复跑与原 PPU 原生开发结果在六场景的逐种子终局全部一致。实际标签改善了单步风险续训的结果，入场策略仍保留。[入场及候选关投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/task29-native-raw.Y83zxs/raw/comparison.json)

三机原型的原生闭环完成。留存四航路种子 67229 从 6 到达、6 碰撞进到 10/2，留存六航路 67231 从 8/4 到 10/2，首次改动前分别逐步核对了 66 和 76 个决策。新种子 73117 下，四航路 10/2→12/0，六航路两组均为 8/4。四个配对空域共 48 架，到达 32→40，碰撞 16→8。按只接受预测危险对减少且不新伤及其他配对的规则，已将同一搜索移植为张量实现，原生和方案 B 主评估入口都可显式启用，单卡运行开销和独立六类效果仍待验证。[留存四航路](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-altitude-audit.Ducgy2/native-crossing4-67229-joint.json) · [留存六航路](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-altitude-audit.Ducgy2/native-crossing6-67231-joint.json) · [新种子四航路](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-altitude-audit.Ducgy2/native-crossing4-73117-joint.json) · [新种子六航路](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-altitude-audit.Ducgy2/native-crossing6-73117-joint.json)

设备张量版在 802 个留存原生状态上的高度指令与 CPU 搜索原型逐项相同，最大差 0，速度和航向逐项一致。原生评估 CLI 复现了新种子两场景的 12/0 和 8/4，所有事件仍按 0.05 s 逐帧采样。回归 46 项通过、1 项跳过。另启动独立六场景诊断，在方案 B 共 576 架和原生共 216 架分别以同权重和新种子比较双机与三机，检查近对头与两种偏航的非退化。当前正式 Stage 2 门槛未通过。[张量与原型指令对比](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-altitude-audit.Ducgy2/tensor-verified.json) · [原生 CLI 结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-altitude-audit.Ducgy2/tensor-native/comparison.json)

单卡任务 32 的常规组和远距观测组各完成 5,364 次新增更新，均为 16.22 次/s，分别约 9,480 和 9,504 轮/h。标准方案 B 的开关投影配对已开始，尚未选模。[常规组实测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/far-conflict-native-overlap.85hrGN/control/metrics.jsonl) · [远距组实测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/far-conflict-native-overlap.85hrGN/far-conflict/metrics.jsonl)

方案 B 独立六场景完成。双机到达 524/576（90.97%）、碰撞 52/576（9.03%），三机为 532/576（92.36%）、44/576（7.64%）。三、四航路和正偏航分别少碰撞 4、2、6 架。六航路和负偏航各多碰撞 2 架。近对头两组均 96/96，到达、碰撞以外的终局均为零。控制环境用时 70.41→74.47 s，单卡开销待测。对退化的两个家族启动逐决策归因。[双机](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-altitude-audit.Ducgy2/new-b-pairs.json) · [三机](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-altitude-audit.Ducgy2/new-b-joint.json) · [退化定位](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-layer-regression.4T0bQK)

回放预填两组已生成最终权重。常规组第 20 步仅有 16 条成熟经验，低于 32 条批量，到第 21 步才开始更新。原脚本的等预算断言阻止了评估。实测常规 298 次、预填 300 次，保留两组 checkpoint 并恢复原计划的同种子闭环筛选。该试验使用相近更新预算，结论按这一差异解释。[实际更新预算](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/replay-prefill-pilot.r3vbJ3/actual-budget.json)

独立原生 0.05 s 开发种子 73471–73473 的三机结果为 196 到达、18 碰撞、2 越界。双机为 194/20/2，全部收益来自四航路，两类偏航与近对头保持相同。该结果使用诊断投影和开发种子。[原生双机](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-altitude-audit.Ducgy2/new-native-pairs/comparison.json) · [原生三机](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-altitude-audit.Ducgy2/new-native-joint/comparison.json)

独立方案 B 反例的逐决策重放复现了六航路和负偏航各多 2 架碰撞。六航路的三机调整把一对原本 182.91 m 的预测高度间隔降至 91.47 m，时间窗从约 103 s 后开始。负偏航调整把一对已经水平重叠的飞机的预测高度间隔从 134.15 m 降至 119.29 m。两者均高于原 91.44 m 阈值，原搜索允许。新增可选裕度检查，保留原已达到 101.44 m 的配对裕度，并限制未来 6 s 内水平重叠的原间隔减少最多 1 m。对原先不够 101.44 m 的远期危险对仍允许恢复至 300 ft。47 项回归通过、1 项跳过，开关裕度后的完整闭环正在复验。[六航路轨迹](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-layer-regression.4T0bQK/crossing6.json) · [负偏航轨迹](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-layer-regression.4T0bQK/yaw_minus15.json)

预填复跑两批共 432 架。开投影时入场、常规、预填依次为 326/104、332/94、330/100 到达/碰撞，近对头均 72/72。关投影时依次为 323/108、296/128、303/126，近对头 70/66/68。预填较常规减少了关投影退化，入场是关投影表现最好的权重，两组续训均未晋升。两组实际更新预算为 298 和 300，试验保留该差异。[预填所有配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/replay-prefill-pilot.r3vbJ3)

任务 32 的两组远距观测筛选结果已落盘。标准方案 B 的 864 架开投影中，入场为 688/172，常规续训为 650/210，远距观测续训为 646/218。关投影依次为 654/206、587/272、592/272。选模保留入场权重，等待独立门槛和原生开发配对。[任务 32 选模](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/far-conflict-native-overlap.85hrGN/development-selection.json)

保留裕度的三机版完成原方案 B 流量复跑。六航路和负偏航恢复到双机的 18、6 架碰撞，另外四类没有比双机更多碰撞，总计 530/576 到达（92.01%）、46/576 碰撞（7.99%）。双机与未保留裕度的三机分别为 524/52、532/44。以新种子 73607 做独立双机配对，并排队复测原生 73471–73473 的同一批流量。故障复验结果不用于正式 Stage 2 准入。[裕度版六类复跑](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-layer-regression.4T0bQK/paired-regression-guard.json)

独立方案 B 种子 73607 共 576 架，双机为 529 到达、46 碰撞，裕度三机为 531 到达、44 碰撞。两组均有 1 架负偏航越界，无超时。收益来自六航路 78/18→80/16，近对头保持 96/96，两种偏航保持 88/96 和 83/96。原生 0.05 s 开发种子 73471–73473 上，裕度三机与未加裕度三机的六类终局一致，为 196 到达、18 碰撞、2 越界，双机为 194/20/2。裕度版在原生保住了四航路的收益，阶段准入阈值仍未达到。[方案 B 双机](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-layer-regression.4T0bQK/fresh-b-pairs.json) · [方案 B 裕度三机](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-layer-regression.4T0bQK/fresh-b-guard.json) · [原生裕度三机](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-layer-regression.4T0bQK/retained-native-guard/comparison.json)

张量三机搜索只打包仍有危险对且存在至少三机约束分量的空域。候选控制量保持张量表示，设备后端的动态形状同步耗时由实测包含。混合 512×20 面板有 51 个符合条件的空域，密集压力面板有 512 个。两种面板的打包与全量命令逐项一致，分别有 2 和 20 个空域实际改变分层。CPU 测时基于 4 次重复，单卡任务 36 已在原队列提交，将测 20 次、报告同步延时和显存峰值。[512 空域控制环境实测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-layer-device-benchmark.bzZKfk/cpu.json) · [单卡任务](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-layer-device-benchmark.bzZKfk/submit.log)

复核两条原生有效的首次三机干预，共 6 架受改飞机。3 架的联合命令同时使用调速和分层，现有策略只能在一个控制步选择一轴。四航路另有两架在 20 km 图内没有邻机，扩展候选能观察到它们的邻机。已在同初始状态复跑只保留高度轴的干预，验证可直接学习的动作是否能维持原生收益。[动作可表达性](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-altitude-audit.Ducgy2/policy-representability.json)

任务 32 的独立标准开发集为 1,828/2,304 到达、466/2,304 碰撞、10 架越界，近对头 382/384。到达和碰撞分别为 79.34% 和 20.23%，Stage 2 继续等待达标。原生开发配对正在运行。[任务 32 独立门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/far-conflict-native-overlap.85hrGN/development-gate.json)

单轴复跑保住三条已验证原生轨迹的到达/碰撞结果。四航路 67229 从 6/6 到 10/2，六航路 67231 从 8/4 到 10/2，新四航路 73117 从 10/2 到 12/0。对两条有留存初态的轨迹生成了 4 条在当前图中可观测的逐机分层标签，现有 TD3 教师数据加载已逐项验证。[原生单轴终局](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-altitude-audit.Ducgy2) · [训练标签验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-layer-native-teacher.V5eMIV/known-example-verification.json)

新种子 73771–73772 各跑六类交通。样本验收要求同种子到达增加、碰撞减少、越界及超时不增加，并且逐机输入中有冲突邻机。早期采集在同一解释器里重复初始化 BlueSky，出现插件退场异常。已停止旧采集，保留诊断产物，新的双机对照和单轴修复回合各在独立进程运行，待完整终局后再筛选训练样本。[隔离版采集](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-layer-native-teacher.V5eMIV/collect_isolated.py)

任务 32 原生六场景完成。开发种子 69473–69475，216 架，按 0.05 s 逐帧检查。开投影且统一使用双机分层时，入场权重 210/6，常规和远距观测均为 194/22。四航路和近对头均为 36/36，两组续训在六航路均从入场的 36/36 到达降至 26/36。关投影常规为 156/60，远距观测为 154/62。各组越界和未到达均为零。任务成功结束，入场权重继续保留。[原生六场景逐种子结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/far-conflict-native-overlap.85hrGN/native-development/comparison.json)

### 协商头梯度与单卡恢复

入场与远距观测续训权重逐项比对，速度协商、高度协商和机动意愿三个头共 6 个张量全部未变，动作头均有变化。阶段 1 训练增加可选两级 ADMM 协商损失，沿用实际调速、分层约束和迭代轮数，用间隔缺口和原始残差优化协商头。该损失对飞控提议和图编码器切断梯度。方案 B 实际训练入口完成 77 次更新，原生方案 A 重叠入口完成 21 次更新，48 项回归通过、1 项跳过。原生高保真微调入口支持继承该参数。[方案 B 短训](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/admm-gradient-calibration.hUoph6/online-smoke/metrics.jsonl) · [原生短训](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/admm-gradient-calibration.hUoph6/native-smoke/metrics.jsonl)

从当前入场权重用四、六航路真实原生状态独立校准 80 步，只改变上述 6 个参数，原控制命令逐项一致。两条校准轨迹的全量损失从 0.175/0.243 降到 0.161/0.223，两条未参与拟合的偏航轨迹也下降。独立标准方案 B 的 288 架为 230/58→235/52，新增 1 架正偏航越界。启用同样双机兜底时两组均 268/20，近对头均为 48/48。原生 0.05 s 新种子 74103–74105 下，在相同双机兜底配置由 201 到达、14 碰撞、1 越界改善到 206/10/0。更大的标准方案 B 配对正在运行。该校准只改变安全层动作，策略本体的泛化仍由关投影学习衡量。[校准记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/admm-gradient-calibration.hUoph6/fit.json) · [标准方案 B](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/admm-gradient-calibration.hUoph6/calibrated-standard.json) · [原生开发配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/admm-gradient-calibration.hUoph6/native-development/comparison.json)

独立进程的单轴修复采集已补齐 12 组。同种子双机和单轴修复均为 132/144 到达、12 架碰撞，无越界或超时，新增合格标签为 0。原生中断时保留了 9 组完整配对与近对头已落盘对照，仅重新采集缺失回合。[原采集](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-layer-native-teacher.V5eMIV/isolated) · [补齐记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/joint-layer-native-teacher.V5eMIV/recovery)

单卡任务 33 的对照组在 3,564 次新增更新后因写入 CPFS 返回 ENOSPC 中断，34、35 因前序不完整退出。旧实测日志和中途权重保留。另一个独立错误来自任务 36 无条件导入可选的 torch_ppu，本节点实际使用自带 PPU CUDA 兼容的 PyTorch。节点实际 fsync 写入 160 MiB 探针已通过。恢复快照逐项校验，三项训练继续沿用原入场权重、种子与预算。批量提交被自动审核拒绝，现按同一张 PPU 一次仅运行一项恢复。修正后微基准已作为任务 37 提交。[原任务日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/topology-replay-native-overlap.YWILJI/control.log) · [恢复快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/recovered-single-ppu.5XWTkf/manifest.json) · [任务 37](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/recovered-single-ppu.5XWTkf/benchmark/submit.log)

任务 37 已完成真实单卡 512×20 基准。混合面板 51 个需检查空域，打包与全量搜索的选择一致。三机搜索算子临时显存由 20.88 MiB 降至 3.87 MiB，耗时 21.27→20.85 ms。包括双机分层的完整兜底段为 3.73→24.89 ms，峰值分配均为 116.95 MiB。512 空域全有残余冲突时完整段为 3.69→24.42 ms。稀疏与密集面板的耗时接近，工作区据此调整了块大小，在相同临时元素上限内减少稀疏面板的分块次数。更改后的单卡耗时待实测。任务 37 终止且队列为空后，仅提交恢复拓扑对照的任务 38。[单卡实测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/recovered-single-ppu.5XWTkf/benchmark/ppu.json) · [任务 38](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/recovered-single-ppu.5XWTkf/topology/submit.log)

协商头校准在独立种子 74231 的标准方案 B 2,304 架复验中，由 1,808 到达、488 碰撞变为 1,801/495，两组均为 7 架越界、1 架未到达，近对头为 384/384。校准候选六航路少碰 10 架，但三航路及两类偏航有回退，保留原入场权重。梯度入口继续作为后续研究选项，正式门槛仍待通过。[大样本入场结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/admm-gradient-calibration.hUoph6/incumbent-full-standard.json) · [大样本校准结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/admm-gradient-calibration.hUoph6/calibrated-full-standard.json)

### CPFS 缓存去重与高度观测对齐

任务 38 失败，对照组最后完整落盘为 2,964 次新增更新。故障同期，控制环境在用户 CPFS 根目录和两个实验子目录写入 4 KiB 均返回 ENOSPC。实验目录有 1,700 份可重建 BlueSky 导航缓存，约 28.0 GB。对旧诊断目录 1,001 份缓存逐一校验 SHA-256，将 997 份完全相同的缓存原位替换为硬链接，估计释放 15.3 GiB，无错误。保留全部路径、内容、checkpoint 和日志。原单卡随后同时写入两份 160 MiB 临时数据并 fsync 成功，已逐项校验原协议和入场权重，恢复拓扑回放任务 39。[中断日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/recovered-single-ppu.5XWTkf/topology/control.log) · [缓存核验](/workspace/group_share/adc-perception-pretrain/zhuy25/yry/experiment/Geo_ADMM_RL/reports/navigation-cache-deduplication.json) · [恢复提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/topology-replay-storage-recovered.gZ7vCY/submit.log)

原生四、六航路及正负偏航的 7,772 个逐机决策中，有 164 个仍有待执行的目标高度。保持其他状态不变，将已选高度替换成当前高度时，旧原生观测逐项相同，但有 107 个逐机决策的时序前瞻风险变化超过 0.1。追溯入口发现旧方案 B 将已选高度传入航路计划高度字段，原生方案 A 使用发机时的航路层，属于两引擎的观测口径差异。[原生观测审计](/workspace/group_share/adc-perception-pretrain/zhuy25/yry/experiment/Geo_ADMM_RL/reports/selected-altitude-observability.json)

新增可选观测配置，两套后端都保留发机时的航路层，另外提供本机尚未完成的目标高度差、邻机的已选目标层差和垂向速度。迁移旧方案 B 权重时，将原目标高度权重映射到新单独通道，新邻机列从零开始。真实方案 B 的 45 个控制步内，检查到 14 个有未完成高度指令的逐机状态和 8 次空域重置，已迁移策略的抽查 logits 与控制命令逐项一致。原生烟测也同时记录到非零航路层偏移和尚未完成的飞控目标。49 项回归通过、1 项跳过。新种子原生 0.05 s 六场景开关投影配对正在运行。[方案 B 迁移烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/selected-altitude-smoke.oU0S3q/plan-b.json) · [原生高度烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/selected-altitude-smoke.oU0S3q/plan-a.json) · [迁移闭环对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/selected-altitude-native-transfer.gltDNz)

正常训练入口的旧权重迁移与二次恢复各完成 13 次更新，二次恢复自动继承新高度观测元数据。[旧权重迁移日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/selected-altitude-smoke.oU0S3q/migrate-train/metrics.jsonl) · [自动继承日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/selected-altitude-smoke.oU0S3q/resume-train/metrics.jsonl)

任务 39 已越过两次写入中断点。对照组完成 5,364 次新增更新，实测 16.22 次/s、约 9,864 轮/h，最终 checkpoint 已落盘。拓扑回放组接力，已完成首批 564 次更新。[对照组指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/topology-replay-storage-recovered.gZ7vCY/control/metrics.jsonl) · [拓扑回放组指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/topology-replay-storage-recovered.gZ7vCY/topology-replay/metrics.jsonl)

未来训练遇到 CPFS 空间或配额错误时采用有限退避。指标写入以单条 JSON 行提交并回滚未完成的尾行，checkpoint 用临时文件写完后原位提交。调度脚本保留训练进程的原始退出码，即使可读日志的镜像写入失败。模拟短写测试和真实方案 B 续训均通过，三个周期与最终 checkpoint 可完整回载，50 项回归通过、1 项跳过。后续混合采样快照仅补入写盘恢复，已重新校验原训练输入和代码，待同一 PPU 完成任务 39 后接续。[正常写盘与回载](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/atomic-storage-smoke.DawBTn/verification.json) · [混合采样快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/mixed-projection-after-storage.qzUIc5/launch-command.txt)

高度观测迁移在方案 B 的新开发流量中，旧权重与迁移权重六类终局完全相同，均为到达 228/288、碰撞 58、越界 2。原生同种子开关投影已完成五个场景，正在完成负偏航。[方案 B 入场](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/selected-altitude-native-transfer.gltDNz/plan-b-baseline.json) · [方案 B 迁移](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/selected-altitude-native-transfer.gltDNz/plan-b-migrated.json)

任务 39 两组各完成 5,364 次新增更新。对照与拓扑回放分别约 9,864 和 9,734 轮/h、16.22 和 16.15 更新/s、84.36% 和 83.15% PPU 利用率，均为 33 个训练窗口样本，峰值 6,344 MiB。交叉组 6/8 航路的回放条数由 72,912→85,871，总数同为 139,464。偏航组 8 航路由 24,277→42,223，总数同为 101,916。近对头回放总量同为 101,916。任务现进入标准开发评测。[吞吐和单卡利用率](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/topology-replay-storage-recovered.gZ7vCY/throughput-summary.json)

高度观测的直接迁移原生复验已完成，开发种子 74521–74523，0.05 s 逐帧。216 架中，开投影到达/碰撞/越界为 167/48/1→172/44/0。两类偏航和近对头均改善，三、六航路有回退。关投影到达/碰撞为 164/52→161/55，近对头由 36/36 降至 34/36。直接迁移仅用于定位同策略在两引擎的输入差异，候选权重未晋升。现行 Stage 2 阶段门槛仍未满足。[原生完整配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/selected-altitude-native-transfer.gltDNz/native/comparison.json)

### 拓扑回放筛选与 Stage 2 开关投影选模

任务 39 标准方案 B 以同种子 69931 比较 864 架。开投影的入场、常规续训和拓扑回放依次为 684/176、670/192、685/170 到达/碰撞。拓扑回放组近对头由入场的 144/144 降为 140/144，负偏航越界由 4 增至 9。关投影三组分别为 639/220、620/243、629/230，拓扑回放近对头 138/144，入场为 142/144。按已有规则保留入场权重。独立门槛和原生开发配对继续运行。用相同原生开发种子 70111–70113 补跑入场和拓扑回放的关投影对照。[选模记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/topology-replay-storage-recovered.gZ7vCY/development-selection.json) · [原生消融补跑](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/task39-native-raw.wLhjY8)

Stage 2 周期选模现对每个候选在相同原生 0.05 s 开发种子执行六场景开关投影。在按总体到达、碰撞、成本排序前，候选需保持入场权重的两种偏航到达和关投影的到达、碰撞、近对头表现，开投影近对头继续要求至少 58/60 且不低于入场。正式种子仍按既有协议。原生烟测确认两个模式的种子逐项匹配，且 12 个已退出 worker 的同内容导航缓存通过硬链接共享至 1 个 inode。51 项回归通过、1 项跳过。Stage 2 仍等待 90%/3% 阶段准入。[烟测与缓存核验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage2-paired-validation-smoke.s3D3VO/verification.json)

任务 39 的独立方案 B 开发门槛已完成。入场权重在 2,304 架中到达 1,803（78.26%）、碰撞 494（21.44%）、越界 7，Stage 2 未获准入。单卡任务继续完成原生独立开发评测，混合采样等这一任务终止后再接续。[独立门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/topology-replay-storage-recovered.gZ7vCY/development-gate.json)

### 拓扑回放原生消融与价值排序审计

任务 39 的入场和拓扑回放权重补齐原生关投影，开发种子 70111–70113，0.05 s 逐帧，共 216 架。入场为 164 到达、50 碰撞、2 越界，拓扑回放为 144/68/4。近对头到达从 36/36 降至 28/36。CPU 补跑与原任务已完成的候选逐种子终局一致，完整跨设备核对等待单卡配对结束。保留入场权重。[原生关投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/task39-native-raw.wLhjY8/raw/comparison.json)

两条原生原始轨迹上的四个经配对复跑验证的避碰方案，比较同一状态下原执行动作与避碰第一步的双评论家最小 Q。入场和拓扑回放各有 1/4 个方案的改动飞机 Q 总和上升，实际碰撞标签与执行动作评论家两组均为 0/4。改用历史方案 B 的高度输入后，四组分别为 1/4、1/4、1/4、0/4。该审计比较的是连续干预方案的首步，适用于定位长程价值排序，最终收益仍由实际闭环确认。[四种权重逐案价值](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-paired-critic-audit.9YA86n/summary.json)

已准备带写盘恢复的 144 秒多步回报快照。两组保持相同的回放成熟时点、入场权重与更新预算，只改变目标中的累积时间跨度，训练使用混合采样的后续选模结果。原生四案价值排序将用于审计短期与延迟风险的学习差异。[多步回报快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/return-window-after-storage.eIOtjW/launch-command.txt)

恢复后的多步回报快照通过真实方案 B 双组烟测，各完成 26 次更新和 12 个完整回合，回放都在第 24 个决策步成熟。两组的价值时间跨度分别为 1 步与 24 步，最终 checkpoint 回载成功。[多步回报入口验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/return-window-after-storage.eIOtjW/smoke/verification.json)

任务 39 已完成原生 0.05 s 六场景配对并正常退出。带同一双机兜底的入场、常规、拓扑三组分别为 202 到达/12 碰撞/2 越界、190/22/4、192/18/5，拓扑另有 1 架未到达。候选关投影为常规 161/52/3、拓扑 144/68/4。拓扑组的 CPU 补跑与单卡六场景逐种子终局一致。选模保留入场权重，空出单卡后已提交任务 40。新对照两组使用执行动作评论家，候选在约 25% 的完整回合关闭保护，采样分布和更新预算保持配对。[原生完整配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/topology-replay-storage-recovered.gZ7vCY/native-development/comparison.json) · [混合采样提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/mixed-projection-after-storage.qzUIc5/submit.log)

### 混合保护采样与延续价值目标

任务 40 两组均以 512 空域完成 5,364 次新增更新。常规续训为 9,497 轮/h、16.19 更新/s、84.42% PPU，混合采样为 9,574 轮/h、16.21 更新/s、82.61% PPU。两组各有 33 个 PPU 样本，Torch 峰值均为 5,893 MiB。回放的三类样本条数分别为 139,464、101,916、101,916。混合组在 230,400 个空域决策步里关闭保护 56,942 步（24.71%），该组别在采样期结束 214 个回合，累计记录 2,274 架到达与 1,082 架碰撞。标准开发配对正在运行。[两组训练汇总](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/mixed-projection-after-storage.qzUIc5/training-throughput-summary.json)

复算任务 40 常规续训权重的双目标评论家。两条原生开投影交叉轨迹共 1,100 个冲突图内逐机决策点，按训练配置重放目标投影后，546 个点的动作改变，改变点的最小 Q 平均绝对差为 1.083。两条关投影轨迹对应为 961/1,349 个点和 0.951。该结果测量价值目标对动作编码的敏感性，候选效果由配对闭环评测判断。[归档轨迹、权重校验和逐案结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-target-action-audit.twEYo1/summary.json)

训练入口新增可选 `--projected-target-action`。启用后，保护样本用同一投影函数和后继状态计算延续动作，关保护样本按目标策略原始动作引导。多步回放使用窗口末端的后继几何，终局飞机不引导。混合采样另记评论家实际抽到的开关保护事件。方案 B 和方案 A（CPU worker 重叠采样）分别完成 26 和 24 次更新，checkpoint 均可回载，保护分组计数与汇总一致。52 项回归通过、1 项跳过。任务 40 和待接续的 144 秒配对沿用冻结快照，新选项待独立长跑验证。[双后端验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-target-action-audit.twEYo1/smoke-verification.json)

任务 40 末期权重重跑原生已验证的四个避碰方案的首步排序。原生观测下，改动飞机的最小 Q 总和偏向避碰的方案数由常规组 2/4 变为混合组 0/4。使用匹配方案 B 历史训练的高度观测时，由 0/4 变为 1/4。延续动作复算在混合组归档检查中仍有 547/1,100 个开投影冲突决策点改变。保留标准开发选模和 144 秒回报配对，以判定闭环与长程回报。[四案首步排序](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-target-action-audit.twEYo1/first-action-mixed-projection.json) · [混合组目标动作复算](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-target-action-audit.twEYo1/mixed-summary.json)

### 30,000 更新长训与原生准入

此前任务 40 已结束。方案 B 开发配对保留原入场权重。原生 216 架开投影的入场、常规、混合三组分别到达 196、205、192 架，碰撞 20、10、18 架。两组续训关投影分别到达 134、142，碰撞 82、72。停止旧的短对照系列，144 秒旧配对快照未提交。[方案 B 选模](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/mixed-projection-after-storage.qzUIc5/development-selection.json) · [原生六场景](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/mixed-projection-after-storage.qzUIc5/native-development/comparison.json)

从 `batched-layer-stage1.TQZC8G` 的指定入场权重开始，单卡任务 42 已启动 30,000 次新增更新。使用方案 B 512 空域、40/30/30 混合采样、24 步（144 s）回报、执行动作评论家与投影目标动作、逐机方向保护和保留 10 m 裕度的三机分层。余弦衰减从入场计数重新起算。既有回报和训练动作权重保持一致，整回合关闭投影的采样比例为零。第一任务 41 因微基准快照缺少辅助脚本于训练前退出，保留日志，在单卡空闲后补齐同一脚本恢复。新任务已实测完成首批 1,270 次新增更新，24 步目标，早期约 12.54 更新/s。入场 Q 排序的原生观测和匹配训练观测均为 1/4。[首次启动日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000.dOeYDJ/ppu-layer-benchmark.log) · [恢复提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/submit.log) · [训练进度](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/training/metrics.jsonl) · [初始四案](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/incoming-q.json)

最新 PPU 分块复测使用同一 512×20 面板，20 次同步采样。51 个需查空域的混合组，三机全段中位耗时由此前 24.89 ms 改善到 6.82 ms，单独三机求解为 2.72 ms，双机兜底全段为 3.68 ms。512 个空域均有残余冲突的密集组，三机全段为 24.14 ms。两组紧凑求解与全量求解选项逐项一致。新安全配置同时用于 Stage 1 训练、原生开发评测及 Stage 2 的正式推理和微调。[真实单卡 PPU panel](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/ppu-layer-benchmark.json)

长训每 10,000 次新增更新保存断点并执行四案 Q 排序、原生 0.05 s 六场景评测。首个开发面板为 70111–70113，达标候选在 75121–75123 复验。两组都需超过 202/216 和按同一安全层复跑的入场权重，保持近对头及两种偏航到达。30,000 更新后择优通过者进入方案 A 9×8、2,500 轮原生训练，回航混合场景使用 ±15°，余弦学习率及开发集选模保持开启。正式 12001–12005 按原生 0.05 s 逐帧覆盖微调选中、当前入场和冻结三种权重的开关投影，历史 174/360 对应同一冻结权重和当时的保护配置。新报告会复跑冻结权重并保留两份口径。52 项回归通过、1 项跳过，新增更新数的断点和余弦衰减另经方案 B 烟测验证。[同权重 174/360 历史表](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/frame-audited-holdout.moQMNS/run/comparison.json) · [相对更新断点验真](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-relative-schedule-smoke.vpBG2A/verification.json)

任务 42 首个连续训练窗共 3,270 次新增更新，平均 12.78 更新/s、约 7,206 轮/h；26 个 PPU 样本平均 77.92%，PPU SMI 峰值 3,206 MiB。原生开发 CLI 在 0.05 s 三种子短测通过了三机参数和缓存归档。已退出任务 40 的 90 个导航缓存校验后合并为 1 个 inode，释放约 1.37 GiB，原生对照 JSON 的 SHA256 一致。[首段吞吐](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/early-throughput.json) · [原生配置烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-native-protocol-smoke.2euMOq/native/comparison.json) · [缓存去重核验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/mixed-projection-after-storage.qzUIc5/native-cache-deduplication.json)

### 长程策略的四案梯度与分层监督审计

对任务 42 入场权重执行四案梯度归因。使用已通过原生逐帧反事实验证的首步避碰指令作为参考，在各状态固定八个 Gumbel 抽样种子，分别计算加权评论家、风险、回航、锚定和在线投影模仿的梯度。匹配方案 B 训练观测时，评论家和合并梯度各在 2/4、1/4 案的抽样中位数降低参考动作损失。六航路提前 120 秒和 60 秒两案，评论家沿参考动作方向的梯度分量为 +0.962 和 +1.850，合并后为 −1.812 和 −0.704，冻结策略锚定项分别为 −2.244 和 −2.595。六航路提前 30 秒的锚定则与参考动作同向。以上是四个归档状态的局部梯度证据，候选的闭环结果仍由独立原生开发评测给出。[四案实测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/incoming.json) · [可复跑脚本](/workspace/group_share/adc-perception-pretrain/zhuy25/yry/experiment/Geo_ADMM_RL/eval/bluesky_actor_gradient_audit.py)

另用四条归档原生交叉与偏航轨迹复算当前三机分层。共 87 条逐机兜底改动，其中 9 条为三机相对双机的新动作。训练中用投影前的 ADMM 可行标志筛选教师，以上 87 条均未进入投影模仿。追加的碰撞追踪入口支持当前三机裕度与关闭投影，并记录 checkpoint 哈希、种子和步长，供万步断点复核。52 项回归通过、1 项跳过。[分层教师覆盖率](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/projection-teacher-coverage.json) · [碰撞追踪入口](/workspace/group_share/adc-perception-pretrain/zhuy25/yry/experiment/Geo_ADMM_RL/eval/bluesky_collision_trace.py) · [长训指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/training/metrics.jsonl)

任务 42 已保存第 10,000 次新增更新的 checkpoint，累计更新计数 132,856，余弦学习率为 0.0000930，训练段 12.84 更新/s。相同四案重新排序，原生观测从 1/4 到 4/4，匹配方案 B 历史训练观测从 1/4 到 3/4，达成 Q 诊断目标。相同状态下，万步权重在匹配方案 B 的三个较早干预案中，评论家梯度已与正确动作同向，合并梯度仍反向，方向冲突主要来自锚定和回航损失。[万步权重](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/training/checkpoint-0132856.pt) · [Q 排序](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/update-10000-q.json) · [梯度归因](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/update-10000.json)

原生 0.05 s 六航路种子 70111 逐帧复跑，入场与万步权重均使用同一三机裕度。到达从 8/12 升至 10/12，碰撞从 4 降至 2。剩余接触在第 84 个决策附近，调速可行界不足，K=2 与自适应 K=4 的高度解均预测冲突。离线离散高度诊断在接触前 33.6、63.6、93.6 秒的记录中仍无解，93.6 秒处该对尚未入冲突图。本单种子支持六航路部分改善，Stage 2 由完整六场景配对及独立面板决定。单卡正在完成首个万步原生开发配对。[入场接触轨迹](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/native-incoming-crossing6-70111/collision-events.jsonl) · [万步接触轨迹](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/native-update-10000-crossing6-70111/collision-events.jsonl)

首轮原生开发面板 70111–70113 的入场权重，同一三机保护下为 204/216 到达、10 碰撞、2 越界。三、四、六航路、近对头、正偏航、负偏航的到达/碰撞分别为 36/0、34/2、30/6、36/0、36/0、32/2。本面板以实际复测的 204/216 作为候选的同口径对照。万步权重的三、四航路已经完成，与入场相同，其余场景仍在评测。[入场逐种子数据](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/native-incoming-70111/comparison.json)

万步原生开发三种子配对完成。入场到达/碰撞/越界 204/10/2，万步权重 188/4/24。三航路保持 36/36，四航路保持 34/36，六航路从 30/36 到 34/36、碰撞 6→2，近对头均为 36/36。正偏航到达 36/36→25/36，负偏航 32/36→23/36，候选未获 Stage 2 准入。任务恢复预定的长训，继续跑到 20,000 和 30,000 更新。[万步对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/native-update-10000-70111/comparison.json) · [原生准入结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/milestone-evaluations.json)

正偏航种子 70111 的逐机原生回放与正式开发计数一致。万步权重的 5 架越界飞机在入场权重下均到达。对这 5 架飞机，万步状态中有 381 次航向误差超过 1°，379 次未提出达到 0.25° 死区的航向指令。仅两次提出朝向目标的回航动作，两次均执行。五架飞机越界前最后 120 秒共 100 次带误差决策，全程没有达到死区的回航提议。381 次均在冲突图内，当前训练的自由态回航监督不覆盖这些状态，配置中的冲突态安全回航权重为零。证据定位到该样本的回航提议不足，等后续断点独立复核是否持续。[新权重逐机](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/yaw-plus-10000-70111/trace.json) · [同槽入场逐机](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/yaw-plus-incoming-70111/trace.json) · [越界归因](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/yaw-plus-70111-attribution.json)

### 回航指令许可与闭环时机

用万步权重在原生 0.05 s 的正负偏航种子 70111 回放，原动作和终局与三种子开发表一致。最终越界的飞机在正、负偏航分别缺少 379、476 次有效转向，其中 379、462 次在保持速度和当前选层时通过图内回航监督的方向检查，同时在实际执行速度和高度下可由逐机保护放行。两方向各有 100、120 次缺失发生在前 120 秒，均可通过即时检查。将这些转向注入闭环后，两方向的到达/碰撞/越界由 7/0/5、6/0/6 变为 10/2/0、8/4/0。多数新增转向集中在前 24 秒，新增接触集中在第 85–87 个决策。局部方向许可可定位策略缺失，回航后续的交叉安全仍需检查。[原生反事实与许可统计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/linked-return-counterfactual.ewzUxd/summary.json)

在相同两条开发流量上，保持同一许可条件，延至横偏达到出界边界的一半（10 km）才补足转向。两次均在第 36–37 个决策首次补足。正偏航为 10/12 到达、2 碰撞，负偏航为 12/12 到达且零碰撞。追加两组独立诊断种子 77621–77622 做同权重对照。诊断始终使用 10,000 更新权重，任务 42 继续原定长训。早期任务 16 曾将图内转向与密集风险、对头锚定组合，伴随两种偏航回退，当前先用逐帧反事实检验何时监督回航。[正偏航延后回航](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/linked-return-counterfactual.ewzUxd/yaw_plus15-delayed/trace.json) · [负偏航延后回航](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/linked-return-counterfactual.ewzUxd/yaw_minus15-delayed/trace.json) · [早期联合对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage1-paired-followup.2QyXxG/development-selection.json)

两组独立回航诊断完成。77621–77622 的正负偏航共 48 架，原策略 35 到达、0 碰撞、13 越界，10 km 补回航变为 40 到达、8 碰撞、0 越界。合并 70111 为 48/0/24→62/10/0。仍有碰撞的飞机中，有 4 架在最后一次补转向约 294–312 秒后碰撞，其余 6 架本身没有补转向，说明指标同时受邻机后续轨迹影响。该延时规则未满足碰撞目标，复跑结果仅作诊断。阶段评测保留现行原生安全协议，后续回航监督须考虑汇聚到交叉区时的间隔与对邻机的影响。[三种子六对时机诊断](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/linked-return-counterfactual.ewzUxd/timing-summary.json)

第二个万步断点的原生三种子面板完成。两万步权重到达/碰撞/越界为 176/20/20，三、四、六航路分别到达 32、32、28 架，近对头保持 36/36，双向偏航到达仍为 25/36、23/36。四案排序在原生观测保持 4/4，在方案 B 匹配观测由 3/4 回落到 1/4。三个较早避碰案的合并梯度仍逆向，锚定项持续抵消部分评论家信号。正负偏航种子 70111 的同口径逐机复跑共 11 架越界，861 个超 1° 的决策中有 856 次缺少有效航向提议。两向共 830 次满足图内监督的即时许可。两个检查点均未通过 Stage 2。预定 30,000 更新训练继续，追加调参以前先结束原因复核。[第二个万步原生配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/native-update-20000-70111/comparison.json) · [四案排序](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/update-20000-q.json) · [梯度](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/update-20000.json) · [逐机归因](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/yaw-update-20000-attribution.json)

进一步在 70111、77621 的两向偏航以相同万步权重做反事实。仅在网络未提出转向时，把同一状态下入场权重朝目标的转向提议交给当前保护器。48 架中，基准到达/碰撞/越界为 30/0/18，10 km 规则为 40/8/0，入场回航参考为 41/4/3。保留已学回航时机减轻了碰撞代价，仍高于最终 ≤3% 门槛。两个反事实均单独归档，未作为新推理规则。已退出的 18 个本轮诊断导航缓存按哈希硬链接，释放 267 MiB。[原生同种子参考对照与缓存](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/linked-return-counterfactual.ewzUxd/navigation-reference-summary.json)

### 原生回航的 144 秒回报审计

负偏航 70111 和正偏航 77622 的两个零碰撞回航分支，都在第 37 个决策首次改变动作。复跑逐机终局与先前记录一致，且首次变化前的航迹、奖励和续值逐项相同。在等权配对的 10 架从越界变为到达的飞机中，前 24 步（144 s）累计训练奖励仅 1 架提高，加上目标双评论家续值后有 8 架提高，完整折扣回报 10 架均提高。两组获救飞机的完整回报增量中位数为 +81.15、+85.65，24 步奖励中位数为 −0.96、−0.52。当前在线评论家在两个相同起点，对三个实际改变的逐机动作都给回航更高的 Q，第一评论家逐机差为 +0.55、+0.71、+0.78。该证据区分了早期奖励不足与已有的评论家偏好。审计按任务 42 的方案 B 奖励与引导续值计算，动力学仍是原生 0.05 s，不改变训练与正式评测。[逐机回报与续值](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/yaw-return-audit.mapxZF/return-audit-summary.json) · [负偏航首步 Q](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/yaw-return-audit.mapxZF/yaw_minus15-70111-first-choice/trace.json) · [正偏航首步 Q](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/yaw-return-audit.mapxZF/yaw_plus15-77622-first-choice/trace.json)

在两个相同的回航起点，八个固定策略采样下，两个评论家梯度对已验证动作的中位分量为 +5.11、+1.95。当前冻结策略锚定使总分量降为 −8.63、−1.48。仅将锚定参考换为当前入场权重，保持其余损失后，总分量为 +37.95、+41.92。前两轮六航路诊断也观察到冻结锚定对早期动作的反向分量。工作区已接入按训练流量筛选锚定的参数。新快照固定以入场权重为近对头和偏航参考，交叉空域不加这一锚定，保留相同训练种子、30,000 更新预算、144 秒回报、目标动作、三机裕度和原生选模。55 项回归通过、1 项跳过，方案 B 真实入口烟测完成且三类均采到。当前任务 42 仍在最终原生评测，新快照等待同卡终态。[起点梯度实测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/guarded-actor-gradient-audit.THAV1m/yaw-return-audit.mapxZF/yaw-policy-gradient-summary.json) · [参数烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/scoped-anchor-smoke.EYZgTc/verification.json) · [下一轮快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/launch-command.txt)

任务 42 已达到 30,000 次新增更新，完成 7,070 个训练回合。报告的训练时长 2,337 秒，吞吐 12.84 更新/s、10,891 轮/h。按 checkpoint 时间与训练段时长切出的 232 个 PPU 样本平均 82.19%，设备占用峰值 3,206 MiB。30,000 更新四案排序在原生和匹配方案 B 观测均为 3/4，六场景最终原生配对仍在进行。[全程训练吞吐](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/30k-training-throughput.json) · [最终断点四案](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/update-30000-q.json)

任务 42 已正常退出。30,000 更新最终原生开发面板到达/碰撞/越界为 178/16/22。对照入场 204/10/2，三航路、四航路、近对头保持 36/36、34/36、36/36，六航路为 24/36、12 碰撞，正负偏航为 23/36、25/36。三个断点均未达到原生准入。自动调度核实任务终止、单卡队列为空后，提交了基于上述轨迹与梯度证据的任务 43。调度已显示 RUNNING，实际配置确认 512 空域、24 步回报、投影目标动作、三机裕度，入场权重只锚定 head_on 与 yaw。Stage 2 门槛与正式留出种子保持冻结。[最终原生逐种子表](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/native-update-30000-70111/comparison.json) · [三断点准入](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/milestone-evaluations.json) · [任务 43 提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/submit.log) · [实际配置](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/training/config.json)

### 规模外推和固定 ADMM 轮数评测入口

新增原生规模评测入口，按 30、50、80 架、共同流量种子与给定训练 seed 组运行。它逐种子存储终局、成本、间距、控制批次耗时和实际并发数，汇总观测冲突图的平均/最大节点度、精确最大团和最大连通分量。多 checkpoint 时先按各训练种子汇总流量，再统计模型之间的均值和样本标准差。开发种子 78323 在原生 0.05 s 用开关保护各走两个控制周期，均验证到实际 30/50/80 架并发。大于 64 节点的位运算和小图穷举对照通过。首轮烟测发现的 NumPy 对角接口和最大团闭包错误均已修正，57 项回归通过、1 项跳过。短烟测未用于规模达标判断。[规模评测入口](/workspace/group_share/adc-perception-pretrain/zhuy25/yry/experiment/Geo_ADMM_RL/eval/bluesky_scalability.py) · [三规模原生烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/scalability-native-smoke.ncQbYd/scale/summary.json)

新增固定 ADMM 轮数冷热启动对照。冷启动使用零初始对偶、固定惩罚和均匀移动权重，策略和几何保护保持相同。K=0 直接执行策略，K>0 的连续两层各执行指定 K 次，不追加密集分量轮次，三机离散兜底作为单独配置记录。单种子六航路 K=2 验真，热/冷分别为 8/12、10/12 到达及 4、2 碰撞，不能单凭该种子选择控制配置。实验入口覆盖 K=0/1/2/3/5；跨步通信、丢包和着色调度仍待接入。[固定轮数入口](/workspace/group_share/adc-perception-pretrain/zhuy25/yry/experiment/Geo_ADMM_RL/eval/bluesky_admm_sweep.py) · [K=2 同种子验真](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/admm-cold-start-native-smoke.OtOeQK/verification.json)

K 矩阵的 9 个原生分支全部通过入口烟测。同一入场权重在独立六航路 78324–78326 上固定 K=2，热启动为 32/36 到达、4 碰撞、1,744 次干预，冷启动为 34/36 到达、2 碰撞、388 次干预。两组均使用三机兜底，密集额外轮次关闭。单场景样本支持后续按流量验证学习热启动的适用范围，当前训练和原生准入仍沿用热启动协议。[K 矩阵验真](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/admm-round-matrix-smoke.cvXlFL/grid/summary.json) · [独立逐种子冷热对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/admm-cold-crossing6-dev.nZqRzQ/paired-summary.json)

任务 43 已存下首个 10,000 更新断点，训练吞吐 12.89 更新/s。四案 Q 排序原生 4/4、匹配方案 B 观测 3/4，与任务 42 的万步诊断达标结果一致。初次原生六场景配对正在进行。[任务 43 四案](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/update-10000-q.json) · [训练指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/training/metrics.jsonl)

任务 43 万步第一组原生开发面板 70111–70113 完成。入场为 204 到达、10 碰撞、2 越界，候选为 205/10/1。六航路到达 30→34、碰撞 6→2，近对头与正偏航均保持 36/36，负偏航 32→35。三、四航路到达各下降 2 和 4 架。正负偏航的 70111 逐机补跑均为 12/12，到达与 PPU 计数一致。第一组通过编排中的准入条件，等待 75121–75123 独立复验，Stage 2 尚未启动。首个训练窗用时 775.9 秒，10,064 轮/h，77 个 PPU 样本平均 79.48%，设备峰值 3,342 MiB。[首组原生开发配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/native-update-10000-70111/comparison.json) · [首段吞吐](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/10k-throughput.json) · [偏航逐机](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/scoped-anchor-10000-gradient.DanFwO/yaw-attribution.json)

选择性锚定后，四条交叉轨迹的梯度诊断按实际配置关闭交叉锚定。匹配训练观测下，合并梯度在四航路提前 120 秒、六航路提前 60 秒与参考动作同向，六航路提前 120 秒与 30 秒两案仍反向。原生配对六航路改善与梯度证据相符，其他交叉拓扑仍需独立检验。[四案梯度](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/scoped-anchor-10000-gradient.DanFwO/update-10000.json)

万步权重关投影后，本次补跑的六航路、正偏航、负偏航三类，在原生开发 70111–70113 上分别由入场 22/14、20/16、32/2 到达/碰撞变为 18/18、22/14、20/16，候选负偏航少了 2 架越界。六航路及负偏航的策略本体尚未保住入场表现。关投影负偏航 70111 复跑 4/12 到达、8 碰撞，与三种子评测逐项一致。4 对接触在约 33–35 秒前均无固定水平计划下的单次分层解，其中 3 对在约 93–95 秒前有可行分层但尚未进入当前观测冲突图。结论限于捕获轨迹和离散动作集合。[三场景关投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/scoped-anchor-10000-native-raw.4CS4DX/paired/comparison.json) · [逐帧接触及前瞻](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/scoped-anchor-raw-yaw-contact.2A0buY/contact/collision-events.jsonl)

任务 43 万步独立原生面板 75121–75123 完成。入场为 195 到达、20 碰撞、1 越界，候选为 190/26/0。近对头保持 36/36，正偏航 32→34，负偏航 33→30，六航路 30→28。首组的 +1 到达未随种子复现，万步候选未获 Stage 2 准入。编排记录两个面板为通过/未通过，长训已恢复进入第二段。[独立入场面板](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/native-incoming-75121/comparison.json) · [独立候选面板](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/native-update-10000-75121/comparison.json) · [双面板准入](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/milestone-evaluations.json)

### ADMM 实际丢包和着色调度

0%、20%、50% 丢包现作用于速度与高度层的逐轮双边 ADMM 交换。丢失时沿用该边最后收到的共识和对偶变量，物理几何检查保持完整。随机流按流量种子隔离，统计排除已结束空域。原生 0.05 s 六航路诊断种子 78323，热启动实际丢失比率为 0%、19.87%、50.67%，冷启动为 0%、20.04%、50.25%。零丢包的终局、控制步、干预次数及成本与此前无注入跑法逐项一致。单位是双边交换，非单向物理报文。[原生丢包六组验真](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/admm-packet-loss-native.EYJmmP/verification.json)

着色调度按独立色组顺序更新。同一轮内后续色组使用前面色组刚收到的共识与对偶变量。测量每层末轮原始残差与双边交换次数。固定 K=2、无丢包、同一六航路开发种子，热启动 Jacobi、着色分别到达 8、10 架，碰撞 4、2 架，实际交换 1,914、3,680 次。速度末轮残差均值 0.151→0.059 m/s，高度 5.238→3.691 m。相同 K 的实际交换预算有差别，正式比较按真实交换数一起报告。单种子只用于机制验真。[着色与 Jacobi 原生验真](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/admm-colored-native.QhaBKc/verification.json)

在保持原生物理 0.05 s、控制周期 6 s 的前提下，可按 6/12/18/30 s 独立调度 ADMM。空窗对照可维持已选的速度、航向、高度目标，或执行关闭协调的策略。11 个控制步的原生验真实际调度 11、6、4、3 次。周期和丢包可组合，丢包分母只计计划交换。此处验真的是空窗基线，滚动预测与跨步延迟不计为已验证。整个通信分支 60 项几何回归通过、1 项跳过。[四周期原生样本](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/admm-cadence-native.34cozd/verification.json) · [矩阵参数透传](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/admm-cadence-matrix-smoke.4hiBY5/grid/summary.json)

任务 43 两万步候选在原生 70111–70113 开保护到达/碰撞/越界为 204/8/4，入场 204/10/2；六航路到达 36/36 且零碰撞，近对头保持 36/36。正偏航到达 36→32、碰撞 0→4，负偏航到达均为 32，候选为 0 碰撞、4 越界，因此未过 Stage 2。关投影负偏航相比万步由 20/16/0 改善到 28/6/2，仍低于入场 32/2/2。四案 Q 原生 3/4、匹配训练观测 2/4。在无交叉锚定的实际损失下，六航路提前 120 s 和 60 s 的合并梯度同向，提前 30 s 仍反向。预定 30,000 更新长训已继续。[两万步原生配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/native-update-20000-70111/comparison.json) · [同种子关投影负偏航](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/scoped-anchor-20000-raw-yaw.KF6ivE/paired/comparison.json) · [梯度数据](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/scoped-anchor-10000-gradient.DanFwO/update-20000.json)

两万步正偏航 70111 复现 10 到达、2 碰撞。新增接触对在约 96 秒前有固定水平计划下的离散分层解，尚未进入冲突图；到约 66 秒时在同一动作集合已无解，最后 K=2 和自适应 K=4 均检出冲突。这与此前关投影负偏航的前瞻覆盖缺口一致，现有证据优先指向观测及干预时机。任务 43 继续训练并已超过 21,000 更新。[正偏航公开逐帧诊断](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/scoped-anchor-20000-yaw-contact.NP83vM/trace/collision-events.jsonl)

### 提前分层复放与接触转移

对任务 43 的偏航接触量化观测门限。两万步开保护正偏航 70111 的目标机对在碰撞前 96 s 相距 40.84 km，预测水平最近 3.97 km，预测最小高度差 110.35 m。它满足潜在跨层可达条件，超出 20 km 感知范围。万步关保护负偏航另有三对在约 93–95 s 前已满足名义圆柱冲突，距离 22.75–38.29 km，同样受距离裁剪。两类均有早期离散方案，正偏航的早期方案为保留已有高度目标。[观测门限与方案](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/scoped-anchor-edge-cause.Qwq5UM/early-missed-edges.json)

两条诊断在原生 0.05 s 对比同前缀闭环，分别在第 65 和第 61 个控制步开始保持可行分层至原接触。正偏航原接触 [4,7] 转为 [7,8]，终局同为 10 到达、2 碰撞。负偏航原接触 [0,7] 消失，但增加 [0,3]，总体同为 4 到达、8 碰撞。原始和干预分支分别逐项校验了 65、61 步的状态和控制前缀。延长观测或保住某对飞机的选层可改变碰撞对象，当前证据还需要全空域同时检验。[两条原生反事实](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/scoped-anchor-edge-cause.Qwq5UM/counterfactual-summary.json)

任务 43 的四案在线 Q 按被改动飞机与全空域并列计算。万步原生从逐机 4/4 到全域 3/4，匹配训练观测从 3/4 到 2/4。两万步原生从逐机 3/4 到全域 0/4，匹配训练观测两种口径均为 2/4。后续训练编排保存这两个口径；当前任务 43 继续使用经校验的快照。[万步原始四案](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/update-10000-q.json) · [两万步原始四案](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/update-20000-q.json)

在接触转移的首个实际改令点，正偏航分支的原生最小 Q 全空域差为 −1.19，负偏航为 +0.082，负偏航第一评论家差为 −0.28。两者真实总碰撞均不变。两条改令原生轨迹与前次反事实逐步一致，支持后续保留全空域终局与双评论家分别记录。60 项回归通过、1 项跳过。[同状态逐机与全域 Q](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/scoped-anchor-edge-cause.Qwq5UM/local-vs-team-q.json)

任务 43 按预算达到 30,000 新增更新与 7,083 个新回合，训练段 2,318 秒，12.94 更新/s、11,000 轮/h，230 个 PPU 样本平均 77.67%，设备峰值 3,342 MiB。最终四案逐机/全域排序在原生观测为 4/4、2/4，匹配方案 B 观测为 1/4、1/4。最终原生面板仍在执行。[30,000 更新资源记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/30k-training-throughput.json) · [最终 Q 审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/update-30000-q.json)

论文同流量对照已预查输入。现存三套 Figure 4 航路修正数组逐 Case 校验为同一哈希，分别有 3、4、6 条路由，包含起终点和初航向。历史 SARL Case 3 的设置为原生物理 0.05 s、决策 6 s，已有飞机结果文件不含发机时刻和初值，后续须在统一流量重放中补齐发机清单。当前 Geo 开发面板的合成航路不并入正式 Case 对照。[Figure 4 路由来源](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/figure4-baseline-input-audit.uye1x3/route-provenance.json)

任务 43 最终权重在首组 70111–70113 到达 208/216（96.30%）、碰撞 6、越界 2，入场为 204/10/2。六航路 30→34，碰撞 6→2；近对头与正偏航保持 36/36，负偏航 32→34。四航路 34→32。首组通过，独立 75121–75123 正在运行。[最终首组配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/native-update-30000-70111/comparison.json)

任务 43 最终独立面板 75121–75123 到达/碰撞/越界为 186/30/0，入场 195/20/1。六航路 30→22，近对头 36→36，正偏航 32→32，负偏航 33→32。三个检查点均未稳定通过，Stage 2 未启动。最终关投影 70111–70113 在六航路、负偏航分别为 14/36 与 26/36，到达较入场 22/36、32/36 均下降。原 144 秒目标和锚定训练的这一方向按单卡止损完成；当前追加证据集中在全空域收益与真实闭环接触转移。[最终独立面板](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/native-update-30000-75121/comparison.json) · [三断点准入](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/milestone-evaluations.json) · [最终关投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/scoped-anchor-30000-native-raw.uYb9Ji/paired/comparison.json)

### 30,000 更新准入汇总与跨流量价值审计

两次 Stage 1 长训从同一入场权重启动，使用 512 空域、24 决策步（144 s）回报、投影目标动作、余弦学习率和 10 m 三机分层裕度。任务 43 额外锚定近对头和偏航。三次检查分别在新增 10,000、20,000、30,000 更新完成，Q 列为原生四案中干预计划所涉及飞机的排序。原生开发数字为到达/碰撞，每面板 216 架。仅首面板通过时追加独立面板。当前准入程序按权重 SHA 和逐种子终局复算，全部断点判定与归档一致。SOFY 任务 43 正常退出，队列为空。[原配置任务](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/guarded-144s-30000-recovery.YO2YL9/milestone-evaluations.json) · [选择性锚定任务](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/milestone-evaluations.json)

| 任务 | 新增更新 | 首步 Q | 70111–70113 | 75121–75123 |
| --- | ---: | ---: | ---: | ---: |
| 入场权重，同安全层 | 0 | 1/4 | 204/10 | 195/20 |
| 42 | 10,000 | 4/4 | 188/4 | — |
| 42 | 20,000 | 4/4 | 176/20 | — |
| 42 | 30,000 | 3/4 | 178/16 | — |
| 43 | 10,000 | 4/4 | 205/10 | 190/26 |
| 43 | 20,000 | 3/4 | 204/8 | — |
| 43 | 30,000 | 4/4 | 208/6 | 186/30 |

任务 43 的最终权重在同一套三机安全层和原生 0.05 s 下按场景配对。每个场景列有 36 架，数值为到达/碰撞。原历史门槛为 202/216；新版安全层下同权重首面板为 204/216。独立面板出现六航路和负偏航回退，Stage 2 尚未被触发，12001–12005 的新正式对照尚未运行。[首面板逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/native-update-30000-70111/comparison.json) · [独立面板逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/incumbent-scoped-anchor-30000.EbStil/native-update-30000-75121/comparison.json)

| 0.05 s 场景 | 首面板入场 | 首面板 30k | 独立面板入场 | 独立面板 30k |
| --- | ---: | ---: | ---: | ---: |
| 三航路 | 36/0 | 36/0 | 36/0 | 34/2 |
| 四航路 | 34/2 | 32/4 | 28/8 | 30/6 |
| 六航路 | 30/6 | 34/2 | 30/6 | 22/14 |
| 近对头 | 36/0 | 36/0 | 36/0 | 36/0 |
| +15° 偏航 | 36/0 | 36/0 | 32/4 | 32/4 |
| −15° 偏航 | 32/2 | 34/0 | 33/2 | 32/4 |
| 合计 | 204/10 | 208/6 | 195/20 | 186/30 |

补充价值审计汇集 6 组独立场景流量，含 8 个到达增加且各项失败数均未增加的干预和 2 个碰撞对象转移、终局计数相同的对照。从同一原生状态比较原命令与干预首条命令，六航路同种子变体一起留出。以所有在场飞机、双评论家平均 Q 作诊断，入场与任务 43 末期分别偏好 7/8 和 5/8 个净收益干预。按 TD3 的双 Q 最小值则是 4/8 与 5/8，其中独立六航路 67231 从 −0.055 变为 −0.321。保留原评论家编码器，按流量分组计算输出头配对损失的留一梯度，固定参数步幅 L2=0.01，两个权重的原生边际各有 6/8 同方向，线性化后均无错排翻转；匹配方案 B 高度输入时分别为 6/8 与 4/8 同方向。正偏航接触转移在末期权重的最小 Q 仍得到 +4.55 的全域逐机平均分差，真实碰撞数未减少。样本均来自开发流量，首步排序使用全程干预的终局作标签。此项未训练或发布新权重。几何回归 60 通过、1 跳过。[逐案观测、配对梯度和出处](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-team-preference-audit.BWMFlh/summary.json)

### 原生四案首步与持续干预复放

用原四案中两组独立原生流量（66314 四航路、66313 六航路）共复放 30 个分支，物理和接触事件都按 0.05 s 逐帧，命令间隔为 6 s。每案对照完整干预、只改首步、跳过首步后持续干预和固定长度干预。开始改令前逐项对齐活动槽号、经纬度、高度、空速、航向及每步终局计数。对照窗口之外使用同一归档命令流。完整干预均复现原来的 12/12 到达、零碰撞；只改首步的 4 案均保留原终局计数，跳过首步后继续干预的 4 案均复现完整干预终局。[30 分支汇总与逐帧接触](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-first-action-timing.2p3jOw/summary.json)

| 原生回放 | 只改首步到达/碰撞 | 跳过首步后到达/碰撞 | 已测最短零碰撞干预窗口 |
| --- | ---: | ---: | ---: |
| 四航路，约提前 120 s | 8/4 | 12/0 | 144 s |
| 六航路，约提前 120 s | 10/2 | 12/0 | 48 s |
| 六航路，约提前 60 s | 10/2 | 12/0 | 12 s |
| 六航路，约提前 30 s | 10/2 | 12/0 | 24 s |

四航路窗口为 24、48、96 s 时均为 10 到达、2 碰撞，144 和 192 s 都为 12/0。窗口长度是被测试的计划生效区间，实际改令次数另记。改动持续时间和后续命令决定了这些配对中的最终收益。首步 Q 排序仍保留用于跟踪，整段干预成功只作为该序列的配对结果。该诊断使用两个开发流量及归档控制，未测纯策略闭环，也未使用 12001–12005。

### 源策略闭环复验与交叉流量时间相关探索

用原始入场权重重新推理四案控制，从起点复现归档指令和终局。每条干预在同一原生 0.05 s 状态开始，未指定机动的轴继续采纳策略当前输出，窗口结束后继续正常闭环。完成 20 个分支，包含源双机层与切换至 10 m 三机层的配对。源策略的跟进控制使六航路提前约 120 s 和 30 s 两案只改首步就从 10 到达/2 碰撞变为 12/0，提前约 60 s 首步 10/2，持续 12 s 后为 12/0。四航路首步维持 8/4，持续 144 s 为 10/2。测试从干预点切换三机层后，各组终局与对应双机层相同。结论来自四案、两组独立开发流量。[原生闭环汇总及逐机回报](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-feedback-interventions.kvzZ0Q/summary.json)

按原生环境结算、任务 43 的奖励塑形、0.99 折扣计算整空域回报，四案有界连续干预的首步回报差均为负，24 步回报差依次为 +88.79、+73.11、+89.39、+94.17。只改首步的两条零碰撞分支其 24 步差为 +85.44 和 +94.66，两条未改善分支分别为 −5.28 和 −1.29。此计算使用原生终局和动作成本，无目标 Q 引导。任务 43 末期原四案局部首步 Q 为 4/4，两条真实首步正例在全空域原生 Q 下仅 1/2，余两案分开记录实际终局。长训编排可同时记录原四案排序和同状态单动作闭环结果。

新增方案 B 全分布选项 `--crossing-exploration-steps`，对交叉流量同一飞机的 Gumbel 路由扰动和连续控制噪声复用多步，策略和保护仍每 6 s 读取新状态。飞机终局或空域复位时清除该身份的探索状态。近对头与偏航继续使用原随机流，给定相同状态时动作逐项相同；目标动作投影和 24 步回报沿用现有入口。窗口设为 8（48 s）时，12 空域、20 架上限的真实方案 B 烟测完成 14 次更新与 12 次超时复位，采样保留交叉/近对头/偏航三类。61 项回归通过、1 项跳过。该短烟测只证明训练和复位接口。[方案 B 日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/crossing-exploration-smoke.eVuu4w/training/metrics.jsonl)

冻结入场权重，在两个新种子各配对 72 个方案 B 空域，每个空域只结算首个完整回合。两组交叉相邻控制步的原始动作类型重复率为 66.3%→92.3% 和 65.7%→92.6%，交付命令的平均 L1 变化为 0.187→0.062 和 0.186→0.058。近对头、偏航的逐空域终局和动作统计一致。交叉到达/碰撞分别为 357/65→361/61 和 357/44→351/50。两组 6/8 航路交叉合计到达 331→333、碰撞 91→89。冻结结果用于确认机制，候选效果由长训后的原生独立面板判断。[固定权重 144 空域配对与逐空域记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/crossing-noise-frozen-pair.k6Ywx3/summary.json)

任务 44 已提交并在原已授权单卡上运行。源码和输入 SHA 均已核验，入场为原 `batched-layer-stage1.TQZC8G` 权重；同任务 43 的目标、采样配比与近对头/偏航锚定，只为交叉组增加 8 步探索噪声复用。实际训练配置确认 512 空域、24 步回报、投影目标动作、10 m 三机层、30,000 更新余弦衰减。首份指标记录 270 次新增更新，11.44 更新/s。每万步自动执行原生四案 Q（同时附单动作闭环证据）和六场景开发配对，两个开发面板均通过才自动启动 2,500 轮方案 A。冻结权重、开发种子与正式留出协议保留。[任务 44 提交记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/crossing-persistent-30000.Ffjkur/submit.log) · [实际配置](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/crossing-persistent-30000.Ffjkur/training/config.json) · [实时指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/crossing-persistent-30000.Ffjkur/training/metrics.jsonl)

任务 44 已完成首个 10,000 更新断点，新增 2,175 轮，训练阶段用时 778.0 s，吞吐 12.85 更新/s、10,064 轮/h。训练窗的 77 个 PPU 样本平均利用率 75.51%，设备采样峰值 3,342 MiB。原生首步 Q 的干预相关飞机与全空域排序分别由 1/4、2/4 变为 3/4、3/4，匹配方案 B 输入的两个计数均为 1/4。两条被源策略闭环验为实际有益的单动作，原生全域 Q 由 1/2 变为 2/2。六场景开发配对已开始。[首段性能归档](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/crossing-persistent-30000.Ffjkur/training-windows.json) · [万步逐案 Q](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/crossing-persistent-30000.Ffjkur/update-10000-q.json)

万步权重在原生 0.05 s 关投影开发集的六航路到达/碰撞为 26/10，入场 22/14；负偏航为 20/16，入场 32/2，入场另有 2 架越界。每组使用种子 70111–70113 的 36 架完整结算。六航路在 70111 与 70112 改善，在 70113 回退。负偏航三种子碰撞均增加。该诊断保留两类结果，单卡六场景开保护配对继续执行。[关投影逐种子结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/persistent-10000-native-raw.zI1PTz/paired/comparison.json)

关投影逐帧复跑复现了六航路 70112 的两对接触消失与负偏航 70111 的三对新增接触。六航路两对在原接触所在控制步起点的实际垂直间隔由 63.86/73.01 m 变为 121.92/213.36 m，水平间隔同时缩短，和分层改善相符。负偏航三对里，一对目标层间隔从 91.44 m 缩至 37.59 m，另一对实际层间隔从 195.88 m 降至 91.44 m，第三对的水平间隔由 14.48 km 缩至 5.16 km，实际高度差由 91.98 m 降至 42.86 m。该对照对齐同一仿真控制时刻，尚未隔离单一控制轴的因果贡献。[两组逐帧轨迹与同刻状态对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/persistent-10000-native-raw.zI1PTz/paired-contact-differences.json)

任务 44 的万步开保护原生面板已完整结算。到达/碰撞由入场 204/10 变为 208/8，候选全部 216 架均为到达或碰撞。三航路 36/0、四航路 34/2、近对头 36/0 对照相同。六航路从 30/6 提升到 34/2，负偏航从 32/2 到 34/2，并消除 2 架越界。正偏航从 36/0 回退至 34/2，按不牺牲偏航条件未获 Stage 2 准入。单卡已继续下一训练段，读到 10,270 次新增更新。[原生六场景逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/crossing-persistent-30000.Ffjkur/native-update-10000-70111/comparison.json) · [准入记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/crossing-persistent-30000.Ffjkur/milestone-evaluations.json)

### 万步负偏航闭环归因

三个关投影负偏航开发种子做了 36 个原生 0.05 s 闭环分支。各分支每 6 s 在自己的当前状态重算两个策略，未使用归档后续指令。两个完整父策略逐种子复现原结果，入场 32 到达、2 碰撞、2 越界，万步 20/16/0。只换回入场路由、使用万步动作幅度后为 29/6/1；只换幅度为 24/12/0。把路由替换限制在现有锚定规则覆盖的已连边或 LNAV 状态，三个种子的终局均与全程路由替换一致。只在未连边且关 LNAV 时替换，合计为 22/14/0。[闭环逐种子矩阵](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/persistent-10000-yaw-components.vShvCU/summary.json)

直接交换已编码命令时，单换速度仍为 20/16/0，单换高度为 28/8/0，单换航向为 23/10/3。同时换高度和航向逐种子复现入场 32/2/2。这些单轴和双轴分支可以同时执行不同父策略的多轴命令，作为因果诊断。保留单一动作模式的路由交换已经恢复 9 架到达。六场景训练和 Stage 2 准入继续使用原协议。

在两个偏航方向的 6 条原生轨迹上分别冻结前 64 个控制步的 9 个时点，用真实万步权重、训练损失及匹配方案 B 的高度输入做梯度复算。负偏航孤立关 LNAV 的 154 个逐机状态中，入场与候选路由只在 4 个状态不同。向入场路由靠拢的局部一阶方向在负偏航三个种子的全部 24 次采样均为正，主要来自导航项。正偏航对应方向在第一、第二个种子为负、第三个为正。该复算衡量冻结早期状态上的梯度投影，与闭环范围替换结合后，优先排查混合训练怎样改变已锚定的交互状态。61 项几何回归通过、1 项跳过。[保存的状态与负偏航梯度](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/persistent-10000-yaw-components.vShvCU/gradient-yaw_minus15-70111/gradients.json) · [正偏航梯度](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/persistent-10000-yaw-components.vShvCU/gradient-yaw_plus15-70111/gradients.json)

任务 44 的第二段已跑满 10,000 更新并保存两万步断点。本段新增 2,484 轮、777.2 s，12.87 更新/s、11,506 轮/h，77 个卡采样平均利用率 79.14%。原生局部/全域 Q 为 3/4、1/4，匹配方案 B 输入为 2/4、1/4。开投影六场景开发评测和关投影重点场景的配对均已启动。[训练窗口](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/crossing-persistent-30000.Ffjkur/training-windows.json) · [两万步四案排序](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/crossing-persistent-30000.Ffjkur/update-20000-q.json)

两万步开保护的原生开发面板由入场 204 到达/10 碰撞变为 200/16。六航路回到 30/6，四航路降至 30/6，近对头仍为 36/0，两个偏航均为 34/2。该断点未获 Stage 2 准入，任务进入最后 10,000 更新。关投影的三个开发场景中，由入场到候选的到达/碰撞分别为六航路 22/14→14/22、正偏航 20/16→22/14、负偏航 32/2→22/14，负偏航入场另有 2 架越界。[开投影配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/crossing-persistent-30000.Ffjkur/native-update-20000-70111/comparison.json) · [关投影配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/persistent-20000-native-raw.C7gc8B/paired/comparison.json)

固定万步权重在两个新方案 B 混合种子各跑 36 个空域。在每个种子的 8 个控制时点各做 3 次策略采样，按流量分解实际 actor 损失，逐项核对损失及梯度加总与正式训练相同。以降低现有锚定覆盖内偏航飞机的入场路由 KL 为正方向，当前总梯度在每个种子均为 11/24 正向。线性去掉交叉梯度为各 10/24，偏航锚定梯度加倍为 11/24、12/24，去掉偏航导航项为 18/24、17/24。只剔除偏航导航梯度与入场路由保持相冲突的方向，一阶估计为 19/24、18/24，按时点分组为 7/8、6/8。此结果将后续诊断聚焦于同一偏航组内“自由回航”和“交互状态路由保持”的梯度冲突。该计算在固定权重上使用局部 SGD 方向，未修改 AdamW 训练或生成新权重。[混合梯度分解](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/persistent-10000-yaw-components.vShvCU/mixed-gradient-summary.json) · [导航梯度局部投影复算](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/persistent-10000-yaw-components.vShvCU/navigation-gradient-local-projection.json)

### 交叉持续探索结案与 AdamW 偏航保持

任务 44 已正常结束，30,000 更新共新增 7,101 轮，训练用时 2,333.7 s，12.85 更新/s、10,954 轮/h。232 个 PPU 样本平均利用率 78.59%，设备采样峰值 3,342 MiB。最终原生四案局部/全域 Q 为 3/4、2/4，匹配方案 B 输入为 3/4、3/4。原生开投影三万步为 206 到达、10 碰撞，入场为 204 到达、10 碰撞、2 越界。六航路从 30/6 到 36/0，四航路从 34/2 到 30/6，正偏航从 36/0 到 34/2，三个断点均未获 Stage 2 准入。关投影三万步的六航路为 10/26、正偏航 22/14、负偏航 20/16，入场分别为 22/14、20/16、32/2，负偏航入场另有 2 架越界。[完整训练窗口](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/crossing-persistent-30000.Ffjkur/training-windows.json) · [原生最终配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/crossing-persistent-30000.Ffjkur/native-update-30000-70111/comparison.json) · [关投影原生复测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/persistent-30000-native-raw.dRUgPF/paired/comparison.json)

用两组各 8 个混合流量控制时点、每时点 2 次路由抽样，回载万步与两万步的真实 AdamW 状态，比较一步更新。每次在一个交通种子上计算梯度，在另一种子同刻状态上复测。另一流量种子上偏航入场路由 KL 下降次数，普通更新在两断点为 14/32 和 5/32。只在梯度中去除导航冲突分量为 25/32 和 15/32。对实际 AdamW 步长反射其增加 KL 的分量为 25/32 和 29/32，按控制时点分别为 13/16 和 16/16。在复算一开始冻结回放，两个断点复用万步策略生成的同一经验。该证据衡量路由保持及跨流量方向，闭环效果仍需完整训练验证。[实际 AdamW 复算](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/persistent-adamw-yaw.9CfXnB/summary.json)

训练新增可选 `--yaw-routing-retention`。保留现有 actor 损失和 AdamW 动量统计，在 AdamW 更新后只反射会增加入场路由 KL 的参数步长分量。保持目标只计算偏航类别中已连边或启用 LNAV 的飞机。当 KL 小于浮点精度时忽略残差，避免相同入场权重受数值噪声牵引。几何回归 62 项通过、1 项跳过。真实方案 B 恢复烟测验证 24 步回放和投影目标动作，36 空域的 5 次 actor 更新共访问 276 个符合条件的逐机状态并真实反射 2 次。[集成烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/yaw-routing-retention-smoke.ktCoJu/linked-yaw/metrics.jsonl)

任务 45 在任务 44 正常终止、SOFY 队列为空后已在原授权单卡接力。初始权重、混合流量、8 步交叉探索和 30,000 更新预算与任务 44 相同，只启用 `--yaw-routing-retention`。预检的入场摘要哈希与四案逐项符号一致，控制环境与原节点 Q 数值最大差为 3.815e-5。初始实测完成 770 更新、385 次 actor 更新，其中反射 261 次，覆盖 30,193 个累计合格偏航逐机状态，初期吞吐 11.63 更新/s。每万步继续按原生四案 Q 与六场景配对诊断。[SOFY 提交](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/yaw-retention-30000.285qhh/submit.log) · [实际训练配置](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/yaw-retention-30000.285qhh/training/config.json) · [实时指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/yaw-retention-30000.285qhh/training/metrics.jsonl)

### Exp 4 原生动态扰动入口

新增原生 `0.05 s` 扰动评测入口，使用独立随机流固定正常航路、插入时刻、非合作飞机和速度噪声。可分别运行中途插入、10% 非合作飞机拒绝实际交付命令、速度增量乘性 ±5% 噪声，并支持同时开启。非合作飞机保持选定航向、高度和 CAS。原生开发种子 87041 对照验真覆盖五臂，所有正常航路哈希一致，速度噪声序列哈希一致，插入均恰在 126 s，独立与混合非合作臂选择同 2 架飞机并维持选定飞控目标。短程只验真扰动与指标入口，未用于安全率结论或 Stage 1 选模。[五臂验真](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/exp4-native-disturbance-smoke.pcJ1yO/verification.json)

任务 45 首个万步断点已保存。原生局部/全域 Q 为 4/4、2/4，匹配方案 B 输入为 2/4、1/4。本段新增 2,175 轮，训练 826.0 s，12.11 更新/s、9,480 轮/h，82 个卡采样平均利用率 79.10%。5,000 次 actor 更新中 3,660 次落入反射分支，覆盖 451,737 个累计符合条件的偏航逐机状态。六场景开投影和三个关键场景关投影开发配对已启动。[原生四案](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/yaw-retention-30000.285qhh/update-10000-q.json) · [单卡窗口](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/yaw-retention-30000.285qhh/training-windows.json)

任务 45 万步关投影开发集采用相同 70111–70113。六航路为 14 到达、22 碰撞，入场 22/14。负偏航为 14/22，入场 32/2 另有 2 架越界。正偏航为 21 到达、14 碰撞、1 越界，入场 20/16/0。冻结状态上的优化器方向改善尚未转化为这些关投影闭环的安全收益。持续按同一协议检查后续断点。[三个场景逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/yaw-retention-10000-native-raw.QfkN8z/paired/comparison.json)

任务 45 的万步开投影开发面板完成，入场为 204 到达、10 碰撞、2 越界，新权重为 197/16/3。六航路从 30/6 变为 28/8，近对头从 36/0 变为 34/2。正偏航维持 36/0，负偏航到达从 32 增至 33。万步断点未获 Stage 2 准入，单卡继续下一段。

在任务 44 归档的相同 170 个原生负偏航锚定逐机状态上，由任务 44 万步到任务 45 万步，入场路由一致数从 156 降至 126，平均 KL 从 0.081 升至 0.308。在归档方案 B 的 1,706 个锚定偏航状态上，一致数从 1,460 降到 1,408，KL 从 0.072 至 0.083。比较使用相同状态及同一冻结参考，可分离状态构成差异。单步复算显示的局部保持收益尚未延续到实际万步权重。[开投影六场景](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/yaw-retention-30000.285qhh/native-update-10000-70111/comparison.json) · [同状态路由对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/yaw-retention-10000-native-raw.QfkN8z/retention-observation-audit.json)

进一步用任务 45 万步权重复算 16 个归档方案 B 锚定梯度，与同一权重在两类原生偏航的参考路由梯度比较。以第 32 控制步分期，较晚的负/正偏航梯度分别只有 7/16、1/16 与方案 B 同向，余弦中位数为 −0.131、−0.194。原生同一批状态仅切换高度输入时，四个分期的梯度余弦为 0.985–1.000。该诊断指向当前样本下的状态分布及梯度迁移问题。未据此改变运行中的训练；最新读取为 11,770 更新。[双精度梯度对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/yaw-retention-10000-native-raw.QfkN8z/retention-gradient-views.json) · [实时训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/yaw-retention-30000.285qhh/training/metrics.jsonl)

### Exp 5 控制步延迟与空窗滚动规划

新增 `--admm-delay-steps 0/1/2`。速度、高度、主求解和回退分别维护消息队列，按 6 s 控制步成熟，抑制跨飞机退出与空域终局的旧状态。高度共识以绝对目标传输，接收时换算到当前高度坐标。固定 K=2、原生六航路开发种子 78323 的冷热六臂均完成，显式零延迟逐项复现即时路径。一、两步延迟的热启动分别消费 817、859 个旧边状态。两步延迟与 12 s 调度、着色、20% 丢包叠加时，95 次协调，丢失 473/2,308 次交换，消费 532 个旧边状态，最小、平均和最大龄期为 2、2.011、8 控制步。物理间隔仍按原生逐帧检查。[延迟与组合验真](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/admm-native-latency.UNJsMy/verification.json)

新增 `--admm-gap-policy mpc`。空窗步用当前感测的速度、航向及已选高度，逐机比较 120 s 有限候选的第一条命令，之后保持目标。用分段爬升和连续水平重叠预测筛掉新增接触对的组合，下一空窗再按新状态求解。该局部近似模型只用于选动作，终局仍由 BlueSky 0.05 s 结算。同一开发流量的 6 s 无空窗路径完全一致。12/18/30 s 实际额外改令 7/9/14 次。18 s 的原生到达/碰撞由 7/4 到 11/0，12 s 与 30 s 终局一致，三个有空窗的周期命令成本均增加。此处是单种子、190 决策上限的机制验真。通信矩阵的按模型结果同时汇总报文数、丢包、按求解次数加权的残差、按实际消费数加权的龄期和 MPC 计数。[四周期配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/admm-native-latency.UNJsMy/verification.json)

最终几何回归 65 项通过、1 项跳过。本轮通信验真的 21 个可重建导航缓存路径逐组按 SHA-256 校验，物理 inode 从 3 个合并为 1 个，保留全部路径。[缓存验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/admm-native-latency.UNJsMy/navdata-deduplication.json)

### Figure 4 同流量输入审计

从历史名义评测的 `Ep1.scn` 抽取 Case 1/2/3 × 5 个流量种子的 `CRE` 和 `ADDWPT`，指令保留呼号、发机时刻、起终点、初航向、初始高度和空速。60 条历史场景日志逐架绑回 Figure 4 航路表，60 条均含 50 架且覆盖对应的全部航路。Case 1 的历史配置使用 upstream 航路，实际起终点逐架匹配修订表。15 份统一清单取对应 Hold+KSS 的发机记录。历史 SARL 开关 KSS 的清单与 Hold+KSS 逐架相同，历史 Hold 关 KSS 在 Case 2/12004、Case 3/12003、Case 3/12004 的首次发机差分别为 60、240、180 s。正式消融可用固定清单消除该差异。[逐 Case 和种子检查](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/figure4-shared-flow.DwYtaF/manifests/comparison.json)

以种子 12001 对三个 Case 在非 fast 的 `bs.sim.step()` 上复放至各自最后一次发机，分别完整核对 50 架的释放帧、经纬度、目的点、初航向、初始高度与 CAS。空场起点显式切换到 OPERATE，使第 6 秒定时发机按原生 0.05 s 时钟触发。历史日志的时间戳只保留整秒，复放将记录时刻作为共同输入。历史 SARL 的 2 km 横向阈值及控制回调采样与当前 Geo 的 5 km 横向阈值及逐帧采样存在差异，历史终局数未纳入论文配对。此次仅验证发机，未运行策略比较，也未使用该流量选择 Stage 1 权重。[Case 1 发机复验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/figure4-shared-flow.DwYtaF/case1-native-all-releases.json) · [Case 2](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/figure4-shared-flow.DwYtaF/case2-native-all-releases.json) · [Case 3](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/figure4-shared-flow.DwYtaF/case3-native-all-releases.json)

### 任务 45 两万步与原生准入复核

两万步断点完成，第二段新增 2,433 轮，实测 12.22 更新/s、10,707 轮/h，82 个训练窗样本的平均 PPU 利用率为 77.83%。原生四案改令飞机/全空域 Q 排序均为 3/4，匹配方案 B 输入分别为 3/4、2/4。[训练窗口](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/yaw-retention-30000.285qhh/training-windows.json) · [两万步四案](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/yaw-retention-30000.285qhh/update-20000-q.json)

原生 0.05 s 的 70111–70113 开投影开发面板为 199 到达、16 碰撞、1 越界，入场为 204/10/2。两万步三、四、六航路到达/碰撞为 34/2、32/4、30/6，近对头与正偏航均为 36/0，负偏航为 31/4，低于入场 32/2。关投影同种子六航路、正偏航和负偏航分别为 16/20、20/16、24/12，入场分别为 22/14、20/16、32/2，入场负偏航另有 2 架越界。两万步未过 Stage 2，继续三万步训练。[六场景开投影配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/yaw-retention-30000.285qhh/native-update-20000-70111/comparison.json) · [三场景关投影配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/yaw-retention-20000-native-raw.jdDWdi/paired/comparison.json)

Stage 2 准入按两个独立原生面板各超过 202/216 执行，同时逐组保持同种子入场的近对头与正负偏航到达数。方案 B 大开发面板不参与这条准入。归档任务的现有判定逐一复算一致，回归为 66 通过、1 跳过。运行中的任务 45 按提交时校验过的源码执行，后续断点同时按更新后的固定门槛复核。

用任务 45 两万步权重和冻结状态，对同一评论家比较原始、投影动作处的梯度。在负偏航原生的 15 个发生改令的控制批里，原始动作梯度沿改令方向的一阶符号有 7 个与精确 Q 差相反；正偏航为 3/17，六航路为 0/10。对应方案 B 偏航样本为 3/16。原生样本的高度通道匹配 Stage 1。此结果只涉及已归档状态与当前评论家，未改变优化器或用作准入。[原始与投影动作 Q 归因](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/projected-critic-input-audit.p7oJxR/update-20000.json)

### 任务 45 三万步结算

SOFY 任务 45 正常结束，退出码为 0。方案 B 在 512 空域从 `batched-layer-stage1.TQZC8G` 跑满 30,000 次新增更新，使用 24 步即 144 s 回报、投影目标动作、余弦学习率和 10 m 三机分层。三个断点均已落盘。训练窗合计 2,462.3 s、7,027 个新增回合，实测 12.18 更新/s、10,274 轮/h，246 个 PPU 样本平均利用率 78.17%，设备显存峰值 3,350 MiB。[训练窗](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/yaw-retention-30000.285qhh/training-windows.json) · [断点及准入](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/yaw-retention-30000.285qhh/milestone-evaluations.json)

原生四案中，改令飞机的首步 Q 排序从入场 1/4 到万步 4/4、两万步 3/4、三万步 2/4。全空域汇总依次为 2/4、2/4、3/4、2/4。三万步方案 B 匹配输入为改令飞机 3/4、全空域 2/4。[三万步逐案分差](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/yaw-retention-30000.285qhh/update-30000-q.json)

原生 BlueSky 0.05 s 逐帧开发集使用种子 70111–70113，每个场景 36 架。表内为到达/碰撞。

| 场景 | 入场开投影 | 10,000 步开投影 | 20,000 步开投影 | 30,000 步开投影 | 入场关投影 | 30,000 步关投影 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 三航路 | 36/0 | 36/0 | 34/2 | 36/0 | 26/10 | 26/10 |
| 四航路 | 34/2 | 30/6 | 32/4 | 32/4 | 28/8 | 18/18 |
| 六航路 | 30/6 | 28/8 | 30/6 | 32/4 | 22/14 | 18/18 |
| 近对头 | 36/0 | 34/2 | 36/0 | 36/0 | 36/0 | 32/4 |
| +15° 偏航 | 36/0 | 36/0 | 36/0 | 34/2 | 20/16 | 24/12 |
| −15° 偏航 | 32/2 | 33/0 | 31/4 | 35/0 | 32/2 | 20/16 |
| 合计，216 架 | 204/10 | 197/16 | 199/16 | 205/10 | 164/50 | 138/78 |

三万步开投影另有 1 架越界，入场为 2 架。三万步达到 205/216，正偏航少 2 架到达，因此按固定门槛复算仍未获 Stage 2 准入。原生 9×8 的 2,500 轮和 12001–12005 的 360 架保留种子配对继续等待合格权重。历史 174/360 是此前的保留集冻结基线，202/216 是入场开发成绩。[三万步开投影逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/yaw-retention-30000.285qhh/native-update-30000-70111/comparison.json) · [六场景关投影逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/yaw-retention-30000-native-raw.kEgy3n/paired/comparison.json)

用同一原生 0.05 s 轨迹重放种子 70111。六航路入场接触对 [1,6] 与 [5,10] 消失，新出现 [1,8]，终局从 8 到达/4 碰撞到 10/2。正偏航新增 [4,10]，接触前的 120 s 历史采样点均在冲突图内且有离散高度解。触发步，10 号机请求爬升约 299 ft，指令原样交付，K=2 和 K=4 均未得到无冲突的高度预测。冻结控制前缀相同，撤回该机一个或五个周期的高度变化后，同一对飞机仍分别晚 6 s、12 s 接触，三个分支均为 10 到达/2 碰撞。修正单条指令只能推迟这次接触。三万步冻结评论家的负偏航、正偏航原始动作梯度与精确 Q 差符号分别在 8/17、6/17 个改令批不一致，诊断使用开发状态。[正偏航逐帧接触](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/yaw-retention-30000-native-contacts.imAyRQ/yaw_plus15-update30000/collision-events.jsonl) · [五步反事实](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/yaw-retention-30000-native-contacts.imAyRQ/hold-five/collision-events.jsonl) · [三万步评论家诊断](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/projected-critic-input-audit.p7oJxR/update-30000.json)

接触前缀相同的双机反事实保持 4、10 号机高度 30 或 144 s，均消除该对接触。30 s 分支出现 [5,9]，终局维持 10 到达/2 碰撞。144 s 分支又出现 [4,11]，终局为 8/4。延缓或冻结局部动作会把接触转移到同图其他飞机，现有证据支持以全连通分量的终局评估改令收益。[30 s 双机分支](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/yaw-retention-30000-native-contacts.imAyRQ/hold-pair-5/collision-events.jsonl) · [144 s 双机分支](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/yaw-retention-30000-native-contacts.imAyRQ/hold-pair-24/collision-events.jsonl)

关投影比较完成全部逐种子结果写入后，控制命令收到结束信号。已校验两份模型 SHA、全部 36 条流量结果及终局计数，并以新工作目录独立复跑最后一组负偏航，逐种子终局、动作成本和间距统计逐项一致。43 份新开发配对与反事实缓存路径经 SHA 核对后从 7 个物理 inode 合并至 1 个，后续两份双机缓存也已校验并链接同一内容。[最后一组独立复验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/yaw-retention-30000-negative-recheck.i5owC4/paired/comparison.json) · [缓存校验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/yaw-retention-30000-native-raw.kEgy3n/navdata-deduplication.json)

### STE、镜像回放与 Stage 2 解禁

任务 45 确认终态后，按更新目标移除硬反射及其参数和日志。Actor 的 Q 用当前状态的实际保护器重投影，前向用与回放一致的执行动作编码，反向走原提议的 Jacobian。目标评论家仍用 24 步后的保护动作输入。锚定权重按每轮新增 learner updates 从初始值线性降到第 2,000 步的零。Plan A 与 B 共用成对镜像回放，每个抽样空域保留原样与左右镜像，联合翻转当前及后继的相对航向正弦、横向误差、近邻/远邻横向位置与速度、物理东向轴及原动作与教师动作的航向量。终局、回报和无符号量在每一对中相同。抽样后的训练批大小保持不变。

冻结任务 45 两万步权重及同一组状态，仅改变首评论家取梯度的位置。以原始与投影动作的实际 Q 差作参照，原生负偏航反号 7/15→1/15，正偏航 3/17→4/17，四航路 1/14→1/14，六航路 0/10→1/10。方案 B 交叉、近对头及偏航为 5/16→9/16、2/16→13/16、3/16→10/16。STE 在这批原生负偏航上提高一致性，其余场景方向仍由闭环验证。冻结任务 44 万步 actor，原生负偏航/正偏航各 324 个逐机状态的原提议与镜像提议有 51/8 次航向模式不一致。对相同观测、实际及正反测试转向，方向保护的镜像误差为 0。67 项回归通过、1 跳过。方案 B CPU 功能烟测完成 17 次新增更新，启用投影目标、STE、镜像和多步回放。[冻结 STE 逐批诊断](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/projected-action-ste-audit.jEbLGa/update-20000.json) · [左右对称逐状态审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/ste-mirror-heading-audit.gDtOjc/summary.json) · [方案 B 实际训练入口烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/projected-ste-smoke.PU8XRi/training/metrics.jsonl)

新开发门槛采用总到达率 ≥95% 或同种子净增至少 4 架，同时各场景到达率 ≥88%、近对头 ≥95%。关键准入使用五个种子的原生逐帧报告复算 SHA 与计数。历史任务 44 的万步断点在 70111–70115 由入场 344/360、14 碰撞提升至 348/360、12 碰撞，六场景到达 60、56、58、60、56、58 架，全部通过。由此采用直接接续路径，使用已验真的历史断点进入方案 A。工作区同时保留修复版方案 B 的 20,000 更新、每 5,000 更新检查并按峰值选模入口。[五种子逐场景配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/ste-historical-native-screen.wYrSpm/paired/comparison.json) · [写入并复算的准入](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-ste-mirror-2500.0hlsC2/stage2-native-admission.json)

任务 46 在任务 45 正常终止、现有单卡队列为空后提交。SOFY 确认运行。方案 A 使用 9 workers × 8 空域、0.05 s 原生物理、2,500 轮全分布混合，保留 144 s 回报、投影目标、10 m 三机层和方向保护，使用 STE 及镜像，余弦小学习率。每 500 轮做五种子开发评估，按到达、碰撞及单机成本排序并检查场景率下限。正式 12001–12005 逐帧评估随同一任务在微调结束后运行，输出三份权重开关投影结果与单机动作成本，并单列论文评测启动门槛（到达 >95%、碰撞 ≤3%）及冲刺门槛。当前已完成 59201–59205 的入场开发评测，训练进程已写出配置。[提交记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-ste-mirror-2500.0hlsC2/submit.log) · [实际训练配置](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-ste-mirror-2500.0hlsC2/stage2/training/config.json) · [初始开发组](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-ste-mirror-2500.0hlsC2/stage2/development-incoming/validation.json)

方案 A 首批实测 72 轮、1,024 更新，12.16 更新/s、3,078 轮/h，Torch 设备峰值 2,396 MiB。锚定权重从 1.25 降至 0.61，符合第 2,000 步归零计划。另一组 Stage 2 开发种子 59201–59205 的微调前分数为 324/360 到达、36 碰撞，双向偏航 52/60 与 44/60。该分数和准入面板的 348/360 分属两批流量，用后续周期评测判断高保真微调是否带来泛化收益。[实时训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-ste-mirror-2500.0hlsC2/stage2/training/metrics.jsonl) · [同一权重的独立开发面板](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-ste-mirror-2500.0hlsC2/stage2/development-incoming/validation.json)

### Stage 2 首个开发断点和 Critic 初始化复算

任务 46 的首个周期断点为 504 轮、6,464 更新。同一原生 59201–59205 开发流量下，开投影由入场 324 到达、36 碰撞变为 298/58，关投影由 246/114 变为 88/272。开投影的三、四、六航路、近对头和正负偏航到达数分别为 54、48、46、58、44、48。单机动作成本由 23.54 增至 51.72。候选未达到开发场景率下限，继续保留入场文件并执行预定训练。[首个周期完整开发结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-ste-mirror-2500.0hlsC2/stage2/development-001640/validation.json) · [选模记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-ste-mirror-2500.0hlsC2/stage2/selection.json)

核对源码和实际配置，方案 A 通过 `--initialize` 只载入 actor，评论家从随机权重起步。固定同一入场 actor 和四案原生状态，方案 B 转入前的评论家对改令飞机及全机汇总均给出 3/4 正向排序，用任务实际初始种子复算的随机评论家为 0/4、0/4。其余四个随机种子为 1–4/4 的改令飞机排序，以及 1–4/4 的全机排序。方案 A 完成 10,000 更新后，两个原生计数均为 2/4。这些状态级证据定位了早期价值信号变化，尚未验证载入评论家就能恢复原生闭环。运行快照保留原训练设计。[初始化四案复算](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage2-critic-initialization-audit.jUbPdC/summary.json) · [方案 A 万步逐案 Q](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage2-10000-q-audit.uleIzQ/q.json)

### Figure 4 同流量策略入口、通信矩阵和原生风场验真

Figure 4 策略入口保留清单中的真实 KL 呼号和释放帧，复用原始 KSS 干预、回航和原 SARL 稀疏指令，将其与 Geo 权重放进同一原生 BlueSky。全部策略每 0.05 s 按 5 km/300 ft 碰撞、5 km 到达和 20 km 越界裁决，单机成本按实际下发的归一化三轴绝对增量加权 [1,2,4]。15 份正式清单、30 份匹配的历史控制配置和各历史 actor 文件已核验。独立四机开发流量 89031 上，六个政策开关的呼号、发机帧、阈值和飞机终局计数逐项一致，原始 Hold+KSS 真实干预 11 次。该小样本只验接口。正式入口会逐个检查 45 份结果的权重、控制器源码、配置、清单和发机记录，按 Case 汇总五个流量种子的均值及样本标准差。[同清单六臂验真](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/figure4-policy-bridge-verified.0ZPR6y/verification.json) · [正式汇总入口](/workspace/group_share/adc-perception-pretrain/zhuy25/yry/experiment/Geo_ADMM_RL/eval/bluesky_figure4_matrix.py)

Exp 5 入口将轮数、调度周期和网络条件列为四个面板，共 61 个控制臂。K=0/1/2/3/5 覆盖不开 ADMM 与各正预算的冷热启动。6/12/18/30 s 各有独立 Hold 与 MPC 面板。网络面板在 K=2 下交叉 0/20/50% 丢包、0/1/2 步延迟、Jacobi/图着色、冷热启动。每臂覆盖同一六场景、五个流量种子。非保留开发流量验证了 18 s 周期、MPC、延迟、丢包和着色叠加的两臂执行，续跑复用原结果 SHA，变更控制步数会被拒绝，worker 的导航缓存指向同一个 inode。两处正式入口会检查任务 46 留出验收标记和所选权重 SHA。[通信开发验真](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exp5-resume-dev.BLECiH/run/summary.json) · [正式通信入口](/workspace/group_share/adc-perception-pretrain/zhuy25/yry/experiment/Geo_ADMM_RL/eval/bluesky_paper_exp5.py)

Exp 4 新增水平两点、各两层的原生风场。种子 89031 的无风、仅风场和联合扰动复用了同一路由哈希。仅风场最大风速 15.29 m/s，实测地速减去空速分量和风速后的残差小于 1.7e-14 m/s。联合组还验证了突发发机帧、两架非合作飞机拒令及既定速度噪声序列。这一开发回放用于验真扰动，没有产生保留集成绩。[原生风场三臂验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-wind-field-dev.XORJYr/verification.json)

### Stage 2 后续断点与单变量续训准备

任务 46 的 1,001、1,500、2,004 轮在原生开发流量 59201–59205 完成评测，开投影到达/碰撞依次为 318/32、327/20、314/38，关投影依次为 106/248、131/211、126/226。三者均未过开发选模。1,500 轮的到达增量来自三、四航路等场景，负偏航只有 43/60。2,004 轮的六航路、正偏航和负偏航分别为 46、52、44 架。入场文件仍为当前选中权重。20,000 次更新的真实断点四案局部/全空域原生 Q 排序为 1/4、2/4，入场为 3/4、3/4。[开发选模与逐场景](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-ste-mirror-2500.0hlsC2/stage2/selection.json) · [20,000 次更新的原生 Q](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage2-20000-q-audit.r6IMAH/q.json)

新增可选的 Critic 权重转入。从同一输入权重复制评论家至在线和目标网络，actor、Adam 初态、更新计数、回报、STE、镜像和锚定设定继续沿用当前训练入口。实际方案 A CPU 烟测两臂配置仅有该开关不同，actor 位级一致，打开时的在线和目标 Critic 与方案 B 输入逐项一致，优化器为空，学习前的样本统计吻合。该测试只验初始化。后续完整对照可用开发评测结算，保留种子只用于入选模型的正式验收。正在运行的单卡快照保留其原设计。[单变量原生功能验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/stage2-critic-warmstart-smoke.IWNmXe/verification.json)

### Exp 2/3 拓扑与规模验真、Exp 6 潜在边对照

Exp 2 固定 30 架并发，在同流量种子下比较 6、13、15、17 航路。后三种超过历史 BlueSky 混合训练采样的 12 航路上限。Exp 3 固定六航路比较 30、50、80 架。正式入口经 Stage 2 门控后执行五流量种子的开关投影配对。独立开发种子 89033 的 14 臂各跑了两个原生控制周期，核对实际并发数和新拓扑航路计数均正确。报告包含最大度和最大团、策略及保护批次耗时、含原生物理的控制周期耗时，以及各流量种子的统计。这组短回放只验真入口。[拓扑与规模验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-topology-scale-dev.NwzS5L/verification.json) · [正式入口](/workspace/group_share/adc-perception-pretrain/zhuy25/yry/experiment/Geo_ADMM_RL/eval/bluesky_paper_generalization.py)

Exp 6 新增 `--without-potential-edges`。去潜在边时保留既有冲突边，重新计算节点度、密度、着色和图门控协调所用邻接。原生物理继续逐帧裁决。种子 89033 的原生六航路同权重开投影配对跑满 100 个控制周期，完整图与仅既有边的平均度为 1.051 与 0.060，最大团为 4 与 2。两臂均结算 2 架碰撞和 10 架未完成。此开发样本验证潜在边确实被移除。几何回归 68 项通过、1 项跳过。[完整图](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-potential-edge-ablation-dev.6av5xL/full/comparison.json) · [仅既有边](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-potential-edge-ablation-dev.6av5xL/existing/comparison.json)

### 任务 46 原生微调与正式留出结算

任务 46 在获授权单卡上完成 2,508 轮、29,824 次更新并正常退出。扣除周期开发评测窗口后，日志时间约 2,289 s，对应 3,945 轮/小时和 13.03 updates/s。同期 228 个 PPU 样本均值为 74.68%，峰值 HBM 2,806 MiB。计入周期评测后，316 个样本均值为 58.72%。训练端最终开发集开投影为 318/360 到达、32 架碰撞，关投影为 124/360 到达、226 架碰撞。五个微调断点均未过选模。选中权重与任务 44 的原始 10k 入场权重字节一致，SHA-256 为 `224776f45909de543ef4c1e8407888a0f20eff89c72d89e0b2686f22a077e479`。[吞吐与设备取样窗口](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-ste-mirror-2500.0hlsC2/training-window-summary.json) · [开发选模](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-ste-mirror-2500.0hlsC2/stage2/selection.json)

同一批冻结原生状态上，入场权重负偏航的 raw／STE 梯度与精确 Q 差分别有 9/15 与 1/15 个反号批次，正偏航为 3/17 与 0/17。四航路为 1/15 与 3/15，六航路为 1/12 与 3/12。训练终轮负偏航为 4/15 与 1/15，正偏航为 3/17 与 3/17，四航路为 7/21 与 4/21，六航路为 2/12 与 4/12。四案首步最小 Q 的原生局部及全空域排序均由入场 3/4 变为终轮 2/4。这批审计使用原生高度观测，分母是各权重在该冻结批中实际改变命令的次数。[入场符号](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage2-final-critic-audit.ePnIBZ/entry-sign.json) · [终轮符号](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage2-final-critic-audit.ePnIBZ/final-sign.json) · [四案 Q](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage2-final-critic-audit.ePnIBZ/final-q.json)

留出集使用 `12001–12005`，每场景 60 架，每臂 360 架。原生 BlueSky 物理及事件采样均为 0.05 s，控制周期 6 s。核验了各臂的流量种子、逐行终局计数和三个 checkpoint 的 SHA-256。表格各格依次为到达架数／碰撞架数及平均单机控制成本。冻结基线的两个偏航场景共 120 架越界，两臂相同。此前记录的 174/360 为历史评测值，本次表格使用同批、同协议重跑的冻结基线。

| 场景 | 冻结基线 开 | 冻结基线 关 | 当前参考 开 | 当前参考 关 | 任务 46 选中 开 | 任务 46 选中 关 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 三航路 | 60/0 · 2.99 | 30/30 · 1.37 | 58/2 · 18.95 | 44/16 · 30.33 | 58/2 · 22.10 | 42/18 · 25.02 |
| 四航路 | 50/10 · 3.48 | 26/34 · 2.07 | 52/8 · 19.26 | 34/26 · 25.42 | 52/8 · 20.82 | 32/28 · 19.78 |
| 六航路 | 44/16 · 4.00 | 26/34 · 2.81 | 50/10 · 17.98 | 34/26 · 29.94 | 41/19 · 22.07 | 30/30 · 21.19 |
| 近对头 | 58/2 · 4.65 | 20/40 · 2.12 | 60/0 · 26.97 | 56/4 · 25.16 | 60/0 · 23.59 | 50/10 · 25.34 |
| 正偏航 | 0/0 · 0.11 | 0/0 · 0.00 | 52/8 · 27.60 | 42/18 · 31.56 | 58/2 · 32.25 | 44/16 · 30.58 |
| 负偏航 | 0/0 · 0.10 | 0/0 · 0.00 | 53/6 · 26.17 | 42/16 · 27.01 | 54/6 · 26.84 | 42/18 · 25.06 |
| 合计 | 212/28 · 2.55 | 102/138 · 1.39 | 325/34 · 22.82 | 252/106 · 28.23 | 323/37 · 24.61 | 240/120 · 24.49 |

任务 46 选中权重开投影到达率 89.72%、碰撞率 10.28%，论文门槛未通过。负偏航 54/60，近对头 60/60。任务 47 只用开发种子选模，继续比较冷／热启动评论家；这张留出表只用于结算任务 46。[六场景逐种子对照、开关投影及三权重哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-ste-mirror-2500.0hlsC2/stage2/formal-stage2/comparison.json) · [正式门控](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-ste-mirror-2500.0hlsC2/stage2/stage2-acceptance.json)

### 单变量 Critic 转入与 Exp 4/6 功能验真

核实任务 46 为 `SUCCEEDED` 且队列为空后，在同一授权单卡提交任务 47。沿用原生 9×8 空域、2,500 轮、训练种子、24 步目标、STE 和镜像。把同一入场 checkpoint 的在线 Critic 同时转入在线与目标 Critic，优化器仍从空状态开始。任务 47 周期开发评测使用 `59201–59205`。快照及输入哈希均通过校验。[提交回执](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-critic-transfer-2500.wZyswa/submit.log) · [单变量快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-critic-transfer-2500.wZyswa/preflight.json)

任务 47 调度已进入 `RUNNING`。两次训练从同哈希入场权重开始，六场景、五种子、开关投影共 60 条开发记录的终局逐条相同。实际训练配置仅 Critic 初始化与输出路径有差异。入场、冷启动 10k、20k、终轮各提供一个 Actor 和一个 Critic，在相同冻结原生观测及两条固定执行命令下做 4×4 交叉复算。更换 Actor 的动作编码来源时，各 Critic 四案的排序计数保持一致。四个 Critic 的局部／全空域正确数依次为 3/4、3/4，2/4、2/4，1/4、2/4，2/4、2/4。这一结果把已观测到的排序退化定位到四案的 Critic 权重差异。闭环效果由任务 47 的开发配对判定。[实际配置和逐条入场核验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-critic-transfer-2500.wZyswa/runtime-match-verification.json) · [权重交叉复算](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-frozen-q-factorial.u9Cd4j/factorial.json)

任务 47 在首个约 500 轮节点完成 501 轮、5,984 次更新。原生开发集开投影为 338/360 到达、22 架碰撞，冷启动首节点 504 轮为 298/360、58 架，入场为 324/360、36 架。六场景到达依次为 60、58、50、60、56、54。六航路的 50/60 未达到 88% 分场景门槛。关投影为 174/360 到达、186 架碰撞，冷启动首节点为 88/360、272 架，入场为 246/360、114 架。当前比较按训练轮次对齐，实际更新数相差 480。任务 47 继续运行，后续用 10k 更新等步数断点复算冻结审计。[热启动首节点逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-critic-transfer-2500.wZyswa/stage2/development-001520/validation.json) · [热启动选模](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-critic-transfer-2500.wZyswa/stage2/selection.json)

10k 等更新数冻结审计中，冷启动的四案首步最小 Q 局部与全空域排序均为 2/4，热启动均为 3/4。保持入场 Actor 的动作编码时结论相同。冷启动负／正偏航、四／六航路的 STE 反号为 0/15、0/17、5/20、2/10，热启动为 0/15、0/17、3/14、0/11。分母为各权重实际发生动作更改的批数。等步数闭环对照使用非保留种子 `59201–59203`，原生 0.05 s 六场景开关投影。[等步数梯度、Q 和权重哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-frozen-q-factorial.u9Cd4j/matched-10000-summary.json)

同一对 10k 权重的 216 架等步数原生开发对照已全部结算。表格各格为到达／碰撞／越界，所有臂未完成数为零。热启动关投影净增 54 架到达、减少 56 架碰撞。开投影净减 18 架到达、增加 18 架碰撞，其中六航路增加 8 架碰撞。四案冻结价值改善和策略本体收益已复现，开投影闭环仍限制晋升。[完整逐种子与采样协议](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-frozen-q-factorial.u9Cd4j/matched-10000-native-development/comparison.json) · [六场景汇总](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-frozen-q-factorial.u9Cd4j/matched-10000-native-development-summary.json)

| 场景 | 冷启动 开 | 热启动 开 | 冷启动 关 | 热启动 关 |
| --- | ---: | ---: | ---: | ---: |
| 三航路 | 34/2/0 | 32/4/0 | 8/28/0 | 24/12/0 |
| 四航路 | 34/2/0 | 32/4/0 | 14/22/0 | 22/14/0 |
| 六航路 | 32/4/0 | 24/12/0 | 12/24/0 | 16/20/0 |
| 近对头 | 36/0/0 | 34/2/0 | 0/36/0 | 18/18/0 |
| 正偏航 | 36/0/0 | 32/4/0 | 14/20/2 | 22/14/0 |
| 负偏航 | 29/2/5 | 29/2/5 | 23/8/5 | 23/4/9 |
| 合计 | 201/10/5 | 183/28/5 | 71/138/7 | 125/82/9 |

热启动第二节点为 1,002 轮、11,744 次更新，原生五种子开投影 314/360 到达、46 架碰撞，关投影 232/360 到达、127 架碰撞，关投影近对头达到 56/60。冷启动相近的 1,001 轮节点开投影 318/360 到达、32 架碰撞，关投影 106/360 到达、248 架碰撞，关投影近对头为 0/60。热启动的策略本体在第二节点接近入场的 246/360，到达改善集中在关投影。开投影六航路仍为 46/60，其余五类为 54、50、60、52、52。该断点未被选中，后续按原计划完成 2,500 轮。[热启动第二节点逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-critic-transfer-2500.wZyswa/stage2/development-002960/validation.json) · [冷启动同阶段逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-ste-mirror-2500.0hlsC2/stage2/development-003120/validation.json)

独立验收入口 `eval.stage_acceptance --guarded` 可直接读取原生正式配对 JSON，并输出三机安全协议下的门槛判定。对任务 46 已归档的正式表复算，输出与训练入口原始判定逐字节一致，SHA-256 为 `fbf66c9f484397c07f51d5f24dc734d0e6e129247e89fb92619ed39766eb3fc1`。[独立验收复算](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-frozen-q-factorial.u9Cd4j/recomputed-task46-acceptance.json)

Exp 5 四个正式矩阵面板的全部 61 个配置在非保留对头种子 `89071` 逐一经过原生 0.05 s 的七控制步烟测。步内预算 9 臂、空窗 Hold 8 臂、空窗 MPC 8 臂、丢包×延迟×着色与冷热启动 36 臂。配置、源权重、种子及每臂输出校验均通过。该短测覆盖执行入口和协议，不用于报告端到端效果。[61 臂运行验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/exp5-full-grid-dev.Jkzrh6/verification.json)

Exp 4 原生入口包含近对头 30 架压力、±15° 开关投影回航及突发插入、非合作、风场三种扰动和组合扰动。开发种子 89061、89062 的两类偏航各执行 120 个控制周期，正偏航开／关投影持续对齐架数分别为 `[6,10]`／`[8,11]`，负偏航为 `[6,10]`／`[5,10]`。持续对齐按航向距目标 2° 内连续三个 6 s 控制周期判定。30 架对头烟测执行 3 个控制周期。此处是开发回放。[Exp 4 开发验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/exp4-yaw-recovery-dev.5K0hGJ) · [正式入口](/workspace/group_share/adc-perception-pretrain/zhuy25/yry/experiment/Geo_ADMM_RL/eval/bluesky_paper_exp4.py)

Exp 6 原生入口串接关闭 ADMM、去潜在边及替换 KSS 的同流量配对，并接收独立训练的一步回报权重。方案 B 入口配对烟测各完成 9 次更新，两臂回放均保留 24 决策成熟窗口，Critic 回报分别为 24 步与 1 步。正式去多步回报仍需相同训练配置的长程对照。几何回归 68 通过、1 跳过。[匹配功能验证](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/exp6-return-matched-smoke.6ISmqk/verification.json) · [正式入口](/workspace/group_share/adc-perception-pretrain/zhuy25/yry/experiment/Geo_ADMM_RL/eval/bluesky_paper_exp6.py)

用烟测的两份实际权重验证了 Exp 6 正式入口的匹配配置校验。相同回报配置作为消融输入时被拒绝。[权重及配置哈希、拒绝记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-frozen-q-factorial.u9Cd4j/exp6-control-gate-verification.json)

已退出的原生训练和烟测工作进程中，两组逐字节相同的导航缓存由 10 和 3 个 inode 分别合并至 1 个，释放约 173 MiB 重复数据。复播材料继续保存于 CPFS。[缓存校验明细](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-ste-mirror-2500.0hlsC2/closed-worker-navdata-deduplication.json)

### 任务 47 六航路逐帧消融与编码器对照

热启动第三节点 1,509 轮、17,584 次更新。五种子原生开投影 319/360 到达、30 架碰撞，关投影 177/360 到达、170 架碰撞。开投影六航路 44/60，正／负偏航 50/60、51/60，近对头 60/60。该断点未获选。20k 等更新数审计中，冷／热启动的四案局部最小 Q 排序是 1/4、2/4，全空域均为 2/4。热启动的局部排序从 10k 的 3/4 回落。20k 热启动四类的 STE 反号为 0/15、1/17、3/14、0/10，冷启动则为 2/15、1/17、6/20、6/10，依次是负偏航、正偏航、四航路、六航路。[第三节点逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-critic-transfer-2500.wZyswa/stage2/development-004420/validation.json) · [20k 等更新数梯度、Q 与权重哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-frozen-q-factorial.u9Cd4j/matched-20000-summary.json)

20k 同权重哈希在开发种子 59201–59203 做原生 0.05 s 六场景闭环。冷／热启动开投影为 200/216 到达、12 架碰撞与 187/216 到达、24 架碰撞。关投影为 105/216、104 架碰撞与 138/216、74 架碰撞。开投影六航路到达／碰撞由 34/2 变为 26/10。随后任务 47 的 2,000 轮节点在五种子上开投影为 307/360、40 架碰撞，关投影为 212/360、132 架碰撞。该节点未获选。[20k 等步数闭环汇总](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-frozen-q-factorial.u9Cd4j/matched-20000-native-development-summary.json) · [逐种子闭环](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-frozen-q-factorial.u9Cd4j/matched-20000-native-development/comparison.json) · [2,000 轮五种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-critic-transfer-2500.wZyswa/stage2/development-005820/validation.json)

在造成 10k 开投影六航路退化的开发种子 59201、59202，用原生 0.05 s 单空域复跑冷热权重和投影开关。八个终局计数均与配对评测一致。七臂控制步数相同，冷启动开投影 59202 的单空域复跑早一控制步完成。热启动六对接触中，复算 K=4 仍标记 3 对有冲突。另有 3 对在执行命令的近似 60 s 预测中保持至少 91.44 m 高差，原生逐帧低于该阈值。预测复算固定当时命令，结论限于这两条回放。[逐帧接触、历史控制和预报对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage2-critic-crossing6-contacts.L095L4/diagnostic-summary.json)

保持同一 Actor 的动作提议，在沿途同一图状态上换入另一 Actor 的 ADMM 参数输出。表格为到达／碰撞，两种子合计 24 架。高度热启动和协商权重的收益落在不同种子。单独关三机兜底后两种子总碰撞相同，59201 的逐控制步命令完全一致，59202 的接触有变化。[逐种子单变量复跑](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage2-critic-crossing6-contacts.L095L4/diagnostic-summary.json)

| 原生闭环配置 | 59201 | 59202 | 合计 |
| --- | ---: | ---: | ---: |
| 冷启动开投影 | 10/2 | 10/2 | 20/4 |
| 热启动开投影 | 6/6 | 6/6 | 12/12 |
| 热启动动作＋冷启动全部 ADMM 输出 | 8/4 | 10/2 | 18/6 |
| 冷启动动作＋热启动全部 ADMM 输出 | 8/4 | 10/2 | 18/6 |
| 热启动动作＋冷启动速度热启动输出 | 6/6 | 6/6 | 12/12 |
| 热启动动作＋冷启动高度热启动输出 | 6/6 | 10/2 | 16/8 |
| 热启动动作＋冷启动协商权重输出 | 8/4 | 8/4 | 16/8 |
| 热启动动作＋关闭三机兜底 | 6/6 | 6/6 | 12/12 |

冷热 10k 的三个 ADMM 输出头与入场参数逐张量相同。共享编码器 82 个张量中有 72 个发生变化，热启动最大绝对变化为 0.03024。因此下一对照保持共享编码器固定，继续优化动作头和 Critic。新增可选 `--freeze-actor-encoder`，沿用同一个推理网络、STE、镜像和损失。原生双臂烟测各跑 36 次更新，实际配置只差此开关和输出路径。冻结臂在线与目标 Actor 的 82 个编码器张量、6 个 ADMM 头张量逐项保持，8 个动作头张量和 Critic 继续更新。两份烟测导航缓存复用同哈希 inode。几何回归 68 通过、1 跳过。[参数漂移](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage2-critic-crossing6-contacts.L095L4/actor-parameter-drift.json) · [原生烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-encoder-freeze-smoke.wCXBOy/verification.json) · [烟测导航去重](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-encoder-freeze-smoke.wCXBOy/navdata-deduplication.json)

已从任务 47 源码快照准备 2,500 轮匹配编码器冻结试验。除加入冻结开关及其转发，两份源文件其余代码、入场权重、训练及开发流量一致。输入哈希已核对。调度决策等任务 47 完整结算后作出。[对照快照与哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-encoder-freeze-2500.StkJ4G/preflight.json)

用真正的入场 Actor 输出替换热启动 10k 的 ADMM 参数，在五个六航路开发种子上，两臂总计均为 44/60 到达、16 架碰撞。59202 减少 2 架，59205 增加 2 架。末端替换的试验结果有种子依赖。编码器冻结的长程对照检验在固定协商表征的条件下重新学习动作的效果。[五种子入场输出替换](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage2-critic-crossing6-contacts.L095L4/entry-consensus-paired.json)

针对近似预测高差达标的三对实际接触，分别从相同控制前缀复跑。只把该双机在触发周期的高度增量置零，三对接触在各自反事实中均消失，每次整程碰撞均从 6 架降到 4 架。其中一次后续出现了不同接触对。从原生观测的高度、选定高度和垂直速度起算，按 B737 的 0.05 s 垂向加速度、限速、目标捕获规则复算，三次实际接触帧的高度差最大误差为 4.1 微米。原近似预测在相关重叠区给出 94.03、93.18、92.96 m，实测接触帧为 91.187、91.438、91.401 m。该证据定位到该类接触的垂向动力学近似。置零已知双机的干预用于验证原因。[三个单拍反事实](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage2-critic-crossing6-contacts.L095L4/native-hold-one-summary.json) · [垂向逐帧误差](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage2-critic-crossing6-contacts.L095L4/forecast-vertical-parity.json)

### 任务 47 最终结算与任务 48 接续

任务 47 正常退出，原生训练累计 2,500 轮、28,944 次更新。日志内训练窗口为 2,276 s，合计 3,954 轮/小时、12.72 updates/s。排除周期开发评测后，227 个 PPU 设备样本均值 73.41%，峰值 HBM 2,806 MiB。计入周期评测后 313 个样本均值 57.21%。训练工作进程和索引中，SHA-256 一致的 10 份导航缓存已硬链接到同一个 inode，释放约 142 MiB 重复数据。[训练与设备取样明细](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-critic-transfer-2500.wZyswa/training-window-summary.json) · [结束后的导航缓存去重](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-critic-transfer-2500.wZyswa/closed-worker-navdata-deduplication.json)

终轮原生五种子开发集开投影 311/360 到达、30 架碰撞、19 架越界，关投影 228/360 到达、114 架碰撞、18 架越界。开投影六场景到达依次为 60、58、46、60、40、47，平均单机控制成本为 23.21。四案首步最小 Q 局部与全空域均为 2/4。四类冻结样本的 STE 反号依次为 1/15、1/17、0/14、0/10。到达峰值出现在 501 轮的 338/360，六航路为 50/60。五个训练断点均未晋升，选中文件仍为入场 SHA-256 `224776f45909de543ef4c1e8407888a0f20eff89c72d89e0b2686f22a077e479`。这个同哈希权重已有任务 46 的正式留出结果。[全部开发断点与选模](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-critic-transfer-2500.wZyswa/stage2/selection.json) · [终轮逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-critic-transfer-2500.wZyswa/stage2/development-007260/validation.json) · [终轮 Q、梯度和权重哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-frozen-q-factorial.u9Cd4j/warm-final-summary.json)

调度确认任务 47 为 `SUCCEEDED`、集群单卡空闲且队列为空后，提交任务 48，在同一张 PPU 上执行冻结共享编码器的 2,500 轮匹配对照。沿用任务 47 的热启动 Critic、入场权重和开发种子，正式留出仍由新权重的开发晋升决定。[任务 47 调度终态](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-encoder-freeze-2500.StkJ4G/previous-task.txt) · [任务 48 提交回执](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-encoder-freeze-2500.StkJ4G/submit.log)

任务 48 调度显示 `RUNNING`，验证时已产生 304 次训练更新。实际配置比较中，唯一训练差异为 `freeze_actor_encoder`。完整入场开发分数与任务 47 一致，开投影 324/360 到达、36 架碰撞，关投影 246/360 到达、114 架碰撞。[实际配置差异与入场复跑](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-encoder-freeze-2500.StkJ4G/runtime-match-verification.json)

### 原生垂向间隔模型的闭环独立对照

在 CPFS 隔离副本中，对一个控制周期内进入水平间隔带的活跃邻机，试用 B737 的 0.05 s 加速度、限速及目标高度捕获动力学，替换该对在水平重叠区内的垂向间隔预测。分层候选、既有双机与三机搜索、门槛及原生闭环裁决沿用当前配置。用三例归档接触的发令前状态检验新实现，逐帧复算与独立脚本一致，三例接触帧高差对原生最大绝对误差约 4.1 微米。两个已知六航路问题种子上，任务 47 的 10k 权重开投影由 12/24 到达、12 架碰撞改善到 18/24 到达、6 架碰撞。该试验同时启用提前检查和逐帧高差复算。六场景对照另加仅提前启用现行搜索一臂，用以分离两个效应。[逐帧模型核验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-vertical-clearance-probe.76aBWR/vertical-forecast-verification.json) · [两种子初测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-vertical-clearance-probe.76aBWR/pilot/comparison.json)

任务 47 的 10k 权重在五个开发种子 `59201–59205` 上跑完开投影三臂。每格为到达／碰撞。三个配置均为 8 架越界、零超时。仅提前触发分层搜索时净增 16 架到达并减少 16 架碰撞。进一步以逐帧动力学替换临近邻机的高差预测后再净增 4 架到达，来自三航路与近对头。两个方法的六航路结果相同。CPU 上五空域一批的六场景平均决策耗时为 25.70、24.24、29.30 ms，平均单机成本为 38.84、39.19、39.35。逐帧路径仍属于开发对照。[六场景三臂逐种子、权重哈希及耗时](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-vertical-clearance-probe.76aBWR/six-families-validated.json)

| 场景 | 现行 | 提前触发现行搜索 | 提前触发并逐帧复算 |
| --- | ---: | ---: | ---: |
| 三航路 | 54/6 | 56/4 | 58/2 |
| 四航路 | 50/10 | 56/4 | 56/4 |
| 六航路 | 44/16 | 50/10 | 50/10 |
| 近对头 | 58/2 | 58/2 | 60/0 |
| 正偏航 | 54/6 | 54/6 | 54/6 |
| 负偏航 | 44/8 | 46/6 | 46/6 |
| 合计 | 304/48 | 320/32 | 324/28 |

在同一五种子上，入场权重的现行与提前触发两臂均为 324/360 到达、36 架碰撞，六个场景的汇总完全相同。逐帧复算为 326/360、34 架碰撞，收益来自一个三航路种子。热启动 10k 的提前触发收益覆盖全部五个种子，入场权重上观察到种子及 checkpoint 依赖性。现有的连续加速度近似在两个已知问题种子上与提前触发、逐帧复算两臂的到达和碰撞完全相同。该小样本未区分两种动力学模型的总体收益。[入场权重六场景三臂](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-vertical-clearance-probe.76aBWR/entry-six-families-validated.json) · [逐种子差值](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-vertical-clearance-probe.76aBWR/paired-effects.json) · [连续加速度初对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-vertical-clearance-probe.76aBWR/two-seed-forecast-comparison.json)

任务 48 首个开发节点为 500 轮、5,984 次更新，与任务 47 的首节点更新数相同。冻结编码器开投影 322/360 到达、32 架碰撞，继续训练。关投影为 235/360 到达、122 架碰撞，同更新数未冻结对照为 174/360、186 架碰撞。冻结臂开投影六航路 50/60 未达 88% 分场景门槛，负偏航为 53/60。[任务 48 首节点原生逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-encoder-freeze-2500.StkJ4G/stage2/development-001520/validation.json)

任务 48 第二节点为 1,006 轮、11,824 次更新。开投影 325/360 到达、32 架碰撞，关投影 224/360 到达、134 架碰撞。开投影六航路仍是 50/60，近对头降到 52/60，负偏航为 56/60。该断点未获选。实际 PPU 10k checkpoint 的在线和目标 Actor 中，82 个共享编码器张量与 6 个 ADMM 头张量全部与入场字节一致，8 个动作头张量均已更新。冻结臂在原生四案的局部与全空域首步最小 Q 排序均为 1/4，未冻结热启动均为 3/4。冻结臂 STE 反号依次为负偏航 1/15、正偏航 1/17、四航路 0/14、六航路 2/10。[第二节点逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-encoder-freeze-2500.StkJ4G/stage2/development-002980/validation.json) · [10k PPU 权重核对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-encoder-freeze-10000-audit.OqCwE1/ppu-parameter-verification.json) · [冻结状态 Q 与梯度](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-encoder-freeze-10000-audit.OqCwE1/matched-q-and-sign-summary.json)

同一对 10k 权重在开发 `59201–59203` 的 216 架原生闭环结算。冻结编码器开投影由未冻结对照的 183/216 到达、28 架碰撞改善至 202/216、12 架碰撞。关投影由 125/216、82 架改善至 164/216、50 架。开投影六航路由 24/36、12 架改善至 30/36、6 架，仍低于 88% 分场景要求。冻结版在当前小样本上的闭环改善和首步 Q 排序下降同时出现。后续五种子补全在同一冻结权重上分离安全层效果。[10k 完整原生配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-encoder-freeze-10000-audit.OqCwE1/native-development/comparison.json) · [216 架汇总](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-encoder-freeze-10000-audit.OqCwE1/native-development-summary.json)

任务 48 第三个开发节点为 1,503 轮、17,584 更新，与任务 47 的 1,509 轮节点等更新数。开投影由未冻结的 319/360 到达、30 架碰撞变为 334/360、22 架碰撞。关投影由 177/360、170 架碰撞变为 248/360、108 架碰撞，关投影近对头由 0/60 变为 60/60。新节点分场景开投影到达为 60、54、52、60、54、54，六航路 `52/60` 是唯一未达到分场景门槛的一项。该节点未选中，原任务只保留周期更新断点和通过选模的权重。[任务 48 节点原生逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-encoder-freeze-2500.StkJ4G/stage2/development-004420/validation.json)

任务 48 的 10k 冻结权重在 360 架开发对照中，现行、提前触发及逐帧复算三臂逐种子的到达、碰撞、越界、未完成数全部一致。各臂 331/360 到达、26 架碰撞、3 架越界。四航路 52/60、六航路 50/60 低于分场景门槛。按各自预测协议与同一入场权重配对，总到达增量分别为 +7、+7、+5，三臂均未晋升。[冻结 10k 六场景三臂](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-vertical-clearance-probe.76aBWR/frozen-10k-six-families-validated.json) · [同协议晋升复算](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-vertical-clearance-probe.76aBWR/frozen-10k-matched-gates.json)

后续 Stage 2 训练入口会把每次开发评估对应的完整 checkpoint 独立归档，并把路径和 SHA-256 记入 `selection.json`。选中权重通过原子硬链接更新，未晋升断点也可复跑。入口烟测使用小网络验证了未晋升断点留存、选中文件链接及源权重完整性。几何回归 68 通过、1 跳过。运行中的任务 48 使用其提交时源码。[归档功能烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/stage2-checkpoint-archive-smoke.WPo4Oz/verification.json)

任务 48 的 2,000 轮节点为 23,264 更新，五种子开投影 333/360 到达、24 架碰撞，关投影 247/360、110 架碰撞，开关投影近对头均为 60/60。相较 1,503 轮节点，六航路从 52 升到 54/60，四航路回落到 50/60，仍未晋升。[2,000 轮节点逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-encoder-freeze-2500.StkJ4G/stage2/development-005840/validation.json)

冻结与未冻结的等 20k 断点在 `59201–59203` 开投影分别为 207/216、6 架碰撞与 187/216、24 架碰撞。冻结版的六航路 30/36，另五项为 36、36、36、33、36。关投影分别为 156/216、58 架碰撞与 138/216、74 架碰撞；冻结版近对头关投影 36/36。冻结版四案首步最小 Q 局部排序为 3/4、全空域 2/4，未冻结分别为 2/4、2/4。冻结版 STE 反号为负偏航 2/15、正偏航 1/17、四航路 1/14、六航路 3/11。同一 20k 权重另跑标准协议五种子开发确认。[20k 原生等步数对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-encoder-freeze-20000-audit.mnMiEy/native-development-summary.json) · [20k 四案 Q、梯度与权重哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-encoder-freeze-20000-audit.mnMiEy/matched-q-and-sign-summary.json)

该 20k 冻结权重的五种子标准协议确认到达 336/360、碰撞 20 架。六场景到达为 60、56、50、60、54、56。与同协议入场相比净增 12 架，六航路 50/60 未达分场景门槛。此次保留的是 20k 周期权重，与 2,000 轮节点的 23,264 更新权重区分。[五种子晋升复算及 SHA-256](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-encoder-freeze-20000-audit.mnMiEy/native-five-gate.json)

### 任务 48 终轮结算

任务 48 调度为 `SUCCEEDED`，共 2,503 轮、29,024 次更新，单卡已释放。扣除周期评测的训练日志窗口约 2,212 s，4,074 轮/小时、13.12 updates/s。窗口内 221 个设备样本平均 PPU 利用率 75.36%，峰值 HBM 2,774 MiB；计入开发评测后 310 个样本均值 58.30%。任务结算后，9 个工作进程缓存与同哈希索引从 10 个 inode 合为 1 个，释放约 142 MiB 重复数据。[训练和设备统计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-encoder-freeze-2500.StkJ4G/training-window-summary.json) · [关闭进程后的导航缓存去重](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-encoder-freeze-2500.StkJ4G/closed-worker-navdata-deduplication.json)

终轮原生五种子开投影 336/360 到达、22 架碰撞、2 架越界，关投影 251/360 到达、106 架碰撞、3 架越界。开投影到达分场景为 60、54、52、60、52、58，六航路和正偏航均低于 88% 门槛。原始入场开／关投影到达为 324/360、246/360，碰撞为 36、114 架。任务 48 的五个开发节点均未获选，保留的权重 SHA-256 仍为 `224776f45909de543ef4c1e8407888a0f20eff89c72d89e0b2686f22a077e479`。保留种子的既有验收适用于该权重。[任务 48 全部开发节点](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-encoder-freeze-2500.StkJ4G/stage2/selection.json) · [原生终轮逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-encoder-freeze-2500.StkJ4G/stage2/development-007280/validation.json)

冻结版 20k 扩至五种子后仍只有六航路一项低于 88%。同一权重在该场景进一步对比现行、提前触发和逐帧复算，三臂每种子的结果一致，合计均为 50/60 到达、10 架碰撞。[冻结 20k 六航路逐种子预测对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-vertical-clearance-probe.76aBWR/frozen-20k-crossing6-comparison.json)

对这 10 架碰撞做同种子独立原生逐帧复跑，终局计数完全复现。五对接触中，现有近似预测均已给出低于 91.44 m 的高差，2 对调速幅度上界不足，K=4 仍标记 3 对冲突，末次发令的固定水平轨迹下有 4 对无法用一次离散分层得到可行方案。提前约 30 s 的五对仍均无可行分层，约 60 s 时有 2 对可行且都已入冲突图，约 90 s 时有 4 对可行但只有 1 对入图。这个样本表明剩余碰撞需要检查更早的观测与分层决策。离散复算固定历史水平轨迹。[六航路五对逐帧接触与历史可行性](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-encoder-freeze-20000-audit.mnMiEy/crossing6-contacts/diagnostics.json)

约 90 s 的四对可行接触在 20/35/50 km 观测下分别覆盖 1/4、4/4、4/4。冻结 20k 权重用既有原生评测 API 单独覆盖观测距离至 35 km，五种子六航路从 50/60 到达、10 架碰撞改善至 54/60、6 架碰撞。59201、59205 各减少两架碰撞，另外三种子终局一致。用 35 km 口径继续配对入场权重与该冻结断点的完整六场景。[可行窗口覆盖](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-encoder-freeze-20000-audit.mnMiEy/crossing6-contacts/early-observation-coverage.json) · [20/35 km 五种子六航路](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-encoder-freeze-20000-audit.mnMiEy/crossing6-observation-range-paired.json)

任务 48 终轮权重的原生四案首步最小 Q 局部、全空域均为 2/4，与任务 47 终轮相同。终轮冻结 Actor 的 STE 反号为负偏航 1/15、正偏航 1/17、四航路 0/14、六航路 2/11。任务 48 终轮训练权重 SHA-256 为 `c9d91f449c4a2efbca61a50ecd0a242156a01f3403cf7bb2245c494a104d913d`。[终轮 Q、梯度及权重哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-encoder-freeze-20000-audit.mnMiEy/terminal-audit/summary.json)

### 35 km 五种子晋升与正式留出

冻结编码器 20k 候选与入场权重在 35 km 口径下完成原生 0.05 s 五种子配对。各格为到达／碰撞，每场景 60 架。候选 343/360 到达（95.28%）、12 架碰撞，入场 329/360、30 架碰撞，净增 14 架。六个分场景均至少 54/60，近对头 60/60，通过既定开发门控。入场 1 架越界，候选 5 架越界。正式留出使用单独种子 `12001–12005`。[完整同协议配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-encoder-freeze-20000-audit.mnMiEy/five-35000/comparison.json) · [门控与两权重哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-encoder-freeze-20000-audit.mnMiEy/five-35000-gate.json)

| 场景 | 入场 35 km | 冻结 20k 35 km |
| --- | ---: | ---: |
| 三航路 | 60/0 | 60/0 |
| 四航路 | 60/0 | 58/2 |
| 六航路 | 46/14 | 54/6 |
| 近对头 | 60/0 | 60/0 |
| 正偏航 | 52/8 | 55/0 |
| 负偏航 | 51/8 | 56/4 |
| 合计 | 329/30 | 343/12 |

在相同冻结原生轨迹上重建 20 km 和 35 km 图。四案首步最小 Q 局部与全空域排序分别保持 3/4、2/4。独立采样的四航路、六航路、正偏航、负偏航 STE 反号在 20 km 为 4/20、3/17、1/14、7/16，在 35 km 为 3/20、3/18、2/15、7/17。这些是改变动作的控制窗统计。35 km 闭环收益应结合更早的图覆盖判断，冻结批的 Q 排序未显示同幅改善。[观测半径的梯度与 Q 复算](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-encoder-freeze-20000-audit.mnMiEy/range-gradient-audit/verification.json)

正式任务 49 固定候选 35 km 部署副本，SHA-256 `ed5458b5e7fe365737f4ac76c9e82c35e1dad4d93cbe3fb7ad3ccdd306ce64f6`。部署比对逐张量校验，优化器及其他 checkpoint 值相同。原有冻结基线和上一版参考权重按自身观测配置复跑，额外加入入场 35 km 对照，四份权重均跑开／关安全层。单卡确认空闲后提交，正式 Exp 1–6 按候选留出 `>95%` 到达且 `≤3%` 碰撞启动。[正式快照与版本校验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/heldout-freeze-20k-range35.pF3Rmn/preflight.json) · [任务 49 提交回执](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/heldout-freeze-20k-range35.pF3Rmn/submit.log)

为正式阶段核验了 15 份 Figure 4 清单、30 份历史配置、3 份 SARL checkpoint 和使用的脚本哈希。Exp 1–6 单卡快照保留正式门控，先执行 Exp 1 和 Exp 5。Exp 6 的一步回报对照按同一初始化、训练配置和 20k 更新数比对。评测新增实际 checkpoint 的更新数与观测口径一致性校验，既有两臂真实烟测通过，模拟两类错配均被拒。几何回归 68 通过、1 跳过。[Figure 4 输入核验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/heldout-freeze-20k-range35.pF3Rmn/paper-inputs-preflight.json) · [Exp 6 校验烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/heldout-freeze-20k-range35.pF3Rmn/exp6-matched-control-verification.json) · [一步回报输入核验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exp1-6-range35.y2xG5C/one-step-dry-run.json)

任务 49 正式留出已完成，原生模拟与事件采样均为 0.05 s，控制周期 6 s。用 `12001–12005` 核验全部 48 场景臂、240 条逐种子记录和四份模型 SHA-256，独立复算与运行门控一致。候选开投影到达率 89.17%、碰撞率 10.56%，关投影到达率 73.89%、碰撞率 25.56%。正式 Exp 1–6 门槛和冲刺门槛均未通过。该留出结果只用于验收。每格为到达／碰撞，每臂 360 架。[正式逐种子配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/heldout-freeze-20k-range35.pF3Rmn/formal/comparison.json) · [验收和平均单机成本复核](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/heldout-freeze-20k-range35.pF3Rmn/verified-holdout-closure.json)

| 场景 | 冻结基线 开 | 冻结基线 关 | 上一版参考 开 | 上一版参考 关 | 入场 35 km 开 | 入场 35 km 关 | 冻结 20k 35 km 开 | 冻结 20k 35 km 关 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 三航路 | 60/0 | 30/30 | 58/2 | 44/16 | 58/2 | 44/16 | 60/0 | 48/12 |
| 四航路 | 50/10 | 26/34 | 52/8 | 34/26 | 54/6 | 36/24 | 48/12 | 42/18 |
| 六航路 | 44/16 | 26/34 | 50/10 | 34/26 | 42/18 | 27/33 | 44/16 | 32/28 |
| 近对头 | 58/2 | 20/40 | 60/0 | 56/4 | 60/0 | 52/8 | 60/0 | 60/0 |
| 正偏航 | 0/0 | 0/0 | 52/8 | 42/18 | 60/0 | 46/14 | 55/4 | 40/18 |
| 负偏航 | 0/0 | 0/0 | 53/6 | 42/16 | 58/2 | 46/14 | 54/6 | 44/16 |
| 合计 | 212/28 | 102/138 | 325/34 | 252/106 | 332/28 | 251/109 | 321/38 | 266/92 |
| 平均单机成本 | 2.55 | 1.39 | 22.82 | 28.23 | 26.01 | 26.33 | 27.68 | 32.59 |

在独立开发种子 `61231–61235`，同一冻结 20k 权重开投影 20 km 为 329/360 到达、30 架碰撞，35 km 为 333/360、22 架碰撞。六航路从 46/60 升到 54/60，正偏航从 57/60 降到 50/60，负偏航从 56/60 降到 53/60。关投影从 255/360、103 架碰撞变为 263/360、92 架碰撞，近对头两臂均为 60/60。扩大范围存在拓扑与偏航的相反效应。后续用同一开发面板区分 Actor 观测和安全层图的范围。[独立五种子开关投影配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/range35-independent-development.LVwUsa/summary.json)

#### Actor 观测与安全层图的独立拆分

固定冻结 20k 权重和开发种子 `61231–61235`，在原生 BlueSky 0.05 s 上分别切换 Actor 观测和安全层图的范围。每格为到达／碰撞，每臂 360 架。[四臂逐场景汇总](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/actor-projection-range-ablations.80ufhV/summary.json)

| 场景 | Actor 20 / 安全层 20 km | Actor 20 / 安全层 35 km | Actor 35 / 安全层 20 km | Actor 35 / 安全层 35 km |
| --- | ---: | ---: | ---: | ---: |
| 三航路 | 56/4 | 56/4 | 60/0 | 60/0 |
| 四航路 | 54/6 | 54/6 | 54/6 | 56/4 |
| 六航路 | 46/14 | 50/10 | 54/6 | 54/6 |
| 近对头 | 60/0 | 60/0 | 60/0 | 60/0 |
| 正偏航 | 57/2 | 55/4 | 52/4 | 50/6 |
| 负偏航 | 56/4 | 56/4 | 53/6 | 53/6 |
| 合计 | 329/30 | 331/28 | 333/22 | 333/22 |
| 越界 | 1 | 1 | 5 | 5 |

仅扩大 Actor 观测已复现六航路的全部到达增益。两类偏航仍回退，四臂中没有一臂满足全部场景下限。同一组冻结状态的四案最小 Q 排序均为局部 3/4、全域 2/4。下表按修改动作的控制窗计数。[冻结状态逐案 Q 与 STE 梯度](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/actor-projection-range-ablations.80ufhV/frozen-audit.json)

| 反号控制窗 | Actor 20 / 安全层 20 km | Actor 20 / 安全层 35 km | Actor 35 / 安全层 20 km | Actor 35 / 安全层 35 km |
| --- | ---: | ---: | ---: | ---: |
| 四航路 | 4/20 | 4/20 | 3/20 | 3/20 |
| 六航路 | 3/17 | 3/18 | 3/17 | 3/18 |
| 正偏航 | 1/14 | 1/15 | 2/14 | 2/15 |
| 负偏航 | 7/16 | 7/17 | 7/16 | 7/17 |

任务 49 的 2,880 架原生留出评测耗时约 1,063 s，PPU 利用率样本均值 3.81%，峰值显存 514 MiB。240 份工作进程导航缓存共享同一 inode。[正式评测吞吐与导航缓存](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/heldout-freeze-20k-range35.pF3Rmn/throughput-and-navdata.json)

#### 任务 50，35 km 原生 Stage 2 训练对照

复用同一张 SOFY PPU。任务 50 已进入运行状态，先评估 35 km 入场权重，再运行 Plan A 9 workers × 8 空域、原生 0.05 s、2,500 轮全分布微调。对照任务 48 的训练源码、入场 checkpoint、固定编码器、Critic 初始化、随机种子 `83721`、24 步回报、STE、镜像和余弦学习率保持一致。训练只将共享观测和安全层图的范围由 20 km 调至 35 km，梯度损失不变。每 500 轮用同一原生五种子开发面板做开关安全层验证和归档。正式留出由独立的开发准入结果触发。[单卡运行状态](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/task-50-status.txt) · [快照校验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/preflight.json) · [启动差异](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/controlled-change.diff)

本机在两种范围分别完成 32 次真实原生 0.05 s 更新。两臂 Stage 2 入口的解析配置仅在范围和输出路径上有差异。使用固定验证分数测试 checkpoint 路径，开发验证读取范围、最终断点元数据及入场回退范围一致。回退副本除观测配置外所有字段逐项相同，原始入场权重 SHA-256 保持 `224776f45909de543ef4c1e8407888a0f20eff89c72d89e0b2686f22a077e479`。几何测试 68 通过、1 跳过。[原生预检](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/smoke/verification.json) · [几何测试](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/geometry-tests.log)

单卡已开始实际更新。完整核对任务 48 与任务 50 的运行时配置，仅范围和输出路径不同。任务 50 入场六场景开投影到达／碰撞为 331/28，关投影为 244/116。开投影六场景分别到达 60、60、46、60、52、53 架，每场景 60 架。任务 48 同面板的 20 km 入场为 324/36，六航路 52/60。当前 35 km 入场只用于同任务后续断点的开发基准。[实际运行配置对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/runtime-match-verification.json) · [入场逐种子开关投影](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/stage2/development-incoming/validation.json)

从通过主开发面板的训练断点按到达、碰撞、成本依次选峰值，使用预先固定的另一面板 `62441–62445` 做四臂配对。包括任务 50 的 35 km 入场与候选，任务 48 的 20 km 断点及其同权重 35 km 部署副本。各臂均包含开关投影。冻结四案 Q 与梯度新入口已精确复现归档审计，候选产生后用同一冻结状态复算。正式留出种子不参与上述开发。[独立开发协议](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/independent-development-protocol.json) · [对照校验和](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/matched-development-inputs.sha256) · [冻结审计复演](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/audit-rehearsal.json)

##### 首个 500 轮断点

任务 50 在 502 轮、1,520 个控制决策和 5,984 次更新处完成原生 0.05 s 验证。任务 48 对照恰好使用相同决策和更新数，两臂均用 `59201–59205`。任务 50 断点 SHA-256 为 `86df926f54835881702d731ce29548cdc72358f0b3ab1722339418cf0ac73812`。每格为到达／碰撞，每臂 360 架。[原始逐种子结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/stage2/development-001520/validation.json) · [同预算配对与吞吐](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/milestone-audits/first-comparison.json)

| 场景 | 20 km 开 | 35 km 开 | 20 km 关 | 35 km 关 |
| --- | ---: | ---: | ---: | ---: |
| 三航路 | 54/6 | 60/0 | 44/16 | 48/12 |
| 四航路 | 54/6 | 56/4 | 44/16 | 42/18 |
| 六航路 | 50/10 | 56/4 | 28/32 | 30/30 |
| 近对头 | 60/0 | 60/0 | 46/14 | 52/8 |
| 正偏航 | 51/4 | 50/4 | 35/22 | 38/16 |
| 负偏航 | 53/6 | 48/8 | 38/22 | 40/20 |
| 合计 | 322/32 | 330/20 | 235/122 | 250/104 |
| 越界 | 6 | 10 | 3 | 6 |
| 平均单机成本 | 22.71 | 30.31 | 23.89 | 30.10 |

相对 35 km 入场，开投影六航路 46→56 到达，总到达 331→330，碰撞 28→20。双向偏航为 50/60 和 48/60，均低于 88% 下限。本轮未晋升。调度确认任务继续运行，已记录 580 轮和 6,944 次更新。STE 在相同冻结原生状态上的四航路／六航路／正偏航／负偏航反号由 9/20、3/19、7/15、11/18 变为 0/20、2/18、5/15、4/18。四案首步最小 Q 的局部排序由 4/4 降为 2/4，全域保持 3/4。交叉场景到达改善，偏航反号减少后闭环到达仍回退。[入场冻结审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/milestone-audits/incoming.json) · [断点冻结审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/milestone-audits/first.json) · [任务状态](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/task-50-status-after-first.txt)

首段 20/35 km 活动训练吞吐分别为 3,908/3,907 轮每小时和 12.99/12.94 次更新每秒。各 47 个 PPU 样本的利用率均值为 74.64%/71.66%，峰值显存均 2,774 MiB。窗口从实际训练配置落盘到首个开发评测 worker 启动。[窗口与采样汇总](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/milestone-audits/first-comparison.json)

##### 正式论文评测入口预检

新快照接受通过验收的任意 Stage 2 入选断点及实际训练配置，并从原生逐种子 JSON 重算验收，核对 6 s 控制和开关投影各臂。使用任务 49 的真实未达标记录验证了入口拒绝路径。单独检查验收数据绑定时，改动原始行或错误标注投影预算也被拒绝。[论文评测快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/paper-suite/run.sh) · [门控预检](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/paper-suite/admission-verification.json)

Exp 5 保留 9 个轮数臂和 36 个丢包 × 延迟 × 调度臂。通信周期与延迟、MPC 开关交叉后，hold 与 MPC 各 24 臂，共 93 臂。原生 0.05 s 在独立开发种子 `60221` 核验 12 s 周期、0/1/2 步延迟、开关 MPC 与冷热启动共 12 臂，确认实际使用了延迟消息和 MPC 重规划。Exp 6 一步对照执行至入选权重的实际控制决策，并核验更新数、推理观测配置和训练变量。20/35 km 两臂各完成 32 决策、32 更新的真实原生对照预检，拒绝更改种子或决策数的样本。几何回归 68 通过、1 跳过。已关闭的 17 个 smoke worker 缓存从 6 个 inode 去重为 1 个。正式 Exp 1–6 仍由正式留出放行。[通信预检](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/smoke/exp5-cadence-delay/verification.json) · [一步回报预检](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/smoke/one-step-generalized/verification.json) · [缓存去重](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/smoke/canonical-navdata/verification.json)

##### 1,000 轮断点与偏航路由根因对照

任务 50 在 1,004 轮和 11,744 次更新的断点完成同一原生 0.05 s 五种子开发评测。开投影 332/360 到达、16 架碰撞、10 架越界、2 架未到达，关投影 246/360 到达、108 架碰撞。开投影各场景到达为 60、54、50、60、53、55。双向偏航均超过 88% 下限且零碰撞，六航路仍低于下限，总到达比 35 km 入场净增 1 架，未晋升。关投影近对头 60/60。该断点 SHA-256 为 `8d983e4e225004938047b54f4af3e7b51b525ad887156ac19f79769e9c74fd9e`。[原生闭环逐种子数据](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/stage2/development-002960/validation.json)

同组冻结状态上，1,000 轮的 STE 四航路／六航路／正偏航／负偏航反号为 8/20、3/18、0/15、3/18，首步四案 Q 排序局部 2/4、全域 3/4。对比 500 轮，偏航排序方向一致率上升，交叉并未同步改善。[1,000 轮冻结审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/milestone-audits/second.json)

针对 500 轮断点的局部偏航退化，CPU 原生 0.05 s 复跑了正负偏航种子 `59201` 与负偏航 `59205`，入场及训练两臂的全部终局计数与原 PPU 六条记录一致。正偏航 `59201` 新越界的 4 架在末 120 s 共 80 个逐机决策中，航向头始终指向目标且幅度达到指令阈值，动作路由从未选择航向模式，实际航向指令均为零。负偏航 `59201` 新越界的 3 架在末 120 s 均有放行的回航转向，窗口起点航向误差已达 128–143°。负偏航 `59205` 由 4 架碰撞增到 8 架，新增接触的多架到接触前均接近目标方向。[六条逐机轨迹与原生终局匹配](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/first-yaw-diagnostic/summary.json)

在同一 500 轮权重上仅恢复入场动作路由头，正偏航 `59201` 和负偏航 `59201` 均从 8/12 到达、4 架越界变为 12/12、零碰撞。负偏航 `59205` 从 4 到达、8 碰撞变为 8 到达、4 碰撞。编码器逐张量相同，实际改动仅 `router.weight` 和 `router.bias`。原 Critic 重算四案 Q 仍为 2/4、3/4，正负偏航 STE 反号分别为 5/15、5/18。该单变量用于诊断，独立五种子 `61231–61235` 的六场景开关投影对照已启动。[三场景路由头闭环对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/first-yaw-diagnostic/router-only-replay.json) · [参数与权重溯源](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/router-independent/provenance.json) · [路由头冻结审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/router-independent/frozen-q-and-gradient.json)

##### 1,000 轮断点六航路接触复核

开发种子 `59202`、`59203` 在 500 轮断点均 12/12 到达、零碰撞，1,000 轮断点均为 8/12 到达、4 架碰撞。四条 CPU 原生 0.05 s 逐帧回放终局与单卡逐项一致。后一个断点四对实际接触在触发步均已入图，固定各步水平计划并模拟原生垂直动态后，四对在接触前约 60 s 的组件内均无可行单次分层组合。两个 `59202` 接触对在约 90 s 前有可行解且均已入图。触发步 K=4 仍预报三对冲突，对另一对的清除预测与实际原生接触不符。这四对说明需排查更早的策略决策与预测误差，仅观察末段迭代数无法解释回退。[逐帧与历史可行性](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/second-crossing6-diagnostic/summary.json)

任务 50 的正式评测入口沿用任务 49 的冻结基线、历史参考及原生版本，增加本轮 35 km 入场配对。入口先用独立原生开发逐种子数据重算综合门槛，再核对入选断点和四案 Q／梯度审计对应同一个 SHA-256。正式口径保持 0.05 s、控制周期 6 s、五种子 `12001–12005` 和四权重开关投影。当前入口仅预检。[正式启动脚本](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/run-formal.sh) · [比较臂校验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/formal-reference-sha256.txt)

##### 独立路由对照完成与 1,500 轮断点

原生 0.05 s 开发种子 `61231–61235` 的六场景、开关投影共 24 臂、120 条逐种子记录全部复算。下表为到达／碰撞，每臂 360 架。替换路由头的冻结审计见上节，同一 Critic 的四案排序仍为局部 2/4、全域 3/4。[完整逐种子与诊断门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/router-independent/closure.json)

| 场景 | 原断点 开 | 替换路由 开 | 原断点 关 | 替换路由 关 |
| --- | ---: | ---: | ---: | ---: |
| 三航路 | 60/0 | 58/2 | 48/12 | 52/8 |
| 四航路 | 56/4 | 58/2 | 42/18 | 36/24 |
| 六航路 | 50/10 | 52/8 | 32/28 | 36/24 |
| 近对头 | 60/0 | 60/0 | 56/4 | 52/8 |
| 正偏航 | 48/6 | 52/8 | 36/20 | 39/21 |
| 负偏航 | 50/8 | 54/6 | 40/20 | 42/18 |
| 合计 | 324/28 | 334/26 | 254/102 | 257/103 |
| 越界 | 7 | 0 | 4 | 0 |

两向偏航开投影各多到达 4 架，六航路与正偏航 52/60 低于 88% 下限。单换路由的结果仅用于诊断。

任务 50 的 1,500 轮断点开投影 328/360 到达、24 架碰撞，关投影 251/360、102 架碰撞。开投影六场景到达为 60、60、46、60、49、53，本轮未晋升。冻结样本 STE 四航路／六航路／正偏航／负偏航反号为 1/20、3/18、1/15、9/18，四案 Q 排序局部和全域均 1/4。断点 17,504 更新，SHA-256 为 `e7fb4b942d09709d91261bb5b53895c15cde9e130f7c509257238cdf64784ec7`。[原生逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/stage2/development-004400/validation.json) · [冻结审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/milestone-audits/third.json)

冻结路由头的 Stage 2 对照以任务 50 为参照。只新增全局训练开关 `--freeze-actor-router`，保持 35 km、入场权重、原生 9×8、2,500 轮、STE 和损失。提交前已在 CPU 原生 0.05 s 做同种子双臂预检，各 96 次更新。开启后在线和目标 Actor 的路由权重与入场一致，速度、高度、航向三个头都继续更新。配置只有冻结开关和输出路径不同。几何测试 68 通过、1 跳过。该对照快照已通过哈希核验，等任务 50 全部断点选模后决定是否使用同一张单卡。[单变量启动快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/controlled-change.diff) · [实际原生预检](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/smoke/freeze-router/verification.json)

##### 2,000 轮断点与匹配更新对照

任务 50 在 2,000 轮、23,184 次更新完成原生 0.05 s 五种子开发评测。开投影到达／碰撞为 332/20，关投影为 237/118。开投影六场景依次到达 60、58、54、60、49、51 架。六航路相较 1,500 轮恢复 8 架，双向偏航低于 88% 下限，整体较入场净增 1 架，未晋升。断点 SHA-256 为 `55d640ed2cd686c7e6dcf3d742ce01b9b7a9e3487841f76772c9b5581430c637`。固定四案局部／全域 Q 排序为 2/4、1/4。STE 在四航路、六航路、正偏航和负偏航分别反号 4/20、5/18、0/15、11/18。[逐种子结果和门槛复算](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/milestone-audits/fourth-closure.json) · [冻结审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/milestone-audits/fourth.json)

任务 48 与任务 50 各锁定 20,000 次更新和 5,024 个控制决策，均以 35 km 部署。前者训练范围为 20 km，后者为 35 km。两臂在原生 0.05 s 的 `59201–59205` 完成六场景开关投影配对。原本机会话结束后只补跑未写出的两组负偏航关投影结果，核验两份原始报告权重、协议及 120 条逐种子记录后形成闭合表。每格为到达／碰撞。[配对权重溯源](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/matched-update-20000/paired-checkpoints.json) · [完整配对闭合](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/matched-update-20000/closure.json)

| 场景 | 20 km 训练 开 | 35 km 训练 开 | 20 km 训练 关 | 35 km 训练 关 |
| --- | ---: | ---: | ---: | ---: |
| 三航路 | 60/0 | 60/0 | 46/14 | 50/10 |
| 四航路 | 58/2 | 58/2 | 41/19 | 46/14 |
| 六航路 | 54/6 | 48/12 | 32/28 | 26/34 |
| 近对头 | 60/0 | 60/0 | 60/0 | 48/12 |
| 正偏航 | 55/0 | 49/6 | 40/16 | 36/20 |
| 负偏航 | 56/4 | 56/2 | 43/16 | 38/22 |
| 合计 | 343/12 | 331/22 | 262/93 | 244/112 |
| 平均单机成本 | 26.47 | 28.93 | 32.51 | 30.38 |

固定冻结状态下，35 km 全程训练的四案 Q 局部／全域排序为 2/4、2/4，20 km 训练再部署到 35 km 的历史同图审计为 3/4、2/4。前者四航路、六航路、正偏航、负偏航 STE 反号分别为 3/20、6/18、0/15、7/18。20k 同预算配对未支持扩大训练范围本身改善总体到达，主要差距来自六航路与正偏航。[35 km 训练冻结审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/matched-update-20000/frozen-audit.json)

冻结路由对照快照接齐独立开发、正式验收与论文评测入口。Exp 6 用相同训练引擎原生复跑 48 控制步、96 次更新，确认一步回报对照的在线和目标 Actor 路由均与初始化权重相同，观测仍为 35 km。该对照只做预检，尚未占用单卡。[配对评测快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/paper-suite/source-sha256.txt) · [Exp 6 路由冻结预检](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/smoke/freeze-router/one-step-control/verification.json)

##### 任务 50 收尾与冻结路由对照接续

SOFY 确认任务 50 成功退出，退出码 0。全量微调完成 2,503 轮、28,944 次更新，五个训练断点均通过原生 0.05 s 开关投影闭环记录核验。终轮开投影 336/360 到达、18 架碰撞、6 架越界，较入场净增 5 架。六航路 52/60、正偏航 51/60 仍低于 88% 下限。五个周期权重均未晋升，未启动正式留出。终轮 SHA-256 为 `a36a42deef2bdef146d6294a50805b28334f5c84bd53c7c32c743da7c3ba7593`。四案 Q 局部／全域排序均为 1/4，四航路、六航路、正负偏航 STE 反号分别为 3/20、4/18、0/15、6/18。[全部断点、门槛和训练窗口](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/stage2-completed-closure.json) · [终轮冻结审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/milestone-audits/final.json)

| 场景 | 35 km 入场 开 | 终轮 开 | 35 km 入场 关 | 终轮 关 |
| --- | ---: | ---: | ---: | ---: |
| 三航路 | 60/0 | 60/0 | 44/16 | 46/14 |
| 四航路 | 60/0 | 60/0 | 40/20 | 44/16 |
| 六航路 | 46/14 | 52/8 | 32/28 | 28/32 |
| 近对头 | 60/0 | 60/0 | 54/6 | 54/6 |
| 正偏航 | 52/8 | 51/4 | 36/24 | 39/16 |
| 负偏航 | 53/6 | 53/6 | 38/22 | 41/18 |
| 合计 | 331/28 | 336/18 | 244/116 | 252/102 |

活动训练用时 2,227.70 s，4,045 轮/小时、12.99 次更新/s。223 条活动训练 PPU 样本均值 67.83%，峰值 HBM 2,774 MiB。计入周期开发评测，317 条样本均值为 50.51%。退出后 369 份导航缓存由 10 个 inode 硬链接去重为 1 个。[吞吐利用率与存储统计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-observation35-2500.WaY3Bx/stage2-completed-closure.json)

调度确认任务 50 终止且单卡空闲后，提交任务 51。该对照复用单卡、2,500 轮、原生 0.05 s 和 35 km，相对任务 50 只冻结 Actor 路由头。使用同一开发周期验证其能否缓解偏航退化并保住交叉性能。[前序终止状态](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/previous-task-terminal.txt) · [提交回执](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/submit.log) · [调度状态](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/task-51-status.txt)

##### 任务 51 运行配置与投影镜像审计

任务 51 入场 SHA-256 与任务 50 相同，为 `88bde89691b217c066fa793d5afa74113ffe92870da7194ad3a4d7728c971bc4`。开关投影的 60 条逐种子记录逐项核对到达、碰撞、越界、未到达与单机动作成本，全部一致。实际训练配置只有 `freeze_actor_router` 和输出路径两项差异。9×8 空域、35 km、原生 0.05 s 与全分布保持配对。[实际配置及入场权重对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/runtime-match-verification.json)

用训练原本采用的镜像变换，从两条已归档原生开发轨迹抽取左右完全配对的 127 个逐机状态。入场路由在 45/127 对选择不同动作模式，任务 50 终轮为 49/127 对。实际 Actor 只在一个方向提出足以超过 0.25° 死区的正确航向修正，入场为 30/127 对，终轮为 41/127 对。所测 14 对度为零的飞机均在两个方向提出纠偏，模式不一致集中于 113 对在冲突图内的飞机。终轮需正向和负向转向时，分别有 51/127、72/127 对提出正确修正。[按冲突图拆分的镜像审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/mirror-symmetry-audit/task50-and-controls.json)

单独检查安全层时，两侧使用同一份动作的严格物理反射及相同协商参数，五份权重均无超过航向死区的左右指令差。仅恢复 500 轮权重的入场路由后，本批的模式选择和纠偏频数恢复到入场的计数。上述结论范围为冻结样本，独立原生闭环仍用于验证实际效果。[镜像审计实现](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/mirror-symmetry-audit/run.py)

原生闭环再验，固定任务 50 终轮权重，取负偏航开发种子 `59201`、`59205`。反射起点及目标经度、翻转初始偏航，其余物理和相同机号保持一致。对镜像空域逐步执行原始轨迹的反射指令时，两组原生逐机终局和控制步完全复现，3,844 个已对齐逐机控制步中冲突图度数与指令镜像均无差异，航向误差的反射残差最大 0.000031°。在同一镜像空域改用策略独立推理，`59201` 由 9 到达／2 碰撞／1 越界变为 9/0/3，`59205` 由 10 到达／2 碰撞变为 8/4。原始空域两组均与任务 50 的 PPU 开发记录一致。[原生镜像闭环三臂](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/native-mirror-pair/summary.json)

对 `59201` 原始和已对齐的反射轨迹，在同机同刻评估策略的原始回航提议。图内有 861 个需纠偏的逐机配对控制步，需正转时 19 次、需负转时 203 次提出正确转向，模式有 327 次不同。图外 60 个逐机配对均在两向提出纠偏。`59205` 图内仅 36 个需纠偏配对，其中 5 次正转、3 次负转正确。方向差随流量而变化。以上是控制步计数，终局影响以两条原生配对回放为准。[原生对齐状态统计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/native-mirror-pair/paired-states.json)

##### 任务 51 首断点与同预算冻结对照

任务 51 在 501 轮、5,984 次更新、1,520 个控制决策完成原生五种子开发评估，与任务 50 的首断点更新数和决策数完全相同。实际 checkpoint 验证在线与目标 Actor 的路由头逐位等于入场，速度、高度和航向头有更新。任务 51 断点 SHA-256 为 `f0f2d3738db8a5d0cf3dc550bd566c0eb05bff44708d986b6fa913e40a4b38dd`。表内为 0.05 s 逐帧到达／碰撞。[实际权重及配对结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/milestone-audits/first-runtime-weights.json) · [逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/stage2/development-001520/validation.json)

| 场景 | 任务 50 开 | 冻结路由 开 | 任务 50 关 | 冻结路由 关 |
| --- | ---: | ---: | ---: | ---: |
| 三航路 | 60/0 | 60/0 | 48/12 | 46/14 |
| 四航路 | 56/4 | 58/2 | 42/18 | 46/14 |
| 六航路 | 56/4 | 52/8 | 30/30 | 32/28 |
| 近对头 | 60/0 | 60/0 | 52/8 | 46/14 |
| 正偏航 | 50/4 | 56/4 | 38/16 | 42/18 |
| 负偏航 | 48/8 | 54/6 | 40/20 | 38/22 |
| 合计 | 330/20 | 340/20 | 250/104 | 250/110 |
| 越界 | 10 | 0 | 6 | 0 |

冻结路由的开投影总到达较入场多 9 架，双向偏航均通过 88% 下限，六航路 52/60 距场景下限差 1 架，未晋升。六航路的场景变化来自 `59201` 从 10 到 12、`59202` 从 12 到 8、`59205` 从 10 到 8。关投影总体到达与同预算对照相同，碰撞多 6 架。[完整门槛核验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/milestone-audits/first-closure.json)

四案首步 Q 局部／全域排序由任务 50 的 2/4、3/4 变为 2/4、2/4。冻结路由四航路、六航路、正负偏航 STE 反号分别为 3/20、4/18、3/15、8/18。同一批 127 组镜像状态的动作模式不一致数和双向纠偏分类保持入场频数。首段活动训练 3,864 轮/h、12.82 更新/s，47 样本 PPU 均值 66.98%，峰值显存 2,774 MiB。相同预算任务 50 为 3,907 轮/h、12.94 更新/s、71.66%。任务 51 继续原定训练预算。[STE 与 Q 审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/milestone-audits/first.json) · [镜像审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/mirror-symmetry-audit/task51-first.json) · [匹配贯通测量](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/milestone-audits/first-paired-throughput.json)

##### 首断点原生交叉路由替换

固定两个同预算首断点，在 CPU 原生 BlueSky 0.05 s 上互换路由头。每一列先标其余 Actor 参数的来源，再标路由头来源。实际断点的 8 条原生对照均逐项复现 PPU 到达、碰撞和越界。替换前确认共享编码器相同，其他参数与保护层保持原样。表内为到达／碰撞／越界，限所选 4 条开发流量。[16 条逐机轨迹和权重哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/first-router-cross-swap/summary.json)

| 开发流量 | 50 + 50 | 50 + 51 | 51 + 50 | 51 + 51 |
| --- | ---: | ---: | ---: | ---: |
| 六航路 59202 | 12/0/0 | 10/2/0 | 8/4/0 | 8/4/0 |
| 六航路 59205 | 10/2/0 | 8/4/0 | 8/4/0 | 8/4/0 |
| 正偏航 59201 | 8/0/4 | 12/0/0 | 10/0/2 | 12/0/0 |
| 负偏航 59201 | 8/0/4 | 12/0/0 | 12/0/0 | 12/0/0 |

在任务 50 的其余参数上，冻结路由所保留的入场路由足以恢复两条偏航流量。用任务 51 的其余参数时，换入任务 50 路由也未恢复上述两条六航路流量，表明回退仍涉及其余动作头的训练。替换结果仅用于解释原生闭环差异，未用于选模。

##### 任务 51 的 1,000 轮断点

1,001 轮断点与任务 50 同为 11,744 次更新、2,960 个控制决策，路由头和目标路由头仍与入场逐位一致。冻结组开投影到达／碰撞为 334/26，同预算对照为 332/16。冻结组各场景开投影依次为 58/2、58/2、50/10、60/0、56/4、52/8，无越界。相对入场净增 3 架，六航路和负偏航低于 88% 场景下限，未晋升。关投影冻结组由同预算对照的 246/108 改善为 256/104，近对头两臂均到达 60/60。冻结组断点 SHA-256 `609eb1d337ec0fc999d2d71434dbd4fdcd7f175f5479c54edf41dc6d277b0bc8`。[实际权重配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/milestone-audits/second-runtime-weights.json) · [原生逐种子和门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/milestone-audits/second-closure.json)

固定四案首步 Q 局部／全域均 2/4。STE 四航路、六航路、正偏航、负偏航反号分别为 3/20、2/18、4/15、10/18。继续按原周期验证剩余训练。上一节 16 条已退出的诊断回放复用同一物理导航缓存 inode。[第二断点 Q 与梯度](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/milestone-audits/second.json)

##### 任务 51 的 1,500 轮断点与精确镜像路由诊断

1,503 轮断点与任务 50 同为 17,504 次更新、4,400 个控制决策，在线及目标路由仍与入场逐位相同。开投影冻结组为 326/360 到达、34 架碰撞，同预算对照为 328/360、24 架碰撞。冻结组开投影各场景依次为 58/2、58/2、48/12、60/0、52/8、50/10，均无越界。关投影冻结组为 266/360、94 架碰撞，较同预算对照净增 15 架、少碰撞 8 架。该断点未晋升。断点 SHA-256 为 `9a60b2bab523757caa1cb1021afefb9c0bcb5ee3f4d7ea4f1cf87cb88ddc1e3c`。[原生逐种子和综合门槛](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/milestone-audits/third-closure.json) · [实际权重对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/milestone-audits/third-runtime-weights.json)

固定四案首步 Q 局部／全域均 2/4，四航路、六航路、正负偏航 STE 反号为 9/20、4/18、2/15、6/18。[第三断点冻结审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/milestone-audits/third.json)

独立诊断仅对任务 51 首断点的左右镜像路由 logits 取平均，其他 Actor 输出与未经修改的模型逐位相同。镜像变换逐位匹配当前训练增强。50 个冻结窗口中验证了镜像模式和 logits 完全一致，539 个活跃逐机动作中有 51 个改变路由模式。在 127 个需要纠偏的镜像配对上，原先只在单侧提出正确转向的 30 对降为 0，两侧都提出正确转向从 58 对增加到 77 对。四案 Q 仍为局部／全域 2/4，四航路、六航路、正负偏航 STE 反号为 3/20、5/17、2/15、9/18。诊断不修改训练代码，另行在 `61231–61235` 验证原生六场景开关投影。[策略清单及样本哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/symmetric-router-diagnostic/symmetric-router-policy.json) · [镜像纠偏复算](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/symmetric-router-diagnostic/mirrored-headings.json) · [诊断冻结 Q 和 STE](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/symmetric-router-diagnostic/q-and-ste.json)

上述独立闭环已覆盖 6 场景 × 4 臂 × 5 种子，共 120 条场景种子记录。两臂使用同一断点、同一 35 km 观测及原生 0.05 s 物理。每格为到达／碰撞，每臂 360 架。[独立闭环结果及报告哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/symmetric-router-diagnostic/closure.json)

| 场景 | 原路由 开 | 对称路由 开 | 原路由 关 | 对称路由 关 |
| --- | ---: | ---: | ---: | ---: |
| 三航路 | 60/0 | 60/0 | 50/10 | 54/6 |
| 四航路 | 58/2 | 56/4 | 40/20 | 40/20 |
| 六航路 | 52/8 | 52/8 | 34/26 | 28/32 |
| 近对头 | 60/0 | 60/0 | 56/4 | 56/4 |
| 正偏航 | 58/2 | 54/6 | 44/16 | 40/20 |
| 负偏航 | 55/4 | 56/4 | 42/18 | 40/20 |
| 合计 | 343/16 | 338/22 | 266/94 | 258/102 |
| 越界 | 1 | 0 | 0 | 0 |

对称路由开投影六航路仍低于 88%，同预算各两臂均出现净退化。原生 CPU 诊断各场景开投影平均控制决策耗时范围从 81.1–87.7 ms 增至 126.7–135.0 ms。精确对称约束在这批样本上未转化为整体安全收益，诊断方案不进入正式推理。

### 任务 51 收尾与原生安全层改造

任务 51 完成 2,504 轮、28,944 次更新。2,005 轮和终轮的冻结四案 Q 排序均为局部／全域 3/4，开投影终轮为 328/360 到达、32 架碰撞，六航路 44/60；关投影为 268/360、92 架碰撞。所有断点均未晋升，按新目标结束冻结路由长程微调。终轮 SHA-256 为 `f16b0b0df360bcd26d7b849842ab9ec14dc6bbc823b2f87cea550f40c300b6c4`。活动训练实测 4,033 轮/h、12.95 次更新/s，223 个 PPU 样本均值 69.72%，峰值 HBM 2,774 MiB。已退出的 369 份缓存由 10 个 inode 合为 1 个。[终局、全部断点和吞吐](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/stage2-completed-closure.json) · [终轮冻结 Q 与 STE](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-frozen-router35-2500.ZfgzTY/milestone-audits/final.json)

新安全层开放三机搜索与提前触发、宽裕度及临近间隔保持联合使用。名义间隔优先，裕度只在名义约束满足条件下参与排序。对预计 6 s 内进入水平间隔带的邻机，Torch 在原生 0.05 s 网格上复算加速度、限速、目标高度捕获和纵向量化，并按配对分块计算重叠窗最低间隔。保留既有默认配置作对照，原生和方案 B 评测及训练均接通显式选项。CPU 的 768 条、每条 1,201 帧的测试轨迹与既有逐帧原型完全一致。几何回归 70 通过、1 跳过。[逐帧原型对齐](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/torch-oracle-parity.json) · [回归结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/geometry-tests-current.log)

在两份固定 35 km 权重上做原生 0.05 s 小样本逐变量对照，六航路种子为 `59202–59203`，正偏航为 `59201–59202`。每格为到达／碰撞。[逐种子和控制耗时](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/pilot-summary.json)

| 安全层 | 入场 六航路 | 20 km 训 / 35 km 部署 六航路 | 入场 正偏航 | 20 km 训 / 35 km 部署 正偏航 |
| --- | ---: | ---: | ---: | ---: |
| 现行 | 20/4 | 22/2 | 22/2 | 20/0 |
| 提前触发 | 20/4 | 22/2 | 22/2 | 20/0 |
| 提前触发与逐帧预测 | 20/4 | 22/2 | 22/2 | 20/0 |
| 提前触发、逐帧预测与 6 m 裕度 | 24/0 | 24/0 | 22/2 | 20/0 |

小样本中，加入 6 m 裕度后两份权重共多到达 6/96 架、少碰撞 6 架，正偏航终局相同。冻结 6 m 配置后升格验证，先测 512 空域 × 20 个机位的混合与密集计算耗时和显存，再跑完整原生开发及正式留出的安全层开关对照。旧安全层同批复跑开关投影，两份权重不训练也不改写。[单卡零训练评测快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/run-ppu.sh) · [输入 SHA-256](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/evaluation-inputs.sha256)

SOFY 已确认任务 51 正常终止后提交任务 52，唯一 PPU 已处于原生开发基线评测阶段。512 空域 × 20 机位的混合面板中，现行／仅提前／加精确／再加 6 m 的安全层中位耗时为 7.03／7.07／9.41／25.89 ms。密集面板为 25.73／25.38／31.15／107.21 ms。6 m 配置的额外峰值分配分别为 303.8 MiB 和 826.9 MiB。测得最重的开销来自加裕度之后的搜索，尚未将该配置投入 512 空域长训。[前序终态](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/previous-task-terminal.txt) · [任务 52 回执](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/submit.log) · [512 空域同步测量](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/ppu-512-layer-benchmark.json)

等变 Actor 已在独立于任务 52 的主线实现并接入 `--equivariant-actor`、训练目标网络、采样以及 checkpoint 加载。原图和镜像图拼为 2B 前向，路由 logits、速度、高度、rho 与 willingness 取偶对称平均，航向与边对偶反号平均，模式在合并后采样。原有回放镜像和 Actor 共用观测变换。单元测试覆盖奇偶字段、边交换结构及梯度，全部几何回归 71 通过、1 跳过。已在同一新安全配置下启动最佳固定权重的零训练小样本闭环，先观察 `61231–61232` 六航路和双向偏航。[等变回归](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/geometry-tests-equivariant.log) · [等变诊断源码哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/equivariant-actor-pilot/actor-source-sha256.txt)

任务 52 已完成原生五种子开发集 `59201–59205`。锁定的新安全层使入场权重由 331/360 到达、28 架碰撞变为 337/360、22 架碰撞；最强权重由 343/360、12 架碰撞变为 339/360、16 架碰撞。最强权重六航路由 54/60 到达增至 56/60，正偏航由 55/60 降至 51/60，负偏航由 56/60 降至 54/60。两份权重的原始关投影终局分别为 244/116、262/93。完整开发结果不支持直接替换最强权重的原保护配置。按同一锁定配置继续比对留出，未用留出选裕度。[开发全量配对与报告哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/development-closure.json)

同一 6 m 精确安全层下，结构等变零训练小样本开投影由 68/72 到达、2 架碰撞变为 64/72、8 架碰撞。六航路同为 22/24，正偏航同为 22/24，负偏航由 24/24 降至 20/24。关投影由 53 到达、18 碰撞变为 46 到达、26 碰撞。固定四案 Q 的局部／全域排序保持 3/4、2/4，负偏航 STE 反号从 9/19 降至 6/19。冻结状态的改善没有转化为这批闭环收益。训练未启动，完整 `61231–61235` 推理对照已在 CPU 启动，两臂使用同一实际权重和安全层。[小样本原生终局](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/equivariant-actor-pilot/summary.json) · [等变冻结 Q 和 STE](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/equivariant-actor-pilot/equivariant-frozen-audit.json) · [独立五种子快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/equivariant-actor-independent/source-sha256.txt)

### 计算与存储交接

轻量单元测试、局部轨迹分析和开发流量探针在控制环境执行。训练和高吞吐正式评估复用已授权的 SOFY 单卡 PPU。共享 workspace 保存代码与报告，环境、模型、checkpoint、数据、缓存和试验产物保存在 CPFS。长训命令显式设置 HF、Torch、pip 与临时目录。调优续训写入独立输出目录并校验输入权重。当前未委派子代理。[CPFS 实验目录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123)

### 任务 52 正式留出与论文摸底

任务 52 正常退出。原生 BlueSky 以 0.05 s 逐帧检测，控制周期 6 s，锁定 `12001–12005`、六场景、每场景 60 架。相同权重内成对比较旧安全层、新安全层与关闭投影。每格为到达／碰撞。[冻结六场景结果与哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/holdout-closure.json)

| 场景 | 预锁定权重 旧安全层 | 预锁定权重 新安全层 | 预锁定权重 关投影 |
| --- | ---: | ---: | ---: |
| 三航路 | 60/0 | 60/0 | 48/12 |
| 四航路 | 48/12 | 54/6 | 42/18 |
| 六航路 | 44/16 | 51/9 | 32/28 |
| 近对头 | 60/0 | 60/0 | 60/0 |
| 正偏航 | 55/4 | 58/0 | 40/18 |
| 负偏航 | 54/6 | 57/2 | 44/16 |
| 合计 | 321/38 | 340/17 | 266/92 |

预锁定的 20 km 训练、35 km 部署权重净增 19 架到达，少碰撞 21 架，平均单机动作成本由 27.681 变为 28.239。入场权重旧／新／关闭投影分别为 332/28、344/16、251/109，平均单机成本 26.015、26.517、26.327。论文摸底沿用先验锁定候选，SHA-256 为 `ed5458b5e7fe365737f4ac76c9e82c35e1dad4d93cbe3fb7ad3ccdd306ce64f6`。其 94.44% 到达触发论文摸底，4.72% 碰撞仍高于最终碰撞上限。保留既有严格晋升门槛。[旧层及关投影原始记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/holdout-baseline/comparison.json) · [新层原始记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/holdout-exact-margin6/comparison.json)

复用单卡 PPU 的任务 53 曾启动 Exp 1，后续由续跑任务 55 承接。执行顺序为 Exp 1、Exp 5、Exp 2/3、Exp 4 及 Exp 6。Exp 1 使用 15 份锁定统一流量清单及三臂原生 0.05 s 对照；Exp 5 同口径预检为轮数 9、通信周期 Hold 8、MPC 空窗 8、丢包／延迟／着色 36，共 61 个配置。冻结快照使用任务 52 实测的精确安全层求解器和同一权重，不混入正在验真的分块优化。[任务 53 回执](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-baseline.akxmXp/submit.log) · [门槛与输入哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-baseline.akxmXp/acceptance.json) · [运行状态](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-baseline.akxmXp/status.txt)

三机层搜索改为只枚举重叠图内相连的三机组，保留原优先级。2,295 个随机空域和 512 个固定轨迹采样空域的输出逐位一致。单线程 CPU 的精确加 6 m 面板中位耗时从 5,512.8 ms 降至 4,107.4 ms（每臂 3 次），线上 512 PPU 耗时仍待复测。几何回归 71 通过、1 跳过。[逐位复算与计时](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/packed-triple/cpu-reference-verification.json)

### 等变 Actor 五种子独立开发对照

同一锁定权重和 6 m 精确层在 `61231–61235` 的原生 CPU 六场景开发集上，开启安全层从 344/360 到达、12 架碰撞变为 345/360、14 架碰撞，关投影从 263/360、92 架碰撞变为 250/360、110 架碰撞。六航路开投影两臂均 54/60，正／负偏航从 55／57 变为 56／55，近对头均 60/60。开投影平均批决策耗时由 79.0 增至 129.0 ms，关投影由 51.1 增至 104.9 ms。这些是 CPU 进程实测，512 空域的 PPU 耗时与显存仍待测。继续在锁定正式种子用 CPU 两臂对照，保持相同输入与真实逐帧终局核验。[标准版交叉](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/equivariant-actor-independent/standard-crossing/comparison.json) · [等变版交叉](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/equivariant-actor-independent/equivariant-crossing/comparison.json) · [标准版回航](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/equivariant-actor-independent/standard-return/comparison.json) · [等变版回航](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/equivariant-actor-independent/equivariant-return/comparison.json)

### 等变正式配对与短程精修

原生 BlueSky 0.05 s 的正式留出 `12001–12005` 独立 CPU 两臂已完整闭合。使用同一权重、35 km 冲突图及 6 m 精确三机层。标准 Actor 的逐场景终局和任务 52 的 PPU 记录一致。表内为到达／碰撞。[两臂逐种子、单机成本和源码输入](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/equivariant-actor-heldout.H3LfyA/closure.json)

| 场景 | 标准开 | 等变开 | 标准关 | 等变关 |
| --- | ---: | ---: | ---: | ---: |
| 三航路 | 60/0 | 60/0 | 48/12 | 36/24 |
| 四航路 | 54/6 | 54/6 | 42/18 | 31/29 |
| 六航路 | 51/9 | 54/6 | 32/28 | 24/36 |
| 近对头 | 60/0 | 60/0 | 60/0 | 58/2 |
| 正偏航 | 58/0 | 58/2 | 40/18 | 34/26 |
| 负偏航 | 57/2 | 53/6 | 44/16 | 31/28 |
| 合计 | 340/17 | 339/20 | 266/92 | 214/145 |
| 平均单机动作成本 | 28.239 | 26.198 | 32.592 | 28.085 |

等变开投影达到 94.17% 到达、5.56% 碰撞，六航路到达多 3 架，负偏航少 4 架。按短程验证的 94% 触发线进入精修，论文摸底沿用预锁定的标准 Actor，最终晋升仍需独立验证。四案冻结 Q 以干预机排序为门槛，入场为 3/4，全活跃机为 2/4。任务 54 复用原生方案 A 的 9×8 空域与 40/30/30 混合流量，运行上限 800 轮，学习率为 0.000024（0.00012 的 1/5）并作余弦衰减。每 200 轮以五种子开发集评估开关投影；Q 低于 3/4 或连续两次低于峰值则早停。有开发晋升才复跑正式留出。任务先对归档的 512×20 面板同卡复测分块与穷举安全层，仅在逐位一致且两个面板均提速时选分块。CPU 的方案 B 小型用例通过 4 次实际更新，几何回归 71 通过、1 跳过。20k 等变方案 B 启动入口已准备，当前按短程例外执行。[任务 54 回执](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-equivariant-refine.ybB3dR/submit.log) · [冻结原生训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-equivariant-refine.ybB3dR/run.sh) · [四案静态复跑](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-equivariant-refine.ybB3dR/tmp/incoming-q-smoke.json) · [最小实际训练](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/training-path-smoke.9VTNba/run.log)

复查发现任务 53 后续 Exp 6 原计划读取 35 km 训练配置。实际摸底权重的源断点在 20 km 下完成 20,000 次更新。本地已验证其 Actor、Critic 和两个目标网络张量与当前 35 km 部署权重逐位相同。为保持单变量消融，任务 55 从任务 53 已落盘的完整 Figure 4 JSON 继续，Exp 6 取原生 20 km 原训练配置和对应源码快照，只改评论家单步回报，再统一在 35 km 部署口径评估。任务 54 与 55 串行复用同一张 PPU。[已完成原始回放](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-baseline.akxmXp/exp1) · [任务 55 回执](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/submit.log) · [更正后的论文套件快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/run.sh)

任务 53 于 Exp 1 阶段取消，已完成前三份 Case 1 清单的三臂结果，共 9 个 JSON。任务 54 目前占用 1 张 PPU，任务 55 已排队，续跑时以内容校验复用这 9 个结果。任务 53 尚未执行 Exp 5 或错误配置的 Exp 6。[取消回执](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/replaced-task53.txt) · [续跑作业](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/submit.log)

任务 54 已完成 PPU 512×20 复测。每臂 20 次同步样本，穷举和分块输出逐位一致。混合面板中位数由 25.69 降至 13.41 ms（降 47.8%），密集面板由 105.97 降至 30.30 ms（降 71.4%）。额外峰值 HBM 分别保持 303.76 MiB 和 826.89 MiB。512×20 的 Actor 前向从 29.89 升至 59.23 ms，额外峰值 HBM 从 1,919.14 升至 3,883.69 MiB。以上是固定状态前向计时。任务 54 已采用分块版并进入原生入场闭环，训练吞吐和新权重终局在阶段产物落盘后汇报。[PPU 基准](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-equivariant-refine.ybB3dR/ppu-packed-benchmark.json) · [已选求解器](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-equivariant-refine.ybB3dR/layer-solver.txt) · [精修实时状态](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-equivariant-refine.ybB3dR/status.txt)

#### 任务 54 早停与后续调度

任务 54 正常结束。首个 200 轮检查点已进行 2,460 次更新，活动吞吐 3,057 轮/h、10.45 更新/s，训练 PyTorch 峰值 2,763 MiB。SOFY 五分钟窗的 5 个样本均值为 64.4%，含检查点评估等待，当前显存 3,862 MiB。[训练日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-equivariant-refine.ybB3dR/stage2/training/metrics.jsonl) · [SOFY 采样](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-equivariant-refine.ybB3dR/ppu-monitor-200.txt)

同一开发集原生开投影从 345/360 到达、14 架碰撞退化到 336/360、18 架碰撞，关投影从 250/360、110 架碰撞退化到 236/360、120 架碰撞。开投影平均单机动作成本由 27.008 降到 24.088，六航路到达由 54/60 降到 52/60，负偏航由 55/60 降到 52/60，近对头保持 60/60。干预机四案 Q 从 3/4 降到 2/4，全活跃机维持 2/4。按门槛停止后续训练，选择仍为等变入场权重，冻结正式集继续保持零训练配对记录。[选模与早停](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-equivariant-refine.ybB3dR/stage2/selection.json) · [首检查点开关投影逐种子](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-equivariant-refine.ybB3dR/stage2/development-000639/validation.json) · [Q 核验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-equivariant-refine.ybB3dR/stage2/development-000639/frozen-q.json)

任务 55 已在原授权单卡上执行 Exp 1，核验并复用了旧任务的 9 个完整结果。任务 56 已在同一卡队列等待，使用 35 km、等变 Actor、已通过 PPU 复测的分块精确安全层和 512 空域，做 20,000 次 24 步回报与投影动作学习。每 5,000 更新存档，以原生开发配对和四案 Q ≥3/4 选择候选；正式留出仅用于验收。[论文续跑状态](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/status.txt) · [20k 任务回执](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-exact-20k.taihJG/submit.log) · [冻结训练清单](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-exact-20k.taihJG/inputs.sha256)

#### 任务 55 运行中校验（原生精确安全层）

对正在运行的任务 55 独立核验 15 份 Figure 4 流量清单和 30 份历史配置。快照时有 12/45 个完整回放，涵盖 Case 1 的前四份三臂配对。已落盘的回放逐臂核对了权重和清单哈希、原生 0.05 s 发机帧、物理判定阈值及每臂 50 架的结果。四份完整配对的 Geo、Hold+KSS、SARL+KSS 到达／碰撞分别为 200/0、198/2、198/2。这组为 Case 1 阶段数据，完整报告仍需 Case 2/3 与第五种子。[已完成配对及逐文件哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp1-in-progress-audit.json)

用任务 55 同一冻结源码、已发布权重与新安全层，在非留出种子运行三项原生 CPU 烟测。18 秒周期在 12 个控制步中实际触发 4 次 ADMM 和 8 次 MPC 空窗重算，这批状态无需调整。着色调度配置 50% 丢包和两步延迟，冷热两臂均记录尝试 480、丢失 232、送达 204、期末在途 44 次，实际使用 100 份延迟对状态，最短年龄为两步。80 架／17 航路烟测实测峰值在场 80，短窗冲突图最大度 3、最大团 2。三个试验均实际执行原生 0.05 s 帧，限于功能验证。[MPC 回放](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-rehearsal.QXRJEg/mpc/summary.json) · [丢包延迟着色回放](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-rehearsal.QXRJEg/delayed-colored/summary.json) · [规模回放](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-rehearsal.QXRJEg/scale/summary.json)

同样的冻结精确安全层在独立开发种子 `89031` 完成基础流量与联合扰动配对。20 架基础流量为 12 到达／8 碰撞，联合流量加入一架突发飞机后为 13 到达／8 碰撞。两臂基础航路哈希一致。突发飞机在计划的 96 s 插入，非合作飞机全程保持指定目标，联合组最大实测风速为 17.00 m/s，原生地速分解残差上界为 2.01e-14 m/s。单种子结果只验证扰动接线，正式五种子结果由任务 55 产出。[联合配对及结果哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-rehearsal.QXRJEg/disturbance-parity.json) · [联合组逐项证据](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-rehearsal.QXRJEg/combined.json)

核查任务 53、54、55 与两批本地烟测的 151 个导航缓存路径，共用两个 inode（PPU 和 CPU 各一个，合计约 31.5 MiB）；现有缓存复用有效。SOFY 实查任务 55 正在运行，任务 56 在同一张 PPU 队列等待。[任务 55 实时状态](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/status.txt)

#### Figure 4 Case 1 闭合与剩余碰撞的单步反事实

任务 55 的 Case 1 五种子已完整闭合。三臂使用同一 50 架发机清单和同一物理判定阈值，核对了发机帧与权重哈希。五种子合计 Geo-ADMM-RL 250/250 到达、零碰撞，SARL+KSS 248/250、2 架碰撞，Hold+KSS 246/250、4 架碰撞。三臂平均单机动作成本分别为 6.416、5.787、0.967。Case 2/3 仍由同一论文任务继续运行。[Case 1 五种子逐臂结果和哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp1-case1-audit.json)

扩展碰撞诊断以复用当前 35 km 权重、等变开关、提前触发、6 m 裕度和精确 B737 预测。选择开发集 `yaw_minus15/61231`（等变多碰撞 4 架）及 `crossing6/61235`（等变少碰撞 2 架），四条原生 0.05 s 轨迹逐一复现历史到达和碰撞数。共五个接触对在最后一个控制步都被冲突图覆盖，回算未见单纯从 K=2 增至 K=4 便清除预测冲突的接触。两处接触前高度间隔为 103.44 m 和 93.23 m，随后执行的新目标造成穿越 91.44 m 层间界限。[逐帧接触、前 120 s 预测、逐机命令与哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/matched-collision-audit.sSpy6o/closure.json)

仅取消一个机位的一次相对高度增量，使原高度目标保持一个 6 s 控制步。两次干预前的 73／76 个控制步日志逐字节一致，保存的动作张量只改变所选机位的垂向分量。负偏航 `[0,1]` 441.95 s 的接触消失，448.20 s 出现 `[0,4]`；六航路 `[0,9]` 457.55 s 的接触消失，459.10 s 出现 `[7,9]`。两轮终局各维持 8/12 到达、4 架碰撞，单步保高的局部收益被邻机后续冲突抵消。当前冻结训练与评测配置保持原样，几何回归 71 通过、1 跳过。[反事实干预和原始结果哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/matched-collision-audit.sSpy6o/hold-last-target-closure.json)

#### 两组剩余碰撞的多机可行性证书

原生 0.05 s 复盘的新增冻结状态入口，在负偏航 `61231` 和六航路 `61235` 的原轨迹各提取四个决策（距所选碰撞控制步 0／30／60／90 s）。两段控制日志与前一轮原轨迹逐字节相同。保持原速度和航向指令，采用直线水平预测、B737 垂向逐帧积分及六种有界高度方案，以整数约束在全在场飞机上求最少改令架数。八个状态的 120 s 窗都在这一有限动作与预测模型下不可行。负偏航最后一个控制步的 6 s 窗有一机保高解，延长至 18 s 即不可行，符合原生单步保高把冲突转移至另一邻机的观测。六航路最后一组第 76 步连 6 s 窗都不可行。这些证书约束了冻结状态下的净避让空间，未对可变速、可转向或多步控制序列作不可行判断。[冻结状态、位姿、动作和约束求解结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/matched-collision-audit.sSpy6o/joint-window.BW0Puf/joint-certificate.json) · [求解器脚本](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/matched-collision-audit.sSpy6o/joint-window.BW0Puf/audit.py)

六航路标准 Actor 的第 61 步存在覆盖 60 s 的一机高度替代方案。将机位 1 的垂向指令从保持改为向上 12 m，仅改一个标量、只执行这个 6 s 控制步，之后将控制交还原策略。干预前 61 步轨迹逐字节一致。原生终局从 **8/12 到达、4 架碰撞** 改善到 **10/12、2 架碰撞**。原有 368.30 s 接触消失，后段仍发生一对接触。这是单条开发流量的闭环净收益，冻结训练与论文评测仍采用原配置。几何回归为 71 通过、1 跳过。[单标量动作、原生接触时间及结果哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/matched-collision-audit.sSpy6o/joint-window.BW0Puf/certified-feedback-closure.json)

SOFY 实查任务 55 正运行 Case 2，Figure 4 已落盘 22/45 臂。任务 56 已入队，等待同一张 PPU。[任务 55 状态](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/status.txt)

#### 临近冲突规则的独立验证与精修验收入口

基于前一轮 60 s 证书，隔离评估单一运行时规则。只替换垂向动作，要求此前冻结投影后的 6 s 预测冲突对数严格减少，并保留所有 18 s 窗内既有的安全配对。八个冻结几何镜像的决策逐位一致。规则在六航路 `61235` 的冻结第 61 步选中同一 12 m 替代动作。但规则连续运行时全局执行两次干预，完整轨迹仍为 8/12 到达、4 架碰撞，上一轮只改一次的局部收益未能复现。[冻结镜像验真](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/matched-collision-audit.sSpy6o/near-window-paired.j2VL5N/frozen-audit.json) · [连续应用的原生结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/matched-collision-audit.sSpy6o/near-window-paired.j2VL5N/diagnostic-crossing/comparison.json)

在独立种子 `67601–67603` 的原生 0.05 s 配对上，固定 35 km 等变权重和精确安全层，两臂四场景结果逐项相同。六航路 32/36 到达、4 架碰撞，负偏航 32/36、4 架，正偏航 31/36、4 架、另 1 架越界，近对头 36/36、零碰撞。每臂总计 **131/144 到达、12 架碰撞**。候选规则识别六次临近风险决策，18 s 约束下未执行改令，因而保留冻结训练和论文评测配置。[独立闭环及统计、权重哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/matched-collision-audit.sSpy6o/near-window-paired.j2VL5N/closure.json)

为方案 B 已通过开发集和冻结 Q 门槛后的短程原生精修，补齐 500–1,000 轮范围、200 轮检查和最终轮数边界。精修早停保留已通过开发集的方案 B 权重时，仍可以做锁定留出验收。精确配置的严格验收要求同一 0.05 s 口径、五种子、六场景、35 km 等变候选和开关投影的 ADMM 预算，发布要求到达率大于 95% 且碰撞率不超过 3%。历史 94% 摸底许可仍通过独立验真。真实历史等变 339/360、20 架碰撞的结果被严格验收拒绝，几何回归 **72 通过、1 跳过**。[真实历史回归结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/matched-collision-audit.sSpy6o/near-window-paired.j2VL5N/historical-regression-rejected.json)

#### Figure 4 Case 2 闭合

任务 55 的 Case 2 已完成五种子三臂原生对照。独立核对 15 个臂的发机帧、每臂 50 架结果、物理阈值、权重及历史脚本哈希。Geo-ADMM-RL 为 **250/250 到达、零碰撞**，SARL+KSS 为 **248/250、2 架碰撞**，Hold+KSS 为 **242/250、8 架碰撞**。三臂平均单机动作成本分别为 5.705、3.733、1.379。Figure 4 共闭合 30/45 个回放。任务 55 继续 Case 3，任务 56 等待同一张 PPU。[Case 2 五种子原始指标与回放哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp1-case2-audit.json)

#### 等变 Actor 到物理 ADMM 的镜像接线修复

对四、六航路和正负偏航的 256 个真实冻结状态，逐项反射局部观测、东西位置和东西地速分量。原等变 Actor 的 logits、速度、高度、航向及边对偶均按约定逐位变换，原始模式无差异。但原版把随镜像反号的边对偶直接用作速度、高度 ADMM 的标量热启动，导致最终指令在 **128/256** 个状态出现镜像差异。速度差最大 4.5 kt，高度差最大 260.63 ft。保持镜像两臂所用热启动数值相同，或同时置零这两个热启动，均消除全部指令差异，定位到对偶接线。[修复前逐项审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/actor-safety-equivariance.CF8wgS/symmetry.json)

Actor 继续输出反号边对偶，同时从同一组头生成镜像平均的标量速度／高度热启动。物理投影、采样、Critic 投影目标、STE、协调损失及原生评测统一使用同号热启动；非等变 Actor 沿用既有数值。重复上述 256 状态测试，执行指令镜像差异降为 **0/256**，Actor 的输出奇偶关系和模式仍逐位一致。方案 B CPU 单步入口完成 4 次真实更新并写入同号热启动元数据，几何回归 **72 通过、1 跳过**，历史 94% 摸底许可仍可验真。[镜像回归](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/actor-safety-equivariance.CF8wgS/symmetry-corrected.json) · [学习烟测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/actor-safety-equivariance.CF8wgS/learn.log) · [证据哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/actor-safety-equivariance.CF8wgS/evidence-sha256.txt)

原生 0.05 s 独立五种子 `61231–61235` 以同一 35 km 等变权重隔离热启动改动。开投影两臂均为 **345/360 到达、14 架碰撞、1 架越界**。六航路到达由 54 升至 56，正偏航由 56 升至 58，四航路由 60 降至 58，负偏航由 55 降至 53，三航路和近对头保持 60/60。冻结镜像的数学一致性已验证，开发流量的负偏航收益仍需学习获得。未用正式留出挑选热启动版本。[六场景、五种子闭环及配对哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/native-exact-joint.lJpK3j/actor-safety-equivariance.CF8wgS/dev-closure.json)

调度确认任务 56 尚在队列且无任何训练记录。先提交同一 PPU 的更正快照任务 57，再取消任务 56。任务 57 沿用 512 空域、35 km、20,000 更新、每 5,000 更新做原生开发配对及四案 Q，新增同号热启动证据校验。运行中的任务 55 和论文基线快照保持原样。[替换回执](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/submit.log) · [旧任务取消记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/replaced-task56.txt) · [新训练输入与源码哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/inputs.sha256)

#### Exp 1 完整闭合，Exp 5 启动

任务 55 的原生 BlueSky Figure 4 已生成 **15 份统一流量清单 × 3 个策略臂，共 45 个回放**。五种子覆盖 `12001–12005`，每份 50 架，按原生 0.05 s 逐帧计结局，控制周期 6 s。下表为到达／碰撞／平均单机动作成本。所有结果无越界与超时，Case 3 三臂均全到达。[正式 Figure 4 汇总](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp1/summary.json)

| Case | Geo-ADMM-RL | SARL+KSS | Hold+KSS |
| --- | ---: | ---: | ---: |
| 1 | 250 / 0 / 6.416 | 248 / 2 / 5.787 | 246 / 4 / 0.967 |
| 2 | 250 / 0 / 5.705 | 248 / 2 / 3.733 | 242 / 8 / 1.379 |
| 3 | 250 / 0 / 3.667 | 250 / 0 / 1.739 | 250 / 0 / 0.438 |
| 合计／平均 | 750 / 0 / 5.263 | 746 / 4 / 3.753 | 738 / 12 / 0.928 |

新增统一套件核验入口，从冻结门槛复算权重及报告哈希，并区分已闭合实验和待运行面板。已在本次 45 个实际 Figure 4 回放验证同发机序列和全部原始文件，并用未完成的实测回放验证缺臂拦截。Exp 5 已转入正式原生运行，首个 K=0 配置完整闭合六场景 × 五种子，共 **266/360 到达、92 架碰撞、2 架越界**，运输与控制延迟面板仍在前方。当前快照独立记录 48 个结果文件哈希，Exp 5 已验证 1/61 个配置。任务 57 等待同一张已授权 PPU，独立六场景的已冻结安全层摸底仍为 340/360 到达、17 架碰撞。[套件验真快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json) · [K=0 原始闭环](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp5/round-budget/k0-raw/comparison.json) · [训练排队回执](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/submit.log)

#### Exp 5 固定轮数前五臂闭合

任务 55 已完成 5/61 个通信配置，覆盖 K=0 及 K=1、K=2 的冷热启动。每臂均为原生 0.05 s、六场景、五种子 360 架，固定实验轮数并停用额外密集轮数；正预算臂共用冻结的精确分层安全配置。批决策耗时为六场景等权平均。

| 配置 | 到达 | 碰撞 | 越界 | 平均单机动作成本 | 平均批决策 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| K=0 | 266 | 92 | 2 | 32.592 | 4.740 |
| K=1 热启动 | 345 | 14 | 1 | 30.086 | 14.312 |
| K=1 冷启动 | 334 | 24 | 2 | 27.359 | 14.137 |
| K=2 热启动 | 335 | 24 | 1 | 28.216 | 14.914 |
| K=2 冷启动 | 337 | 22 | 1 | 27.535 | 14.932 |

K=1 热启动相对 K=0 的五种子到达增量分别为 15、14、22、12、16，碰撞下降为 14、14、22、12、16。K=1 热启动比同预算冷启动多到达 11 架、少碰撞 10 架，成本增加 2.727。K=2 已呈现不同排序，未由此调整冻结训练和部署口径。套件核验现在从匹配原始回放按配置生成逐种子差分、成本和耗时，并在 CPFS 原子提交出现短暂无效句柄时有限重读，已用真实正式产物验证。回归 72 通过、1 跳过。SOFY 仍显示任务 55 正在运行、任务 57 等待同一卡。[验证快照及文件哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json) · [轮数原始分面汇总](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp5/round-budget/summary.json) · [训练任务回执](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/submit.log)

#### Exp 5 固定轮数九臂闭合

固定预算分面九臂均已完成，任务 55 已启动通信周期 Hold 分面。新增四臂为同一原生 0.05 s 六场景、五种子、每臂 360 架，保持所有其余控制参数固定。

| 配置 | 到达 | 碰撞 | 越界 | 平均单机动作成本 | 平均批决策 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| K=3 热启动 | 340 | 18 | 2 | 27.758 | 16.084 |
| K=3 冷启动 | 337 | 20 | 3 | 27.471 | 16.611 |
| K=5 热启动 | 341 | 16 | 3 | 27.595 | 18.954 |
| K=5 冷启动 | 338 | 18 | 4 | 27.351 | 18.892 |

固定预算内，K=1、2、3、5 的热启动相对同预算冷启动净到达分别为 +11、−2、+3、+3，碰撞减少分别为 +10、−2、+2、+2。多数冷热配对的收益随种子改变。K=1 热启动为分面中到达最高的一臂，K=5 热启动相较该臂多耗时 4.64 ms，少到达 4 架、多碰撞 2 架。以上为锁定留出的消融证据，冻结部署安全配置保持原样。套件核验按相同 K、通信周期、丢包、延迟和调度条件生成冷热逐种子差分。固定预算分面汇总 SHA-256 为 `b6bd187f7c6926c204880660886376c34704c6cfd530796013b0a8616f2e469c`。核查 290 个当前导航缓存路径共用同一 15.73 MiB inode。[九臂原始摘要](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp5/round-budget/summary.json) · [扩展配对和产物哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json)

#### Exp 5 通信周期 Hold 分面闭合

任务 55 已闭合 Hold 空窗的四档周期、冷热启动共 8 臂，Exp 5 累计核验 **17/61** 个配置。每臂固定 K=2、无额外密集轮数，原生 BlueSky 0.05 s 六场景五种子共 360 架。耗时为六场景等权的平均批决策耗时。

| 周期 | 热启动到达／碰撞／越界 | 冷启动到达／碰撞／越界 | 热／冷实际协调次数 | 热／冷平均批决策 ms |
| --- | ---: | ---: | ---: | ---: |
| 6 s | 335 / 24 / 1 | 337 / 22 / 1 | 5,725 / 5,733 | 15.597 / 15.818 |
| 12 s | 335 / 20 / 5 | 330 / 26 / 4 | 2,925 / 2,932 | 9.637 / 9.692 |
| 18 s | 337 / 20 / 3 | 337 / 19 / 4 | 1,979 / 1,991 | 7.528 / 7.554 |
| 30 s | 329 / 25 / 6 | 330 / 24 / 6 | 1,226 / 1,237 | 5.870 / 5.965 |

6 s 组冷热都逐种子复现固定 K=2 对照的终局及平均单机动作成本。18 s 组相对 6 s 组实际协调减少约 65%，热启动多到达 2 架，冷启动到达数持平；两臂碰撞分别少 4、3 架，越界较多。30 s 组协调进一步减少，两臂均出现到达退化。结果已按通信触发步、投影次数及包计数逐种子审核。核验器在真实 MPC 与延迟烟测日志上验证了空窗步数和消息年龄，并拒绝了篡改协调次数、收包总数或最小消息年龄的输入。几何回归 72 通过、1 跳过。任务 55 已实际启动 MPC 分面，任务 57 仍在同一张 PPU 排队。[Hold 原始分面](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp5/cadence-hold/summary.json) · [逐种子配对、传输计数及产物哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json)

#### Exp 5 MPC 空窗分面闭合

任务 55 已闭合 MPC 空窗四档周期、冷热启动共 8 臂，并完成丢包／延迟／着色分面的首个臂，Exp 5 累计核验 **26/61**。同一组原生 0.05 s 六场景五种子固定 K=2。下表为每臂 360 架终局，括号内为同周期、同启动方式相对 Hold 的净到达及碰撞下降。

| 周期 | 热启动到达／碰撞／越界（净到达、碰撞下降） | 冷启动到达／碰撞／越界（净到达、碰撞下降） |
| --- | ---: | ---: |
| 6 s | 335 / 24 / 1（0、0） | 337 / 22 / 1（0、0） |
| 12 s | 343 / 13 / 4（+8、7） | 346 / 8 / 6（+16、18） |
| 18 s | 342 / 12 / 6（+5、8） | 338 / 12 / 10（+1、7） |
| 30 s | 339 / 8 / 13（+10、17） | 332 / 8 / 20（+2、16） |

6 s 周期没有空窗，两臂均逐种子复现 Hold 终局，MPC 重规划次数为 0。12 s 冷启动 MPC 实测重规划 2,959 次、调整 235 架次，将累计模型预测接触由 195 降至 78，增加平均批决策 1.466 ms。30 s 下冷热各只剩 8 架碰撞，但越界升至 13 和 20 架。模型预测接触的累计数与原生终局碰撞分别记账。消融结果用于论文通信对照，冻结正式部署仍为独立六场景 340/360 到达、17 架碰撞。

套件审核按相同 K、周期、启动方式和五种子配对 MPC 与 Hold，对实际重规划、改令架次及预测接触进行核算；实测坏计数拦截通过，几何回归 72 通过、1 跳过。MPC 分面摘要 SHA-256 为 `61cc87d9b10f53c95ba7db8979f2391b46ab2ddba63f3c0a9ceec376f424b39b`。本轮核查过的 695 个导航缓存路径共用一个 15.73 MiB inode。SOFY 确认任务 55 运行网络分面，任务 57 等待同一张 PPU。[MPC 原始分面](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp5/cadence-mpc/summary.json) · [八组 Hold 配对与哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json)

#### Exp 5 零丢包热启动延迟三档

在固定 K=2、热启动 Jacobi、0% 丢包下，原生 0.05 s 六场景五种子配对结果如下。延迟以 6 s 控制步为单位，实际年龄按使用次数加权。

| 配置延迟 | 到达／碰撞／越界 | 平均单机动作成本 | 平均批决策 ms | 实际使用延迟状态次数 | 消息年龄最小／均值／最大 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 步 | 335 / 24 / 1 | 28.216 | 15.775 | 0 | — |
| 1 步 | 346 / 10 / 4 | 27.232 | 16.535 | 43,219 | 1 / 1.073 / 57 |
| 2 步 | 333 / 24 / 3 | 29.044 | 16.246 | 40,212 | 2 / 2.110 / 147 |

延迟一步相对同配置零延迟多到达 11 架、少碰撞 14 架。五种子净到达分别为 +1、+4、0、0、+6。延迟两步相对零延迟少到达 2 架、碰撞持平。三个配置分别尝试、送达 82,552、89,982、86,998 次交换，结束时无未投递交换。延迟设置控制首次可送达时刻；接收端按有效边读取最近送达状态，边暂时移出冲突图后重新出现时可再次使用缓存，故实际年龄可远高于配置延迟。最大年龄 147 步见于近对头种子 `12002`。该缓存语义属于冻结通信消融，仅用来解释本次配对。三档已将 Exp 5 核验推至 **28/61**。[逐种子计数、配置与原始文件哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json)

20% 丢包零延迟首臂已归档，Exp 5 核验 **29/61**。同样固定 K=2、热启动 Jacobi，85,012 次交换实际丢失 16,937 次（19.923%），送达 68,075 次。终局为到达 333、碰撞 25、越界 2，平均单机动作成本 28.824、平均批决策 16.322 ms。相对同条件 0% 丢包少到达 2 架、多碰撞 1 架；五种子净到达依次为 −2、−3、0、0、+3。原始配对 SHA-256 为 `a651568f5f75a4958bcddec9a42dae13d90d24420ebe7be3db8abe062b3e1001`。[原始配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp5/loss-delay-coloring/k2-warm-loss-0.2-delay-0/comparison.json)

#### Exp 5 热启动 Jacobi 20% 丢包三档闭合

任务 55 已完成热启动 Jacobi 下首次完整的 20% 丢包三档延迟，Exp 5 核验 **31/61**。固定 K=2，原生 0.05 s 六场景五种子，每臂 360 架。配对差分固定延迟，以同条件零丢包为对照。

| 延迟控制步 | 到达／碰撞／越界 | 相对零丢包到达／碰撞变化 | 实测丢包率 | 平均单机动作成本 | 平均批决策 ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 333 / 25 / 2 | −2 / +1 | 19.923% | 28.824 | 16.322 |
| 1 | 340 / 16 / 4 | −6 / +6 | 19.806% | 27.382 | 16.825 |
| 2 | 329 / 28 / 3 | −4 / +4 | 19.892% | 28.032 | 16.666 |

一步延迟的六架差异来自种子 12002（少到达 4 架、多碰撞 4 架）和 12004（少到达 2 架、多碰撞 2 架）。两步延迟下三组种子变化分别为 12001 少到达 4 架、12003 少 2 架，12004 多 2 架。一步／两步实际使用延迟状态 42,874／39,462 次，平均消息年龄 1.124／2.133 步。冻结网络消融仍沿用有效边的最近消息缓存，最长消息年龄 153／76 步。一步和两步原始报告 SHA-256 分别为 `3b5f6bbd33c2d05605c427f2dabb3f6c9b500d185626e3cc515740d8f0005cf3` 和 `67f72be746302db80bd51c156eac2c2481e75669aa623aee6c3e8aa2513a1229`。

套件核验器新增匹配丢包、延迟及调度的单变量差分，并汇总实际丢包率和按使用次数加权的消息年龄。完整配置集合核验得到 66 组单变量配对，另用真实回放与畸变年龄测试复核。几何回归 72 通过、1 跳过。任务 55 已进入 50% 丢包，任务 57 等待同一 PPU。核查时任务 55 的 966 个导航缓存路径共用一个 15.73 MiB inode。[配置、配对和原始产物哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json)

#### Exp 5 热启动 Jacobi 网络九臂闭合

热启动 Jacobi 的 0%、20%、50% 丢包 × 0、1、2 步延迟九臂已齐，Exp 5 核验 **34/61**。固定 K=2，原生 BlueSky 0.05 s，六场景、五种子、每配置 360 架。新增 50% 面板的实测如下，配对变化以同延迟的零丢包为对照。

| 延迟控制步 | 到达／碰撞／越界／未完成 | 到达／碰撞配对变化 | 实测丢包率 | 平均单机动作成本 | 平均批决策 ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 337 / 22 / 1 / 0 | +2 / −2 | 49.929% | 29.557 | 16.158 |
| 1 | 346 / 10 / 3 / 1 | 0 / 0 | 49.972% | 28.965 | 16.888 |
| 2 | 339 / 18 / 3 / 0 | +6 / −6 | 49.955% | 27.089 | 16.914 |

50% 丢包一步延迟的总到达和碰撞与零丢包持平，但种子 12002 少到达 3 架，12003 和 12004 合计多到达 3 架。唯一未完成出现在种子 12002 的负偏航场景。两步延迟的五种子净到达依次为 +1、+2、+1、−2、+4。五种子汇总随丢包率非单调；一步、两步下使用的缓存状态平均年龄分别为 1.407、2.412 步，均高于同延迟的零丢包配置。三臂原始报告 SHA-256 依次为 `fb62cc2ec00dd61574d20be663c974ef14c9cf7235c44ae5ae265cf3517abef7`、`1217ea6879b391353fe19a0c098b4ad52d4b74931b258d5ec9e5dfbb87e2b9ab`、`da0d10825f711a003b22609edf39341bf98b581649a773a73c9f9f443183529c`。任务 55 已进入冷启动 Jacobi，冻结部署与训练准入继续按各自的原生评测口径核验。[九臂逐种子配对、通信计数和哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json)

#### Exp 5 冷热 Jacobi 零丢包延迟配对闭合

任务 55 已完成冷启动 Jacobi 的 0、1、2 步零丢包延迟，并启动冷启动 20% 丢包。Exp 5 累计核验 **37/61**。固定 K=2、35 km，原生 BlueSky 0.05 s 六场景五种子，每配置 360 架。

| 延迟控制步 | 冷启动到达／碰撞／越界 | 热启动到达／碰撞／越界 | 冷／热单机动作成本 | 冷／热平均批决策 ms |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 337 / 22 / 1 | 335 / 24 / 1 | 27.535 / 28.216 | 16.200 / 15.775 |
| 1 | 341 / 16 / 3 | 346 / 10 / 4 | 27.776 / 27.232 | 16.751 / 16.535 |
| 2 | 327 / 30 / 3 | 333 / 24 / 3 | 34.329 / 29.044 | 16.445 / 16.246 |

零延迟冷启动逐场景和逐种子复现固定 K=2 冷启动的 30 次回放，终局、单机成本与控制步数均一致。热启动在一步延迟多到达 5 架，差异来自四航路 +2、六航路 +2、正偏航 +1；两步多到达 6 架，来自四航路 +4、六航路 +2、负偏航 +2、正偏航 −2。两档延迟的五种子差分有正有负。两步冷启动还在回放结束时留下 2 次在途交换，已单独计入 87,138 次总交换的守恒检查。新增三臂原始报告 SHA-256 依次为 `c235d1a36e9b00a0c8b1d19bb971859b4ea3b9a615a211ebb0cdcdf415bd76e6`、`d40eba739844551623f2982afa1202a65d8771e5780e5af74925902d0f8bbd7a`、`9890866d3f613cabcd7bdf9df422beffb9dc019dcdb350f4b070e00e7757061d`。

审核扩充至冻结原生安全配置、35 km 观测、投影状态、记录冲突图及通信随机种子。在实际回放上拒绝了单独改动联合分层裕度、观测半径和通信种子偏移的输入。几何回归 72 通过、1 跳过。[逐种子配对、实测通信计数与原始哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json)

#### Exp 5 冷启动 Jacobi 20% 丢包三档闭合

任务 55 已闭合冷启动 Jacobi 20% 丢包的三档延迟，Exp 5 累计核验 **40/61**。本组仍为原生 BlueSky 0.05 s、固定 K=2、35 km、五种子六场景，每臂 360 架。配对差分以同延迟冷启动零丢包为对照。

| 延迟控制步 | 到达／碰撞／越界 | 相对零丢包到达／碰撞变化 | 实测丢包 | 单机动作成本 | 平均批决策 ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 335 / 24 / 1 | −2 / +2 | 19.928% | 27.715 | 16.355 |
| 1 | 335 / 18 / 7 | −6 / +2 | 19.821% | 27.790 | 16.983 |
| 2 | 334 / 22 / 4 | +7 / −8 | 19.795% | 32.138 | 16.937 |

一步延迟的 6 架到达损失来自六航路多碰撞 2 架、正负偏航各多越界 2 架。两步延迟的净到达主要来自正负偏航，四航路多到达 2 架，六航路少到达 2 架。相对同丢包同延迟热启动，冷启动三档净到达为 +2、−5、+5，碰撞变化为 −1、+2、−6；其中两步档在偏航多到达 11 架、六航路少到达 6 架。一步／两步实际使用缓存状态 42,491／40,465 次，平均年龄 1.122／2.149 步。三档原始报告 SHA-256 依次为 `c95e9518f30fe82a738ea9d9384a16ab94b188401d9007d6314a498851516129`、`1a562a2992b2259ae6f5c1f64240728c0d1cc20b2679899a571bea07ca2113c6`、`eb099528a4004d839d08a17997f0b367ed9e8c6fd7807c5944a03a1b863b9a03`。任务 55 已进入冷启动 50% 丢包，任务 57 仍等待同一张 PPU。[逐种子配对、实测通信与产物哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json)

#### Exp 5 Jacobi 网络对照闭合

Jacobi 调度的 0%、20%、50% 丢包 × 0、1、2 步延迟 × 冷热启动共 18 个配置已全部闭合，形成 9 组原生冷热配对。Exp 5 核验 **43/61**，任务 55 已开始图着色。所有结果使用 K=2、35 km、原生 BlueSky 0.05 s 六场景和五种子。新增的 50% 丢包冷启动结果如下。

| 延迟步 | 冷启动到达／碰撞／越界／未完成 | 热启动到达／碰撞／越界／未完成 | 冷启动实测丢包率 | 冷／热单机动作成本 | 冷／热批决策 ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 340 / 18 / 2 / 0 | 337 / 22 / 1 / 0 | 50.012% | 27.373 / 29.557 | 16.524 / 16.158 |
| 1 | 323 / 34 / 3 / 0 | 346 / 10 / 3 / 1 | 49.927% | 27.702 / 28.965 | 17.180 / 16.888 |
| 2 | 331 / 26 / 3 / 0 | 339 / 18 / 3 / 0 | 49.911% | 29.615 / 27.089 | 17.015 / 16.914 |

50% 丢包一步延迟下，热启动净增到达 23 架且减少碰撞 24 架，五种子到达差依次为 +2、+7、+10、+2、+2。三、四、六航路分别多到达 4、8、8 架，两个偏航场景合计多到达 3 架，近对头持平。两步延迟净增 8 架，种子 12004 减少 2 架。零延迟下冷启动多到达 3 架。全部九组里热启动到达领先 5 组、冷启动领先 4 组。Jacobi 实际尝试交换为每协调步约 14.42–15.43 次；着色后同一轮内按颜色更新，后续将与 Jacobi 一并核算实际交换量和决策耗时。

新增三档冷启动原始 SHA-256 依次为 `01eb69cbe1ae64c9f117c473b2c622a0640b98ce9d6a7b6b1f7b4ac6bd3bed0a`、`29ca7db88eeb0e676aa0a8706aca9de87767fea8a0fcc18a596bb21bd3ce222d`、`a5fc078d732955d207003b54eb0a84e690217513fb358d503ec500d7a8c44739`。任务 57 等待同一张 PPU。[完整原生冷热配对、通信计数和源哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json)

#### Exp 5 热启动零丢包图着色配对闭合

零丢包热启动下，0、1、2 步延迟的图着色全部与 Jacobi 完成原生五种子配对，Exp 5 核验 **46/61**。两调度固定 K=2、35 km、0.05 s、六场景，每配置 360 架。交换量为总尝试交换除以协调步数。

| 延迟步 | 着色到达／碰撞／越界 | Jacobi 到达／碰撞／越界 | 着色／Jacobi 每协调步交换 | 着色／Jacobi 批决策 ms |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 338 / 20 / 2 | 335 / 24 / 1 | 29.151 / 14.420 | 21.078 / 15.775 |
| 1 | 346 / 10 / 4 | 346 / 10 / 4 | 30.869 / 15.434 | 22.004 / 16.535 |
| 2 | 333 / 24 / 3 | 333 / 24 / 3 | 29.677 / 14.838 | 22.194 / 16.246 |

零延迟着色增加 3 架到达、减少 4 架碰撞，单机成本从 28.216 升到 30.696。按求解次数加权的速度／高度原始残差从 0.0862 m/s／1.660 m 降至 0.0679 m/s／1.331 m。1、2 步有延迟的 60 组逐场景逐种子记录，各项物理统计、成本、控制步数、冲突图、残差和消息年龄均精确一致，尝试和送达交换数精确翻倍，批决策增加 5.469／5.949 ms。现有延迟信道只在步开始时交付数据，同一步中较早颜色的发送不进入后续颜色的共识更新。零丢包时同边同到达步保留最后一个状态。此观察适用于当前冻结信道，20%、50% 丢包仍单独核验。

审核器现输出每协调步交换量和求解次数加权残差，并以原生回放验证实测值和非法残差拦截。几何回归 72 通过、1 跳过。1,366 个导航文件路径共用单个 15.73 MiB inode。着色三臂原始 SHA-256 依次为 `2e591907d67f316631587004843e044a56579025d83773f62d9b910a821b9f74`、`7994737bd776496fc5e185c11a9a6aca9d4a52f1cb3d072f77dc9fe1452836a9`、`9417254095ea2aaae13c57296cf645af2494cb786fc60adfa890b3a62be19cbd`。任务 55 已启动热启动 20% 丢包着色，任务 57 等待同一张 PPU。[逐种子配对、交换残差与哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json)

#### Exp 5 热启动着色 20% 丢包配对闭合

热启动图着色的 20% 丢包三档已与同延迟 Jacobi 完成原生配对，Exp 5 核验 **49/61**。两调度固定 K=2、35 km、0.05 s，六场景五种子共 360 架每配置。

| 延迟步 | 着色到达／碰撞／越界 | Jacobi 到达／碰撞／越界 | 着色／Jacobi 每协调步尝试交换 | 着色／Jacobi 批决策 ms | 着色／Jacobi 单机动作成本 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 340 / 18 / 2 | 333 / 25 / 2 | 29.677 / 14.852 | 21.715 / 16.322 | 30.578 / 28.824 |
| 1 | 344 / 12 / 4 | 340 / 16 / 4 | 30.536 / 15.317 | 22.875 / 16.825 | 26.742 / 27.382 |
| 2 | 333 / 24 / 3 | 329 / 28 / 3 | 29.783 / 14.629 | 22.736 / 16.666 | 29.785 / 28.032 |

零延迟着色多到达 7 架、少碰撞 7 架。一步多到达 4 架、少碰撞 4 架，差异均来自六航路的种子 12002 和 12004。两步多到达 4 架、少碰撞 4 架，其中负偏航 +4、四航路 +2、六航路 −2；种子 12004 少到达 2 架。着色实际丢包为 19.964%、19.854%、19.926%，每协调步的尝试交换约为 Jacobi 两倍，平均批决策增加 5.394、6.050、6.070 ms。对照固定 ADMM 轮数，实际尝试交换量随调度变化。在延迟信道，同边同到达步最后一次被接收的状态生效；有丢包时，附加发送机会也随调度变更。

着色三档原始报告 SHA-256 依次为 `9700b916678da4959fbe5c7c0405ef8df87312d4c1ccc3841e38744b46c82b78`、`e03f06cad5388b3689bb535e7e64a29842148397a09532c1703bf4096e6e2d04`、`9633b140f7efe34abb08071aec1869f93163240d2164a39e3fd17478be1a73f1`。任务 55 已开始 50% 丢包热启动着色，任务 57 等待同一张 PPU。[逐种子配对、实测通信与哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json)

#### Exp 5 热启动图着色九组配对闭合

0%、20%、50% 丢包 × 0、1、2 步延迟的九组热启动调度配对已闭合，Exp 5 核验 **52/61**。任务 55 已切入冷启动图着色。下表为最后三组 50% 丢包对照，固定 K=2、35 km、原生 BlueSky 0.05 s 六场景五个留出种子，每配置 360 架。

| 延迟步 | 着色到达／碰撞／越界／未完成 | Jacobi 到达／碰撞／越界／未完成 | 着色／Jacobi 每协调步尝试交换 | 着色／Jacobi 批决策 ms | 着色／Jacobi 单机动作成本 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 338 / 20 / 2 / 0 | 337 / 22 / 1 / 0 | 29.515 / 14.714 | 21.813 / 16.158 | 29.894 / 29.557 |
| 1 | 336 / 20 / 4 / 0 | 346 / 10 / 3 / 1 | 29.859 / 14.696 | 22.940 / 16.888 | 26.940 / 28.965 |
| 2 | 338 / 18 / 4 / 0 | 339 / 18 / 3 / 0 | 30.216 / 14.740 | 22.753 / 16.914 | 29.436 / 27.089 |

50% 丢包一步延迟，着色比 Jacobi 少到达 10 架、多碰撞 10 架。四、六航路分别少到达 4、8 架，正偏航少到达 1 架，负偏航多到达 3 架；五种子差值为 0、−3、−3、−2、−2。着色的加权速度和高度残差由 0.628 m/s、5.428 m 降至 0.574 m/s、5.328 m，飞行终局同时退化。两步延迟碰撞总数持平，着色多越界 1 架。其 176,764 次交换中 4 次未在回放结束前抵达，已完成守恒核算。

九组热启动配对中，着色到达数领先 5 组、持平 2 组、落后 2 组；对照固定轮数，每协调步尝试交换约为 Jacobi 的两倍。50% 丢包三档着色原始 SHA-256 依次为 `4587557168e23e0b2a99eb0540a1080d1a8c9c9ffc7a33a31feecd2adbf41cac`、`fcd6322784b2c280058a3f4da8841a13b4bfde126e322160f3765402c71b66e0`、`3641db32d10ed7208a8d3508ffd6f95308f3db2f8146927c5368aa70b716fdd9`。[全部热启动逐种子调度配对、残差及源哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json)

#### Exp 5 冷启动零丢包图着色配对闭合

冷启动图着色的零丢包三档已与同延迟 Jacobi 完成原生配对，Exp 5 核验 **55/61**。固定 K=2、35 km、BlueSky 0.05 s，六场景五个留出种子，每配置 360 架。

| 延迟步 | 着色到达／碰撞／越界 | Jacobi 到达／碰撞／越界 | 着色／Jacobi 每协调步尝试交换 | 着色／Jacobi 批决策 ms | 着色／Jacobi 单机动作成本 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 340 / 18 / 2 | 337 / 22 / 1 | 29.806 / 14.489 | 21.299 / 16.200 | 26.983 / 27.535 |
| 1 | 341 / 16 / 3 | 341 / 16 / 3 | 30.046 / 15.023 | 21.801 / 16.751 | 27.776 / 27.776 |
| 2 | 327 / 30 / 3 | 327 / 30 / 3 | 29.143 / 14.572 | 21.722 / 16.445 | 34.329 / 34.329 |

零延迟图着色多到达 3 架、少碰撞 4 架。四航路和负偏航各多到达 2 架，正偏航少到达 1 架并多越界 1 架。一步和两步的冷启动回放逐场景、种子精确复现 Jacobi 的物理、成本、残差与消息年龄统计，尝试、送达和在途数严格翻倍。两步实际在途为图着色 4 次、Jacobi 2 次。纳入热启动后，零丢包正延迟的 120 组逐场景逐种子记录均满足这一等价性；这与冻结信道按控制步交付、同边同到达步保留最后状态的实现相符。

三档着色原始报告 SHA-256 依次为 `64434bfc91ab7c4ccb030efc6fc5c83958f04cc57fe4b681e3a0700b98b9dd2e`、`a52ce6d65621fb29e27026d355f7dbe4dac68f3484dbdbf3144ac97ab205eaea`、`81d689404913b856f3c9a0f7a934154bc6d905c1ede298a23334907b5c04264b`。任务 55 已启动冷启动图着色的 20% 丢包三档，任务 57 仍等待同一张 PPU。[逐种子配对、实际交换与产物哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json)

#### Exp 5 冷启动着色 20% 丢包配对闭合

冷启动图着色的 20% 丢包三档已与同延迟 Jacobi 完成配对，Exp 5 核验 **58/61**。两调度固定 K=2、35 km，使用原生 BlueSky 0.05 s、六场景五种子和每配置 360 架。

| 延迟步 | 着色到达／碰撞／越界 | Jacobi 到达／碰撞／越界 | 着色／Jacobi 每协调步尝试交换 | 着色／Jacobi 批决策 ms | 着色／Jacobi 单机动作成本 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 334 / 24 / 2 | 335 / 24 / 1 | 28.876 / 14.708 | 21.840 / 16.355 | 27.517 / 27.715 |
| 1 | 333 / 24 / 3 | 335 / 18 / 7 | 29.667 / 14.818 | 23.188 / 16.983 | 27.253 / 27.790 |
| 2 | 329 / 28 / 3 | 334 / 22 / 4 | 29.354 / 14.684 | 22.928 / 16.937 | 34.313 / 32.138 |

冷启动下着色三档净到达依次为 −1、−2、−5，碰撞增加 0、6、6。一步延迟多出的碰撞来自三航路、六航路、正偏航各 2 架；两个偏航场景各少越界 2 架。两步延迟的净到达损失来自正偏航 −5、四航路 −2、负偏航 +2，种子差异为 0、−6、+5、−2、−2。热启动同丢包三档的到达增量为 +7、+4、+4。结果显示启动方式会改变图着色的收益。两种调度固定轮数，着色实测每协调步交换接近两倍，平均批决策增加 5.485–6.205 ms。

三档着色原始报告 SHA-256 依次为 `ac1463ab2ebbe22265dc0cd4e7fea3da6438c9693ee5c2046367757329190a17`、`dadaedea390b47ba94420711c84e481576174741a73d3494ac5ae2b595797cc8`、`9ca3751b36c474796988df501fb9009230562949554bfc46281df922f52f51b1`。复核任务 55 下 1,726 个导航缓存路径共用单个 15.73 MiB 文件。任务 55 开始最后三档 50% 丢包冷启动着色，任务 57 仍等待同一张 PPU。[逐种子配对、实测通信与哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json)

#### Exp 5 全部 61 配置闭合

任务 55 完成并通过 Exp 5 全矩阵核验。固定轮数 9、通信周期 Hold 8、MPC 空窗 8、丢包／延迟／调度 36，共 **61/61**。每配置使用原生 BlueSky 0.05 s、六场景、五个锁定种子及 360 架。最终三档为 50% 丢包冷启动图着色。

| 延迟步 | 着色到达／碰撞／越界 | Jacobi 到达／碰撞／越界 | 着色／Jacobi 每协调步尝试交换 | 着色／Jacobi 批决策 ms | 着色／Jacobi 单机动作成本 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 332 / 26 / 2 | 340 / 18 / 2 | 29.116 / 14.711 | 21.827 / 16.524 | 27.510 / 27.373 |
| 1 | 334 / 22 / 4 | 323 / 34 / 3 | 29.853 / 14.474 | 23.134 / 17.180 | 26.954 / 27.702 |
| 2 | 330 / 26 / 4 | 331 / 26 / 3 | 29.942 / 14.641 | 22.979 / 17.015 | 33.619 / 29.615 |

50% 丢包冷启动下，一步延迟着色多到达 11 架、少碰撞 12 架。三、四航路各多到达 4 架，正、负偏航各多到达 1、2 架，六航路和近对头持平。零延迟着色少到达 8 架、两步少到达 1 架。三档原始 SHA-256 依次为 `91b892877551242a16d4c3c8baf1f54b8936a8e8d8be5df6bcfb339a16feda19`、`1a01b4a8878c8a4ffa32fe6f5e7631c61495f6e53a02c319bfc4cf5a6d1108c0`、`d93340b5e05a53cc78481e4850815df3dd0b70b9ec619861f2dc577b432571f8`。

完整网络分面核验 24 组丢包、24 组延迟、18 组调度单变量配对，另有 18 组同网络条件冷热配对。调度配对中着色到达领先／持平／落后为 7／4／7，碰撞减少／持平／增加为 7／7／4。固定 K=2 的着色每协调步实际尝试交换为 Jacobi 的 1.96–2.06 倍，平均批决策增加 5.05–6.20 ms。此前复核的 120 个零丢包正延迟场景种子层配对终局、物理统计与残差一致。结果是冻结部署的通信消融，正式部署原生留出仍为 340/360 到达、17 架碰撞。

逐一重算 61 个原始报告、四个分面清单和总清单的 SHA-256 均与审核一致。Exp 5 总清单为 `3d488a8141a4fd8cdf399aba9747d375ce241c1ff978eb76076d8c8fbb0d4a56`，收尾核验快照为 `e7af4f1faa140a1e1cc4ad71f3e6762ec55b1cb9e39d89d25ea5a425b1a4d417`。任务 55 转入 Exp 2/3 原生拓扑泛化及规模评测。[冻结收尾核验快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp5-closure-audit.json) · [原始总清单](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp5/summary.json)

#### Exp 2/3 原生泛化与规模外推闭合

任务 55 完成两套正式矩阵，使用冻结 35 km 权重、原生 BlueSky 0.05 s、种子 12001–12005 和逐流量开关投影配对。开投影启用精确垂向预测、提前分层、6 m 高度裕度和 10 m 联合预留。图统计来自实际活动图，记录逐种子峰值并发、最大度和最大团。两面板均逐种子核对终局、模型哈希和原生参数。

**Exp 2 拓扑零样本泛化**　固定 30 架峰值并发，每格 150 架，13、15、17 航路超过历史训练的 12 航路上限。表中耗时为五并行世界的一次批决策。

| 航路 | 开投影到达／碰撞 | 关投影到达／碰撞 | 开／关单机动作成本 | 开／关最大度、最大团 | 开／关平均批决策 ms | 开／关 p95 ms |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 6 | 106 / 44 | 50 / 100 | 27.535 / 29.830 | 19、8 / 17、8 | 38.985 / 3.187 | 24.734 / 3.983 |
| 13 | 74 / 76 | 29 / 121 | 33.695 / 34.990 | 22、13 / 23、12 | 19.032 / 3.211 | 23.609 / 3.799 |
| 15 | 53 / 97 | 30 / 120 | 37.276 / 37.565 | 27、14 / 20、13 | 18.908 / 3.090 | 23.844 / 3.591 |
| 17 | 56 / 94 | 33 / 117 | 35.594 / 37.786 | 22、13 / 21、13 | 19.226 / 3.295 | 23.077 / 4.107 |

13、15、17 航路开投影到达率为 49.3%、35.3%、37.3%，相对关投影多到达 45、23、23 架。投影减少碰撞，密集未见拓扑仍有 76–97 架碰撞。峰值最大团从六航路的 8 增至 13–14。六航路开投影的批均值明显高于 p95，另一面板复跑的五种子物理数据逐项一致，批均值为 18.972 ms、p95 为 24.063 ms。两次耗时分别保留，尚未确认尾部高耗时成因。

**Exp 3 规模外推**　固定六航路，每组总架数为峰值并发乘五种子。

| 峰值并发 | 开投影到达／碰撞 | 关投影到达／碰撞 | 开／关单机动作成本 | 开／关最大度、最大团 | 开／关平均批决策 ms | 开／关 p95 ms |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 30 | 106 / 44 | 50 / 100 | 27.535 / 29.830 | 19、8 / 17、8 | 18.972 / 3.282 | 24.063 / 4.216 |
| 50 | 162 / 88 | 70 / 180 | 31.021 / 31.855 | 22、8 / 18、9 | 25.719 / 3.591 | 39.216 / 4.601 |
| 80 | 244 / 156 | 107 / 293 | 35.341 / 35.022 | 22、9 / 19、10 | 57.343 / 5.564 | 99.048 / 5.853 |

开投影到达率从 70.7% 降至 64.8%、61.0%，80 架峰值并发开投影少碰撞 137 架，仍有 156/400 架碰撞。平均批决策 57.343 ms，p95 99.048 ms。以上为专门的压力外推组。两个面板所有组合的越界和未完成均为 0。合并清单 SHA-256 为 `beb859056568f0d2e1815d2aef2d2170e038f6b2936dbb6a92d93e99f282cc4e`。统一核验确认 Exp 1、2、3、5 完成，任务 55 已进入 Exp 4 近对头原生压力回放。[拓扑和规模原始清单](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp2-3/summary.json) · [统一结果与源哈希](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json)

#### Exp 4 对头压力与偏航回航闭合

任务 55 已完成 Exp 4 的三组原生压力对照。冻结 35 km 权重、BlueSky 0.05 s、6 s 决策周期、五种子 12001–12005。偏航回航按航向与目标方位误差 ≤2° 连续三个控制步判定，持续对齐和最终到达分开统计。

| 场景 | 开投影到达／碰撞／越界 | 关投影到达／碰撞／越界 | 开／关持续对齐 | 开／关平均单机动作成本 | 开／关批决策均值 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30 架近对头并发 | 150 / 0 / 0 | 148 / 2 / 0 | — | 15.625 / 71.852 | 29.441 / 3.310 |
| +15° 偏航 | 58 / 0 / 2 | 40 / 18 / 2 | 53/60 / 51/60 | 43.565 / 44.938 | 16.027 / 3.215 |
| −15° 偏航 | 57 / 2 / 1 | 44 / 16 / 0 | 55/60 / 51/60 | 46.217 / 47.324 | 16.266 / 3.133 |

偏航关投影时正负方向各有 85% 飞机达到持续对齐，两方向仍合计 34 架碰撞。投影使偏航总到达由 84/120 升至 115/120，碰撞由 34 降至 2。两方向和开关投影条件下，对齐成功样本的首次持续对齐中位时间均为 30 s。对头批决策均值高于 p95 22.525 ms，报告保留原始两项统计，未推断异常时延成因。

对头原始报告 SHA-256 为 `4cb67f1b380a74e6f23b71176255c8b162a7e332cfea8f1b75b38beb7d89e2b6`，偏航原始报告为 `c95e7df944a10e96a162499297e91f3e576ac54ff4aab35174e66f481d327da8`。Exp 4 仍在执行 25 次环境扰动。套件审核已核验原生突发插入时刻、非合作目标记录和实际测得风矢量，并拒绝缺失目标、无风及不一致风速观测；几何回归 72 通过、1 跳过。Exp 6 冻结入选权重的目标为原生历史 20 km 训练的 5,024 决策、20,000 更新，单步控制保留 24 步回放成熟延迟和 35 km 部署口径。[对头原始配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp4/head-on-pressure/comparison.json) · [偏航原始配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp4/yaw/comparison.json)

#### Exp 4 原生环境扰动 25 次配对闭合

五种子 12001–12005 的六航路环境扰动全部闭合。每种子复用同一组 20 架基础航路，突发与联合条件再插入 1 架，采用原生 BlueSky 0.05 s、冻结 35 km 策略与精确安全层。耗时为单世界决策均值再按种子等权，开关投影压力组的五世界批耗时见上一节。

| 扰动 | 总飞机 | 到达 | 碰撞 | 越界 | 单机动作成本 | 平均单世界决策 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 无扰动 | 100 | 70 | 30 | 0 | 21.386 | 20.131 |
| 突发插入 | 105 | 79 | 26 | 0 | 20.609 | 19.655 |
| 非合作目标 | 100 | 70 | 30 | 0 | 19.098 | 20.168 |
| 风场 | 100 | 74 | 26 | 0 | 61.440 | 19.741 |
| 联合扰动 | 105 | 78 | 20 | 7 | 53.642 | 19.928 |

核验单项与联合条件的插入参数、非合作槽位、风场配置逐种子一致。插入时刻实测 78、108、144、144、150 s，每个非合作场景有两架记录锁定目标。风场峰值 17.125 m/s，原生地速分解最大误差约 $2.01\times10^{-14}$ m/s。风场到达变化依次为 +4、+4、0、0、−4，种子间方向不同。非合作组总体到达与碰撞数持平，种子内差异仍存在。联合条件有 7 架越界，且突发与联合条件总架数增加，按各自分母读到达率。

完整 Exp 4 清单 SHA-256 为 `b13320cbd9ea75e2b524654923095c3ebb88527192ce3432c61cf7443b87da0a`。统一审核已闭合 Exp 1–5。任务 55 于 Exp 4 结束后进入 Exp 6 单步回报训练，实测 300 决策／1,104 更新，约 11.99 updates/s、234.62 空域步/s。当前计数满足同一 24 步回放成熟延迟后每决策 4 次更新的协议，目标 5,024 决策／20,000 更新，结束后执行原生配对消融。[Exp 4 原始清单和逐种子数据](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp4/summary.json) · [原生套件审核](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json) · [单步回报训练日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/one-step.log)

#### Exp 6 单步训练同预算核对

任务 55 继续运行原生 Plan A 单步回报对照。历史多步权重与当前单步控制在同一 1,580 个仿真决策均为 **6,224 次更新**，回放成熟均为 24 个决策，Critic 目标分别使用 24 步和 1 步。两臂累计吞吐为 12.985／13.068 updates/s、237.330／238.850 空域步/s。单步进程的峰值 Torch 分配约 2,362 MiB。最终固定停在入选权重的 5,024 决策、20,000 更新。此处吞吐为训练过程快照，正式消融尚待原生回放。

控制端套件核验新增独立重算单步对照配置、断点计数和训练证据；潜在边消融要求同一权重、其他原生配置和开投影预算，KSS 消融要求同权重、复用 Exp 1 各自发机清单和阈值，逐文件对照原始回放。已用当前训练配置和真实历史权重元数据验证匹配，拒绝了人为改动的更新计数和 Figure 4 碰撞阈值，几何回归 72 通过、1 跳过；Exp 6 最终产物统一审核仍待该任务跑完。Exp 4 的 55 个导航路径共用一个 15.73 MiB 文件。当前原生训练的 9 个 worker 有独立导航缓存，共约 141.57 MiB，运行期间保持原路径，结束后可按内容核对合并。[单步训练指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/one-step/training/metrics.jsonl) · [套件核验快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/suite-audit.json)

Exp 6 单步 10k 断点已核验。CPU 独立加载当前单步与历史多步断点，两者均为 2,524 个仿真决策、10,000 次更新，训练观测半径均为 20 km。单步／历史多步断点 SHA-256 分别为 `7239683f7d8887e0e36a962572429e295e03c7f74fcdcd8d74e991a28dc49ff3` 与 `5f82ec6fb3762e9b04e90c6a898fd75b73146dafe3c8bab8e20e4299096df6b7`，两者权重各自独立。单步进程随后记录 10,944 次更新，原生配对仍等待同预算 20,000 次更新。

#### Exp 6 同预算单步权重闭合，原生回放启动

单步和历史多步训练均达到 **5,024 决策、20,000 更新**，训练观测半径 20 km、回放成熟 24 决策，原生部署观测半径 35 km。终点累计吞吐单步／多步为 13.189／13.086 updates/s、238.544／236.675 空域步/s，完成场景 1,700／1,715，峰值 Torch 分配 2,361.69／2,360.51 MiB。已用独立控制端加载最终权重、核对训练配置及唯一回报阶数变化。单步部署权重 SHA-256 为 `c795096b6596de9af98236ceb7a83023d1d4ed5b628c15021f55b3fdfb3fbfb6`，训练收尾清单 SHA-256 为 `dda0b7d8ec0c0e2eaadc0921be62cb7843214d53fe6057436e4bc2393d817713`。

任务 55 已转入原生四臂回放。KSS 替代分面预置了 15 份已经核验的 Exp 1 Geo 对照，以复用同一冻结权重、BlueSky 提交、Figure 4 实现、流量清单和安全参数。续跑入口逐份重查来源与终局阈值，KSS 替代臂仍执行 15 份独立原生回放。复用清单 SHA-256 为 `aa950cc8051b8903dae6b408441c1f7e9d1bea0c11ded6145aafbe5c5381fd21`。训练退出后将九个已完成 worker 的缓存按 SHA-256、权限、属主一致性验证后原子链接到既有导航文件，释放 148,445,433 字节重复内容。任务 57 的冻结输入哈希、Plan B 仓库版本和等变 Actor／安全层／训练门控源码匹配已复核，等待同一卡。[单步权重收尾证据](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/one-step/summary.json) · [Geo 对照复用清单](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp6/reused-geo-reference.json)

#### Exp 6 投影、回报阶数和潜在边原生对照闭合

原生 BlueSky 0.05 s、35 km、五种子 12001–12005 的六场景消融已对齐。每臂 360 架。闭环回报对照是训练原始 20 km、部署统一 35 km、相同 5,024 决策和 20,000 更新的独立权重。潜在边对照复用多步权重及保护参数，仅关闭潜在边构造。

| 配置 | 总到达／碰撞／越界／未完成 | 平均单机动作成本 | 六场景平均批决策 ms |
| --- | ---: | ---: | ---: |
| 24 步回报，开投影 | 340 / 17 / 3 / 0 | 28.239 | 19.586 |
| 1 步回报，开投影 | 328 / 28 / 3 / 1 | 31.156 | 16.931 |
| 24 步回报，关投影 | 266 / 92 / 2 / 0 | 32.592 | 3.432 |
| 1 步回报，关投影 | 251 / 106 / 3 / 0 | 27.705 | 3.416 |
| 24 步回报，开投影，去潜在边 | 306 / 54 / 0 / 0 | 10.643 | 17.024 |

| 场景 | 多步开投影到达／碰撞 | 单步开投影到达／碰撞 | 多步关投影到达／碰撞 | 单步关投影到达／碰撞 | 去潜在边到达／碰撞 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 三航路 | 60 / 0 | 60 / 0 | 48 / 12 | 44 / 16 | 60 / 0 |
| 四航路 | 54 / 6 | 56 / 4 | 42 / 18 | 38 / 22 | 52 / 8 |
| 六航路 | 51 / 9 | 44 / 16 | 32 / 28 | 30 / 30 | 46 / 14 |
| 近对头 | 60 / 0 | 60 / 0 | 60 / 0 | 58 / 2 | 54 / 6 |
| +15° 偏航 | 58 / 0 | 57 / 0 | 40 / 18 | 43 / 14 | 48 / 12 |
| −15° 偏航 | 57 / 2 | 51 / 8 | 44 / 16 | 38 / 22 | 46 / 14 |

多步回报开投影总计多到达 12、少碰撞 11，五种子净到达依次为 +2、+3、0、+1、+6。六航路和负偏航分别多到达 7、6 架，四航路少到达 2 架。关投影时多步组仍多到达 15、少碰撞 14。全部恢复的潜在边在多步权重开投影时增加 34 架到达、减少 37 架碰撞，正负偏航共贡献 21 架净到达、24 架净减碰撞。动作成本按每个名义飞机累计，较高碰撞率会改变实际飞行决策长度，应与终局并读。

四臂原始报告 SHA-256 为 `bf19420806288a3129c5298511027cb63ee0fa5c560311d1ed24c95fb23b02ea`，去潜在边原始报告为 `f3534ea47c30774be577e6005b883cb83123ba3be79880f49e8266e6ab89c81f`。已核对两个原生面板的权重、仿真步长、决策周期及其他开投影配置。任务 55 后续只剩 KSS 替代，已预置 15 个同清单 Geo 参考回放。[四臂原始配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp6/projection/comparison.json) · [潜在边原始配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp6/existing-edges-only/comparison.json)

#### Exp 6 KSS 替代 Case 1 闭合

Case 1 的五份 Geo+KSS 原生回放全部完成，并与 Exp 1 复用的 Geo 原生参考逐种子配对。两臂均 **250/250 到达、零碰撞**。动作成本按权重 1、2、4 合计，Geo 单机均值 6.416，Geo+KSS 6.668。速度／高度／航向未加权轴成本 Geo 为 5.966／0.205／0.010，Geo+KSS 为 6.264／0.146／0.028。原生保护替代保留同一 35 km Geo Actor、权重、发机帧、种子和终局判定；历史 KSS 保护自身使用 20 km 近邻范围、最多 17 邻居、航路恢复和稀疏命令执行，因此这是完整保护模块的替换实验。

逐文件复核同一初始清单、物理判定阈值、历史 KSS 解析配置与来源哈希。五个替代回放 SHA-256 依次为 `b5770f39a4e6387cbb13d3d0bbbe94ec367df15fa80b69911aa78a0020548c53`、`96caec2256207122811c4263ab6f77f9057ef8e181e3ebf3e223d916bee6`、`1e597fdc96ff0f73afa3e0074a203eb98d90df7332bb34e5b0c4abd92c3ac4c8`、`99230a1dd4255b740e50f86e92a0bfc5510d47ad99a49318bf8ef2d078fc6241`、`0b60dbdf08a14cb765bbd8b178f34e0a5c3864fc81aa61fc563eb68250216869`。统一核验已加入 KSS 替代与 Exp 1 历史 KSS 来源一致性，现有基线矩阵审核通过，几何回归 72 通过、1 跳过。任务 55 已进入 Case 2，任务 57 等待同一 PPU。[KSS 替代原生逐种子回放](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp6/kss-substitution)

#### Exp 6 KSS 替代 Case 2 闭合

Case 2 的五种子原生配对均已完成，Geo 与 Geo+KSS 同为 **250/250 到达、零碰撞**。平均单机动作成本分别为 5.705 与 5.634。Geo+KSS 减 Geo 的逐种子成本为 −0.266、−0.583、−0.140、+0.843、−0.206；速度／高度／航向未加权轴均值分别由 5.064／0.277／0.022 变为 5.143／0.135／0.055。较少的高度动作抵消了速度和航向动作成本的增长。两臂逐种子复核同一发机清单、冻结 Geo 权重和原生物理判定，替代模块的哈希与历史 KSS 对照一致。

Case 2 五个 KSS 替代回放 SHA-256 依次为 `26300db6f0d37fdb62ae1486109236e3e3386dda39e392062033e5b196be932d`、`007f1b5be7a229e283984fe7d09667aad240a2cfde71e7da8dbbb91095058fd4`、`7731d178f7fcda514f56379802e4a297d152fe8421ada6b9eede478900ac73b7`、`6335dd6e3e8e95ac966bcc4298c62bddd35e6ae524d6a87f67b8109da9476803`、`f0f8dcb539d48044656eef5c3abad73376a4e364c6bfe920db7c1d3294288688`。另外逐项比对了单步训练及部署断点，模型、优化器及计数均完全一致，仅观测半径元数据由 20 km 变为 35 km；训练初始化权重仍匹配 `224776f45909de543ef4c1e8407888a0f20eff89c72d89e0b2686f22a077e479`。任务 55 已启动最后一个 Case 的五个种子，任务 57 等待同一 PPU。[KSS 原生逐种子回放](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp6/kss-substitution)

### 论文基线 Exp 1–6 验收与等变 Stage 1 接续

任务 55 正常退出（退出码 0）。统一审核读回 Exp 1–6 全部正式结果，重新计算 **182 个**实际产物哈希，冻结快照 SHA-256 为 `eb5dc28f1a4709c6c7da914c43c687273aa31f172590517169d0fdec660b132d`。全部实验基于 94.44% 原生留出的冻结权重 `ed5458b5e7fe365737f4ac76c9e82c35e1dad4d93cbe3fb7ad3ccdd306ce64f6`。需要按场景区分压力外推、通信消融和独立部署门槛。

| 实验 | 已核验结果 |
| --- | --- |
| Exp 1 | 15 份 Figure 4 清单及 45 次原生配对。Geo 750/750 到达、0 碰撞，SARL+KSS 746/750、4 碰撞，Hold+KSS 738/750、12 碰撞。 |
| Exp 2 | 30 架并发原生零样本。13／15／17 航路的开投影到达为 74／53／56 架，每组 150 架。 |
| Exp 3 | 六航路 30／50／80 架峰值并发。开投影到达 106/150、162/250、244/400，五世界平均批决策 18.972／25.719／57.343 ms，原始图度与最大团逐种子保留。 |
| Exp 4 | 30 架对头压力开投影 150/150 到达、0 碰撞，正负偏航开／关投影到达 115/120／84/120，25 次环境扰动全部核验。 |
| Exp 5 | 61 个通信配置，含轮数、空窗 Hold／MPC、丢包、延迟与图着色，完整逐种子配对已冻结。 |
| Exp 6 | 回报阶数、关闭投影、潜在边与 KSS 替代全部配对。多步／单步开投影 340/360／328/360 到达，去潜在边 306/360。 |

Exp 6 的 Geo 和 Geo+KSS 都在同一 15 份 Figure 4 清单上 **750/750 到达、0 碰撞**。Case 1／2／3 的单机动作成本 Geo 为 6.416／5.705／3.667，Geo+KSS 为 6.668／5.634／3.611，总均值分别为 5.263 和 5.304。仅保护模块替代，Geo 参考复用了同权重、同清单的 Exp 1 回放，KSS 替代 15 次均独立执行。Case 3 五份新回放 SHA-256 依次为 `0856ca85de6546f5f7ec10fa9f8972862fb68c6512ca6f2ebfdc45b3e8e256f2`、`cbaaf1957d5a781616cfa1a8c3e9ff9fc23f603c3cfc976fa487145922112a98`、`aaca567450646669e13b23e9b26d566ccc8bfd4210e24e0aff96ef386bb42ca7`、`63c8946f61d59be68f786a264a901b85c7c45fa2e3f630eaf2409882c6f4d98b`、`2987ac3527cf29efddd14cf6c7ee8071da41defd9031aae429c5432b972fc8a8`。完整 Exp 6 清单 SHA-256 为 `41dc3257f4e24828965bacdc2f465924364f43dc0d29733ab9904dbb4fac5969`。论文任务中 2,165 个导航路径共用一个 15.73 MiB 文件。

SOFY 确认任务 57 已接手原单卡 PPU，实际配置为 Plan B、512 空域、35 km、镜像等变 Actor、物理 ADMM 偶对称热启动、精确垂向预测／6 m 裕度、24 步目标和投影 STE。目标 20,000 更新，每 5,000 更新核对原生开发与四案 Q。入场历史权重 Q 原生 3/4、同 Plan B 观测 2/4。实际首批达到 1,770 更新／200 决策，10.44 updates/s、604.17 空域步/s，峰值 Torch 3,356.59 MiB，单次 PPU 采样 60%。新训练尚未产出通过门控的断点。冻结部署原生留出仍为 340/360 到达、17 碰撞，严格超过 95% 且碰撞 ≤3% 要求至少 343 架到达且至多 10 架碰撞。[论文基线冻结审核](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/paper-baseline-audit.json) · [Exp 6 全部原生结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-exact-resume.kzHGIZ/exp6/summary.json) · [阶段 1 训练指标](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/training/metrics.jsonl)

#### 等变 Stage 1 的 5,000 更新门控

任务 57 已落盘 5,000 更新断点。独立加载确认 523 决策、1,024 训练场景、35 km 镜像等变 Actor、物理 ADMM 偶对称热启动，权重 SHA-256 为 `31d3ff0bfad77a27197fa17ecf62c3ffdb679ea318b45cbe62e8a05fd38db067`。四案 Critic Q 原生排序为 **4/4**，较入场 3/4 提高一案；Plan B 对齐观测为 2/4，原生活跃飞机全体口径 3/4。原生四个变化飞机 Q 差依次为 +0.003304、+0.160439、+0.137146、+0.637505，第一案余量较小。四案文件 SHA-256 为 `a1aff7eb8fdc61d749c97689c64424a741ce09b35fbfaa9b62979a012282b43f`。

原生 BlueSky 0.05 s、开发种子 61231–61235、每组 360 架。下表为同种子到达／碰撞。

| 场景 | 入场权重 | 5,000 更新 |
| --- | ---: | ---: |
| 三航路 | 60 / 0 | 60 / 0 |
| 四航路 | 58 / 2 | 60 / 0 |
| 六航路 | 54 / 6 | 52 / 8 |
| 近对头 | 60 / 0 | 60 / 0 |
| +15° 初始偏航 | 53 / 4 | 60 / 0 |
| −15° 初始偏航 | 57 / 2 | 58 / 2 |
| 总计 | 342 / 14 | 350 / 10 |

到达率 95.0%→97.22%，碰撞率 3.89%→2.78%，越界 4→0，平均单机动作成本 27.195→35.039。五种子到达差为 0、+2、+2、+3、+1。六航路 52/60（86.67%）差一架达到 88% 单场景门槛，因此首面板未通过，按既定流程跳过第二开发面板。独立留出与部署权重继续保持冻结。里程碑快照 SHA-256 为 `06aa6ae9ebd8de0370808f8c02b556027d5a809415f27ed32b653a2db8b7e79b`；入场及候选原生报告哈希分别为 `a794eaf6117f6c7783190c08dbc1d77848c330104b42feda07b3d92373a28952`、`088074e8e503a16ec34098f784b609b979cee54e870a704ba4acda0be8a518c5`。

用 256 个真实原生冻结状态独立复核本次训练权重，共 2,477 次活动飞机决策。原始 Actor 通道最大镜像误差为 0，模式差异为 0，安全投影后镜像指令差异为 0。镜像报告 SHA-256 为 `cad85c9fc94b62f49e12a5c2d7102758d5a205e0e1dbd09ddcceca6b9627af97`。5k 训练吞吐 10.45 updates/s、7,707 场景/小时、560 空域步/s，峰值 Torch 3,356.59 MiB。评测结束后任务已继续至 6,770 更新。[四案 Critic Q](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/update-05000-q.json) · [原生开发配对候选](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/native-update-05000-61231/comparison.json) · [训练后镜像等变证据](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/checkpoint-audit-05000.LGANNT/symmetry.json)

#### 等变 Stage 1 的 10,000 更新门控和原生轨迹

任务 57 的 10,000 更新断点为 1,023 决策、2,279 训练场景，累计吞吐 10.447 updates/s、8,571 场景/小时。原生四案 Q 为 **3/4**，Plan B 对齐观测 **1/4**，原生全活动飞机口径 **2/4**。权重 SHA-256 为 `e4e8e29705d8d23f9ab9c41b1b21da640c34f86656c4d4f07fd54d69311dabfc`，四案 Q 报告为 `4b73dc44d61b92ad406af3b6c74db4cebeab32db0d11c106e3df0cc81e73f826`。

原生 BlueSky 0.05 s 开发种子 61231–61235 的新权重共 **319/360 到达、16 碰撞、25 越界**，入场为 342/360、14 碰撞、4 越界。候选六场景到达依次为三航路 58、四航路 60、六航路 50、近对头 60、正偏航 47、负偏航 44。25 架越界均来自偏航，正偏航 13、负偏航 12。到达逐种子变化为 −2、−2、−5、−3、−11，平均单机动作成本 27.195→23.128。原生 Q 达线，闭环总到达和部分单场景门控失败，按既定流程保留任务继续训练并维持正式留出冻结。10k 原始开发回放 SHA-256 为 `4c8eefb509574e9b84b718021ca58675e19a4c8aaaf5512c3c8789de770510a0`。

用同原生安全参数在控制端各复跑一个代表性开发种子，均复现 PPU 终局。

- 六航路 61233 为 5k 相对入场新增两架碰撞的唯一种子。入场 12/12 到达，5k 10/12，接触发生在第 79 个控制步的第 86 帧，机位 4、6 的水平间隔 4.988 km、垂直 39.110 m。接触前 34.3 s 已在冲突图内；到接触控制步，K=2 高度及自适应附加轮的预测均仍冲突。仅调速不可行的诊断下界未触发，现有轨迹不支持把该碰撞单独归因于缺边或协商轮数。接触证据 SHA-256 为 `c0d7d0ebb450c85c7340666d1d81796c189ec1955253efbb4a86e3a38eec5ede`。
- 负偏航 61235，5k 12/12 到达，10k 6/12 到达且 6 架触发 20 km 横航迹退出。机位 6–11 的首次高于航向死区的命令在 5k 出现在 0–84 s，10k 推迟到 216–354 s，其中一架始终没有。10k 六架退场前最后 60 个飞机决策仅 1 个航向指令高于死区，投影未改变这 60 个航向指令。本组回航命令延迟发生在策略输出端。两个控制日志 SHA-256 为 `66df717cb8d79ef41f8ea747bba935fa1f22661245ad2b51ecf3d9d4c56f77dc` 与 `9f4f74af495b9fee300d59dfb7d2b67a526c6192e99abf640ef92d287cbc98fd`。

任务已继续越过 12,770 更新，计划仍按 15k、20k 执行原生开发门控。[10k 原生开发结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/native-update-10000-61231/comparison.json) · [六航路逐帧接触](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/crossing6-61233-audit.mKc1gg/step5k-cpu/collision-events.jsonl) · [负偏航 10k 控制轨迹](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/yaw-minus-61235-audit.LI6mWm/step10k/control-history.jsonl)

#### 等变 Stage 1 的 15,000 更新双面板验证

任务 57 的 15k 断点实际为 1,523 决策、3,546 训练场景。权重 SHA-256 为 `5a0982e5767b5544942035cf70fe28cf5fb6e46492657c923eaefec59555b1fe`，四案 Q 报告为 `2a245e09c59f4e4eb2370be9c2aa3521594d6b4db18b959f28e1a8781eb1dd93`。Critic Q 在原生变更飞机及全活动飞机上均为 **4/4**，Plan B 匹配观测两种口径均为 **3/4**。独立加载证实 35 km 等变 Actor 与偶对称物理热启动；另用此前 256 个冻结原生状态复查 2,477 次活动飞机决策，Actor 最大镜像误差、模式差异及投影命令差异均为 0，镜像报告 SHA-256 为 `dcf53b7d16e08c0de7d6cb754642c0fdca943f62331c77c20454f45495742416`。

| 原生 0.05 s 开发组 | 入场到达／碰撞／越界 | 15k 到达／碰撞／越界 | 入场→15k 单机动作成本 | 门控 |
| --- | ---: | ---: | ---: | --- |
| 61231–61235 | 342 / 14 / 4 | 351 / 8 / 1 | 27.195→26.618 | 通过 |
| 75121–75125 | 316 / 38 / 6 | 326 / 34 / 0 | 29.282→27.436 | 未通过 |

第一组六航路由 54/60 升到 56/60，正偏航 53/60→60/60、负偏航 57/60→59/60，近对头 60/60；五种子净到达为 +2、+2、+2、+1、+2。第二组四航路从 50/60 降至 46/60，六航路从 38/60 升至 46/60，二者仍低于 88% 单场景门槛。第二组近对头仍 60/60，正负偏航达到 56/60 与 60/60，到达净增为 +8、+2、−1、+1、0。15k 未获 Stage 2 准入，原生正式留出仍冻结。

第一组 15k 原始回放 SHA-256 为 `b52fef8e84ff62902471a9e4178bd3329b42566a3333ed7734bcc6cb31d0ee5c`，第二组入场／15k 为 `687aa9ab0b80e6239cc6b10e2f65219f2e98441472b665994f6a6b91418a2b4e`／`c40707920b3d2b162d2218b8b41d5a29e8eb5ede9e60edb00b6cca7111018e5d`。当次双面板里程碑快照为 `9eb21d4b8fe04fdc80e6dc3efb2d508e62d12040aed0d120844499d5e6c055cd`。15k 记录吞吐为 10.435 updates/s、8,881 场景/小时、542.48 空域步/s。评测后训练已经超过 15,270 更新，继续至 20k。[第一开发组](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/native-update-15000-61231/comparison.json) · [第二开发组](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/native-update-15000-75121/comparison.json) · [15k 镜像审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/checkpoint-audit-15000.tO9kpI/symmetry.json)

#### 等变 Stage 1 的 20,000 更新收尾和四航路机制

任务 57 正常退出，状态 `stage1-held-by-native-development`。目标 20,000 更新完整执行，共 2,023 仿真决策、4,729 训练场景，训练秒数 1,915.77，吞吐 10.440 updates/s、8,886 场景/小时、540.66 空域步/s，峰值 Torch 3,356.59 MiB。四个断点的原生 Q 依次 4/4、3/4、4/4、4/4，Plan B 匹配观测依次 2/4、1/4、3/4、3/4。20k 权重 SHA-256 为 `cdd2eb5687d705eee73b20489e55f62f54c9a938c4dfc16f9fa1927c03d83d8d`，四案 Q 为 `0fcd5acdf7b8542fd3eab6928ef162fd8d460968273de3e660d4e2a7b41daedc`。最终里程碑清单为 `574934e603febeb47f3d8a16ca2761e5f1db3a80a92ccced4b62ff4ec03371cd`。

20k 的第一组原生开发为 332/360 到达、14 架碰撞、14 架越界，单机动作成本 31.850，低于入场 342/360 到达。六航路 52/60，正偏航 52/60，负偏航 48/60，近对头 60/60。原始报告 SHA-256 为 `438ba6e0e02542f3c76e0c3ff1522fecab245492c4e916877b3fbd341afeadcb`。15k 为唯一通过第一组的断点，第二组四、六航路均为 46/60。四个断点均未取得短程 Stage 2 准入；正式留出继续使用既有冻结结果 340/360 到达、17 架碰撞。218 个导航路径共用两个 NumPy 版本兼容的实际文件，总 31.46 MiB。

独立开发种子 75123 的四航路 CPU 原生复跑复现 PPU 终局。入场 10/12 到达，15k 降到 8/12，实际接触对为 [0,6] 和 [1,7]。两对在接触前 60–63 s 已进入 35 km 冲突图，离散高度精确可行性在接触步均失败。后者在决策 68–71 的选定高度差约 103.44 m，此后多次改变，到接触时只剩 34.06 m。在同一 15k 权重上预设仅让 1、7 号机从控制决策 71 到 80 保持已选择高度，两原接触均消失，出现另一对 [3,5] 的 91.28 m 擦边接触，终局为 10/12、2 架碰撞。原始／反事实事件文件 SHA-256 为 `8c73de6a6087d7c898fdd42b49c1468bb195d2c2bd89da48e3580df992f8623c`／`f30db0376b0d5be048eb0d659d3ccb3f601171fc1563a3d3494f18df769bb9c1`。

决策 71 的全图精确垂向预测覆盖 19 个水平重叠对。保持指定两机高度后，[1,7]／[7,9] 的预测最小垂直间隔从 12 m 增到 103.44 m，却使 [5,7] 从 96.50 m 降到 5.06 m。全图离散求解遍历每机六档高度且固定速度航向，至少留下 1 个预测冲突；扩展为四档速度 × 六档高度后，仍至少留下 1 个，航向固定。该结论限于这个冻结状态、有限候选及水平匀速预测。实际转移的冲突对和预测新冲突对不同，后续闭环重新决策也影响终局。有限候选无法同时消解的证据用于定位提前量和分量级约束。

[最终里程碑](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/milestone-evaluations.json) · [20k 原生开发结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/native-update-20000-61231/comparison.json) · [全图保留高度的精确预测](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/crossing4-75123-audit.H9HYYJ/vertical-forecast-decision71.json) · [全图调速分层可行性](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/crossing4-75123-audit.H9HYYJ/global-speed-vertical-oracle-decision71.json)

四航路预接触窗口已冻结。原生回放重现 8/12 到达、4 架碰撞；决策 54、59、64、69、71 的原生状态按文件逐个记哈希。将相同四档调速和六档高度候选用在各状态、固定航向，按 91.44 m 判定时最少预测冲突依次为 **0、2、2、1、1**，按 97.44 m 裕度为 **0、3、3、4、1**。决策 54 的现有命令本已满足单步预测；决策 59 距两次实际接触还有约 81–91 s，单步离散候选已无法同时分离全部重叠对。本试验定位到需要考察连续控制步的协同决策，有限候选、固定航向和水平匀速假设限制了结论。五个冻结状态与最优解清单 SHA-256 为 `d1e268d22dbbc9c56fee20ac898476fc8df977a83f6a43bbaf43a40bec56bcd1`。[五步提前量可行性](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/crossing4-75123-audit.H9HYYJ/global-speed-vertical-lead-window.json)

#### 四航路提前调速与持续分层的原生闭环对照

对独立开发种子 75123 的同一 15k 权重，做原生 BlueSky 0.05 s 逐帧对照，控制周期 6 s，35 km 图、K=2、密集附加 2 轮、精确垂向预测、6 m 裕度、10 m 联合预留。干预均在现有投影之后施加，属于机制诊断。提前调速在决策 54–60 仅降低 0、1 号机的速度指令；持续分层在决策 54–78 仅覆盖 0、1、6、7 号机的高度指令。参考组复现两次接触的机位、原生帧和垂直间隔。

| 干预 | 到达／碰撞（共 12 架） | 原生接触 |
| --- | ---: | --- |
| 无 | 8 / 4 | [0,6] 决策 72 帧 65；[1,7] 决策 74 帧 12 |
| 仅提前调速 | 10 / 2 | [1,7] 决策 74 帧 69 |
| 仅持续分层 | 10 / 2 | [5,6] 决策 73 帧 109 |
| 提前调速加持续分层 | 10 / 2 | [5,6] 决策 73 帧 109 |

四臂均无越界和超时。调速去掉一对原接触；分层方案去掉两对原接触并引入一对新接触，组合未额外减少碰撞。机位和区间按该轨迹设定，收益只归因于这一原生反事实。实验脚本 SHA-256 为 `686eda261f964c6a09b7c9877817f68a920ba923e861e8c3364b86c6292af516`。参考、仅调速、仅分层、组合的接触文件 SHA-256 依次为 `fcc6324547d56673a584681ed4d7c31938f35e348dfc5a6f71dabd3650492e27`、`f487a4588ac485e9de03fa40191f4331ea11f8bc624dc675ba52f1c828a93428`、`77d72a12257bf7914fcb00f78caaf53b5bb9972cb4c1245cd56af7880754ed29`、`2492e31c5353c500559cabb53a5914e90a8fc6231ef02fb97096bf1def472f77`。[复现实验脚本](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/crossing4-75123-audit.H9HYYJ/closed-loop-timing.9CBuQj/replay_intervention.py) · [组合控制和接触](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/crossing4-75123-audit.H9HYYJ/closed-loop-timing.9CBuQj/combined/schedule.json)

#### 12 s 协商与 MPC 空窗的双开发组验证

复用任务 57 的 15k 权重与原生 BlueSky 0.05 s 逐帧判定。所有组的控制周期均为 6 s，35 km 图、精确垂向预测、6 m 裕度、10 m 联合预留和 K=2 协调保持一致。第二组固定三臂对照包括每 6 s 协调、每 12 s 协调加保持原指令、每 12 s 协调加 MPC 空窗。每组 360 架，下表为到达／碰撞。

| 场景 | 61231–35 每 6 s 协调 | 61231–35 12 s MPC | 75121–25 每 6 s 协调 | 75121–25 12 s Hold | 75121–25 12 s MPC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 三航路 | 58 / 2 | 60 / 0 | 58 / 2 | 58 / 2 | 60 / 0 |
| 四航路 | 58 / 2 | 60 / 0 | 46 / 14 | 52 / 8 | 58 / 2 |
| 六航路 | 56 / 4 | 58 / 2 | 46 / 14 | 40 / 20 | 54 / 6 |
| 近对头 | 60 / 0 | 60 / 0 | 60 / 0 | 60 / 0 | 60 / 0 |
| +15° 偏航 | 60 / 0 | 60 / 0 | 56 / 4 | 58 / 2 | 58 / 2 |
| −15° 偏航 | 59 / 0 | 57 / 2 | 60 / 0 | 56 / 4 | 60 / 0 |
| 总计 | 351 / 8 | 355 / 4 | 326 / 34 | 324 / 36 | 350 / 10 |

第一组 MPC 仍有 1 架越界，第二组无越界或超时。第一组单机动作成本 26.618→23.330，第二组同次原生 CPU 复跑每 6 s 配置为 27.425、Hold 为 23.093、MPC 为 25.688。第二组 CPU 与原 PPU 的 30 个逐种子四分类结果完全一致；动作成本存在 0.011 的运行环境差异。两组的每个场景均超过 88% 到达门槛，近对头 60/60。第一组负偏航少到达 2 架，仍为 57/60。单纯延长指令保持时间未解决第二组的六航路，MPC 在同一协调周期下使六航路净增 14 架到达。

两组 MPC 原始报告 SHA-256 分别为 `1b6f7de17c4a6a7a0c2ffc3d46395488db555d135e8e4c8c02a58d8348cfe95e` 和 `dea764a10075463ded753b352107bdbe3ce2385b5ecb937b9c68a6b45d0b59f5`。对照审核 SHA-256 为 `4f29b40181bdb7f433073351b36a0605f7e04aea8a12d4337b7e4b0ef27de382`。模型原生四案 Q 4/4、Plan B 对齐观察 3/4。正式留出继续冻结。Stage 2 已支持按两组同口径选择 checkpoint，同时保留原生 6 s 投影闭环训练及现有投影 STE、144 s 多步目标，小学习率为 Stage 1 的 1/5。正式配对时各投影臂共享 12 s MPC 配置，历史冻结的每 6 s 结果作为单独锚点。单组缺失及误用 Hold 的准入被拒绝，几何、安全层与发布协议回归 73 通过、1 跳过。[双组逐种子审核](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/native-mpc-development.YbJpa4/paired-summary.json) · [模型阶段准入文件](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/native-mpc-development.YbJpa4/stage2-native-admission.json)

SOFY 任务 58 已复用原单卡 PPU，独立精修预算 600 轮，9 个原生 worker × 8 空域，每 200 轮做双开发组和四案 Q，Q <3/4 则早停并保留此前最佳权重。副本入场原生 PPU 评测复现 355/360、350/360 以及 4/4 Q。首段 200 轮共 639 个仿真决策、2,460 次更新，累计训练约 3,096 场景/小时、10.577 updates/s。冻结留出等待开发选峰完成。

Figure 4 Geo 臂已接入同一 MPC 空窗实现。合成开发流量种子 83015 的两个发机帧 0／15 均精确复现，原生 0.05 s 回放 2/2 到达、0 碰撞，并记录 20 个 MPC 空窗。几何回归为 73 通过、1 跳过。原有全套论文基线经新审核器重新核验，Exp 1–6 六组及 182 份证据全部通过。Exp 1／5 的新运行快照将在正式到达严格 >95% 且碰撞 ≤3% 后使用。[任务 58 快照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-mpc-refine.FdBoEZ/run.sh) · [Figure 4 合成流量冒烟](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/native-mpc-development.YbJpa4/figure4-mpc-smoke.LyglBE/geo.json) · [待正式准入论文运行脚本](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-mpc-exp1-exp5.wrrcMn/run.sh)

任务 58 的 200 轮断点未晋升。第一组 355/360 到达、0 碰撞、5 越界，入场为 355/4/1；第二组 342/360 到达、10 碰撞、8 越界，入场为 350/10/0。13 架越界均来自正负偏航。第二组六航路 52/60，低于 88% 下限，两组近对头均为 60/60。关投影时第一组入场→断点为 232/127→214/140，第二组为 216/144→229/123，表明原始策略收益也随流量变化。四案 Q 为 3/4，六航路 30 s 案的变化飞机 Q 差为 −2.162079；任务继续执行 400 轮检查，保留入场权重为当前最佳。断点 SHA-256 `22a6b5ff957b419c81f2a3f7273b18f6faf208f824b2a54350755eb4a07d621a`，Q 报告 `29a22f1ed096802894bf895be0a033a0ffbcbc57504d34f92c58f5bed6b83975`。[第一组](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-mpc-refine.FdBoEZ/stage2/development-000639/61231/validation.json) · [第二组](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-mpc-refine.FdBoEZ/stage2/development-000639/75121/validation.json) · [四案原始 Q](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-mpc-refine.FdBoEZ/stage2/development-000639/frozen-q.json)

任务 58 在 400 轮触发连续两次开发落后早停，共 1,227 仿真决策、4,812 次更新，实测 3,147.04 场景/小时、10.516 updates/s，Torch 峰值 2,787.54 MiB。第一组 345/360 到达、4 碰撞、11 越界，第二组 335/360、8 碰撞、17 越界，两组越界均来自偏航；近对头仍各 60/60，第二组正负偏航分别 52/60、51/60。平均单机动作成本 27.967／29.799，四案 Q 3/4。400 轮断点 SHA-256 `b692a4682575c76f082fd734f22127121a1e4902f46e288fe8685ac03ea2b983`，Q 文件 `5e282f96e3f734f5c72130e9b4d9646875639facb1d037a55fd37e44d6ebce8e`。200、400 轮均未晋升，正式选中的仍为 15k 入场权重 `5a0982e5767b5544942035cf70fe28cf5fb6e46492657c923eaefec59555b1fe`，配合已锁定的 12 s MPC，随后才进入正式五种子留出。[早停选模记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-mpc-refine.FdBoEZ/stage2/selection.json) · [400 轮第一组](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-mpc-refine.FdBoEZ/stage2/development-001227/61231/validation.json) · [400 轮第二组](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-mpc-refine.FdBoEZ/stage2/development-001227/75121/validation.json)

#### 15k 等变策略与 MPC 的正式留出验收

任务 58 沿用两组开发预锁定的 15k 入场权重，以原生 BlueSky 0.05 s 对六场景种子 12001–12005 做 360 架／臂逐帧配对。控制周期 6 s，投影臂统一用 12 s ADMM 通信与 MPC 空窗、35 km 图、6 m 精确垂向裕度、10 m 三机联合预留。200 和 400 轮精修断点均无晋升，正式候选为 15k 原权重。每格为到达／碰撞／平均单机动作成本。

| 场景 | 早期冻结开 | 早期冻结关 | 发布前权重开 | 发布前权重关 | 15k MPC 开 | 15k 关 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 三航路 | 51 / 0 / 3.393 | 30 / 30 / 1.366 | 60 / 0 / 18.531 | 48 / 12 / 20.480 | 60 / 0 / 19.704 | 40 / 20 / 28.785 |
| 四航路 | 47 / 2 / 3.817 | 26 / 34 / 2.068 | 58 / 2 / 18.633 | 42 / 18 / 20.662 | 58 / 2 / 19.286 | 26 / 34 / 25.104 |
| 六航路 | 36 / 10 / 5.467 | 26 / 34 / 2.814 | 47 / 13 / 21.557 | 32 / 28 / 26.752 | 58 / 2 / 22.363 | 24 / 36 / 27.970 |
| 近对头 | 59 / 0 / 4.775 | 20 / 40 / 2.117 | 60 / 0 / 13.814 | 60 / 0 / 35.396 | 60 / 0 / 14.126 | 42 / 18 / 28.337 |
| +15° 偏航 | 0 / 0 / 0.052 | 0 / 0 / 0.000 | 56 / 0 / 36.845 | 40 / 18 / 44.938 | 60 / 0 / 39.274 | 44 / 16 / 43.009 |
| −15° 偏航 | 0 / 0 / 0.051 | 0 / 0 / 0.004 | 60 / 0 / 36.240 | 44 / 16 / 47.324 | 59 / 0 / 37.882 | 44 / 16 / 35.915 |
| 总计 | 193 / 12 / 2.926 | 102 / 138 / 1.395 | 341 / 15 / 24.270 | 266 / 92 / 32.592 | 355 / 4 / 25.439 | 220 / 140 / 31.520 |

15k+MPC 正式到达 **355/360（98.61%）**、碰撞 **4/360（1.11%）**、越界 1、超时 0。五个种子到达 72、70、70、72、71，对配对发布前权重净增 3、5、4、1、1 架。四架碰撞均来自四和六航路；近对头 60/60，正负偏航分别 60/60、59/60。相对原部署历史冻结结果 340/360、17 架碰撞，净增 15 到达、少 13 碰撞；发布前权重在同一新协议的配对结果为 341/360、15 架碰撞。早期冻结报告中的 174/360 属于其原协议，在当前配对 MPC 协议下重测为 193/360。

关投影时 15k 为 220/360 到达、140 架碰撞，发布前权重为 266/360、92 架碰撞，策略本体消融未达标。开投影的正式发布门槛和冲刺门槛已达成，整体验收标志仍保留此消融失败。正式配对 SHA-256 `21819414146fb512ccbfb45acf14e27f828028eace83bcf50f5644ae69df9be7`，发布证明 SHA-256 `74084927974cca880e1920ac7a810e93a74b225d5938910154f22e0ea5527af7`。SOFY 58 正常退出并释放 PPU；SOFY 59 已提交，在同一张卡上运行 15 份 Figure 4 统一清单和 61 个 Exp 5 通信配置。历史 SARL+KSS 与 Hold+KSS 只读复用同一份已经过发布量和发机帧审核的 30 份回放。

[正式逐种子配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-mpc-refine.FdBoEZ/stage2/formal-short-refinement/comparison.json) · [验收结果与闸门](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-mpc-refine.FdBoEZ/stage2/stage2-acceptance.json) · [新权重的论文任务](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-mpc-exp1-exp5.wrrcMn/run.sh)

SOFY 已确认任务 59 正在运行。首份新权重 Figure 4 清单 Case 1／12001 原生完成 50/50 到达、0 碰撞，50 个发机帧齐备，MPC 空窗证据已记录。回放 SHA-256 `1df273da814b153c34e8fee7a8a3c273e9115b258a92c502467248d11bac0bee`。[首份回放](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-mpc-exp1-exp5.wrrcMn/exp1/case1-12001-geo.json)

#### 98.61% 发布权重的全矩阵接续

任务 59 的 Figure 4 已独立核对 Case 1、2 各五种子，三臂使用相同清单哈希和逐帧发机序列。新 Geo 分别 250/250、250/250 到达，均为零碰撞，平均单机动作成本分别为 5.367、4.695。Case 3 及 Exp 5 仍由该任务接续。在任务 58 与任务 59 的目录检查到 560 个导航路径共用同一物理文件，实际文件 15.73 MiB。

Exp 2、3、4、6 已接入发布时的原生 0.05 s 与 12 s ADMM／MPC 口径。Exp 5 保持矩阵自带的逐配置通信变量，未改变各臂定义。以开发种子 75123 跑通 30 架 13 航路配对，开投影四个决策有两次 MPC，关投影为零次。组合扰动验证了突发航班插入时刻、非合作目标和实际风场，并记录 200 次 MPC 空窗、40 架次改令。73 项回归通过、1 项跳过；完整摸底基线再次核验 182 份证据。

任务 60 已提交至同一单卡 SOFY 作业队列，启动前检查任务 59 的两个面板完成并退出成功。Exp 6 单步对照沿用任务 57 的 Plan B 代码快照、种子、初始化、512 空域、35 km 等变 Actor 和 24 决策回放成熟窗口，在 1,523 决策、15,000 更新处停止，唯一训练变量是 Critic 回报步数。KSS 替代复用已验真的同清单 Geo 臂。最终审核要求终版同一权重的 Exp 1–6 全部闭合。接续源码与输入清单 SHA-256 为 `16f0b25306ac4f50bbb19b8eec5272e90a6422297d54e5b7a3b9c7d21e4983c7`。[任务 60 脚本](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-mpc-exp1-exp5.wrrcMn/run-completion.sh) · [外推与扰动冒烟](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/mpc-paper-coverage-smoke.n1dv6l)

#### 98.61% 发布权重的 Exp 1 闭合

任务 59 完成全部 15 份 Figure 4 统一清单、45 个三臂回放，每臂 750 架。独立审核逐文件核对原生 0.05 s 物理步长、50 个发机帧、清单哈希、阈值、权重、12 s MPC 空窗，以及与摸底版相同的 30 份历史策略对照。每格为到达／碰撞／平均单机动作成本。

| 场景 | Geo 终版 | SARL+KSS | Hold+KSS |
| --- | ---: | ---: | ---: |
| Case 1 | 250 / 0 / 5.367 | 248 / 2 / 5.787 | 246 / 4 / 0.967 |
| Case 2 | 250 / 0 / 4.695 | 248 / 2 / 3.733 | 242 / 8 / 1.379 |
| Case 3 | 250 / 0 / 3.305 | 250 / 0 / 1.739 | 250 / 0 / 0.438 |
| 总计 | 750 / 0 / 4.455 | 746 / 4 / 3.753 | 738 / 12 / 0.928 |

相对同清单摸底 Geo，三个案例保持 750/750 到达、零碰撞，总平均单机动作成本从 5.263 降至 4.455，下降 15.34%。Exp 1 独立核对 46 份证据。汇总 SHA-256 为 `cf6bd38794643c70a82dbb25930106919ec44d2f74b76098ff6abb444363f018`，闭合审核 SHA-256 为 `47bdf9be7f04caea76bab2a6aecf35e6e59df60d1ee9d7d0b343c40467addec2`。[逐种子统一清单对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-mpc-exp1-exp5.wrrcMn/exp1/summary.json) · [Exp 1 独立审核](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-mpc-exp1-exp5.wrrcMn/exp1-closure-audit.json)

Exp 5 首批独立核验通过 2/61。K=0 无投影的到达／碰撞／越界／超时为 220/140/0/0，逐场景逐种子的 30 个结果与正式关投影完全相同。K=1 热启动为 338/20/2/0。六场景的平均批决策耗时为 5.318／15.122 ms。两配置原生结果 SHA-256 为 `8d87de4c2e4f1027d16fd670f0bd56e2df27c511f614644beb49a60a67a838e1`、`1db4ecbd6f3ff194e370c88260ef314e74cd64e2d160e0591a7b32f99763715a`。SOFY 再次确认任务 59 运行中，同一单卡作业中任务 60 已排队。

#### 98.61% 发布权重的 Exp 5 固定轮数分面

同一 15k 权重、原生 0.05 s 逐帧判定、五种子六场景，每臂 360 架。九个固定轮数配置全部闭合并独立核验。此分面使用固定 ADMM 轮数，无部署配置的密集额外两轮及 12 s MPC。

| 配置 | 到达 | 碰撞 | 越界 | 平均单机动作成本 | 平均批决策耗时 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| k0-raw | 220 | 140 | 0 | 31.520 | 5.318 |
| k1-warm | 338 | 20 | 2 | 30.112 | 15.122 |
| k1-cold | 347 | 12 | 1 | 27.030 | 14.528 |
| k2-warm | 350 | 10 | 0 | 27.022 | 15.711 |
| k2-cold | 345 | 14 | 1 | 26.975 | 15.838 |
| k3-warm | 344 | 14 | 2 | 26.522 | 16.915 |
| k3-cold | 337 | 22 | 1 | 26.877 | 17.286 |
| k5-warm | 337 | 22 | 1 | 26.349 | 19.692 |
| k5-cold | 342 | 17 | 1 | 26.628 | 19.769 |

全部配置均无超时。同一轮数内热启动相对冷启动的净到达，K=1、2、3、5 依次为 −9、+5、+7、−5；对应少碰撞 −8、+4、+8、−5。K=2 热启动在固定轮数分面到达最多、碰撞最少，为 350/360、10 架碰撞。K=5 热／冷耗时约 19.7 ms。此处为六场景等权的每批决策均值。

单个配置的结果哈希与五种子冷热配对已封存。固定轮数清单 SHA-256 为 `16d65258fc6bf0f75a971500aab5b9ce2d442d1f5b5f09cfcdea3033f6142183`，独立审核 SHA-256 为 `ef56902a934b85f0b377c893dadca4401c966b303bb7fc9897fac7cf35e0f7b5`。[固定轮数逐配置审核](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-mpc-exp1-exp5.wrrcMn/exp5-round-budget-audit.json)

训练证据复核确认 Stage 1 为 512 空域 Plan B、35 km 等变、24 步回报及精确垂向 6 m，四个里程碑的原生四案 Q 为 4/4、3/4、4/4、4/4。Stage 2 为原生 Plan A 9×8 空域，预设 600 轮，学习率 2.4e-5 为 Stage 1 的 1/5，实际在 200／400 轮检查为 3/4、3/4，连续两次回落后关闭训练并保留 15k 入场权重。[Stage 1 原始里程碑](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-equivariant-even-admm-20k.JVJ3Kn/milestone-evaluations.json) · [Stage 2 选峰和早停](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/native-exact-mpc-refine.FdBoEZ/stage2/selection.json)

首个 6 s Hold 热启动已核验，Exp 5 累计 10/61。30 个逐场景逐种子的四分类终局与单机动作成本逐项复现固定 K=2 热启动，符合 6 s 周期没有空窗的对照。该配置结果 SHA-256 为 `bba46fbc1bab6b74f4efca76b3a17b80d3458e57014026bbda04b41364027f56`。[6 s 原生对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-mpc-exp1-exp5.wrrcMn/exp5/cadence-hold/k2-warm-period-6/comparison.json)

#### 98.61% 发布权重的 Exp 5 Hold 周期分面

固定 K=2，原生 0.05 s 逐帧、6 s 控制步、五种子六场景，每臂 360 架。6／12／18／30 s 的冷热启动共八臂已独立核验，累计通过 17/61。每格为到达／碰撞／越界／协调次数／平均批决策耗时（ms）。

| 周期 | 热启动 | 冷启动 |
| --- | ---: | ---: |
| 6 s | 350 / 10 / 0 / 5726 / 16.514 | 345 / 14 / 1 / 5727 / 16.430 |
| 12 s | 327 / 32 / 1 / 2879 / 10.180 | 330 / 28 / 2 / 2908 / 10.381 |
| 18 s | 340 / 18 / 2 / 1951 / 8.085 | 337 / 18 / 5 / 1952 / 8.321 |
| 30 s | 330 / 24 / 6 / 1209 / 6.373 | 334 / 20 / 6 / 1209 / 6.418 |

全部八臂无超时。热启动的 12、18、30 s 组相对 6 s 组实际协调次数分别减少 49.7%、65.9%、78.9%，到达分别少 23、10、20 架。6 s 冷启动的逐种子终局与单机动作成本也复现固定 K=2 冷启动。两次运行的墙钟耗时均按各自进程实测。

Hold 清单 SHA-256 为 `f8db937bbc875a555b63f21384a643254c46701f44e73f5e331a8da25ba1b8bd`，独立审核 SHA-256 为 `b699b68c76dacfb2cbab8f39dd8ecef22c09484db0299184236c64f3f2f3647b`。[Hold 全部逐种子与通信计数](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-mpc-exp1-exp5.wrrcMn/exp5-cadence-hold-audit.json)

在同一本地进程载入已锁定 Plan B 的 rl_gpu 模块后，原先因该模块未导入而跳过的定时发机测试实际执行，完整几何与协议回归 74 项通过。

MPC 首批 6 s／12 s 热启动配对已验真，Exp 5 累计 19/61。6 s 无通信空窗，MPC 重规划为零，逐项复现同周期 Hold 的 30 个终局和单机动作成本。12 s 固定 K=2 热启动由 Hold 的 327 到达／32 碰撞／1 越界变为 MPC 的 351／8／1；五个种子净增到达和少碰撞分别均为 +6、+12、+2、+4、0。全程 2,885 次空窗重规划、268 架次改令。模型预测接触从 195 降至 58，原生逐帧碰撞独立计数。相比同周期 Hold，平均单机动作成本增加 3.737，平均批决策耗时增加 1.951 ms。6 s、12 s MPC 原生报告 SHA-256 分别为 `a258e5c76ddba8dd5b001ea22728faae5e5473eee50bad8d8fb67bb028f33a9c`、`45dd4d8a03e45ac0c6cd4381ffe7c2735ab4dfe7e9c27b486e2ba56e274c4281`。[12 s MPC 原生逐种子报告](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-mpc-exp1-exp5.wrrcMn/cadence-mpc/k2-warm-period-12/comparison.json)

#### 98.61% 发布权重的 Exp 5 MPC 周期分面

固定 K=2，原生 0.05 s 逐帧、6 s 控制步、五种子六场景，每臂 360 架。八个 MPC 周期臂与相同周期和启动方式的 Hold 全部配对，累计通过 25/61。表中差值均为同种子 MPC 相对 Hold 的变化。

| 周期与启动 | 到达／碰撞／越界 | 净到达／少碰撞 | 重规划／改令架次 | 平均批决策耗时变化 ms | 平均单机动作成本变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 s 热 | 350 / 10 / 0 | +0 / +0 | 0 / 0 | -0.051 | +0.000 |
| 6 s 冷 | 345 / 14 / 1 | +0 / +0 | 0 / 0 | +0.097 | +0.000 |
| 12 s 热 | 351 / 8 / 1 | +24 / +24 | 2885 / 268 | +1.951 | +3.737 |
| 12 s 冷 | 351 / 6 / 3 | +21 / +22 | 2894 / 245 | +1.542 | +2.182 |
| 18 s 热 | 353 / 2 / 5 | +13 / +16 | 3886 / 270 | +2.447 | +2.682 |
| 18 s 冷 | 353 / 0 / 7 | +16 / +18 | 3891 / 267 | +1.879 | +2.959 |
| 30 s 热 | 346 / 4 / 10 | +16 / +20 | 4813 / 309 | +2.655 | +3.293 |
| 30 s 冷 | 343 / 6 / 11 | +9 / +14 | 4825 / 276 | +2.494 | +2.778 |

均无超时。6 s 冷热启动无 MPC 空窗，逐种子终局及平均单机动作成本分别复现同周期 Hold。12／18／30 s 在八组配对中的六个有空窗配置均多到达、少碰撞。18 s 冷启动零碰撞并有 7 架越界；30 s 热／冷启动保留 10／11 架越界。原生实际碰撞与 MPC 内部预测接触各自计数。

八个 MPC 配置清单 SHA-256 为 `59a3d22c401c4f1679dba2e1a30bf3503f613cab4fad8bbf3f6b20bba6720893`，包含 18 份通信周期证据与五种子配对的审核 SHA-256 为 `4118aa5499f8b781f7443b7e51f7b4872ca6ebfbcd8d13183a47b5e908a8bc6f`。任务 59 已推进到 36 个丢包／延迟／着色配置，其首个零丢包零延迟 Jacobi 回放已落盘三场景。[MPC 周期独立配对审核](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-mpc-exp1-exp5.wrrcMn/exp5-cadence-mpc-audit.json)

任务 58 与任务 59 目录的本轮导航缓存快照为 1,269 个路径共用同一文件，实际文件 15.73 MiB。

网络首个零丢包、零延迟 Jacobi 配置已独立核验，Exp 5 累计 26/61。350 到达、10 碰撞、0 越界，30 个逐种子四分类终局与平均单机动作成本逐项复现固定 K=2 热启动。80,986 条消息全部送达，观察丢包和在途消息均为零。原生报告 SHA-256 为 `030d27b56a80807d044f6dbc140d8cd0adb02728d6b86b97d7e27a6fb90c3847`。[首个网络对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-mpc-exp1-exp5.wrrcMn/exp5/loss-delay-coloring/k2-warm-loss-0-delay-0/comparison.json)

#### 终版 Exp 5 的 Jacobi 热启动网络九臂

固定 K=2 热启动和 Jacobi，原生 0.05 s，五种子六场景、每臂 360 架。0／20／50% 丢包与 0／1／2 步发送延迟的九个配置全部核验，Exp 5 累计 34/61。消息年龄按被使用的缓存状态加权，单位为 6 s 控制步。

| 设定丢包 | 发送延迟步 | 到达／碰撞／越界 | 实测丢包 | 平均／最大缓存消息年龄 | 平均批决策耗时 ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0 | 350 / 10 / 0 | 0.000% | — | 16.395 |
| 0% | 1 | 340 / 18 / 2 | 0.000% | 1.073 / 85 | 16.937 |
| 0% | 2 | 343 / 16 / 1 | 0.000% | 2.157 / 53 | 16.805 |
| 20% | 0 | 342 / 18 / 0 | 19.886% | — | 16.654 |
| 20% | 1 | 340 / 18 / 2 | 19.760% | 1.120 / 49 | 17.365 |
| 20% | 2 | 340 / 20 / 0 | 19.799% | 2.203 / 155 | 17.439 |
| 50% | 0 | 341 / 18 / 1 | 49.886% | — | 16.766 |
| 50% | 1 | 341 / 16 / 3 | 50.014% | 1.407 / 67 | 17.437 |
| 50% | 2 | 344 / 14 / 2 | 50.020% | 2.449 / 58 | 17.442 |

九臂均无超时，记录了 6 对丢包和 6 对发送延迟的同种子单变量比较。消息按仿真控制步排队到达，冲突边重新出现时可复用该飞机对最近收到的缓存状态，因此实际使用年龄存在长尾。
20% 丢包、2 步发送延迟配置最长使用年龄 155 步出现在 head_on/12004，该配置的平均年龄为 2.203 步。独立审核含九份原生结果哈希，SHA-256 为 `e713e06a1565d18dfc1310328f85c4f2aeecc5aa74e1a02227e2520cd05c8f97`。[Jacobi 热启动逐种子审核](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-mpc-exp1-exp5.wrrcMn/exp5-network-jacobi-warm-audit.json)

#### 终版 Exp 5 的 Jacobi 冷启动与冷热配对

原生 0.05 s、固定 K=2。Jacobi 的 18 个冷热启动配置全部核验，Exp 5 累计通过 43/61。每个配置同五种子六场景、360 架。冷热差值表示热启动相对同丢包率和发送延迟冷启动的变化。

| 设定丢包 | 发送延迟步 | 冷启动到达／碰撞／越界 | 实测丢包 | 冷启动批决策耗时 ms | 热相对冷净到达／少碰撞 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0 | 345 / 14 / 1 | 0.000% | 16.417 | +5 / +4 |
| 0% | 1 | 336 / 22 / 2 | 0.000% | 16.978 | +4 / +4 |
| 0% | 2 | 332 / 26 / 2 | 0.000% | 16.888 | +11 / +10 |
| 20% | 0 | 345 / 14 / 1 | 19.830% | 16.991 | -3 / -4 |
| 20% | 1 | 338 / 20 / 2 | 19.732% | 17.649 | +2 / +2 |
| 20% | 2 | 333 / 24 / 3 | 19.810% | 17.542 | +7 / +4 |
| 50% | 0 | 347 / 12 / 1 | 49.910% | 17.083 | -6 / -6 |
| 50% | 1 | 339 / 20 / 1 | 50.014% | 17.510 | +2 / +4 |
| 50% | 2 | 329 / 28 / 3 | 50.000% | 17.673 | +15 / +14 |

九个冷启动臂全部无超时。冷启动在零丢包零延迟条件下逐项复现固定 K=2 冷启动的 30 个终局和平均单机动作成本。热启动在九组配对的七组多到达、少碰撞，两组反向差异出现于 20% 和 50% 丢包且无发送延迟。50% 丢包、2 步延迟时，热启动净增 15 架到达、少 14 架碰撞。

Jacobi 审核封存 18 份原生结果哈希、9 组冷热配对、12 组丢包与 12 组发送延迟的同种子单变量对照。审核 SHA-256 为 `4729c5d8c084dbf6c5fc3ee14cb4e9ee2f7613f705f9a0a27cb21aa9df324459`。首个零丢包零延迟着色回放已开始。[Jacobi 完整逐种子审核](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-mpc-exp1-exp5.wrrcMn/exp5-network-jacobi-audit.json)

首个着色与 Jacobi 同条件配对已验真，Exp 5 累计 44/61。K=2 热启动、零丢包零发送延迟时，着色为 347 到达／12 碰撞／1 越界，Jacobi 为 350／10／0；着色的平均批决策耗时多 4.810 ms。结果 SHA-256 为 `d7fb8b65d817a2c68b695ecdb1cbde98b3c8c0a3a063cb8f013ac52b2328c1ce`。任务 59 运行中，任务 60 已排队。[首个着色逐种子对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/paper-mpc-exp1-exp5.wrrcMn/exp5/loss-delay-coloring/k2-warm-colored-loss-0-delay-0/comparison.json)

#### v2 前置 Step 1 的 v1 续跑记录

先按指定命令复核同步源码，共 80 项通过。Task 60 原日志确认在 Exp 6 创建预置目录时抛出 FileExistsError。Task 61 在 CPFS 新目录采用固定 v1 控制器，覆盖修复后的 Exp 6 入口并加入审计器的只读继承入口。原 Exp 1–5 通过继承审核，匹配的单步权重复用 1,523 决策和 15,000 更新，SHA-256 为 `4b0fd82280a72dc8dd705da5ff04f2f1984b836d73933a4104c23872340fb1d7`，15 份预置 Geo 回放与 Exp 1 逐字节相同。全程复用同一个现有导航缓存。

| 原生 0.05 s 六场景 | 到达／碰撞／越界／超时（360 架） |
| --- | ---: |
| 24 步开保护 | 355/4/1/0 |
| 1 步开保护 | 348/10/2/0 |
| 24 步关保护 | 220/140/0/0 |
| 1 步关保护 | 193/167/0/0 |
| 24 步去潜在边 | 332/28/0/0 |

24 步权重的开关保护臂逐项复现正式留出的 30 个逐种子终局。匹配对照 SHA-256 为 `7e242afe308d9efd55eb7ab7f56a483ca6c77c3e2bf08e53d71164f42c335832`，去潜在边报告为 `de9a61bc813320d082952c8f92fe367fa3f00dac0390730c833637ec4d12de1c`。KSS 替代仍在补跑，最终全矩阵封存待该子面板闭合。任务 62 在同一单卡后排队，其启动入口先检查 Task 61 退出码和 Exp 1–6 全通过。[v1 新目录续跑](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1-exp6-resume.aVT9yx/run.sh)

v2 首次开启横航迹的迁移核验发现冻结 15k Actor 的预留第 9 列有 256 个非零权重（L2＝1.8018347025）。在同一合成双机图启用横航迹后，冻结 Actor 的最大 logits 差为 0.0208756924。工作区补齐 Actor、Critic、目标网络和锚点首次启用时的列清零，以及已载入优化器的对应缓冲清零；复测最大 logits 差为 0。选定高度观测仍沿用 Plan B 的反号迁移。80 项回归复跑全通过（34.52 s）。本项属于 v2 初始化核验，Task 61 和 Task 62 继续使用提交时的独立源码快照。

##### Step 1 正式封存

Task 61 正常退出，完整审计再次独立复跑与封存结果逐项相同。核验 182 份证据、Exp 1–6 齐全及 Exp 5 的 61 个配置，继承原始 Exp 1–5，补跑 Exp 6。发布权重继续是 SHA-256 `5a0982e5767b5544942035cf70fe28cf5fb6e46492657c923eaefec59555b1fe`。

| Figure 4 替代 | Geo 到达／碰撞／平均单机成本 | Geo+KSS 到达／碰撞／平均单机成本 |
| --- | ---: | ---: |
| Case 1 | 250 / 0 / 5.367 | 248 / 2 / 7.074 |
| Case 2 | 250 / 0 / 4.695 | 248 / 2 / 5.544 |
| Case 3 | 250 / 0 / 3.305 | 250 / 0 / 3.752 |
| 总计 | 750 / 0 / 4.455 | 746 / 4 / 5.457 |

完整审计 SHA-256 `d8db038bf42b0538122f407b577b0f6734b91f0b99a371e3f5539316630b11e0`，Exp 6 汇总 `415ec038e3647d1236873ce291fd8cad13a1ecd8647a242a3ed96f8f06dcf578`，15 份 KSS 替代汇总 `f370cd0b6626fdc66d274df5b301f8ad0a5856768ea4d042c7bcd53f98990a15`。Task 62 在封存完成后接手单卡，进入双开发面板。[v1 全矩阵封存](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1-exp6-resume.aVT9yx/paper-final-audit.json) · [Exp 6 最终汇总](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1-exp6-resume.aVT9yx/exp6/summary.json)

#### Step 2 启动修复与原生对照

Task 62 的 Python 入口缺少主进程保护，原生 worker 的 spawn 重新执行评测后碰到已建输出目录，未产出有效对照。保留失败日志，另建 CPFS 目录补齐入口保护。实际单 worker、4 个原生控制步的启动冒烟通过；另一个 2 边 3 步合成探针验证尝试发送 6 次、送达 4 次、在途 2 次，零重复。重提前原命令 80 项通过（20.22 s）。Task 63 已运行双开发面板，配置 SHA-256 清单的摘要为 `070fde2c0c7ff3a5b9db872d82611b737517d590446d4c8d637ca460d0752177`。正式种子入口由两个开发面板及其逐种子 MPC 预测约束共同控制。[失败日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-fixed-transport-mpc.1zArep/validation.log) · [新目录运行入口](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-fixed-transport-mpc.1rPxqL/run.sh)

##### Step 2 双开发面板实测

冻结同一个 15k 权重，原生 BlueSky 0.05 s，12 s MPC；对照为旧控制器的同权重，同面板前期开保护协议。v1.1 两组均未通过原生开发准入。每组 30 个逐场景逐种子的预测接触总量均满足计划值不高于保持值。

| 开发种子起点 | 旧到达／碰撞 | 新到达／碰撞／越界 | 六航路旧→新到达 | 保持→计划预测接触 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 61231 | 355 / 4 | 341 / 18 / 1 | 58→50 / 60 | 209→133 |
| 75121 | 350 / 10 | 337 / 22 / 1 | 54→48 / 60 | 276→177 |

两份原生报告 SHA-256 依次为 `9a1c4abfedff63b6089b0bab5753d8247ac69f9e663251a98acde16f83f03b3c`、`07fe10a2ac0a6ba84e0ceaf59a123ab4496dd569dfd5fc107feffe27594e4fb3`。因开保护闭环回退，Task 63 后续周期与延迟探针转到开发种子，正式留出保持封存，v2 训练未提交。预测计数满足单调约束仍需与原生碰撞独立判定。[Step 2 进展与原始报告索引](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-fixed-transport-mpc.1rPxqL/validation-in-progress.json)

##### Step 2 六航路回退定位

用固定同一 15k 权重、同一个原生 0.05 s CPU 切片，在六航路的 61231、75121 两个开发种子做 ADMM 和 MPC 的四臂交叉。两臂全旧与全新分别复现对应 PPU 的逐种子终局。

| ADMM／空窗 MPC | 61231 到达／碰撞 | 75121 到达／碰撞 |
| --- | ---: | ---: |
| old-old | 12 / 0 | 10 / 2 |
| new-old | 12 / 0 | 10 / 2 |
| old-new | 10 / 2 | 8 / 4 |
| new-new | 10 / 2 | 8 / 4 |

这两个种子的终局随 MPC 实现切换。四臂 SHA-256 为 `e59a8986e72b855d40b665aafd1172496d8f7fa3eeae1df7332996f4f3aed6f9`。逐帧回放 SHA-256 为 `0cf83ce6a02aa66a9986037686706ce46f2641db1c4cab2ab80c75ed45b89b56`。61231 第 73 决策把 0、1 号机预测水平最近点由 4996.658 m 提到 5004.178 m，并把预测接触从 1 降到 0；第 74 决策第 82 帧在 4999.892 m 水平、3.724 m 垂向触发真实接触。两个冻结状态及计算的证据 SHA-256 为 `af82e7290a018cdbd946f3f54b19693b28e15492685b34293e0e8ddfd7d7cd69`。
沿用现行 ADMM 已有的 20 m 水平裕度做独立反事实，只扩大 MPC 的试验半径。61231 恢复至 12/12 到达、0 架碰撞；75121 仍为 8/12、4。该试验不改 Task 63 控制器。反事实汇总 SHA-256 为 `3fa22f4a8676bdbfca761b7fae93ff481c1a240cd75be228eeee5c3c36a4402f`。
旧版 18／30 s 偏航已在同一个 61231 开发面板补齐冷热参考。热启动 18 s 的偏航越界为 0→0，30 s 为 3→3。历史正式 7–11 为部分配置的六场景总越界，与本组开发偏航分项分开报告。[四臂和逐帧诊断](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-component-diagnosis.1Wj3a6) · [开发偏航同种子参考](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-cadence-yaw-reference.RTQHRb/reference-summary.json)

##### Step 2 周期配对和六航路候选验证

61231–61235 的八个通信周期配置均完成。每个配置六个场景 × 五个种子，所有 240 条场景种子记录都满足 `planned_predicted_pair_contacts <= hold_predicted_pair_contacts`。18 s 热／冷启动的偏航分项分别为 116 到达／4 碰撞／0 越界、114／6／0；30 s 分别为 115／2／3、116／2／2。四个配置逐种子结果均与旧版同开发种子偏航参考一致。30 s 热启动仍超过 2 架的门槛。配对审计 SHA-256 为 `b34c0a7000dfc9c450bc4705fd6f55f85c91a33aeb599e0fcff817e964ebb79c`。[周期配对与全部预测接触数](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-fixed-transport-mpc.1rPxqL/development-cadence-audit.json)

75121 六航路的 3／4 号机从第 65 决策就进入预测冲突，第 76 决策第 101 帧真实碰撞。全层候选会带来第三机接触，试验加入半层后仍为 8 到达／4 碰撞。故障前六次空窗的冻结状态上，单机 41×41 调速／调高候选及双机分别 21×21 的组合均未找到满足双方与第三机分离的候选，评估固定航向、其他飞机保持已选指令。对应决策追踪、单机网格、双机网格 SHA-256 依次为 `d15a82824b4ec211ff06f89d67991c0a93d2ae6ffa6c5c639330e71be60975a6`、`6d4d1eda24c72e107de15a73e9b05f54477403354caf172feabc0ba2a5fd3106`、`2888133c77c032a6b34167da701fc7c8226a3774d87a6bc403c5f6dbdbe900d1`。[冻结候选](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-gap-candidates.XTxLuN/candidates.json) · [单机网格](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-gap-local-feasible.RGk5CN/nominal-peer-candidate-screen.json) · [双机网格](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-gap-pair-feasible.Mu8lnt/pair-compatibility.json)

在只改变局部候选集的独立原生 0.05 s 试验中，加入旧版已有的 ±5° 纯航向候选，保持邻机预测与零通信决策。61231 六航路由 10 到达／2 碰撞变为 12／0，预测接触 1→0；75121 由 8／4 变为 10／2，预测接触 19→15。该候选只在试验副本中启用，双开发面板验证另行记录。汇总 SHA-256 为 `2f1ba293da7cf74b8cdbbaecd080e8ee6499996801990433953a6cea6bbecd7a`。[航向候选原始报告](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-conflict-heading-probe.oTF3N1/summary.json)

##### Step 2 两条 MPC 修复的独立原生对照

加入 ±5° 冲突候选后，原生 0.05 s、12 s 周期的 61231–61235、75121–75125 两个开发面板均为 **355/360 到达、4 碰撞、1 越界**，近对头均 60/60，六航路均 58/60，预测接触分别 146→34 和 183→52；60 条场景种子记录均通过预测接触计数检查。两面板 SHA-256 依次为 `3a22eeab1676483c46dd51b96f7b14e181a4c5bd2265e2ee3d2edf8f29cf7450`、`8808c588ea6eaaa8e7b262ded88f0d46f0bdc81bf6a4b69f07369d66c191c5b5`。[61231–61235](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-heading-dev-probe.XaapwR/panel-61231/comparison.json) · [75121–75125](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-heading-dev-probe.XaapwR/panel-75121/comparison.json)

30 s 热启动三个越界样本为 61234 的负偏航和正偏航、61235 的负偏航，发生在零起始编号的第 68、81、87 决策，目标方位误差分别 +24.254°、−21.861°、+16.140°。三机越界前各有 25 次空窗，均无本机预测冲突，能执行的 Actor 纯航向提议分别为 0、1、1 次。只要有方位误差就回正的反事实让 61235 由 11 到达／0 碰撞／1 越界变为 8／4／0。用本机横向偏差及变化率预测未来 120 s 是否越出 20 km 边界，再在空窗与无冲突时限幅回正，三个场景均由 11／0／1 改为 12／0／0。该反事实将横向偏差仅送给 MPC，冻结 Actor 原输入仍为零。配对审计 SHA-256 为 `b630abeea8eb98c017b1306c99f0019f6965c92cb7e46bab8bea85ab98e009f4`。[偏航逐种子配对和原始轨迹 SHA-256](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-yaw-corridor-probe.zFEV7H/paired-yaw-diagnosis.json)

两项改动已经在新 CPFS 源码候选组合，重新跑 12 s 双开发面板及 18／30 s 冷热偏航。首次候选进程在启动前发现冻结源码缺少当前横向偏差常量，已同步工作树中的横向偏差模块及兼容迁移；旧的失败日志保留在候选目录中。组合 MPC 与原生调用点 SHA-256 分别为 `23c0a155e4831f0859362f0b7a7ba3ef1fd8af2b02d72e381989ec38c20ffe6d`、`0e805831da04b26cc6bbf8dd05271d9b760c8f259cd4fa9ac40a67c7dde73a94`。历史检查点、历史正式评测及 Task 63 源码均保持冻结。[组合开发试验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-mpc-corridor-integration.i8XA4r)

组合候选的原生 0.05 s 双开发门控均通过。61231–61235 从 Task 63 的 341 到达／18 碰撞／1 越界升至 **356／4／0**（上一版 355 到达），75121–75125 从 337／22／1 升至 **355／4／1**（上一版 350 到达）。预测接触分别为 146→34 和 186→53，逐场景种子违例均为 0。两份报告 SHA-256 为 `0923243c2fb51435d645f5fe0c36a0e2225f8f8e2ea9c61124aaedf1654411f4`、`64ec79b9c490b7dc14c4223db6d706cf3e3b5c471c38ca8b6bd73c44b66a0255`，双面板审计为 `85af51af7c70a03ce11ad9389b102ab3127c87e82b747c10c8eaa9a58a03dbfc`。[双开发门控](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-mpc-corridor-integration.i8XA4r/development-summary.json)

| 61231–61235 偏航，原生 0.05 s | 原到达／碰撞／越界 | 组合到达／碰撞／越界 | 组合报告 SHA-256 |
| --- | ---: | ---: | --- |
| 18 s 热启动 | 116／4／0 | 120／0／0 | `1b59939d51c23b89bc028d43cb77e425327e2d981d06365670a323853c24cdf5` |
| 18 s 冷启动 | 114／6／0 | 120／0／0 | `486b2f0eb925f7db49e233b704afd9a839ae6e03ab364ac7a61e9d0268375b96` |
| 30 s 热启动 | 115／2／3 | 119／0／1 | `28a32be583dcfc8460bf5694405c915e87ca4cbb6a83a5e04c986cb4ce2c710a` |
| 30 s 冷启动 | 116／2／2 | 120／0／0 | `5aaafc205d8b798f85407aec951a3255fb8dcce37f4c9acdc94dc6e849f56934` |

四组开发偏航共 40 条场景种子记录满足预测接触单调约束。组合候选 80 项回归全部通过（6.31 s），日志 SHA-256 `66eb2a2259ab202866e5ebb5c3060572d2a54a0f3280fdcc924df3b127a45f4f`。快照回归显式复用共享原作者求解器夹具。[组合回归与周期报告](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-mpc-corridor-integration.i8XA4r)

SOFY Task 64 在已授权单卡队列中排在 Task 63 之后。64 将重新核对 63 的全部 12 个传输报告哈希和活跃边计数，先测原生双开发面板与 18／30 s 六场景，再在通过后跑正式留出。Task 64 输入清单 SHA-256 为 `908e670ef627a7d8ba1a585a063212f739f3aabf97d99bc10e0a1c8e43445d5b`。截至此记录，正式留出尚未打开，v2 训练尚未提交。[Task 64 快照和提交记录](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-fixed-transport-mpc.kC4evi)

##### Step 2 延迟通信矩阵封存和单卡交接

Task 63 已退出（进程退出码 0），延迟通信 12 个配置全部完成独立复核。共 360 条场景种子记录、475,432 次发送。逐求解通道活跃边每控制步发送一次，所有发送数与逐步计数相等，重复发送为 0，已使用消息的龄期均不早于设定延迟。20% 档合计 31,475/157,948（19.927444%），50% 档合计 79,191/157,854（50.167243%）。原控制器仍因双开发质量和 30 s 热启动偏航越界被阻止晋升。输运报告 SHA-256 为 `ac123a9df38f3cee5f5808e1af1f4aecf0bdcb7c3e28562e598ac49accec587a`，Task 64 已接手同一单卡并核对该封存结果。[Task 63 封存报告](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-fixed-transport-mpc.1rPxqL/validation-summary.json) · [Task 64 复验](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-fixed-transport-mpc.kC4evi)

| 原生 0.05 s、K=2 | 发送 | 丢包 | 实测丢包率 | 对应报告 SHA-256 |
| --- | ---: | ---: | ---: | --- |
| k2-warm-loss-0-delay-1 | 40,453 | 0 | 0.000000% | `8044a1133612dae340dced2c6d55c6640b97235f1ccbb4fd3f65de94a3e5e54d` |
| k2-warm-loss-0-delay-2 | 39,546 | 0 | 0.000000% | `3a56ce085aa40c7e706336d4f3429166eab28d7d4d1afc49f05ffbc6952bf84b` |
| k2-warm-loss-0.2-delay-1 | 39,830 | 7,914 | 19.869445% | `ae30caba16a5250309bf8f5755d2360815658bed12ab370e53c68165bc016f25` |
| k2-warm-loss-0.2-delay-2 | 39,627 | 7,892 | 19.915714% | `877cf32da5b88a5572e1d1b33254c8b6d17e0488a96716fa37cb410599c8307c` |
| k2-warm-loss-0.5-delay-1 | 39,263 | 19,706 | 50.189746% | `fc3569d3e1ca48bc5177c5f958215b4544d44343df1c6551a5d0e79a693cb5a5` |
| k2-warm-loss-0.5-delay-2 | 39,788 | 19,997 | 50.258872% | `91dddb53608d933eb807ce68390b65ecf728c75d4c7afcdb83b9a44118a4f73a` |
| k2-cold-loss-0-delay-1 | 39,470 | 0 | 0.000000% | `90fc062afa6402b9f6042282479f97db4186bda7acff75a5a26559f48ac9a8a8` |
| k2-cold-loss-0-delay-2 | 40,161 | 0 | 0.000000% | `815b2bd7b8958c4b90d20360314299dd0eb4c646e56bd6881d48e35e92f619b6` |
| k2-cold-loss-0.2-delay-1 | 38,589 | 7,709 | 19.977196% | `115377b0fa8e6ae8120f9bd23e3af97b67b500bdd373a6382c97dffc8bbaebe9` |
| k2-cold-loss-0.2-delay-2 | 39,902 | 7,960 | 19.948875% | `08ebc8820ecb757348e3d4d0287c11ec60284d59b6396942216545e7425ffa91` |
| k2-cold-loss-0.5-delay-1 | 39,074 | 19,568 | 50.079337% | `51e4e668c74bcfd8941519c9517bb5afaa62df044bb3e11dc8685af2c07595e8` |
| k2-cold-loss-0.5-delay-2 | 39,729 | 19,920 | 50.139696% | `5cd77331dbf32ae8c917f099640cde5424170aae5153fddc6f4bbbc2333f1cf3` |

Task 64 单卡原生 0.05 s 双开发面板均已复现组合候选的到达、碰撞及越界。61231 为 **356／4／0**，预测接触 146→34，报告 SHA-256 `e6857845cd217280ef53083f5c8ecaf1c9961cdc23ae05c8a97928d6701a358e`；75121 为 **355／4／1**，预测接触 185→53（CPU 观察为 186→53），报告 SHA-256 `37c414e8672d6a1e25f2b77ad97921417308c062de6e822e38e9b1c728de2893`。两面板逐场景种子预测计数违例为 0，18／30 s 全场景复验进行中。[单卡双开发结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-fixed-transport-mpc.kC4evi/validation-in-progress.json)

独立复用原镜像审计的四组 256 个冻结原生状态，共 2,477 个活动机决策。相同冻结 15k 权重下，Actor、标准 ADMM 投影和组合空窗 MPC 的速度、高度、航向指令反射误差均为 0。镜像审计 SHA-256 为 `4eb156875218b215e7a8ecb109d41de249ec9729da7ff083094171441365ce3d`。[256 状态镜像复核](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-mpc-mirror-audit.uiQNGe/symmetry.json)

Task 64 的 18／30 s 单卡六场景开发周期验收已完成。18 s 热、冷均为 358 到达／2 碰撞／0 越界，预测接触分别 203→37、209→44；30 s 热为 359／0／1、236→39；30 s 冷为 360／0／0、230→39。四组逐种子预测接触违例均为 0，偏航越界依次 0、0、1、0。原始 SHA-256（18 热、18 冷、30 热、30 冷）分别为 `e5fa6c9d84e3e50da4f4e29b66b54484d14b53d60269083518d5998417aae728`、`da3ed8acbeac0de39edf63f5356f49f09a8c71483e8114dd4d44de92d28983f8`、`80d79947c26da9c6f16ad81953bf79dfb3e3a3c845a4cf349b4cc399dae72e89`、`46a0a4fe12191dcf0c0c1a638e7c3f57f516cba7cfd4dfba283aa473082dbb7e`。双开发和周期门控通过后，Task 64 首次打开本轮的 12001–12005 正式留出。[开发周期与正式留出进度](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-fixed-transport-mpc.kC4evi/validation-in-progress.json)

Task 64 的 12 s 正式留出为 **352/360 到达（97.78%）、8/360 碰撞（2.22%）、0 越界**，冻结 v1 为 355／4／1。三、四、六航路各 58／60 到达并各 2 碰撞，近对头和正偏航均 60／60，负偏航 58／60、2 碰撞。全场景单机平均动作成本为 26.496213，预测接触 198→61，30 条场景种子记录无计数违例。正式报告只用于验收。SHA-256 为 `5deeb007fec395736537d0569c89c89aeb86ed5f9e0086368879b4e552352773`。6／12／18／30 s 正式配对继续执行。[正式 12 s 原始报告](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-fixed-transport-mpc.kC4evi/formal-holdout/comparison.json)

20k Stage 1 入口已在独立 CPFS 目录静态准备，提交门控为 Task 64 完整通过并退出。采样会复用同一零通信空窗控制器，空域重置时同步横向偏差历史。候选源码 80 项回归通过。入场权重四案 Q 为 4/4（报告 SHA-256 `50a663168aeba946c9a8187e84b19fc7b26ed0c2bfa2d2b47e2536a7cc378428`），256 状态镜像误差为 0，三个热启动头权重差的计算在原权重自比时均为 0。还未提交或启动训练。[Stage 1 预检](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-v2-joint-admm-mpc-20k.2MDSXG/preflight)

##### Step 2 正式周期对照通过

Task 64 已结束，调度进程退出码 0，`validation_passed=true`。双开发、四组开发周期、12 s 正式留出与八组正式周期共 450 条场景种子记录满足预测接触约束。正式 18 s 热／冷偏航越界 4→0／5→0，30 s 热／冷 7→2／6→1。12 s 正式成绩为 352／360 到达、8 碰撞、0 越界，较冻结 v1 多 4 碰撞，仍低于 v2 最终晋升要求；本阶段接收按通信与 MPC 指标执行。逐报告哈希均已核对，最终 summary SHA-256 为 `241184a9adaf8bae6c78a43d6866e12e14936586cc924797be863bc78672c7b4`。[Step 2 正式验收](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v1_1-fixed-transport-mpc.kC4evi/validation-summary.json)

| 正式原生 0.05 s 周期 | 冻结版到达／碰撞／越界／未决 | v1.1 到达／碰撞／越界／未决 | 偏航越界 | v1.1 预测接触 | SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| k2-warm-period-6 | 350／10／0／0 | 349／10／1／0 | 0→1 | 0→0 | `9f9e2bc7cf0c5bc5adc958c3bce06cdf02971a2b6e0f62c9992eb0940d2d5cde` |
| k2-warm-period-12 | 351／8／1／0 | 347／12／1／0 | 1→1 | 212→77 | `2ae1738f9b6f8564d35e47352496e36ef41bdf69244cecbc9ed13ec947e70c71` |
| k2-warm-period-18 | 353／2／5／0 | 356／4／0／0 | 4→0 | 188→35 | `757a9a7aa9f4bc0b22d7a6a6acc0a9ac0b7bc2f3a921b732b628f8bddc992e4c` |
| k2-warm-period-30 | 346／4／10／0 | 353／4／2／1 | 7→2 | 225→44 | `b72c9b5cc819971549ca86ecf00acf3609cfa49834569d45d3c5c643c543a712` |
| k2-cold-period-6 | 345／14／1／0 | 342／16／2／0 | 1→2 | 0→0 | `e0377ce72f29e6369e8317ce8c7729f8200fad69f49528fecdfa07d3b94ad951` |
| k2-cold-period-12 | 351／6／3／0 | 350／10／0／0 | 3→0 | 220→80 | `12ace919b0b1aec9b5c15f3a19cec65d9e24adb35c3093c9da821739b4a4ec51` |
| k2-cold-period-18 | 353／0／7／0 | 350／10／0／0 | 5→0 | 226→70 | `0ddbc6f46cae6526400a286a8e59844595b45bb0e47ed54bc050396773692d2e` |
| k2-cold-period-30 | 343／6／11／0 | 353／6／1／0 | 6→1 | 233→66 | `86e3d8f264ff856f84f0052f994126069c8470594f78919dc5c46ab23033668e` |

##### Step 3 Task 65 联合训练已启动

Task 65 已在原 1 卡 PPU 开始运行，初始化使用冻结 15k 权重和 Critic。实际配置 SHA-256 `0aa3380239a972a436e00f1daae50246cffdc6bd7ad4a8dd0a5b96310b19afea`，完整输入清单 SHA-256 `562a0e02d9cf40023eeac87801be27deffd5e1862dc5003f9cfd0cbe4f04b167`，经核对的 Step 2 终审 SHA-256 `241184a9adaf8bae6c78a43d6866e12e14936586cc924797be863bc78672c7b4`。实际参数为 Plan B、512 空域、35 km、0.25 s 物理、6 s 控制、12 s MPC、24 步多步回报，`coordination_weight=0.15`、裕度 6 m、`projection_weight=0.25`、10% 原始采样、选中高度与横向偏差观测。学习率沿用上一轮的 1.2e-4 及 20k 更新余弦衰减，按本轮更新数保存 5k／10k／15k／20k。首条记录到 270 次更新，8.6776 updates/s，累计协调损失均值 4.9194e-08。四个里程碑均执行双原生开发面板开／关投影、四案 Q、256 状态镜像及三个热启动头差分。v2 正式种子门控保持封存。[Task 65 确切配置](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-v2-joint-admm-mpc-20k.2MDSXG/training/config.json) · [训练日志](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-v2-joint-admm-mpc-20k.2MDSXG/training/metrics.jsonl)

##### Step 3 入场无投影对照与观测迁移归因

冻结 15k 权重在原生 0.05 s 的无投影对照为 61231–61235 **232/360** 到达、75121–75125 **216/360** 到达。各场景到达数依次为三航路、四航路、六航路、近对头、正偏航、负偏航，第一面板 42／27／28／44／42／49，第二面板 44／26／20／40／40／46。报告 SHA-256 分别为 `b1f2d83087b4b3a3d8371ca7f84682d53f1b29c746893cc4781b72333ff98b46`、`f668100900b707819bd2a94c2f0dcb62c1f506dfc9ab4a6e269a792fb1aa3bf3`。[冻结权重无投影开发面板](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v2-raw-development-reference.Xp5sKR)

为区分新增观测与学习的作用，使用训练同样的权重迁移，仅打开选中高度与横向偏差观测，零更新重跑四案首步 Q。在原生观察中，Q 排序由冻结权重的 4/4 变为 3/4，失败的六航路 30 s 案 Q 差为 −1.19612885。5k 同案为 −2.083965，四案仍为 3/4。负号在学习前的原生观察迁移已经出现，5k 负差进一步扩大。Plan B 匹配观测的零更新排序为 3/4。零更新审计 SHA-256 `2e2ac48fe8ba42a2d0e314d993fd0aa4f029a6c43e55b84ffefbed464d6a3225`。[独立零更新 Q 审计](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v2-initial-observation-q-audit.2F63q4/critic-q.json)

四案历史脚本使用默认 20 km 建图。补充探针按 checkpoint 实际 35 km 观测重建同一批冻结状态，仅改变建图范围。冻结、零更新迁移、5k 的变更飞机排序分别为 3/4、1/4、3/4，六航路 30 s 案对应 Q 差为 −101.0690、−72.0774、−3.1339。因测试口径不同，里程碑保留原同口径四案排名。35 km 补充探针 SHA-256 为 `cab1f1861151ed2a9cab478cd663559327bbbea89af4e3f7092de30885be2f59`。[35 km 对照](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v2-initial-observation-q-audit.2F63q4/q-profile-range.json)

##### Step 3 Task 65 5,000 更新里程碑

配置累计至 523 决策、987 轮、9.3611 updates/s、6652.34 轮/h，Torch 峰值 3222.5 MiB。截至归档时全任务 PPU 样本均值 38.43%（120 个样本，含原生审计与启动）。断点 SHA-256 为 `6369b73d7cc239955fed9d6d8cea1c87ccca4e217e400e64b74c2bcb57d7519c`，里程碑摘要 SHA-256 为 `25ec777fad5bb20ed396b0c3385e49b81f6a443daac8216f051dcf3a999b8634`。[里程碑摘要](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-v2-joint-admm-mpc-20k.2MDSXG/milestone-05000/summary.json)

| 双开发面板 | 开投影到达／碰撞／越界 | 关投影到达／碰撞／越界 | 冻结 v1 关投影到达 | 预测接触违例 | 报告 SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| 61231–61235 | 338／22／0 | 207／136／17 | 232/360 | 0 | `9c3828ee63832e5c88f5e4dfa201ffb505ae4c177bfcec90315703eb2b7e3ec3` |
| 75121–75125 | 347／10／1 | 189／152／19 | 216/360 | 0 | `1d7a86fc3cc96cef2774cdd48f541fce851f911858dbc538965cdc7345ab524a` |

四案 Q 原生变更飞机 3/4、全活动飞机 3/4，Plan B 匹配观测 3/4。Q 差为 crossing4-120-native=+1.1652031, crossing6-120-native=+0.19997787, crossing6-30-native=-2.0839653, crossing6-60-native=+0.32979584，报告 SHA-256 `2a69c0b229fa7f6cf39600fb09e2f50165b5953eb62ee6497dcfa1a93d5c40df`。
镜像在 256 个冻结状态的最大误差为 0，报告 SHA-256 `c343663732343b323a38876bc79ba6dceb47c3fbaedbb995e86c49e8a63b6580`。三个热启动头 L2 权重差为 consensus=0.9350287264, altitude_consensus=1.101504323, willingness=0.2294299065。双开发与诊断综合晋升为 未通过。


仅迁移观测、不做更新的原生 0.05 s 无投影闭环对照，61231–61235 为 **265/360**、75121–75125 为 **251/360**，相对冻结权重分别增加 33、35 架。5k 权重随后降到 207/360 和 189/360，相对零更新迁移少 58、62 架。初始关投影完整结果的六场景到达分别为 48／40／32／60／38／47 和 44／34／24／56／44／49。两个零更新报告 SHA-256 为 `cf45b21764fc0de9576f8a2e04a991119973284ee0065b1fba0368995dc5d363`、`23ee7aef4a000d616c46c276cf673c29387534723a7d8fa865bf34c7d296003d`。5k 的 75121 面板开投影另有 2 架未决。训练已在 5k 审计后继续；至 5k 共有 27,313/267,776 个空域步走无保护轨迹。[零更新同种子闭环](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v2-initial-observation-dev-raw.nCOEyC) · [5k 全结果](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-v2-joint-admm-mpc-20k.2MDSXG/milestone-05000/summary.json)

在同一批 256 原生冻结状态上配对原始动作，零更新到 5k 的正偏航分层模式由 64 降至 6，负偏航由 40 降至 7，调速模式分别由 270→323、253→296。偏航且无冲突图边的 20 个状态均保留正确回航；有图边且需回航的正／负偏航状态中，正确转向分别由 20→22、0→14。原始策略闭环退化伴随明显的偏航分层提议减少。诊断 SHA-256 `8355c5fe0defbb9a685d5a2cc13a536ca51dbe8d9a51926bf66f3393cd2a5eee`。[冻结状态原始动作配对](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v2-initial-observation-q-audit.2F63q4/raw-action-5k.json)

5k 前训练区间单独按 PPU 采样时间对齐，排除后续原生评测。53 个训练区间样本的 PPU 平均利用率为 **80.91%**，峰值 100%。[原始 PPU 采样](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-v2-joint-admm-mpc-20k.2MDSXG/ppu-utilization.log)

开发逐帧归因按零更新→5k 额外碰撞量挑出三个显著回退样本，配对零更新、5k、10k 关投影。近对头 75125 的到达／碰撞为 12／0→4／8→2／10，前 65 个决策中的非零调速指令 748→612→110。正偏航 61235 的到达／碰撞／越界为 12／0／0→6／4／2→1／0／11，前 55 个决策的非零航向修正 23→14→3，受修正飞机 12→11→3。负偏航 75121 的终局为 12／0／0→5／6／1→6／2／4。配对堵点包含早期调速和偏航回正减少，具体接触帧与原始轨迹哈希见摘要 `0dbe5fd5c5e800155450ecc1d491fae6e5ad5d8e62f907432e88a07dfcf07472`。[三组成对原生轨迹](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate3/v2-raw-regression-traces.9Ahc9S/raw-regression-paired.json)

##### Step 3 Task 65 10,000 更新里程碑

配置累计至 1,023 决策、2,059 轮、9.3299 updates/s、6915.72 轮/h，Torch 峰值 3222.5 MiB。训练活动区间 PPU 利用率均值 80.32%（107 个样本）。断点 SHA-256 为 `2d5663490f6e9dd683af74259224920d844206344d88a91659d043576e623b6d`，里程碑摘要 SHA-256 为 `15fd7d91f92dad1dd96d44421b0e8d29c52c8b5944869f2be235885eab31f995`。[里程碑摘要](/dataset-cpfs5/vtm-foundation-pretrain/personal/zhuy25/Geo_ADMM_RL_20261123/bluesky-gate2/b35-v2-joint-admm-mpc-20k.2MDSXG/milestone-10000/summary.json)

| 双开发面板 | 开投影到达／碰撞／越界 | 关投影到达／碰撞／越界 | 冻结 v1 关投影到达 | 预测接触违例 | 报告 SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| 61231–61235 | 350／6／4 | 169／152／39 | 232/360 | 0 | `cf6fb9d1ada05a2643d774da3f967518bc739cf5b06542f9228f0289416fb065` |
| 75121–75125 | 346／10／1 | 159／156／45 | 216/360 | 0 | `6a92edbf75917e3f16bb24e78c00ff4100c2cdb7bf5e935525f18e6d5813e9b2` |

四案 Q 原生变更飞机 2/4、全活动飞机 3/4，Plan B 匹配观测 2/4。Q 差为 crossing4-120-native=-0.35467911, crossing6-120-native=+0.13114929, crossing6-30-native=-3.7714691, crossing6-60-native=+1.1956329，报告 SHA-256 `6655b1bfa6ba9efedd8ab692582c1c6fe2abf132be5adb1a7cb95292eadaa471`。
镜像在 256 个冻结状态的最大误差为 0，报告 SHA-256 `31c866d4646d3dd36c82aa8cd75de9b9fe5e643f95b631710625dac806e8bede`。三个热启动头 L2 权重差为 consensus=1.252249118, altitude_consensus=1.771196032, willingness=0.3249069103。双开发与诊断综合晋升为 未通过。

