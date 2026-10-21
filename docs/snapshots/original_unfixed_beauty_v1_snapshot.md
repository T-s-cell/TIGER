# 快照归档说明：original_unfixed_beauty_v1_snapshot

- 日期：2026-10-21
- 分支：`repro/original-unfixed-beauty`
- **训练代码 SHA：`aeb3a9715185376aed70f1b522308e866a427957`**（tag `original_smoke_beauty_pass`；全部训练与评估在该提交代码上完成，**零改动**，仅命令行路径/设备覆盖。本说明文件为其上的纯文档提交，不含代码改动）
- 运行目录：`iota:/dev_data/wlt/tiger_runs/original_unfixed_beauty_20261020_234558`（run_info.txt、artifact_sha256.txt、06_logs/ 全部日志）
- 报告：`original_unfixed_beauty_report_v1_1.md`（/home/wlt/Rec/temp/、运行目录 06_logs/、审计包内；v1 原文留痕不覆盖）
- 性质声明：**本快照不标记验收通过，不标记论文复现成功**。定位为"非官方复现，原仓库默认配置"（aeb3a97；保留原 STE 梯度相消问题，原仓库默认配置与论文文本描述并不相同）的未修复梯度完整基线，用途是后续受控对照的"不修复"侧。

## 归档要点

1. **端到端完成**：数据复用核验（含 salesRank 原版 embedding，哈希逐项与 Phase 1 相等 + ItemID 行序对齐 PASS）→ RQ-VAE 3000 epochs 原仓库默认（AdamW 1e-3、Sinkhorn sk_epsilons=[0,0,0.003]；Best Collision 0.0401）→ 四位 ID 闸门 **PASS**（(12101,4)、12101/12101 唯一、各位 ∈[0,255]、ItemID 链对齐）→ rec 原仓库默认（Adam 1e-4、vocab 1025 无 user token、early_stop=10）94 轮自然早停、best 第 84 轮 → 独立 test 复评 + 新进程 reload **PASS**（logits max|diff|=0.0）。
2. **结果**：test（best 第 84 轮，sample-weighted）R@5/10/20 = 0.0367/0.0586/0.0904、N@5/10/20 = 0.0239/0.0309/0.0389；= 论文 Table 1（Beauty）的 74–90%；≈ v3 快照（`paper_align_lr004_beauty_v3_snapshot`）的 4.0–4.7 倍。非法预测率**未统计**（原版评估无该统计逻辑）。
3. **对照边界**：与 v3 的对照**非受控**（数据 salesRank、RQ-VAE 配置、rec 配置同时不同），无法区分各因素（数据、RQ-VAE 配置、rec 配置、梯度状态）的相对贡献；与论文数值对照为非官方复现观察，不构成复现成功声明。
4. **基线记录（仅记录，非闸门）**：三位编码不同前缀 12075、重复余量 26（率 0.215%）、涉碰撞物品 51、25 组（24×2+1×3）、最大组 3（generate_code 日志"23 组"为重分配中间统计，不同源）；码本使用 [123,256,256]/256，第 1 层 48%。
5. 环境异常已登记（报告 v1.1 §7）：2026-10-21 13:45–20:18 iota 自本地不可达伴随 ssh 主机密钥告警，机器实际 uptime 16 天未重启、训练无影响，判为网络路径异常。

## 保留记录（提交 SHA、配置、权重与产物哈希）

### 配置摘要

- RQ-VAE（aeb3a97 默认，未覆盖）：AdamW lr=1e-3，epochs=3000，bs=1024，linear 调度 warmup=50，wd=1e-4，layers=[512,256,128,64]，码本 3×256×32，β=0.25，kmeans_init，sk_epsilons=[0.0,0.0,0.003]，sk_iters=50，eval_step=50，seed=2024，grad clip 1.0
- rec（aeb3a97 默认，未覆盖）：Adam lr=1e-4，最多 200 epochs，early_stop=10（valid NDCG@20 batch-mean），vocab=1025（无 user token），T5 4+4 层、6 heads×64、d_model=128、d_ff=1024，bs=256，seed=2025，beam=30，infer_size=96，topk [5,10,20]；参数 4,590,848
- 数据：Phase 1 冻结版（含 salesRank），item_emb.parquet sha256 e8294f33…
- 环境：iota A800-80GB，torch 2.5.1+cu121，Python 3.10.22，FP32

### 权重与产物哈希（sha256）

| 产物 | 哈希 |
|---|---|
| 训练代码提交 | `aeb3a9715185376aed70f1b522308e866a427957`（零改动；本提交为其上的纯文档） |
| 输入 item_emb.parquet | `e8294f337829bbdd421ad411fe489a84fbf78729dd0060a3040afa9487b4cf32` |
| codes.npy（四位 ID） | `65561c596cceb87e207418badf12f0a0b38fb626ae34a54179445b03dba5dd1d` |
| tiger_best.pth（best 第 84 轮） | `483ab258bf868bfac36da86a7de3a55c7456a350f98da74d54ab8be40fc19d36` |
| tiger_best_last_smoke.pth（第 94 轮） | `bc96c9f74a109dff12e3d44aaaaab86b54482d5f1475970c1a78ff284fd27df1` |
| rec_train_summary.json | `f31a5392bfe04a9f8ecf36752e1ee9b8d24235dee09c62ceb332405ae271a67d` |
| 原生 test_per_sample.npz | `d45a9a21483e6953dcf61caf527d57e72f3393ab38720a38835a1c51ba3ca201` |
| test_eval/eval_report.json | `4d4f7903916352304d350a92f121bfd674030323d2b7d4b2f813f79b3faa58ac` |
| reload_check/eval_report.json | `ca2645ba9a21b0b293a1de4fe60a3791c3d19d75fdf9fa248ee1701ac9384c44` |

RQ-VAE 权重与数据逐文件哈希见运行目录 `02_rqvae/Oct-21-2026_07-49-45/`、`06_logs/data_sha256_copied.txt`、`artifact_sha256.txt`。大文件（权重、codes、逐样本 npz、item_emb）保留在 iota 运行目录，不入审计包（<50MB）。

### 结果速览（test，best 第 84 轮，原生与独立复评逐位一致）

| 口径 | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |
|---|---|---|---|---|---|---|
| batch-mean | 0.03675 | 0.05865 | 0.09050 | 0.02388 | 0.03088 | 0.03891 |
| sample-weighted | 0.03671 | 0.05858 | 0.09042 | 0.02385 | 0.03085 | 0.03887 |

## 边界

- 单次运行（RQ-VAE、rec 各一次），无多种子重复；GPU 非确定性意味着重跑轨迹微小分歧
- 未启动（等指示）：同配置修复 vs 不修复受控对照、salesRank 消融、user token 消融、层 1 使用率专项观察、多种子重复
