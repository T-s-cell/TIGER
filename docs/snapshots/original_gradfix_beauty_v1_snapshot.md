# 快照归档说明：original_gradfix_beauty_v1_snapshot

- 日期：2026-10-23（实验）；2026-10-26（用户独立复核后冻结为**梯度修复实验的负结果**，归档 v1.1）
- 分支：`repro/original-gradfix-beauty`
- **训练代码 SHA：`fb2ab7b22d4ab6a05570e95f0cd5d64febe7b45c`**（= aeb3a97 训练源码 + `rqvae/models/rq.py` 两行 STE 梯度修复；全部训练与评估在该提交代码上完成。本说明文件为其上的纯文档提交，不含代码改动）
- 对照基线：`original_unfixed_beauty_v1_snapshot`（c3ac44c，训练代码 aeb3a97 零改动）
- 运行目录：`zeta:/dev_data/wlt/tiger_runs/original_gradfix_beauty_20261023_104730`（预检目录 iota:`..._20261023_134256`）
- 报告：`original_gradfix_beauty_report_v1_1.md`（/home/wlt/Rec/temp/、审计包内；v1 原文留痕不覆盖）
- 审计包：`audit_original_gradfix_beauty_v1_1.tar.gz`（858,833B，sha256 `9b14ffb5f96774d268838e9ce05218b5a440fb3f0779a520437c5e66343ea71e`，本地 /home/wlt/Rec/temp/ 与 zeta 运行目录各一份且均预解压，manifest 38 项两端 `sha256sum -c` 全 OK）；v1 包 `4e980d4f…` 两端保留留痕
- 性质声明：**本快照不标记验收通过，不标记论文复现成功**。定位为"非官方复现，原仓库默认配置"下修复 STE 梯度缺陷（rq.py 两行）与未修复基线的**同配置受控对照**。

## 冻结结论（2026-10-26，用户原文）

> **本次梯度修复按预期生效，但默认配置下六项测试指标下降约 2.9%–6.9%。**

由于两侧各运行一次且使用不同 GPU（基线 iota A800-80GB ↔ 本轮 zeta RTX 3090-24GB），不能进一步断言"修复必然有害"或"原作者有意如此设计"。用户独立复核项：两份 22,363 样本命中矩阵逐位一致；12,101 个四位 ID 唯一合法；码本使用率与碰撞统计吻合；审计包哈希全部匹配；gradcheck 与 reload PASS。

## 归档要点

1. **梯度检查（训练前置闸门）10/10 PASS**：索引逐位相等；原版 g_rq_l2=g_rq_l3=0（缺陷复现），修复版三层均非零；重构梯度系数 1（非 3 倍）；g_full = g_recon + Σg_li/3 线性一致。前向数值在浮点容差内一致（索引逐位相等、out max|diff| 2.98e-08）。
2. **端到端完成**：数据复用核验（8 文件哈希与基线逐项相等 + ItemID 对齐 ALL PASS）→ RQ-VAE 3000 epochs（Best Loss 0.0015628，Best Collision 0.1208）→ 四位 ID 闸门 **PASS**（(12101,4)、12101/12101 唯一、各位∈[0,255]、ItemID 链一致）→ rec 原仓库默认 85 轮自然早停（early_stop=10）、best 第 75 轮（valid NDCG@20 batch-mean 0.05303）→ 独立 test 复评（pos_index 与训练阶段**逐位一致**）→ 新进程 reload **PASS**（logits max|diff|=0.0）。
3. **结果（test，best 第 75 轮，sample-weighted，两侧均由各自 test_per_sample.npz float64 重算）**：R@5/10/20 = 0.035639/0.055315/0.084202、N@5/10/20 = 0.022892/0.029220/0.036502；相对基线（R@5 0.03671 / R@10 0.05858 / R@20 0.09042、N@5 0.02385 / N@10 0.03085 / N@20 0.03887）**六项全部更低（−2.92%～−6.87%）**。非法预测率未统计（原版评估无该统计逻辑）。
4. **RQ-VAE 行为变化（观察，不归因）**：验证碰撞率上升（0.121 vs 0.040）、第 1 层码本使用率大幅下降（[12,202,256] vs [123,256,256]）、三位重复余量上升（194/1.603% vs 26/0.215%；176 组/370 物品/最大 4 vs 25 组/51 物品/最大 3）、rec 最佳轮次提前（75 vs 84）、轮数更少（85 vs 94）、每轮墙钟约为基线 81%（~368s vs ~454s）。
5. **对照边界**：代码层面单因素（与基线差异仅 rq.py 两行，数据逐位相同、配置逐项相同）；但硬件不同（A800-80GB ↔ RTX 3090-24GB）且两侧各单次运行、GPU 非确定性——"低 3–7%"仅为对该基线单次运行的**观察**，不外推为一般结论；归因份额需同机多种子重复（未做）。
6. 过程异常已登记（报告 v1.1 §8）：zeta 网络受限（环境整包自 iota 拷贝 2.89GiB）、watcher 两度误报（误报日志留档）、rec 首启 dataset_path 崩溃一次后重启、巡检 tqdm 行计数≈真实轮数 2 倍（权威以 rec_train_summary.json 为准）。

## 保留记录（提交 SHA、配置、权重与产物哈希）

### 配置摘要

- RQ-VAE（aeb3a97 默认，未覆盖）：AdamW lr=1e-3，epochs=3000，bs=1024，linear 调度 warmup=50，wd=1e-4，layers=[512,256,128,64]，码本 3×256×32，β=0.25，kmeans_init，sk_epsilons=[0.0,0.0,0.003]，sk_iters=50，eval_step=50，seed=2024，grad clip 1.0
- rec（aeb3a97 默认，未覆盖）：Adam lr=1e-4，最多 200 epochs，early_stop=10（valid NDCG@20 batch-mean），vocab=1025（无 user token），bs=256，seed=2025，beam=30，infer_size=96
- 数据：Phase 1 冻结版（含 salesRank），item_emb.parquet sha256 `e8294f33…`（与基线逐位相同）
- 代码差异：仅 `rqvae/models/rq.py` 两行——`residual = residual - x_res.detach()`（恢复层 2/3 commitment 对 encoder 的梯度）+ return 前 `x_q = x + (x_q - x).detach()`（重构梯度系数钉回 1）；另新增 `scripts/gradcheck_{run,compare}.py`（检查脚本，非训练路径）
- 环境：zeta GPU 0（RTX 3090-24GB），torch 2.5.1+cu121（整包自 iota 复刻，Python 3.10.22、numpy 1.26.4），FP32

### 权重与产物哈希（sha256）

| 产物 | 哈希 |
|---|---|
| 训练代码提交 | `fb2ab7b22d4ab6a05570e95f0cd5d64febe7b45c`（本提交为其上的纯文档） |
| RQ-VAE best_collision_model.pth（所选，用于 codes） | `04be09d4fccdfa04f23e3f06c58c3d5c1a3b797ac48bf5cae92b283e7073ed06` |
| RQ-VAE best_loss_model.pth | `422545ae1cd52dfac25f0d4861805ba2c1fb5995b19d7ed26a8cdf6ff3ecf27f` |
| codes.npy（四位 ID） | `ec53e6417c058245a2a0fa3065cb6d8c37ecb3a35376b9267825a302a9f106a3` |
| item_emb.parquet（复用基线） | `e8294f337829bbdd421ad411fe489a84fbf78729dd0060a3040afa9487b4cf32` |
| tiger_best.pth（best 第 75 轮） | `637ac4cf149bacce8c5669aa79a73a80e85564f2813b2ddcf54d206afbe418a9` |
| tiger_best_last_smoke.pth（第 85 轮） | `cd83155f25da36a710a0f243a98f9b1ec16abafa813ef56384f44adebdf67313` |
| rec_train_summary.json | `5be055c3c7d42aae96b9815388b228d1f4f52e4b978decbb89b633b2615cba19` |
| native test_per_sample.npz（04_rec） | `cfbf2c819a2384954e1cb0e6bfae0bc13159be61e0aaddf78d6454b5e004e85b` |
| test_eval/test_per_sample.npz（独立复评） | `164819e9d013fa5cc1b4ebd54e0ccd71bde449784818a3ec0e02ad26eeb4aac1` |
| test_eval/eval_report.json | `ef7ea3927f9e886d17f8d8bf563df1d87ad28fafdc024c9f81a36d37a7cf6668` |
| reload_check/eval_report.json | `a49f71ab91589bae70e251e6352ec835a18c6cbda180d319c1fc3c173e0b2117` |
| 审计包 v1.1 | `9b14ffb5f96774d268838e9ce05218b5a440fb3f0779a520437c5e66343ea71e` |
| 审计包 v1（留痕） | `4e980d4fa083d3aca0b0f1b374665263cbea952baa49be4ca84825252aa0bb9b` |

### 结果速览（test，best 第 75 轮，native 与独立复评逐位一致）

| 指标 | 基线（未修复，ep84） | 本轮（修复，ep75） | 相对差 |
|---|---|---|---|
| Recall@5 | 0.03671243 | 0.03563923 | −2.92% |
| Recall@10 | 0.05857890 | 0.05531458 | −5.57% |
| Recall@20 | 0.09041721 | 0.08420158 | −6.87% |
| NDCG@5 | 0.02385491 | 0.02289219 | −4.04% |
| NDCG@10 | 0.03084847 | 0.02921972 | −5.28% |
| NDCG@20 | 0.03886931 | 0.03650219 | −6.09% |

sample-weighted（float64 逐样本重算，重算脚本与结果在审计包 04_rec/）；batch-mean 并列见报告 v1.1 §5。

## 边界

- 单次运行（RQ-VAE、rec 各一次），硬件与基线不同（已注明），无多种子重复；GPU 非确定性意味着重跑轨迹微小分歧
- 非官方复现，原仓库默认配置（与论文文本描述不同，已在基线报告登记）
- 未启动：同机同配置多种子重复、salesRank 消融、user token 消融、层 1 使用率专项观察
