# 快照归档说明：paper_align_lr004_beauty_v3_snapshot

- 日期：2026-10-21
- 分支：`repro/paper-aligned-beauty`
- **训练提交 SHA：`7f74b45f27b417dc517e22d6841a0631d25f4661`**（全部训练与评估在该提交完成；本说明文件为其后的文档提交，不含代码改动）
- 运行目录：`iota:~/tiger_runs/paper_align_lr004_beauty_20261020_170252`
- 报告：`paper_align_lr004_beauty_report_v3.md`（/home/wlt/Rec/temp/ 与审计包内）
- 性质声明：**本快照不标记验收通过，不标记论文复现成功**。定位为"TIGER 方法复现，训练超参数有所调整"（RQ-VAE lr 0.4→0.04）的阶段快照。

## 归档要点

1. **RQ-VAE 塌缩已缓解（非"解决"）**：lr=0.04 下 20,000 epochs 无塌缩；固定模型终态层使用率 [247,256,256]（第 1 层 ep500 低谷 [103,223,255] 后单调恢复）；验证碰撞率 best 0.0221、终态 0.0383 平台；recon（样本加权）0.0677 = 常数基线 0.147997 的 −54.3%。限定表述：仅证明本实现该配置下 20k epochs 稳定，不外推论文 lr=0.4 的行为。
2. **全流程完成**：文本编码 → RQ-VAE 20k → 四位语义 ID（12101/12101 唯一、各位 ∈[0,255]）→ rec 200,000 steps（loss 5.48→2.475 单调、lr 调度逐位精确）→ valid/test 全量评估 → 新进程 reload PASS（logits max|diff|=0.0）。端到端实测 4.6h（A800 单卡）。
3. **推荐效果未达到论文水平**：test Recall@5/10/20 = 0.0082/0.0145/0.0225、NDCG@5/10 = 0.0045/0.0065（sample-weighted）；≈ 均匀随机基线的 13.6 倍（模型学到明确信号），但仅为论文 Table 1 Beauty 值（Recall@10 0.0648 / NDCG@10 0.0384）的约 22%/17%。差距与已登记偏差（lr 调整、实现假设、单次运行）相关，未做归因分解。非法预测率 0.0（670,890 候选全合法）。
4. **审计待补齐**：见下节未完成项。

## 未完成项（审计待补齐）

1. **完整预测证据**：评估逐样本产物 `valid/test_per_sample.npz` 仅含 `pos_index`（22363×30 命中/排名布尔矩阵）；**逐样本 beam 候选 token 数组未落盘**（`tiger_final_ref.npz` 仅 32 条固定样本的 preds）。独立复算指标所需的完整候选编码待补（需从 ckpt 重新生成或扩展 evaluate.py 落盘）。
2. **统计脚本修复**：派生脚本 `gate_review.py` 第 99 行存在运算符优先级 bug（`'%.4e' % (sum(cb)+sum(cm)) / 1` → TypeError），崩溃于 latent/final 统计打印之前（崩溃前已打印的 recon/codebook/commitment 数值有效）；`sw_stats.py`、`final_measures.py` 等派生脚本未经第二人独立复审。
3. **报告误述修正**（报告 v3 已核实存在以下误述，待出修正版）：
   - §4"第 4 位仅 10 个碰撞组参与消解"**有误**：实际为 **366 个碰撞组**（829 个 item 涉及，最大组 10）。第 4 位的 10 个不同取值是**组内位置索引 0..9**（唯一三元组记 0；碰撞组内按 ItemID 升序记 0..size−1），"10/10 使用率"≠碰撞组数。ID 唯一性结论本身不受影响（12101/12101 唯一已按四位元组直接验证）。
   - §3 表头与 §9 复现命令中 `--diagnose_every 10` **有误**：实际运行参数为 `--diagnose_every 100`（run_info.txt 记录；报告 §3 表格取样点均为 100 的倍数或末 epoch，与 100 一致，表格数值本身不受影响）。
   - §1"评估逐样本产物（beam token 数组与命中/排名布尔矩阵）"**与实际不符**：npz 仅含命中/排名布尔矩阵（见未完成项 1）。

## 保留记录（提交 SHA、配置、权重与产物哈希）

### 配置摘要

- RQ-VAE（训练提交 7f74b45）：epochs=20000，**lr=0.04**（论文 0.4，登记偏差），seed=2024，Adagrad（initial_accumulator_value=0.1, eps=1e-7, lr_decay=0），constant 调度，β=0.25，encoder layers=[512,256,128]，码本 3×256×32，bs=1024，kmeans_init，grad clip 1.0，diagnose_every=100，eval_step=50
- rec（论文配置未调整）：200,000 steps，T5 4+4 层、6 heads×64、d_model=128、d_ff=1024、ReLU、dropout 0.1，vocab=3025（1+1024+2000），bs=256，lr=0.01 恒定 10,000 步后 inverse-sqrt，Adam(0.9,0.999,eps 1e-8,wd 0)，seed=2025，beam=30，infer_size=96，topk 5/10/20
- 环境：iota A800-80GB，torch 2.5.1+cu121，Python 3.10.22，FP32（pip freeze 见审计包）

### 权重与产物哈希（sha256）

| 产物 | 哈希 |
|---|---|
| 训练提交 | `7f74b45f27b417dc517e22d6841a0631d25f4661` |
| 输入 item_emb.parquet | `2bb409894ec92e133738ec4b5e9867bb4ea7f70455d320f6afb2af39212461ac` |
| RQ-VAE final_model.pth | `f80e42a66f5369581e99f068b144a95da47f9ce739cbe60625c5381d53cb39ef` |
| codes.npy（四位 ID） | `89701b045e95721fdf889ed8ad7403032a64d8bb3f95ce610e76e54812665c7a` |
| tiger_final.pth（rec 权重） | `232d423b586fb4931fd560307b11274fffba43c4efb43f6f15eecda4dc58b038` |
| tiger_final_ref.npz（32 条参考输出） | 见运行目录（iota 保留） |
| valid_per_sample.npz | `a4a0914629754be609b978067533a4fe711ea35e3f19d19116986bbaa3f13988` |
| test_per_sample.npz | `a3691ccc082bdb3d6fef535d884199453b0c813741b9c7b6dc298fb2acde7371` |
| reload eval_report.json | `4626f83c0b08597a27384c706d137192f73baaf1dbf3ea14758b27901237eca2` |
| user_mapping.npy | `23ca50dc892c32570f5809020fc294b544d3ee7afda007f34aa95bfee42737c2` |
| user_bucket_map.json | `d4045d73d50c7e411dc2488d46d45ba038096f7d59fef70784db59442b378ffa` |
| data/（train/valid/test parquet 等） | 见运行目录 `run_info.txt` 内逐文件哈希 |

大文件（权重、codes、逐样本 npz、item_emb）保留在 iota 运行目录，不入审计包（<50MB）。

### 结果速览（sample-weighted）

| split | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 | 非法率 |
|---|---|---|---|---|---|---|---|
| valid | 0.01203 | 0.01994 | 0.03036 | 0.00701 | 0.00952 | 0.01216 | 0.0 |
| test | 0.00823 | 0.01449 | 0.02249 | 0.00451 | 0.00650 | 0.00853 | 0.0 |

## 边界

- 单次运行（RQ-VAE、rec 各一次），无多种子重复；CUDA embedding 反向原子累加导致重跑轨迹微小分歧（初始化与首步锚点逐位一致）。
- 未启动（等指示）：Sinkhorn 等抗塌缩变体、warmup/优化器更换实验（未触发条件）、超参搜索、多种子重复。
- 已知事故（不影响产物）：2026-10-21 02:15 rec 首次启动出现双进程重复，02:18 终止并清理后单进程重启；该时段无产物落盘。
