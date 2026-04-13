# 评测标准与方法论

**更新时间**: 2026-04-13

---

## 一、Memory Benchmark 评测标准

### 1.1 LongMemEval (ICLR 2025)

**五大核心能力**：

| 能力 | 描述 | 示例 |
|------|------|------|
| **Information Extraction (IE)** | 从长对话中提取特定信息 | "我在动物园看到了什么？" |
| **Multi-Session Reasoning (MR)** | 跨会话整合信息 | 聚合、比较多个会话的信息 |
| **Temporal Reasoning (TR)** | 时间感知推理 | 理解时间顺序、时间戳 |
| **Knowledge Updates (KU)** | 动态更新知识 | 用户偏好改变时更新记忆 |
| **Abstention (ABS)** | 识别未知信息 | 正确回答"我不知道" |

**评测指标**：
- QA Accuracy (LLM grader)
- Recall@k (检索准确率)
- NDCG@k (排序质量)

**当前 SOTA**：LiCoMemory 73.8% accuracy, ENGRAM-R +21.8 pp

---

### 1.2 MemoryAgentBench (2026)

**四大核心能力**：

| 能力 | 描述 | 对应 Engram 设计 |
|------|------|-----------------|
| **Accurate Retrieval (AR)** | 精准检索 | Elo 竞争检索 |
| **Test-Time Learning (TTL)** | 运行时学习 | Hebbian 可塑性 |
| **Long-Range Understanding (LRU)** | 长程理解 | 碎片整合、全局认知 |
| **Conflict Resolution (CR)** | 冲突解决 | 知识更新、衰减机制 |

**评测指标**：
- SubEM (子串精确匹配)
- ROUGE-F1
- Model-evaluated F1 (GPT-4o)

**数据集规模**：
- 1.44M tokens 对话
- 172k tokens 长文本摘要
- 262k tokens 冲突解决

---

### 1.3 Minerva Benchmark

**原子任务**：
- Search（搜索）
- Recall & Edit（召回编辑）
- Match & Compare（匹配比较）
- Spot the Differences（发现差异）
- Compute on Sets/Lists（计算）
- Stateful Processing（状态处理）

**复合任务**：
- Processing Data Blocks
- Theory of Mind（心智理论）

**特点**：自动生成测试、可解释评估

---

### 1.4 评测指标汇总

| 指标 | 公式/描述 | 适用场景 |
|------|----------|----------|
| **QA Accuracy** | 正确回答数 / 总问题数 | 问答质量 |
| **Recall@k** | 前k个结果包含答案的比例 | 检索质量 |
| **NDCG@k** | 考虑排序位置的检索质量 | 检索排序 |
| **F1** | 2PR/(P+R) | 精确率召回率平衡 |
| **SubEM** | 答案子串精确匹配 | 冲突解决 |
| **ROUGE** | n-gram 重叠 | 摘要质量 |
| **Efficiency** | tokens/query, latency | 效率 |

---

## 二、Hebbian Learning（赫布学习）

### 2.1 核心原理

**经典表述**：
> "一起放电的神经元，会连接在一起"（Cells that fire together, wire together）

**数学公式**：
```
Δw_ij = η · o_i · o_j
```
- `Δw_ij`: 神经元 j → i 的连接权重更新
- `η`: 学习率
- `o_i`, `o_j`: 神经元激活值

### 2.2 为什么不需要传统训练

| 特性 | 传统深度学习 | Hebbian Learning |
|------|-------------|------------------|
| 学习方式 | 反向传播、梯度下降 | 前向传播、局部规则 |
| 监督信号 | 需要标签 | 无监督 |
| 全局优化 | 需要 | 不需要 |
| 在线学习 | 批量更新 | 实时更新 |
| 可塑性 | 权重冻结后固定 | 持续可塑 |

### 2.3 与 Engram 的结合

**Hebbian 规则应用于记忆形成**：
1. **新记忆编码**：当神经元 A 和 B 同时激活 → 连接加强
2. **记忆检索**：激活神经元 A → 通过加强的连接激活 B
3. **联想回忆**：部分激活 → 通过连接网络扩散 → 完整记忆

**H-Mem 网络（研究参考）**：
- 存储 branch：key-vector × value-vector → Hebbian 更新关联矩阵
- 召回 branch：query key → 检索关联矩阵 → 返回 value
- 可实现 one-shot 记忆和问答任务

### 2.4 在 Engram 中的实现

```python
class HebbianUpdate:
    """Hebbian 学习规则应用于神经元连接"""
    
    def update_connection(self, neuron_a, neuron_b, learning_rate=0.1):
        """
        当神经元 A 和 B 同时激活时，更新连接强度
        
        Δstrength = η × activation_a × activation_b
        """
        # 检查是否同时激活
        if neuron_a.is_active and neuron_b.is_active:
            delta = learning_rate * neuron_a.activation * neuron_b.activation
            connection = neuron_a.get_connection_to(neuron_b)
            connection.strength += delta
            
            # 应用上限约束
            connection.strength = min(connection.strength, MAX_STRENGTH)
```

---

## 三、Karpathy autoresearch 框架

### 3.1 核心思想

**让 AI agent 自主进行实验迭代**：
1. Agent 读取 `program.md`（研究指令）
2. Agent 修改 `train.py`（模型代码）
3. 训练 5 分钟（固定时间预算）
4. 评估 `val_bpb`（验证指标）
5. 决定保留或丢弃
6. 重复

**效率**：~12 实验/小时，~100 实验过夜

### 3.2 关键设计

| 设计 | 描述 | 对我们的启示 |
|------|------|-------------|
| **单一文件修改** | Agent 只修改 train.py | 限制修改范围，可追溯 |
| **固定时间预算** | 每次实验 5 分钟 | 公平比较不同方案 |
| **单一指标** | val_bpb | 目标明确，不混乱 |
| **program.md 接口** | 人类通过 md 文件指导 agent | 类似我们的 SPRINT.md |

### 3.3 应用于 Engram 项目

**可借鉴的设计**：
- 创建 `experiment.py` 作为 agent 修改的目标
- 创建 `metrics.py` 定义评测指标（基于 benchmark）
- 固定实验预算（如 100 轮对话测试）
- Agent 自主尝试不同的 Elo 参数、衰减率配置
- 自动记录每轮实验结果

**自主进化循环**：
```
1. 读取 SPRINT.md 和 pitfalls.md
2. 修改 memory/elo.py 或 memory/decay.py
3. 运行 benchmark 测试（如 LongMemEval 子集）
4. 记录 metrics（Accuracy, Recall@k）
5. 如果改进 → 保留，否则 → 回滚
6. 更新 pitfalls.md（新发现）
7. 重复
```

---

## 四、Engram 项目评测方案

### 4.1 选择 Benchmark

**主选**：LongMemEval
- 覆盖面广（5 核心能力）
- 有成熟评测框架
- 与对话场景高度相关

**补充**：MemoryAgentBench 的冲突解决任务
- 测试知识更新能力
- 测试衰减机制

### 4.2 评测指标

| Sprint | 主要指标 | 次要指标 |
|--------|----------|----------|
| Sprint 1 | Recall@5, NDCG@5 | QA Accuracy |
| Sprint 2 | QA Accuracy | F1 |
| Sprint 3 | QA Accuracy (multi-session) | Efficiency |
| Sprint 4 | Multi-hop Reasoning | Temporal Reasoning |

### 4.3 版本追踪

每次 Sprint 完成后运行评测：
```
docs/benchmark-results/
├── v0.1-baseline.json      # Sprint 0（当前代码）
├── v0.2-elo-competition.json  # Sprint 1
├── v0.3-stability.json     # Sprint 2
└── ...
```

---

