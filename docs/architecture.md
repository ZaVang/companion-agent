# Engram 记忆系统架构设计

**版本**: 0.2
**更新时间**: 2026-04-13
**设计者**: 用户 + Ari

---

## 一、神经科学基础

### 1.1 神经元基础

**神经元数量动态性**：
- 人脑约 860 亿神经元
- 神经元会死亡（细胞凋亡、损伤）
- 成人海马体每天产生约 700 个新神经元（神经发生）
- **设计启示**：神经元数量不应固定，支持动态增删

**突触连接**：
- 单个突触是单向的（轴突 → 树突）
- 两个神经元之间可形成双向连接（两个独立突触）
- 突触强度可通过 LTP/LTD 调节
- **设计启示**：保持双向连接设计，但区分信号流向

### 1.2 记忆分区

**海马体 - 新皮层系统**：
| 分区 | 功能 | 对应设计 |
|------|------|----------|
| 海马体 | 短期/情景记忆编码 | ShortTermMemory |
| 新皮层 | 长期记忆存储 | EpisodicMemory |
| 前额叶 | 工作记忆 | 当前对话上下文 |

**睡眠转化（记忆巩固）**：
- 非REM睡眠：海马体 → 新皮层传输
- REM睡眠：记忆整合、情绪处理
- **设计启示**：需要"睡眠"机制做记忆固化

### 1.3 Default Mode Network (DMN)

**来源**: Stanford Menon 教授 2023 年论文《20 years of the default mode network》

**核心发现**：
- DMN 是空闲时大脑的默认活动网络
- 功能：自我参照、社会认知、情景记忆、语义记忆、思维漫游
- 创造连贯的"内在叙事"
- 与 salience network（显著性网络）协同：外部刺激 → salience 激活 → DMN 抑制

**对 Engram 的启示**：

| DMN 功能 | Engram 设计 |
|---------|-------------|
| 自我参照 | 记忆与 persona 绑定 |
| 情景记忆 | Engram 的碎片化存储 |
| 思维漫游 | 空闲时的记忆整合 |
| 内在叙事 | Reflection 机制 |

---

## 二、核心设计理念

### 2.1 记忆是碎片化存储

**一条记忆 = 一组神经元的集合**

```
Engram {
    neurons: Set[NeuronCell]
    represent: NeuronCell      # 代表神经元
    strength: float            # 整体强度
    summary: str               # LLM 生成的摘要
}
```

**问题**：神经元强度在变化，如何保证稳定想起？

**解决方案**：集体稳定性 + 激活阈值
- 记忆的"可想起程度" = 所有成员强度的聚合值
- 超过 N% 的成员强度 > T 时，记忆可被激活
- 代表神经元稳定性 > 成员神经元

### 2.2 记忆是塑形，不是存储

**每次激活都在重塑记忆**

| 传统数据库思维 | 神经思维 |
|--------------|---------|
| 写入 → 存储 → 读取 | 激活 → 重塑 → 输出 |
| 静态不变 | 动态演化 |
| 查询是读取 | 查询是激活 |

**实现方式**：
- 每次检索后，更新神经元的 strength/Elo
- 激活的神经元之间连接强度增加
- 长时间不激活的记忆逐渐衰减

### 2.3 遗忘是不可达

**遗忘 ≠ 删除**

- 记忆还在，但触发条件变苛刻
- "入口"变少：连接强度衰减、神经元死亡
- 特定线索可以瞬间激活（"顿悟"）

**实现方式**：
- 衰减机制：strength × decay_rate
- 不同记忆类型不同衰减率
- 冲击力高的记忆衰减极慢

### 2.4 因果推断

**记忆系统应该学习"什么导致什么"**

- 连接 ≠ 因果
- 因果关系需要从激活序列中学习
- A 激活后，B 是否以高于随机的概率激活？

**实现方式**：
- 记录激活序列
- 统计共现频率
- 构建因果图（未来 Sprint）

---

## 三、系统架构

### 3.1 分层架构

```
┌─────────────────────────────────────────────────────┐
│                   Agent Layer                        │
│  (Persona, Behavior, Response Generation)           │
├─────────────────────────────────────────────────────┤
│                   Memory Layer                       │
│  ┌───────────────┐  ┌────────────────┐              │
│  │ ShortTerm     │  │ Episodic       │              │
│  │ Memory (STM)  │  │ Memory (LTM)   │              │
│  │               │  │                │              │
│  │ 海马体模拟    │  │ 新皮层模拟     │              │
│  └───────────────┘  └────────────────┘              │
├─────────────────────────────────────────────────────┤
│                   Neuron Layer                       │
│  ┌─────────────────────────────────────────────┐    │
│  │  NeuronCell: 神经元单元                      │    │
│  │  - strength (Elo)                           │    │
│  │  - connections (双向)                       │    │
│  │  - decay_rate (动态)                        │    │
│  └─────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────┤
│                   Storage Layer                      │
│  (PostgreSQL, EventStream, Embedding Index)         │
└─────────────────────────────────────────────────────┘
```

### 3.2 核心组件

#### NeuronCell（神经元）

```python
class NeuronCell:
    event_id: UUID
    event_type: Literal['chat', 'thought', 'reflection', 'experience', 'perception']
    strength: float              # Elo 评分
    decay_rate: float            # 动态衰减率（根据冲击力）
    outgoing_connections: Set    # 出连接
    incoming_connections: Set    # 入连接
    created_at: DateTime
    last_activated: DateTime
```

#### Engram（记忆痕迹）

```python
class Engram:
    neurons: Dict[str, List[NeuronCell]]  # 按类型组织
    represent: NeuronCell                  # 代表神经元
    strength: float                        # 整体强度
    summary: str                           # LLM 摘要
    impact_score: float                    # 冲击力评分
    activation_threshold: float            # 激活阈值
```

#### Memory System（记忆系统）

```python
class ShortTermMemory:
    """海马体模拟 - 短期记忆"""
    sequences: Dict[str, Engram]  # 按 audience 组织
    max_capacity: int
    
    def consolidate(self) -> List[Engram]:
        """固化到长期记忆"""
        
class EpisodicMemory:
    """新皮层模拟 - 长期记忆"""
    registry: Dict                 # 元数据索引
    engram_managers: Dict          # 按 audience 管理
```

### 3.3 核心机制

#### Elo 竞争机制 (elo.py)

```python
class EloCompetitor:
    """Elo 竞争系统"""
    
    # K-factor 动态调整
    HIGH_ACTIVITY_K = 16    # 高频激活神经元
    DEFAULT_K_FACTOR = 32   # 默认
    LOW_ACTIVITY_K = 64     # 低频激活神经元
    
    def get_combat_power(self, event_id):
        """战斗力 = sqrt(Elo) * log(activation_count + 1)"""
        ...
    
    def update_after_retrieval(self, retrieved_ids, all_candidate_ids):
        """检索后更新所有候选神经元的 Elo"""
        ...
```

**核心公式**:
- 战斗力: `power = sqrt(elo) × log(activations + 1)`
- 期望胜率: `E = 1 / (1 + 10^((R_opponent - R_player) / 400))`

#### 动态衰减系统 (decay.py)

```python
# 基础衰减率（每天）
BASE_DECAY_RATES = {
    'chat': 0.995,           # 对话
    'perception': 0.990,     # 感知
    'thought': 0.992,        # 思考
    'reflection': 0.998,     # 反思
    'experience': 0.985,     # 体验
}

def calculate_decay_rate(event_type, impact_score):
    """adjusted_rate = base_rate + impact_score × (1 - base_rate)"""
    base = BASE_DECAY_RATES[event_type]
    return base + impact_score * (1 - base)

def apply_decay(strength, decay_rate, time_days):
    """strength = strength × decay_rate^time_days"""
    return strength * (decay_rate ** time_days)
```

#### 统一检索系统 (unified_retriever.py)

```python
class UnifiedRetriever:
    """
    综合评分公式:
    score = weighted(similarity, elo_strength, decay_factor, recency)
    """
    
    def retrieve(self, neurons, query_embedding, ...):
        """执行统一检索"""
        ...
```

#### 记忆稳定性系统 (stability.py)

```python
class StabilityManager:
    """集体稳定性机制"""
    
    def calculate_engram_stability(self, engram):
        """
        稳定性 = aggregate(member_contributions)
        - 代表神经元有 1.5x 加成
        - 支持多种聚合方法
        """
        ...
    
    def check_activation_threshold(self, engram, threshold=0.3):
        """检查是否达到激活阈值"""
        ...
```

#### Elo 竞争机制（伪代码）

```python
def elo_competition(query_embedding, candidate_neurons):
    """
    Elo 竞争：神经元竞争激活信号
    
    1. 计算相似度（信号强度）
    2. 相似度 × Elo = 战斗力
    3. 按战斗力排名
    4. 更新 Elo（胜者+，败者-）
    """
    for neuron in candidate_neurons:
        similarity = cosine(query_embedding, neuron.embedding)
        combat_power = similarity * neuron.strength
    
    ranked = sort_by(combat_power, descending=True)
    
    # Elo 更新
    for winner, loser in pairs(ranked):
        winner.strength += k_factor * (1 - expected_win_rate)
        loser.strength -= k_factor * expected_win_rate
```

#### 动态衰减系统（伪代码）

```python
# 记忆类型 → 基础衰减率
DECAY_RATES = {
    'chat': 0.995,           # 对话，中等衰减
    'thought': 0.998,        # 思考，较慢衰减
    'reflection': 0.999,     # 反思，慢衰减
    'experience': 0.990,     # 经历，快衰减
    'perception': 0.980,     # 感知，最快衰减
}

def calculate_decay_rate(neuron, impact_score):
    """
    根据冲击力调整衰减率
    
    冲击力高（差点被撞）→ 衰减极慢（0.9999）
    冲击力低（每天遛狗）→ 衰减快（0.99）
    """
    base_rate = DECAY_RATES[neuron.event_type]
    
    # 冲击力调整：冲击力越高，衰减越慢
    impact_factor = 1 - impact_score * 0.01  # 0.99 ~ 1.0
    
    return base_rate * impact_factor
```

#### DMN 模式（空闲整合）

```python
class DefaultModeNetwork:
    """
    模拟 DMN：空闲时的记忆整合
    
    触发条件：
    - 无外部输入超过 N 分钟
    - 定期"睡眠"调度
    """
    
    def integrate_memories(self):
        """
        空闲时整合记忆：
        1. Reflection：自动反思近期记忆
        2. Consolidation：STM → LTM 固化
        3. Association：发现记忆间的隐含关联
        4. Cleanup：清理低强度神经元
        """
        self.auto_reflection()
        self.consolidate_to_ltm()
        self.discover_associations()
        self.cleanup_weak_neurons()
```

---

## 四、数据流

### 4.1 记忆形成流程

```
用户输入 → EventStream → 创建 NeuronCell
                            ↓
                      计算 embedding
                            ↓
                      加入 ShortTermMemory
                            ↓
                      建立 neuron 连接
                            ↓
                      更新 Elo 评分
```

### 4.2 记忆检索流程

```
Query → Embedding → 候选神经元
                         ↓
                   Elo 竞争排序
                         ↓
                   激活阈值过滤
                         ↓
                   扩散激活（沿连接）
                         ↓
                   组装 Engram
                         ↓
                   更新 strength
```

### 4.3 记忆固化流程（睡眠/DMN）

```
触发 DMN 模式
    ↓
扫描 ShortTermMemory
    ↓
识别高价值记忆（strength > threshold）
    ↓
生成 Engram summary（LLM）
    ↓
迁移到 EpisodicMemory
    ↓
清理 STM
```

---

## 五、与原设计对比

| 维度 | 原设计 | 新设计 |
|------|--------|--------|
| **Elo 机制** | 只有 strength 字段 | 完整竞争机制 |
| **衰减** | 一刀切 0.995 | 动态衰减率 |
| **记忆稳定性** | 无 | 集体稳定性 + 激活阈值 |
| **因果推断** | 无 | 激活序列学习 |
| **DMN 模式** | 无 | 空闲整合机制 |
| **神经元数量** | 固定 | 支持动态增删 |
| **睡眠固化** | 无 | STM → LTM 转化 |

---

## 六、技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 神经元存储 | PostgreSQL | 持久化神经元数据 |
| 连接图 | NetworkX | 神经元连接网络 |
| 向量索引 | pgvector | Embedding 检索 |
| LLM | OpenAI/Anthropic | Summary 生成 |
| 调度 | Cron/APScheduler | DMN 模式触发 |

---


---

## 七、学习规则：Hebbian 可塑性

### 7.1 为什么选择 Hebbian Learning

**传统深度学习 vs Hebbian**：

| 维度 | 深度学习 | Hebbian |
|------|---------|---------|
| 训练方式 | 反向传播、梯度下降 | 局部规则、前向传播 |
| 数据需求 | 大量标注数据 | 无需标注（无监督） |
| 在线学习 | 批量更新 | 实时更新 |
| 记忆形成 | 权重冻结后固定 | 持续可塑 |
| 生物学合理性 | 低 | 高（模拟突触可塑性） |

**Hebbian 的优势**：
1. **One-shot learning**：不需要反复训练
2. **在线更新**：每次交互都在学习
3. **联想记忆**：自动形成关联网络
4. **无监督**：不需要标签

### 7.2 Hebbian 规则在 Engram 中的应用

**连接强度更新**：
```python
def hebbian_update(neuron_a, neuron_b, learning_rate=0.1):
    """
    Hebbian 规则：一起激活的神经元，连接加强
    
    Δw = η × A_a × A_b
    """
    if neuron_a.active and neuron_b.active:
        delta = learning_rate * neuron_a.activation * neuron_b.activation
        connection = neuron_a.get_connection_to(neuron_b)
        connection.strength = min(connection.strength + delta, MAX_STRENGTH)
```

**与 Elo 结合**：
- Hebbian 决定连接是否形成/加强
- Elo 决定检索时的竞争力
- 两者独立但互补

### 7.3 H-Mem 网络（参考）

来自论文《Hebbian Memory Networks》：
- 存储模式：key × value → Hebbian 更新关联矩阵
- 召回模式：query key → 检索关联矩阵 → 返回 value
- 可实现 one-shot 联想记忆

---

## 八、自主进化框架

### 8.1 借鉴 Karpathy autoresearch

**核心循环**：
```
修改代码 → 运行测试 → 评估指标 → 保留/丢弃 → 重复
```

**应用于 Engram**：
```
1. 读取 SPRINT.md（当前目标）
2. 读取 pitfalls.md（已知陷阱）
3. 修改目标文件（elo.py, decay.py 等）
4. 运行 benchmark（LongMemEval 子集）
5. 记录 metrics（Accuracy, Recall@k）
6. 如果改进 → git commit + push
7. 如果退化 → 回滚 + 更新 pitfalls
8. 重复直到 Sprint 完成
```

### 8.2 实现要点

| 要素 | autoresearch | Engram 项目 |
|------|-------------|-------------|
| 修改目标 | train.py | memory/elo.py, memory/decay.py |
| 指令文件 | program.md | SPRINT.md, pitfalls.md |
| 时间预算 | 5 分钟/实验 | 100 轮对话/实验 |
| 评估指标 | val_bpb | QA Accuracy, Recall@k |
| 版本控制 | git | git（已配置） |

---

