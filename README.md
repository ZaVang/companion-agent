# Engram 记忆系统

**版本**: 1.1.0  
**更新时间**: 2026-04-15

Engram 是一个受神经科学启发的 AI 伴侣记忆系统，模拟人类记忆的形成、巩固、检索和遗忘机制。

---

## 核心概念

### 神经科学基础

| 概念 | 说明 |
|------|------|
| **神经元 (NeuronCell)** | 记忆的基本单元，类似生物神经元的激活-抑制机制 |
| **记忆痕迹 (Engram)** | 一组神经元的集合，代表完整的记忆片段 |
| **Hebbian 学习** | "一起激活的神经元，连接在一起" |
| **动态衰减** | 低价值记忆逐渐遗忘，高冲击记忆长期保留 |
| **Elo 竞争** | 神经元通过竞争获取激活资源 |

### 五大记忆模块

```
┌─────────────────────────────────────────────────────────────┐
│                      Memory System                          │
├─────────────────────────────────────────────────────────────┤
│  ShortTermMemory  │  EpisodicMemory  │  WorkingMemory      │
│  (海马体)         │  (新皮层)         │  (前额叶)            │
├─────────────────────────────────────────────────────────────┤
│                     NeuronCell Layer                         │
│  Elo Competition  │  Decay Scheduler  │  Stability Manager  │
├─────────────────────────────────────────────────────────────┤
│                    Supporting Modules                        │
│  DMN Consolidation  │  Causal Inference  │  Scene Awareness  │
│  Resonance Engine   │  Emotional Impact  │  Dynamics         │
└─────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 环境要求

| 项目 | 要求 |
|------|------|
| Python | >= 3.10, < 3.14 (tested on 3.13) |
| OS | Linux / macOS / Windows |
| 内存 | >= 4GB (推荐 8GB+) |

### 安装

**方式一：完整安装（包含ML模型）**
```bash
pip install -r requirements.txt
```

**方式二：核心安装（无ML模型，更轻量）**
```bash
pip install numpy scipy pydantic fastapi uvicorn pytest
```

> 注意：text2vec、jieba、gensim 为可选依赖，需要时单独安装：
> ```bash
> pip install torch text2vec jieba gensim
> ```

### Python 3.13 兼容性

本项目已适配 Python 3.13，主要变更：
- `text2vec` 和 `jieba` 采用延迟导入（lazy import）
- 模型加载失败时系统会优雅降级，不会崩溃
- 如遇到 `pkgutil.ImpImporter` 错误，请升级 jieba：
  ```bash
  pip install --upgrade jieba
  ```

### 基本使用

```python
from memory import MemorySystem, EmotionalImpact

# 初始化系统
system = MemorySystem()

# 添加带情绪的记忆
emotion = EmotionalImpact(valence=0.8, arousal=0.9, dominance=0.7)
neuron = system.add_memory(
    content="今天完成了重要的项目演示",
    event_type="experience",
    emotion=emotion
)

# 检索记忆
results = system.retrieve("项目演示", top_k=5)
for neuron in results:
    print(f"找到记忆: {neuron.event_type}")
```

---

## 模块架构图

```mermaid
graph TB
    subgraph "Agent Layer"
        Persona[Persona]
        Brain[Brain]
    end

    subgraph "Memory System"
        STM[ShortTermMemory]
        EM[EpisodicMemory]
    end

    subgraph "Neuron Layer"
        Neuron[NeuronCell]
        Engram[Engram]
        Connection[Connection]
    end

    subgraph "Core Mechanisms"
        Elo[Elo Competition]
        Decay[Decay Scheduler]
        Stability[Stability Manager]
    end

    subgraph "Advanced Modules"
        DMN[DMN Consolidation]
        Causal[Causal Inference]
        Scene[Scene Awareness]
        Resonance[Resonance Engine]
        Emotion[Emotional Impact]
        Dynamics[Dynamics Manager]
    end

    subgraph "Optimization"
        Index[Memory Index]
        Batch[Batch Processor]
        Viz[Visualization]
    end

    Persona --> Brain
    Brain --> STM
    STM --> EM
    EM --> Neuron
    Neuron --> Connection
    Neuron --> Elo
    Neuron --> Decay
    Neuron --> Stability
    Elo --> DMN
    Decay --> DMN
    Stability --> DMN
    DMN --> Causal
    Causal --> Resonance
    Emotion --> Decay
    Dynamics --> Neuron
    Index --> Neuron
    Batch --> Elo
    Viz --> Neuron
```

---

## 核心类说明

### NeuronCell（神经元）

```python
class NeuronCell:
    event_id: UUID1              # 唯一标识
    event_type: str               # chat/perception/thought/reflection/experience
    strength: float               # 强度值 (Elo 评分)
    decay_rate: float             # 衰减率
    impact_score: float           # 冲击力评分 [0, 1]
    activation_threshold: float   # 激活阈值
    is_consolidated: bool         # 是否为稳固记忆
    
    # 情绪字段 (Sprint 7)
    emotional_valence: float      # 效价 [-1, 1]
    emotional_arousal: float      # 唤醒度 [0, 1]
    emotional_dominance: float   # 主导性 [0, 1]
    
    # 连接
    outgoing_connections: Set     # 出向连接
    incoming_connections: Set     # 入向连接
```

### Engram（记忆痕迹）

```python
class Engram:
    engram: Dict[str, List[NeuronCell]]  # 按类型组织的神经元
    represent: UUID1                      # 代表神经元 ID
    strength: float                       # 整体强度
    summary: str                          # LLM 生成的摘要
    scope: Literal["full", "partial"]     # 加载范围
```

### MemorySystem（统一入口）

```python
class MemorySystem:
    elo: EloCompetition                  # Elo 竞争系统
    decay: DecayScheduler               # 衰减调度器
    stability: StabilityManager          # 稳定性管理器
    dmn: DMNMode                        # DMN 巩固模式
    causal: CausalInference             # 因果推理
    scene: SceneAwareRetrieval          # 场景感知
    resonance: ResonanceEngine          # 共振引擎
    dynamics: NeuronDynamics            # 动态管理器
    index: MemoryIndex                  # 记忆索引
```

---

## 使用示例

### 1. 添加记忆

```python
from memory import MemorySystem, EmotionalImpact, SceneContext

system = MemorySystem()

# 基础记忆
neuron = system.add_memory(
    content="用户说他喜欢科幻电影",
    event_type="chat",
    actor="user",
    audience=["assistant"]
)

# 带情绪的记忆
emotion = EmotionalImpact(valence=0.7, arousal=0.6, dominance=0.5)
neuron = system.add_memory(
    content="我们一起看了星际穿越",
    event_type="experience",
    emotion=emotion
)

# 带场景的记忆
scene = SceneContext(location="电影院", time="evening")
neuron = system.add_memory(
    content="约会看电影",
    event_type="experience",
    scene=scene
)
```

### 2. 检索记忆

```python
# 基础检索
results = system.retrieve("科幻", top_k=5)

# 带场景过滤
scene = SceneContext(location="电影院")
results = system.retrieve("电影", scene=scene)

# 按类型过滤
results = system.retrieve("喜欢", event_types=["chat", "experience"])
```

### 3. 触发 DMN 巩固

```python
result = system.run_dmn_consolidation()
print(f"巩固了 {result.consolidations} 个记忆")
print(f"修剪了 {result.prunings} 个弱连接")
```

### 4. 动态管理

```python
# 获取需要巩固的神经元
weak_neurons = system.dynamics.get_neurons_to_strengthen(threshold=0.3)

# 获取需要遗忘的神经元
dying_neurons = system.dynamics.get_neurons_to_forget()

# 执行出生（创建新神经元）
new_neuron = system.dynamics.birth_neuron(
    content="新的兴趣点",
    reason=BirthReason.NEW_INTEREST
)
```

---

## 项目结构

```
companion-agent/
├── agent/                    # Agent 核心
│   ├── agent.py
│   ├── brain.py
│   └── persona.py
├── memory/                   # 记忆系统核心
│   ├── neuron.py            # 神经元单元
│   ├── engram.py            # 记忆痕迹
│   ├── event.py            # 事件流
│   ├── memory.py           # 短/长期记忆
│   ├── elo/                # Elo 竞争机制
│   ├── decay/              # 动态衰减
│   ├── stability/          # 稳定性管理
│   ├── dmn/                # DMN 巩固
│   ├── causal/             # 因果推理
│   ├── scene/              # 场景感知
│   ├── resonance/          # 共振引擎
│   ├── emotion/            # 情绪处理
│   ├── dynamics/           # 动态增删
│   ├── optimization/       # 性能优化
│   └── viz/                # 可视化
├── docs/                    # 文档
│   ├── ARCHITECTURE.md     # 架构设计
│   ├── API.md              # API 文档
│   ├── DATA_MODEL.md       # 数据模型
│   ├── USAGE.md            # 使用指南
│   └── SPRINTS.md          # Sprint 记录
├── tests/                   # 测试
└── main.py                 # 入口
```

---

## 文档导航

- **[架构设计](docs/ARCHITECTURE.md)** - 系统架构、模块职责、设计决策
- **[API 文档](docs/API.md)** - 所有公开接口、参数说明、示例代码
- **[数据模型](docs/DATA_MODEL.md)** - 核心数据结构、字段说明
- **[使用指南](docs/USAGE.md)** - 安装、基础/高级用法、最佳实践
- **[Sprint 记录](docs/SPRINTS.md)** - 每个 Sprint 的目标、决策、限制

---

## 依赖说明

### 核心依赖（必须安装）

| 包名 | 版本 | 用途 |
|------|------|------|
| numpy | >=1.24.0 | 数值计算 |
| scipy | >=1.10.0 | 科学计算 |
| pydantic | >=2.0.0 | 数据验证 |
| fastapi | >=0.100.0 | API 框架 |
| pytest | >=7.0.0 | 测试框架 |

### 可选依赖（按需安装）

| 包名 | 版本 | 用途 | 备注 |
|------|------|------|------|
| torch | >=2.0.0 | 深度学习 | text2vec 需要 |
| text2vec | latest | 中文文本嵌入 | 首次加载需下载模型 |
| jieba | latest | 中文分词 | Word2Vec 模式需要 |
| gensim | latest | Word2Vec | 可选的轻量嵌入方案 |
| pymilvus | >=2.3.0 | 向量数据库 | 大规模存储需要 |

### 常见问题

**Q: ImportError: No module named 'torch'**

A: torch 是可选依赖。如果不需要 embedding 功能，可以忽略此警告。如需使用：
```bash
pip install torch
```

**Q: AttributeError: module 'pkgutil' has no attribute 'ImpImporter'**

A: 这是 Python 3.12+ 和旧版 jieba 的兼容性问题：
```bash
pip install --upgrade jieba
```

**Q: embedding 功能不可用**

A: 检查以下内容：
1. 确认 torch 已安装
2. 确认模型文件存在于 `models/text2vec-base-chinese-paraphrase/`
3. 查看 `_available` 属性：`manager._available`

---

## 许可证

MIT License
