# Engram 记忆系统 API 文档

**版本**: 1.0.0  
**更新时间**: 2026-04-14

---

## 一、MemorySystem 公开 API

### 1.1 初始化

```python
from memory import MemorySystem, MemorySystemConfig

# 默认配置
system = MemorySystem()

# 自定义配置
config = MemorySystemConfig(
    auto_decay=True,
    auto_consolidation=True,
    auto_dynamics=True,
    enable_scene=True,
    enable_index=True
)
system = MemorySystem(config)
```

### 1.2 核心方法

#### add_memory() - 添加记忆

```python
def add_memory(
    self,
    content: str,
    event_type: str = 'chat',
    emotion: Optional[EmotionalImpact] = None,
    scene: Optional[SceneContext] = None,
    actor: str = 'system',
    audience: List[str] = None,
    metadata: Dict = None
) -> NeuronCell
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `content` | str | 必填 | 记忆内容 |
| `event_type` | str | 'chat' | 事件类型 |
| `emotion` | EmotionalImpact | None | 情绪影响 |
| `scene` | SceneContext | None | 场景上下文 |
| `actor` | str | 'system' | 行动者 |
| `audience` | List[str] | None | 受众 |
| `metadata` | Dict | None | 额外元数据 |

**返回**：创建的 NeuronCell

**示例**：

```python
from memory import EmotionalImpact

emotion = EmotionalImpact(valence=0.8, arousal=0.9, dominance=0.7)
neuron = system.add_memory(
    content="完成了一个重要项目",
    event_type="experience",
    emotion=emotion,
    actor="user",
    audience=["assistant"]
)
```

#### retrieve() - 检索记忆

```python
def retrieve(
    self,
    query: str,
    scene: Optional[SceneContext] = None,
    top_k: int = 10,
    event_types: List[str] = None
) -> List[NeuronCell]
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | 必填 | 查询文本 |
| `scene` | SceneContext | None | 场景上下文过滤 |
| `top_k` | int | 10 | 返回数量 |
| `event_types` | List[str] | None | 事件类型过滤 |

**返回**：匹配的 NeuronCell 列表

**示例**：

```python
# 基础检索
results = system.retrieve("项目", top_k=5)

# 带场景过滤
scene = SceneContext(location="办公室", time="morning")
results = system.retrieve("工作", scene=scene)

# 按类型过滤
results = system.retrieve("喜欢", event_types=["chat", "experience"])
```

#### run_dmn_consolidation() - 运行 DMN 巩固

```python
def run_dmn_consolidation(
    self,
    threshold_strength: float = 0.5,
    prune_below: float = 0.1
) -> DMNResult
```

**返回**：DMNResult 对象

```python
class DMNResult(BaseModel):
    success: bool
    consolidations: int = 0      # 巩固数量
    prunings: int = 0            # 修剪数量
    new_associations: int = 0   # 新关联数量
    messages: List[str] = []     # 日志消息
```

**示例**：

```python
result = system.run_dmn_consolidation(threshold_strength=0.6)
print(f"巩固: {result.consolidations}, 修剪: {result.prunings}")
```

---

## 二、核心类 API

### 2.1 NeuronCell

```python
from memory import NeuronCell
```

#### 构造函数

```python
neuron = NeuronCell(
    event_id=uuid.uuid1(),
    event_type="chat",
    create_time=datetime.now(),
    actor="user",
    audience=["assistant"],
    strength=1.0,
    decay_rate=0.995,
    impact_score=0.5,
    activation_threshold=0.3,
    is_consolidated=False
)
```

#### 方法

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `connect_to(other)` | None | 创建到另一个神经元的连接 |
| `disconnect_from(other)` | None | 断开连接 |
| `apply_decay(reference_time)` | float | 应用衰减，返回新强度 |
| `boost_strength(factor)` | float | 增强强度 |
| `set_emotion(valence, arousal, dominance)` | float | 设置情绪 |
| `is_retrievable(cue_strength)` | bool | 检查是否可检索 |

**示例**：

```python
# 创建连接
neuron1 = NeuronCell(event_type="chat", create_time=datetime.now(), actor="user")
neuron2 = NeuronCell(event_type="thought", create_time=datetime.now(), actor="assistant")
neuron1.connect_to(neuron2)

# 应用衰减
new_strength = neuron1.apply_decay(datetime.now())

# 设置情绪
impact = neuron1.set_emotion(valence=0.7, arousal=0.8, dominance=0.6)

# 检查可检索性
if neuron1.is_retrievable(cue_strength=0.5):
    print("可以检索")
```

### 2.2 Engram

```python
from memory import Engram, Events
```

#### 构造函数

```python
engram = Engram(
    actor=["user"],
    audience=["assistant"],
    time=datetime.now(),
    summary="这是一段对话记忆",
    scope="partial"
)
```

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `add_event(events)` | Events/List | None | 添加事件 |
| `add_neurons(neurons)` | NeuronCell/List | None | 添加神经元 |
| `remove_neuron(neuron)` | UUID1/NeuronCell | None | 移除神经元 |
| `get_neuron_by_id(id)` | UUID1 | NeuronCell | 按 ID 获取 |
| `get_all_neurons()` | - | Set[NeuronCell] | 获取所有神经元 |
| `recall(timestamp, create_copy)` | datetime, bool | Engram | 时间回溯 |
| `retain_relevant_neurons(max_turns)` | int | None | 保留相关神经元 |
| `to_json(filename)` | str | None | 保存到 JSON |
| `from_json(filename)` | str | Engram | 从 JSON 加载 |

**示例**：

```python
from memory import ChatEvent

# 创建并添加事件
engram = Engram(actor=["user"], audience=["assistant"], time=datetime.now())

event = ChatEvent(
    actor="user",
    content="我喜欢科幻电影",
    create_time=datetime.now()
)
engram.add_event(event)

# 保存和加载
engram.to_json("my_memory.json")
loaded = Engram.from_json("my_memory.json")
```

### 2.3 EventStream

```python
from memory import EventStream, ChatEvent, ThoughtEvent
```

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `add_event(events)` | Events/List | None | 添加事件 |
| `filter(*conditions)` | Callable | EventStream | 条件过滤 |
| `get_event(id)` | UUID1 | Events | 按 ID 获取 |
| `clear()` | - | bool | 清空 |
| `__len__()` | - | int | 事件数量 |

**示例**：

```python
stream = EventStream()

# 添加事件
stream.add_event([
    ChatEvent(actor="user", content="Hello", create_time=datetime.now()),
    ThoughtEvent(actor="assistant", content="Thinking...", create_time=datetime.now())
])

# 过滤
filtered = stream.filter(
    lambda e: e.actor == "user"
)

print(f"事件数: {len(stream)}")
```

---

## 三、机制模块 API

### 3.1 Elo Competition

```python
from memory import EloCompetition, EloConfig, EloCompetitor
```

#### 配置

```python
config = EloConfig(
    initial_elo=1000.0,      # 初始 Elo
    min_elo=100.0,           # 最小值
    max_elo=2000.0,          # 最大值
    default_k_factor=32     # 默认 K 值
)
elo = EloCompetition(config)
```

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `register_neuron(neuron_id)` | UUID1 | None | 注册神经元 |
| `get_elo(neuron_id)` | UUID1 | float | 获取 Elo |
| `update_after_retrieval(winners, losers)` | List, List | None | 更新 Elo |
| `expected_win_probability(p, o)` | float, float | float | 计算期望胜率 |

**示例**：

```python
# 注册
elo.register_neuron("neuron-1")
elo.register_neuron("neuron-2")

# 更新
elo.update_after_retrieval(
    winners=["neuron-1"],
    losers=["neuron-2"]
)

# 获取
current_elo = elo.get_elo("neuron-1")
```

### 3.2 Decay Scheduler

```python
from memory import DecayScheduler, DecayConfig, calculate_decay_rate
```

#### 基础衰减率

| 事件类型 | 衰减率 |
|---------|--------|
| chat | 0.995 |
| perception | 0.990 |
| thought | 0.992 |
| reflection | 0.998 |
| experience | 0.985 |

#### 函数

```python
# 计算衰减率
rate = calculate_decay_rate(
    event_type="chat",
    impact_score=0.8
)
# rate ≈ 0.997

# 全局调度器
scheduler = get_global_scheduler()
scheduler.apply_decay(neurons, reference_time=datetime.now())
```

### 3.3 Stability Manager

```python
from memory import StabilityManager, calculate_engram_stability
```

#### 函数

```python
# 计算 Engram 稳定性
stability = calculate_engram_stability(
    neurons=[neuron1, neuron2, neuron3],
    method='arithmetic'  # arithmetic, harmonic, geometric, max, min
)

# 检查激活阈值
is_active = check_activation_threshold(
    neuron=strength_value,
    threshold=0.3
)
```

### 3.4 DMN Mode

```python
from memory import DMNMode, DMNConfig
```

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `consolidate(engram)` | Engram | bool | 巩固记忆 |
| `prune_connections(engram, threshold)` | Engram, float | int | 修剪弱连接 |
| `associate(engram1, engram2)` | Engram, Engram | bool | 建立关联 |

**示例**：

```python
dmn = DMNMode()

# 巩固
dmn.consolidate(engram)

# 修剪
pruned = dmn.prune_connections(engram, threshold=0.1)
```

---

## 四、高级模块 API

### 4.1 Emotional Impact

```python
from memory import EmotionalImpact, ImpactMapper
```

#### 构造函数

```python
emotion = EmotionalImpact(
    valence=0.7,      # [-1, 1] 正面 vs 负面
    arousal=0.8,     # [0, 1] 平静 vs 激动
    dominance=0.6    # [0, 1] 被动 vs 主动
)
```

#### 方法

```python
# 转换为冲击力
impact = emotion.to_impact_score()

# 转换为衰减率
decay = emotion.to_decay_rate()

# 使用映射器
mapper = ImpactMapper()
decay = mapper.map_emotion_to_decay(emotion, base_decay=0.995)
```

### 4.2 Scene Awareness

```python
from memory import SceneContext, SceneAwareRetrieval
```

#### 构造函数

```python
scene = SceneContext(
    location="电影院",
    time="evening",
    activity="约会"
)
```

#### 方法

```python
retrieval = SceneAwareRetrieval()

# 映射神经元到场景
retrieval.map_neuron_to_scene("neuron-id", scene)

# 获取场景中的神经元
neurons = retrieval.get_neurons_for_scene(scene)

# 获取场景历史
history = retrieval.get_scene_history(scene)
```

### 4.3 Neuron Dynamics

```python
from memory import NeuronDynamics, BirthReason, DeathReason
```

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `birth_neuron()` | - | NeuronCell | 创建新神经元 |
| `kill_neuron()` | - | NeuronDeath | 标记死亡 |
| `get_neurons_to_strengthen()` | threshold | List | 获取需巩固的 |
| `get_neurons_to_forget()` | - | List | 获取会遗忘的 |
| `get_birth_candidates()` | - | List | 获取候选 |

**示例**：

```python
dynamics = NeuronDynamics()

# 创建新神经元
new_neuron = dynamics.birth_neuron(
    reason=BirthReason.USER_REVELATION
)

# 获取需遗忘的
dying = dynamics.get_neurons_to_forget()
for death in dying:
    print(f"遗忘: {death.neuron_id}, 原因: {death.reason}")
```

---

## 五、数据模型

### 5.1 配置类

```python
# Elo 配置
class EloConfig(BaseModel):
    initial_elo: float = 1000.0
    min_elo: float = 100.0
    max_elo: float = 2000.0
    default_k_factor: int = 32

# Decay 配置
class DecayConfig(BaseModel):
    decay_rate_range: Tuple[float, float] = (0.98, 0.9999)
    base_decay_rates: Dict[str, float] = {...}
    impact_decay_factor: float = 0.5

# Memory System 配置
class MemorySystemConfig(BaseModel):
    elo_config: EloConfig
    decay_config: DecayConfig
    stability_config: StabilityConfig
    auto_decay: bool = True
    auto_consolidation: bool = True
```

### 5.2 状态类

```python
class NeuronEloState(BaseModel):
    neuron_id: str
    elo: float
    activation_count: int
    win_count: int
    last_activation: datetime

class NeuronDecayState(BaseModel):
    neuron_id: str
    strength: float
    decay_rate: float
    last_decay: datetime
```

---

## 六、使用示例汇总

### 完整示例

```python
from memory import (
    MemorySystem,
    EmotionalImpact,
    SceneContext,
    NeuronCell
)

# 初始化
system = MemorySystem()

# 1. 添加记忆
emotion = EmotionalImpact(valence=0.8, arousal=0.9, dominance=0.7)
scene = SceneContext(location="办公室", time="morning")

neuron = system.add_memory(
    content="完成季度报告",
    event_type="experience",
    emotion=emotion,
    scene=scene,
    actor="user",
    audience=["assistant"]
)

# 2. 检索
results = system.retrieve("报告", top_k=5)

# 3. DMN 巩固
result = system.run_dmn_consolidation(threshold_strength=0.5)

# 4. 检查动态
dying = system.dynamics.get_neurons_to_forget()
strengthening = system.dynamics.get_neurons_to_strengthen()

print(f"找到 {len(results)} 条记忆")
print(f"巩固 {result.consolidations} 条")
print(f"遗忘 {len(dying)} 条")
```
