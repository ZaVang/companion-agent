# Engram 记忆系统使用指南

**版本**: 1.0.0  
**更新时间**: 2026-04-14

---

## 一、安装步骤

### 1.1 环境要求

- Python 3.10+
- pydantic
- numpy

### 1.2 安装

```bash
# 克隆项目
git clone <repository-url>
cd companion-agent

# 安装依赖
pip install -r requirements.txt
```

### 1.3 验证安装

```bash
python -c "
from memory import MemorySystem
system = MemorySystem()
print('Engram 安装成功！')
"
```

---

## 二、基础用法

### 2.1 初始化系统

```python
from memory import MemorySystem

# 使用默认配置
system = MemorySystem()

print(f"Elo 系统: {system.elo}")
print(f"衰减调度: {system.decay}")
print(f"稳定性管理: {system.stability}")
```

### 2.2 添加记忆

#### 基础记忆

```python
# 添加对话记忆
neuron = system.add_memory(
    content="用户说今天天气很好",
    event_type="chat",
    actor="user"
)
print(f"创建神经元: {neuron.event_id}")

# 添加思考记忆
neuron = system.add_memory(
    content="助手在思考如何回应",
    event_type="thought",
    actor="assistant"
)

# 添加经验记忆
neuron = system.add_memory(
    content="我们讨论了旅行计划",
    event_type="experience",
    actor="user"
)
```

#### 带情绪的记忆

```python
from memory import EmotionalImpact

# 正面高唤醒（应该慢衰减）
excited = EmotionalImpact(valence=0.8, arousal=0.9, dominance=0.7)
neuron = system.add_memory(
    content="完成了重要项目演示",
    event_type="experience",
    emotion=excited
)

# 负面低唤醒（会快衰减）
neutral = EmotionalImpact(valence=-0.3, arousal=0.2, dominance=0.4)
neuron = system.add_memory(
    content="日常通勤",
    event_type="experience",
    emotion=neutral
)
```

#### 带场景的记忆

```python
from memory import SceneContext

# 定义场景
scene = SceneContext(
    location="咖啡馆",
    time="afternoon",
    activity="工作"
)

# 添加带场景的记忆
neuron = system.add_memory(
    content="讨论了新功能设计",
    event_type="chat",
    scene=scene,
    actor="user"
)
```

### 2.3 检索记忆

#### 基础检索

```python
# 检索最相关的 5 条记忆
results = system.retrieve("天气", top_k=5)

for neuron in results:
    print(f"类型: {neuron.event_type}, 强度: {neuron.strength:.2f}")
```

#### 带场景过滤

```python
# 只在特定场景检索
scene = SceneContext(location="咖啡馆")
results = system.retrieve("工作", scene=scene)

# 按时间检索
from datetime import datetime, timedelta
scene = SceneContext(time="afternoon")
results = system.retrieve("功能", scene=scene)
```

#### 按类型过滤

```python
# 只检索对话和经验
results = system.retrieve(
    "计划",
    event_types=["chat", "experience"]
)
```

### 2.4 访问神经元属性

```python
neuron = system.add_memory(
    content="测试记忆",
    event_type="chat"
)

# 基本属性
print(f"ID: {neuron.event_id}")
print(f"类型: {neuron.event_type}")
print(f"创建时间: {neuron.create_time}")
print(f"强度: {neuron.strength}")
print(f"衰减率: {neuron.decay_rate}")
print(f"冲击力: {neuron.impact_score}")

# 连接
print(f"出向连接: {len(neuron.outgoing_connections)}")
print(f"入向连接: {len(neuron.incoming_connections)}")

# 情绪
if neuron.emotional_valence != 0:
    print(f"情绪效价: {neuron.emotional_valence}")
    print(f"唤醒度: {neuron.emotional_arousal}")
    print(f"主导性: {neuron.emotional_dominance}")
```

---

## 三、高级用法

### 3.1 使用 Engram 管理记忆组

```python
from memory import Engram, ChatEvent
from datetime import datetime

# 创建 Engram
engram = Engram(
    actor=["user"],
    audience=["assistant"],
    time=datetime.now(),
    summary="一次关于旅行的对话"
)

# 添加多个事件
engram.add_event([
    ChatEvent(
        actor="user",
        content="我想去日本",
        create_time=datetime.now()
    ),
    ChatEvent(
        actor="assistant", 
        content="日本是很棒的选择！",
        create_time=datetime.now()
    )
])

# 保存
engram.to_json("japan_trip.json")

# 加载
loaded = Engram.from_json("japan_trip.json")
```

### 3.2 使用 Elo 竞争系统

```python
from memory import EloCompetition, EloConfig

# 自定义配置
config = EloConfig(
    initial_elo=1000.0,
    min_elo=100.0,
    max_elo=2000.0,
    default_k_factor=32
)

elo = EloCompetition(config)

# 注册神经元
elo.register_neuron("neuron-1")
elo.register_neuron("neuron-2")

# 模拟竞争
elo.update_after_retrieval(
    winners=["neuron-1"],
    losers=["neuron-2"]
)

# 查看评分
print(f"Neuron-1 Elo: {elo.get_elo('neuron-1')}")
print(f"Neuron-2 Elo: {elo.get_elo('neuron-2')}")
```

### 3.3 手动应用衰减

```python
from memory import get_global_scheduler
from datetime import datetime, timedelta

# 获取全局调度器
scheduler = get_global_scheduler()

# 模拟时间流逝
future_time = datetime.now() + timedelta(days=7)

# 应用衰减
neurons = list(system._neurons.values())
scheduler.apply_decay(neurons, reference_time=future_time)

# 查看衰减结果
for neuron in neurons[:3]:
    print(f"{neuron.event_id}: {neuron.strength:.4f}")
```

### 3.4 使用 DMN 巩固

```python
# 运行 DMN 巩固
result = system.run_dmn_consolidation(
    threshold_strength=0.5,
    prune_below=0.1
)

print(f"巩固成功: {result.success}")
print(f"巩固数量: {result.consolidations}")
print(f"修剪数量: {result.prunings}")
print(f"新关联: {result.new_associations}")

for msg in result.messages:
    print(f"  - {msg}")
```

### 3.5 使用动态管理

```python
# 获取需要遗忘的神经元
dying = system.dynamics.get_neurons_to_forget()
print(f"即将遗忘 {len(dying)} 个神经元")

for death in dying:
    print(f"  {death.neuron_id}: {death.reason}")

# 获取需要巩固的神经元
weak = system.dynamics.get_neurons_to_strengthen(threshold=0.3)
print(f"需要巩固 {len(weak)} 个神经元")

# 创建新神经元
new_neuron = system.dynamics.birth_neuron(
    content="新兴趣点",
    reason="USER_REVELATION"
)
```

### 3.6 使用情绪工具

```python
from memory import EmotionalImpact, ImpactMapper

# 创建情绪
emotion = EmotionalImpact(
    valence=0.7,
    arousal=0.8,
    dominance=0.6
)

# 转换为冲击力
impact = emotion.to_impact_score()
print(f"冲击力: {impact:.2f}")

# 转换为衰减率
decay = emotion.to_decay_rate()
print(f"衰减率: {decay:.4f}")

# 使用映射器
mapper = ImpactMapper()
decay = mapper.map_emotion_to_decay(emotion, base_decay=0.995)
```

---

## 四、最佳实践

### 4.1 记忆添加策略

```python
# ✅ 推荐：使用具体的事件类型
system.add_memory("用户说喜欢科幻电影", event_type="chat")
system.add_memory("一起看了星际穿越", event_type="experience")

# ❌ 避免：所有记忆都用 chat
system.add_memory("看了电影", event_type="chat")  # 应该用 experience
```

### 4.2 情绪标注策略

```python
# ✅ 推荐：高情绪价值记忆标注
excited = EmotionalImpact(valence=0.9, arousal=0.9, dominance=0.8)
system.add_memory("求婚成功！", emotion=excited)

# ✅ 推荐：日常低价值记忆不标注或低唤醒
neutral = EmotionalImpact(valence=0.0, arousal=0.2, dominance=0.5)
system.add_memory("常规会议", emotion=neutral)
```

### 4.3 检索策略

```python
# ✅ 推荐：使用场景上下文
scene = SceneContext(location="电影院")
results = system.retrieve("电影", scene=scene)

# ✅ 推荐：结合多种过滤
results = system.retrieve(
    "重要",
    event_types=["chat", "experience"],
    top_k=10
)

# ❌ 避免：大量检索不清理
# 每次检索后系统状态会改变
```

### 4.4 性能优化

```python
# ✅ 推荐：批量操作
system.batch.batch_decay(neurons, reference_time)

# ✅ 推荐：使用索引
system.index.search_by_strength(threshold=0.5)

# ✅ 推荐：定期运行 DMN 巩固
system.run_dmn_consolidation()
```

### 4.5 调试建议

```python
# ✅ 使用可视化
from memory import NetworkVisualizer

viz = NetworkVisualizer()
html = viz.visualize_engram(engram)

# ✅ 使用追踪
from memory import MemoryTracer

tracer = system.tracer
history = tracer.history.get_history(neuron_id)
```

---

## 五、常见问题

### Q1: 如何理解"记忆是塑形"？

**A**: 在 Engram 中，记忆不是静态存储的。每次激活都会改变神经元的状态：
- 强度可能增加（Elo 更新）
- 连接可能强化
- 衰减会影响强度

这模拟了真实神经网络的动态重塑特性。

### Q2: 遗忘是如何工作的？

**A**: 遗忘不是删除神经元，而是通过两种机制实现：
1. **强度衰减**：`strength *= decay_rate^days`
2. **入口减少**：连接衰减或断裂

当一个记忆变得"不可达"（低于激活阈值），就相当于被遗忘了。

### Q3: 如何防止重要记忆被遗忘？

**A**: 有几种策略：
1. 定期激活重要记忆（检索会增强）
2. 设置高情绪值（慢衰减）
3. 使用 DMN 巩固强化
4. 设置高初始强度

```python
# 定期复习
results = system.retrieve("重要内容")
```

### Q4: 为什么需要场景感知？

**A**: 场景感知模拟了人类记忆的"情境依赖性"：
- 在咖啡馆聊过的话题更容易在咖啡馆回忆
- 工作场景检索优先返回工作相关记忆
- 时间上下文帮助时间排序

### Q5: 如何调试记忆系统？

```python
# 1. 查看神经元状态
neuron = system._neurons["some-id"]
print(neuron.strength, neuron.decay_rate)

# 2. 查看历史记录
history = system.tracer.history.get_history(neuron_id)
for entry in history:
    print(f"{entry.timestamp}: {entry.change_type}")

# 3. 生成可视化
viz = NetworkVisualizer()
html = viz.visualize_engram(engram)
```

### Q6: 性能问题如何解决？

```python
# 1. 使用索引
system.index.search_by_time(start, end)

# 2. 批量操作
system.batch.batch_decay(large_neuron_list)

# 3. 延迟加载
engram = engram.from_json("memory.json", scope="partial")

# 4. 定期清理
dying = system.dynamics.get_neurons_to_forget()
for death in dying:
    system.dynamics.kill_neuron(death.neuron_id)
```

---

## 六、代码示例汇总

### 示例 1: 完整的用户交互流程

```python
from memory import MemorySystem, EmotionalImpact, SceneContext

# 初始化
system = MemorySystem()

# 用户说"我今天加薪了！"
emotion = EmotionalImpact(valence=0.9, arousal=0.8, dominance=0.7)
system.add_memory(
    content="用户说：我今天加薪了！",
    event_type="chat",
    actor="user",
    emotion=emotion
)

# 助手思考
system.add_memory(
    content="助手思考：应该祝贺用户",
    event_type="thought",
    actor="assistant"
)

# 用户回应
system.add_memory(
    content="用户说：谢谢！",
    event_type="chat",
    actor="user"
)

# 添加为经验
system.add_memory(
    content="用户今天获得了加薪",
    event_type="experience",
    actor="user",
    emotion=emotion
)

# 稍后检索
results = system.retrieve("好事")
print(f"找到 {len(results)} 条相关记忆")
```

### 示例 2: DMN 巩固流程

```python
# 添加足够多的记忆
for i in range(10):
    system.add_memory(
        content=f"记忆 {i}",
        event_type="chat" if i % 2 == 0 else "experience"
    )

# 运行巩固
result = system.run_dmn_consolidation(
    threshold_strength=0.5,
    prune_below=0.1
)

# 检查结果
if result.success:
    print(f"巩固了 {result.consolidations} 条记忆")
    print(f"修剪了 {result.prunings} 条弱连接")
```

### 示例 3: 时间回溯

```python
from datetime import datetime, timedelta

# 假设现在是 7 天后
now = datetime.now()
past = now - timedelta(days=7)

# 添加新记忆
system.add_memory("现在的记忆", event_type="chat")

# 回溯到过去
engram_copy = system._engrams["some-id"].recall(past, create_copy=True)
```

---

## 七、配置参考

### 默认配置

```python
MemorySystemConfig(
    # Elo 配置
    elo_config=EloConfig(
        initial_elo=1000.0,
        min_elo=100.0,
        max_elo=2000.0,
        default_k_factor=32
    ),
    
    # Decay 配置
    decay_config=DecayConfig(
        decay_rate_range=(0.98, 0.9999),
        base_decay_rates={
            "chat": 0.995,
            "perception": 0.990,
            "thought": 0.992,
            "reflection": 0.998,
            "experience": 0.985
        }
    ),
    
    # 行为配置
    auto_decay=True,
    auto_consolidation=True,
    auto_dynamics=True,
    
    # 功能开关
    enable_scene=True,
    enable_index=True
)
```
