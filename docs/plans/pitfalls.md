# 陷阱知识库

记录 engram 项目开发过程中踩过的坑，供所有 subagent 参考。

---

## 架构陷阱

### [架构] 不要把 Elo 当成简单的计数器
- **错误做法**: 每次检索到就 +1
- **正确做法**: Elo 是竞争机制，需要计算战斗力、排名、更新
- **原因**: 简单累加会导致热门记忆无限增长，失去平衡

### [架构] 衰减参数不能一刀切
- **错误做法**: 所有记忆用同一个 decay_rate
- **正确做法**: 根据冲击力/情绪类型动态调整
- **原因**: "差点被撞"应该衰减极慢，"每天遛狗"刺激停止后快速衰减

### [架构] Reflection 不是定期总结
- **错误做法**: 每天/每小时自动触发 reflection
- **正确做法**: 在特定条件下触发（记忆冲突、强度异常、新知识关联）
- **原因**: 机械式 reflection 会产生噪音

### [架构] 稳定性不能只看单个神经元
- **错误做法**: 只看代表性神经元的 strength
- **正确做法**: 聚合所有成员神经元的贡献
- **原因**: 碎片化记忆中任何成员都可能丢失，需要集体稳定性

---

## 实现陷阱

### [实现] 神经元连接是双向维护的
- **问题**: 只更新 outgoing_connections 忘了 incoming_connections
- **正确**: connect_to() 和 disconnect_from() 必须同时维护两端

### [实现] Embedding 不能直接存 JSON
- **问题**: JSON 文件冗余大，读取慢
- **正确**: 使用 PostgreSQL 或专门的向量数据库

### [实现] 记忆检索不只是相似度
- **问题**: 只用 embedding 相似度排序
- **正确**: 相似度 × Elo strength × 时间衰减 × 场景权重
- **原因**: 纯相似度会导致"昨天聊过的话题"被"三年前类似话题"覆盖

---

## 概念陷阱

### [概念] 记忆不是存储，是塑形
- **错误思维**: 把记忆当数据库，存进去取出来
- **正确思维**: 记忆是神经网络的权重变化，每次激活都在重塑

### [概念] 遗忘不是删除，是不可达
- **错误思维**: 遗忘 = 删除神经元
- **正确思维**: 遗忘 = 入口变少，记忆还在但触发条件苛刻

### [概念] 检索是推理的一部分
- **错误思维**: 检索是独立的前置步骤
- **正确思维**: 检索应该主动、联想驱动，是思考过程的延伸

---

## 测试陷阱

### [测试] 不要用静态数据测试动态系统
- **问题**: 用固定输入测试记忆系统，期望固定输出
- **正确**: 测试记忆系统的"变化"——Elo 是否上升、衰减是否生效
- **原因**: 记忆系统本质是动态的，每次交互都在改变状态

### [测试] 测试冲击力衰减需要长时间跨度
- **问题**: 用短时间测试衰减机制
- **正确**: 模拟长时间跨度（天/周/月）验证衰减曲线
- **原因**: 冲击力记忆的衰减周期可能很长

---


---

## 方法论陷阱

### [方法论] 不要用传统训练思维
- **错误思维**: 需要大量数据训练神经元网络
- **正确思维**: Hebbian 学习是无监督、在线更新的
- **原因**: 每次交互本身就是"训练"，不需要额外的训练阶段

### [方法论] 不要用单一指标评估
- **错误做法**: 只看 QA Accuracy
- **正确做法**: Recall@k + NDCG@k + QA Accuracy + Efficiency
- **原因**: 不同 Sprint 关注不同能力，需要多维度评估

### [方法论] 不要跳过 benchmark 测试
- **错误做法**: 凭感觉认为"应该变好了"
- **正确做法**: 每次修改后运行 benchmark，记录具体数值
- **原因**: 直觉不可靠，数据才是证据

---

## Benchmark 相关

### [Benchmark] LongMemEval 的 5 大能力
1. **Information Extraction** - 从长对话中提取信息
2. **Multi-Session Reasoning** - 跨会话整合
3. **Temporal Reasoning** - 时间感知
4. **Knowledge Updates** - 知识更新
5. **Abstention** - 识别未知

### [Benchmark] MemoryAgentBench 的 4 大能力
1. **Accurate Retrieval** - 精准检索
2. **Test-Time Learning** - 运行时学习
3. **Long-Range Understanding** - 长程理解
4. **Conflict Resolution** - 冲突解决

### [Benchmark] 评测指标选择
| Sprint | 主要指标 | 原因 |
|--------|----------|------|
| Sprint 1 | Recall@k | 关注检索质量 |
| Sprint 2 | QA Accuracy | 关注问答质量 |
| Sprint 3 | Efficiency | 关注效率 |
| Sprint 4 | Multi-hop | 关注推理能力 |

---


---

## Sprint 11 新增陷阱（2026-04-14）

### [实现] EmbeddingManager 懒加载不能放在 __init__
- **问题**: 在 `__init__` 中加载 `sentence-transformers` 模型会阻塞整个 MemorySystem 实例化
- **正确做法**: 改为在首次 `embed()` 调用时才加载模型（lazy loading）
- **原因**: 模型文件可能不存在或网络不可用，同步阻塞影响可用性

### [实现] NeuronCell 没有 `id` 属性，只有 `event_id`
- **问题**: 代码中有地方用 `neuron.id` 访问 NeuronCell，但该类只有 `event_id` 属性
- **正确做法**: 添加 `id` property 别名指向 `event_id`，或统一使用 `event_id`
- **原因**: 接口一致性，避免 AttributeError

### [测试] 验收命令中 positional argument 顺序必须与函数签名匹配
- **问题**: SPRINT.md 验收命令 `ms.add_memory('测试记忆', 'test-user')` 把第二个参数当作 event_type，但实际签名是 `(content, event_type, emotion, scene, actor, ...)`
- **正确做法**: 使用 keyword argument 明确指定：`ms.add_memory('测试记忆', actor='test-user')`
- **原因**: keyword argument 更健壮，函数签名变化时不易出错
