# Iteration 13 Plan

## 待完成任务（按依赖顺序）

1. **T1-embedding-timeout**: Embedding 加载超时优化
   - 目标：给 HTTP 请求加 5 秒 timeout，加载失败时优雅降级（返回 None，不阻塞）
   - 依赖：无
   - 验收：`timeout 5 python3 -c "from memory.system import MemorySystem; ms=MemorySystem(); ms.add_memory('test')"` 不卡住

2. **T2-reflection-attribute**: Reflection 暴露为 MemorySystem 属性
   - 目标：添加 `ms.reflection: ReflectionTrigger` 属性，暴露 `ms.trigger_reflection()` 便捷方法
   - 依赖：无
   - 验收：`python3 -c "from memory.system import MemorySystem; ms=MemorySystem(); print('has reflection:', hasattr(ms, 'reflection'))"` 输出 True

3. **T3-postgres-storage**: PostgreSQL 持久化层
   - 目标：创建 `memory/storage/postgres.py` 模块，实现 `PostgresStorage` 类，支持神经元、Engram、Embedding 的 CRUD，提供 JSON → PostgreSQL 迁移脚本
   - 依赖：T1（T3 依赖 T1，因为存储层需要 embedding 不阻塞才能正常初始化）
   - 验收：`python3 -m pytest tests/test_postgres_storage.py -v --tb=short` 通过

4. **T4-tests-green**: 全部测试通过
   - 目标：新增 `tests/test_postgres_storage.py`，确保全部测试无 regression
   - 依赖：T1、T2、T3
   - 验收：`python3 -m pytest tests/ -v --tb=short` 显示全部 passed

## 相关陷阱（从 pitfalls.md 筛选）

- **[实现/懒加载]** EmbeddingManager 懒加载不能放在 `__init__`：在 `__init__` 中加载 `sentence-transformers` 模型会阻塞整个 MemorySystem 实例化。正确做法：改为在首次 `embed()` 调用时才加载模型（lazy loading）。T1 和 T3 均受影响。
- **[实现/存储]** Embedding 不能直接存 JSON：JSON 文件冗余大、读取慢。正确做法：使用 PostgreSQL 或专门的向量数据库。直接影响 T3。
- **[架构/Reflection]** Reflection 不是定期总结：机械式 reflection 会产生噪音。正确做法：在特定条件下触发（记忆冲突、强度异常、新知识关联）。T2 应避免定时触发的实现方式。
- **[测试/验收命令]** 验收命令中 positional argument 顺序必须与函数签名匹配：SPRINT.md 验收命令中 `ms.add_memory('测试记忆', 'test-user')` 把第二个参数当作 event_type，但实际签名是 `(content, event_type, emotion, scene, actor, ...)`。正确做法：使用 keyword argument 明确指定：`ms.add_memory('测试记忆', actor='test-user')`。T1/T2/T3/T4 的验收命令均需注意。
- **[实现/NeuronCell]** NeuronCell 没有 `id` 属性，只有 `event_id`：代码中有地方用 `neuron.id` 访问 NeuronCell，但该类只有 `event_id` 属性。正确做法：添加 `id` property 别名指向 `event_id`，或统一使用 `event_id`。涉及 T2/T3。

## 验收命令（从 SPRINT.md 原样复制）

```bash
cd /tmp/companion-agent-test

# T1: Embedding 超时测试（应在 5s 内完成）
timeout 5 python3 -c "
from memory.system import MemorySystem
ms = MemorySystem()
# 模拟 embedding 加载失败场景
from memory.embedding import EmbeddingManager
original = EmbeddingManager._try_load_model
EmbeddingManager._try_load_model = lambda self: setattr(self, '_available', False) or None
n = ms.add_memory('test', event_type='chat', actor='user')
print('Embedding timeout OK, neuron created:', n.event_id)
EmbeddingManager._try_load_model = original
"

# T2: Reflection 属性检查
python3 -c "
from memory.system import MemorySystem
ms = MemorySystem()
print('has reflection:', hasattr(ms, 'reflection'))
if hasattr(ms, 'reflection'):
    print('reflection type:', type(ms.reflection).__name__)
"

# T3: PostgreSQL 存储测试
python3 -m pytest tests/test_postgres_storage.py -v --tb=short 2>&1 | tail -10

# T4: 全部测试
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -5
```
