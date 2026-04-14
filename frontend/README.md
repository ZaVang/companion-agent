# Engram Memory Visualizer

Engram 记忆系统的 React 前端可视化界面。

## 功能特性

- 🧠 **记忆网络可视化** - D3.js 力导向图展示神经元和连接关系
- ⚡ **实时动态** - Elo 竞争、衰减过程动画
- 🔍 **记忆检索** - 高亮激活神经元，显示检索路径
- 🧠 **DMN 整合** - 触发默认模式网络整合过程
- 📊 **统计面板** - 网络统计、Elo 排名、衰减曲线

## 技术栈

- React 18 + TypeScript
- Vite 构建工具
- D3.js 网络可视化
- TailwindCSS 样式

## 快速开始

### 1. 安装依赖

```bash
cd frontend
npm install
```

### 2. 启动后端服务

```bash
cd ..
python -m service.app
```

后端运行在 http://localhost:5000

### 3. 启动前端开发服务器

```bash
cd frontend
npm run dev
```

前端运行在 http://localhost:3000

### 4. 访问

打开浏览器访问 http://localhost:3000

## 项目结构

```
frontend/
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
├── index.html
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── components/
    │   ├── MemoryNetwork.tsx      # 记忆网络可视化
    │   ├── NeuronNode.tsx         # 神经元节点详情
    │   ├── EngramCard.tsx         # Engram 卡片
    │   ├── Timeline.tsx           # 时间线
    │   ├── DecayChart.tsx         # 衰减曲线
    │   ├── EloRanking.tsx         # Elo 排名
    │   └── ControlPanel.tsx       # 控制面板
    ├── hooks/
    │   └── useMemorySystem.ts     # API 调用 hook
    ├── types/
    │   └── memory.ts              # TypeScript 类型
    └── styles/
        └── index.css
```

## API 接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/memory/network` | GET | 获取网络结构 |
| `/api/memory/stats` | GET | 获取统计信息 |
| `/api/memory/memories` | POST | 添加记忆 |
| `/api/memory/retrieve` | POST | 检索记忆 |
| `/api/memory/dmn` | POST | 触发 DMN |
| `/api/memory/elo/compete` | POST | 触发 Elo 竞争 |
| `/api/memory/config` | PUT | 更新配置 |

## 核心概念

### 神经元类型

| 类型 | 颜色 | 描述 |
|------|------|------|
| chat | 🔵 蓝色 | 对话记忆 |
| thought | 🟣 紫色 | 思考过程 |
| reflection | 🟢 绿色 | 反思总结 |
| perception | 🟡 黄色 | 感知记忆 |
| experience | 🔴 红色 | 经验记忆 |

### 节点大小

节点大小表示记忆强度 (strength)，强度越高节点越大。

### 节点颜色

- 正常状态：按类型着色
- 激活状态：发光效果
- 稳固记忆：边框标记

## License

MIT
