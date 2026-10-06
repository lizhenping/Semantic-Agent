# knowledge/snapshots/ — 知识快照根目录（不入库）

本目录存放知识快照（KnowledgeSnapshot 的持久化发布），**内容不提交到 git**，
仅此 README 占位。快照可由 `python cli.py analyze` 从数据集重新构建。

## 布局

```
knowledge/snapshots/<benchmark>/<database>/<snapshot_id>/
```

- 默认根目录 `knowledge/snapshots/`，可用 `SEMANTICSQL_KNOWLEDGE_ROOT` 覆盖
- 会话内同时保留一份自包含 K1-K6 拷贝（`runs/<...>/<session>/knowledge/`）
- 复用已有快照跳过构建阶段：`generate --reuse-knowledge`
