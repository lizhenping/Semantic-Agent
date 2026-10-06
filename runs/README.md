# runs/ — 运行产物根目录（不入库）

本目录存放合成管线的会话产物，**内容不提交到 git**，仅此 README 占位。

## 会话布局

```
runs/<benchmark>/<database>/<session-id>/
├── manifest.json              # 会话清单（配置、快照 ID、时间）
├── knowledge/                 # 自包含的 K1-K6 知识层
│   ├── k1_schema.jsonl
│   ├── k2_domain.jsonl
│   ├── k3_field.jsonl
│   ├── k4_column.jsonl
│   ├── k5_table.jsonl
│   └── k6_relations.jsonl
├── candidates.initial.jsonl   # 生成阶段候选（q/r/s + 执行反馈与结果行）
├── candidates.final.jsonl     # 准入终态汇总（decision + 矫正轮次统计）
├── reflection.jsonl           # Eq.4 诊断→矫正迭代明细
└── training.accepted.jsonl    # 训练准入记录（question/think/answer/
                               #   correction_history/execution 含结果行采样）
```

- 根目录默认 `runs/`，可用 `SEMANTICSQL_OUTPUT_ROOT` 覆盖
- `run-benchmark` 支持断点续跑：已有足够样本的库自动跳过
