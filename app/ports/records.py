"""产物记录端口（app/ports/records.py）

RunRecordStore：一次运行会话（session）的候选、反思轨迹、训练导出
和指标落盘。存储布局见 doc/03-physical-model.md §6：

    runs/<benchmark>/<database>/<session_id>/
    ├── manifest.json
    ├── knowledge/*.jsonl
    ├── candidates.initial.jsonl
    ├── reflection.jsonl
    ├── candidates.final.jsonl
    └── training.accepted.jsonl

约束：不覆盖中间轮次；training.accepted 只包含 accepted。
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RunRecordStore(Protocol):
    """运行产物存储协议。"""

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        """写入/更新 manifest.json（配置、模型、方言、知识快照、代码版本）。"""
        ...

    def write_knowledge_layer(self, layer: str, records: list[dict]) -> None:
        """写 K 层记录到 knowledge/<layer>.jsonl（发布快照时调用）。"""
        ...

    def append_candidates(self, candidates: list[dict], stage: str = "initial") -> None:
        """追加候选记录（stage: initial / final）。"""
        ...

    def append_reflection(self, traces: list[dict]) -> None:
        """追加反思轨迹（每个候选一条完整轨迹）。"""
        ...

    def append_training(self, records: list[dict]) -> None:
        """追加训练样本（仅 accepted；调用方负责过滤，实现再校验）。"""
        ...

    def count_training(self) -> int:
        """已落盘训练样本数（断点续跑判断用）。"""
        ...

    def session_dir(self) -> str:
        """本会话产物目录（调试/审计用）。"""
        ...
