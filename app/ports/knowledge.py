"""知识存储端口（app/ports/knowledge.py）

- KnowledgeStore：K 层 JSONL 记录的持久化协议（load/save/append/exists）。
- KnowledgeQuery：知识快照上的只读查询协议（生成与验证共用）。

反思不能写回共享知识：写接口只供知识构建阶段使用。
"""

from typing import Protocol, Union, runtime_checkable


@runtime_checkable
class KnowledgeStore(Protocol):
    """K 层记录存储协议（JSONL 适配器与内存测试实现均满足）。"""

    def load(self, layer: str) -> list[dict]:
        """加载某层知识（返回 JSON 记录列表；不存在返回 []）。"""
        ...

    def save(self, layer: str, data: Union[list[dict], dict]) -> None:
        """保存某层知识（覆盖写；仅构建阶段调用）。"""
        ...

    def append(self, layer: str, record: dict) -> None:
        """追加一条记录（审计语义：不覆盖历史）。"""
        ...

    def exists(self, layer: str) -> bool:
        """某层知识是否存在且非空。"""
        ...

    def list_layers(self) -> list[str]:
        """列出所有非空层。"""
        ...
