"""SQL 解析端口（app/ports/parsing.py）

确定性校验依赖 SQL AST。解析失败必须显式暴露（SYNTAX_ERROR），
不允许静默返回空错误列表（doc/03-physical-model.md §9 风险 5）。
"""

from typing import Protocol, runtime_checkable

from app.contracts.parsing import ParsedSql


class SqlParseError(ValueError):
    """SQL 无法解析（含方言不支持的结构）。调用方必须转为
    SYNTAX_ERROR 诊断，而不是当作"没有错误"。"""

    def __init__(self, sql: str, detail: str):
        super().__init__(f"SQL 解析失败: {detail}")
        self.sql = sql
        self.detail = detail


@runtime_checkable
class SqlParser(Protocol):
    """SQL 解析协议（sqlglot 适配器 / 测试 Fake 实现）。"""

    def parse(self, sql: str) -> ParsedSql:
        """解析 SQL；失败抛 SqlParseError（绝不静默成功）。"""
        ...
