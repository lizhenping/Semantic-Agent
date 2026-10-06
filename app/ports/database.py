"""数据库端口（app/ports/database.py）

DatabaseGateway 统一 SQLite/MySQL 访问：开连接、读 schema、采样、
只读执行。连接凭据只在本端口与 adapter 之间流转。

doc/03-physical-model.md §5 接口草案：
    open / read_schema / sample / execute_readonly
"""

from typing import Any, Protocol, runtime_checkable

from app.contracts.database import (
    DatabaseContext,
    DatabaseSnapshot,
    DatabaseSource,
)
from app.contracts.knowledge import SchemaMetadata


@runtime_checkable
class DatabaseGateway(Protocol):
    """数据库访问网关协议。"""

    def open(self, source: DatabaseSource) -> DatabaseContext:
        """打开只读连接并建立 DatabaseContext（失败 -> DatabaseConnectionError）。"""
        ...

    def probe(self, context: DatabaseContext) -> DatabaseSnapshot:
        """采集数据库事实快照（表数、行数、结构摘要哈希）。"""
        ...

    def read_schema(self, context: DatabaseContext) -> SchemaMetadata:
        """提取 K1 schema 元数据（表/列/主外键/采样值/行数/熵）。"""
        ...

    def execute_readonly(
        self, context: DatabaseContext, sql: str, limit: int | None = None,
        row_cap: int | None = None, keep_rows: int | None = None,
    ) -> dict[str, Any]:
        """只读执行 SELECT。

        limit：>0 且 SQL 未写 LIMIT 时追加 LIMIT（仅供探针限流；答案落盘传 None）。
        row_cap / keep_rows：不改写 SQL，只限制客户端取回量——
        逐行计数到 row_cap 即停（rows_capped=True），仅留存前 keep_rows 行。
        失败抛 SQLExecutionError；成功返回受控结果摘要
        {success, row_count, columns, data, rows_capped}。
        """
        ...

    def close(self, context: DatabaseContext) -> None:
        """释放连接。"""
        ...


class DatabaseConnectionError(RuntimeError):
    """数据库连接失败（任务级失败，不得生成没有数据库身份的样本）"""

    def __init__(self, database_id: str, detail: str):
        super().__init__(f"数据库连接失败 [{database_id}]: {detail}")
        self.database_id = database_id
        self.detail = detail


class SQLExecutionError(RuntimeError):
    """SQL 执行失败（B 类数据库反馈，进入诊断而不是直接丢弃）"""

    def __init__(self, sql: str, detail: str, error_code: str = ""):
        super().__init__(f"SQL 执行失败: {detail}")
        self.sql = sql
        self.detail = detail
        self.error_code = error_code


class SchemaExtractionError(RuntimeError):
    """schema 提取失败"""

    def __init__(self, database_id: str, detail: str):
        super().__init__(f"schema 提取失败 [{database_id}]: {detail}")
        self.database_id = database_id
        self.detail = detail
