"""数据库契约（app/contracts/database.py）

对应概念模型 §3.1/§3.2 与多数据库模型 §4：
- DatabaseSource   可访问的物理数据库及其连接配置（凭据绝不进入知识/Prompt/日志）
- DatabaseSnapshot 一次分析绑定的数据库事实快照（schema 摘要、采样策略、内容哈希）
- DatabaseContext  一次生成任务绑定的运行期上下文（连接句柄只在运行期存在）

一个数据库一个 Context；跨库联邦不在第一阶段范围内。
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DatabaseDialect(str, Enum):
    """SQL 方言。执行与解析必须携带方言，不能用 SQLite 假定覆盖 MySQL。"""

    SQLITE = "sqlite"
    MYSQL = "mysql"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DatabaseSource(BaseModel):
    """连接目标 + 方言 + 身份。生命周期由应用装配，供一次或多次任务使用。"""

    database_id: str                       # 稳定身份（如 spider/aan_1）
    dialect: DatabaseDialect = DatabaseDialect.SQLITE
    benchmark: str = ""                    # 所属 benchmark（用于分层存储）
    # --- sqlite 连接参数 ---
    sqlite_path: Optional[str] = None
    # --- mysql 连接参数（凭据不得进入知识库、Prompt、样本或普通日志）---
    host: str = ""
    port: int = 0
    database: str = ""                     # MySQL 中的 schema 名
    username: str = ""
    password: str = ""

    @property
    def display_name(self) -> str:
        """可安全打印的身份标识（不含凭据、不含完整路径）。"""
        if self.dialect == DatabaseDialect.SQLITE:
            name = self.sqlite_path or ""
            return self.database_id or name.replace("\\", "/").rsplit("/", 1)[-1]
        return f"{self.host}:{self.port}/{self.database or self.database_id}"

    def redacted_params(self) -> dict:
        """日志/manifest 可用的连接参数（凭据已抹除）。"""
        return {
            "database_id": self.database_id,
            "dialect": self.dialect.value,
            "benchmark": self.benchmark,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "password": "***",
        }


class DatabaseSnapshot(BaseModel):
    """数据库事实快照：反思可复现性的边界。

    相同数据库名称不等于相同数据库状态；反思过程不得在中途
    无记录地切换数据库内容。
    """

    database_id: str
    dialect: DatabaseDialect = DatabaseDialect.SQLITE
    schema_digest: str = ""                # schema 结构摘要哈希
    content_hash: str = ""                 # 采样内容哈希
    sampling_policy: str = ""              # 采样策略描述
    table_count: int = 0
    row_count_total: int = 0
    captured_at: str = Field(default_factory=_utc_now_iso)


@dataclass
class DatabaseContext:
    """一次生成任务绑定的数据库上下文（概念模型 §4）。

    connection/engine 是运行期对象（由 DatabaseGateway 注入），
    不序列化、不进入任何产物；跨任务不可复用为可变对象。
    """

    source: DatabaseSource
    snapshot: DatabaseSnapshot
    engine: Any = None                     # SQLAlchemy engine（运行期）
    knowledge_snapshot_id: str = ""        # 知识构建完成后回填

    @property
    def database_id(self) -> str:
        return self.source.database_id

    @property
    def dialect(self) -> DatabaseDialect:
        return self.source.dialect
