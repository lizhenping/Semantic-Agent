"""SQLAlchemy 数据库适配器（app/adapters/database_sqlalchemy.py）

合并旧 infra/database.py（DatabaseManager/SchemaExtractor/SQLExecutor）
与 tools/analysis/schema_extraction.py 的 K1 抽取逻辑。

行为保持 + 风险修复（doc/03-physical-model.md §9）：
1. 删除重复的 SQLExecutor 类定义（风险 1）。
2. execute_readonly 不追加 LIMIT，把"查询全部结果"的语义与展示/采样保护分开
   （风险 4）：limit 仅用于探针的便捷限流；答案落盘执行走
   limit=None + row_cap/keep_rows，SQL 文本不被改写，客户端按
   row_cap 流式取回（内存占用与结果规模无关，命中即 rows_capped=True）。
3. 每个 DatabaseSource 独立 engine，绑定在自己的 DatabaseContext 上，
   不再全局共享（风险 3 的多库隔离基础）。
4. SQLite 执行带确定性超时闸门（SEMANTICSQL_SQL_TIMEOUT，默认 15s）。
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from app.contracts.database import (
    DatabaseContext,
    DatabaseDialect,
    DatabaseSnapshot,
    DatabaseSource,
)
from app.contracts.execution import DEFAULT_RESULT_ROWS_LIMIT
from app.contracts.knowledge import (
    ColumnInfo,
    SchemaMetadata,
    TableInfo,
)
from app.ports.database import (
    DatabaseConnectionError,
    SQLExecutionError,
    SchemaExtractionError,
)

logger = logging.getLogger(__name__)


def classify_sql_error(error_msg: str) -> str:
    """把 DB 错误字符串映射到 ErrorType 值（供执行反馈分类）。"""
    from app.contracts.diagnosis import ErrorType

    msg = error_msg.lower()
    if "unknown column" in msg or ("column" in msg and "not found" in msg):
        return ErrorType.COLUMN_NOT_FOUND.value
    if "table" in msg and ("doesn't exist" in msg or "not found" in msg):
        return ErrorType.TABLE_NOT_FOUND.value
    if "syntax error" in msg or "sql syntax" in msg:
        return ErrorType.SYNTAX_ERROR.value
    if "data type" in msg or "type mismatch" in msg:
        return ErrorType.TYPE_ERROR.value
    if "ambiguous" in msg:
        return ErrorType.COLUMN_NOT_FOUND.value  # ambiguous column 归为列问题
    if "join" in msg:
        return ErrorType.JOIN_INVALID.value
    return ErrorType.EXECUTION_FAILED.value


class SQLAlchemyDatabaseGateway:
    """DatabaseGateway 的 SQLAlchemy 实现（MySQL + SQLite）。

    一个 gateway 实例可服务多个 DatabaseSource，但每个 source 的
    engine 缓存在各自 DatabaseContext 上，互不可见。
    """

    def __init__(
        self,
        sql_timeout: Optional[float] = None,
        sample_values_per_column: int = 5,
    ):
        if sql_timeout is None:
            try:
                sql_timeout = float(os.getenv("SEMANTICSQL_SQL_TIMEOUT", "15"))
            except ValueError:
                sql_timeout = 15.0
        self.sql_timeout = sql_timeout
        self.sample_values_per_column = sample_values_per_column
        self._engines: dict[str, Any] = {}

    # ---------- 生命周期 ----------

    def open(self, source: DatabaseSource) -> DatabaseContext:
        if source.dialect == DatabaseDialect.SQLITE:
            if not source.sqlite_path or not os.path.exists(source.sqlite_path):
                raise DatabaseConnectionError(
                    source.database_id,
                    f"sqlite 文件不存在: {source.sqlite_path}",
                )
            url = f"sqlite:///{source.sqlite_path}"
        elif source.dialect == DatabaseDialect.MYSQL:
            url = (
                f"mysql+pymysql://{source.username}:{source.password}"
                f"@{source.host}:{source.port}/{source.database}?charset=utf8mb4"
            )
        else:
            raise DatabaseConnectionError(
                source.database_id, f"不支持的方言: {source.dialect}")

        try:
            engine = create_engine(url, pool_pre_ping=True, echo=False)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            raise DatabaseConnectionError(source.database_id, str(e)) from e

        # engine 缓存：同一 source 复用连接池
        cached = self._engines.get(source.database_id)
        if cached is not None:
            try:
                cached.dispose()
            except Exception:
                pass
        self._engines[source.database_id] = engine

        context = DatabaseContext(
            source=source,
            snapshot=DatabaseSnapshot(
                database_id=source.database_id, dialect=source.dialect),
            engine=engine,
        )
        context.snapshot = self.probe(context)
        logger.info("✅ 数据库连接成功: %s (%s)",
                    source.display_name, source.dialect.value)
        return context

    def close(self, context: DatabaseContext) -> None:
        engine = getattr(context, "engine", None)
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass
        self._engines.pop(context.database_id, None)
        logger.info("🔒 数据库连接已关闭: %s", context.database_id)

    # ---------- 快照 ----------

    def probe(self, context: DatabaseContext) -> DatabaseSnapshot:
        """采集数据库事实快照（结构摘要哈希 + 规模统计）。"""
        engine = self._require_engine(context)
        try:
            insp = inspect(engine)
            table_names = insp.get_table_names()
            structure: dict[str, list[dict]] = {}
            total_rows = 0
            for t in table_names:
                cols = [
                    {"name": c["name"], "type": str(c.get("type", "")),
                     "nullable": bool(c.get("nullable", True))}
                    for c in insp.get_columns(t)
                ]
                structure[t] = cols
                try:
                    with engine.connect() as conn:
                        n = conn.execute(
                            text(f'SELECT COUNT(*) FROM "{t}"')).scalar()
                        total_rows += int(n or 0)
                except Exception:
                    pass
            digest_raw = json.dumps(structure, ensure_ascii=False, sort_keys=True)
            return DatabaseSnapshot(
                database_id=context.source.database_id,
                dialect=context.source.dialect,
                schema_digest=hashlib.sha256(digest_raw.encode()).hexdigest()[:16],
                sampling_policy=f"distinct_values_top{self.sample_values_per_column}",
                table_count=len(table_names),
                row_count_total=total_rows,
            )
        except Exception as e:
            raise SchemaExtractionError(context.database_id, str(e)) from e

    # ---------- K1 schema 提取 ----------

    def read_schema(self, context: DatabaseContext) -> SchemaMetadata:
        engine = self._require_engine(context)
        if context.source.dialect == DatabaseDialect.SQLITE:
            schema = self._read_sqlite_schema(engine, context.database_id)
        else:
            schema = self._read_mysql_schema(engine, context.source)
        logger.info("📊 K1 schema 提取完成: %s（%d 表）",
                    context.database_id, len(schema.tables))
        return schema

    def _require_engine(self, context: DatabaseContext):
        engine = getattr(context, "engine", None)
        if engine is None:
            raise DatabaseConnectionError(context.database_id, "连接未初始化")
        return engine

    # ---------- SQLite ----------

    def _read_sqlite_schema(self, engine, database_name: str) -> SchemaMetadata:
        insp = inspect(engine)
        tables: list[TableInfo] = []
        for table_name in insp.get_table_names():
            tables.append(self._sqlite_table_info(engine, insp, table_name))
        return SchemaMetadata(database_name=database_name, tables=tables)

    def _sqlite_table_info(self, engine, insp, table_name: str) -> TableInfo:
        columns_raw = insp.get_columns(table_name)
        pk_info = insp.get_pk_constraint(table_name)
        fk_info = insp.get_foreign_keys(table_name)

        pk_names = pk_info.get("constrained_columns", []) if pk_info else []
        fk_columns: set[str] = set()
        fk_targets: dict[str, tuple] = {}
        for fk in fk_info:
            sources = fk.get("constrained_columns", []) or []
            targets = fk.get("referred_columns", []) or []
            target_table = fk.get("referred_table")
            for index, source in enumerate(sources):
                fk_columns.add(source)
                if target_table and index < len(targets):
                    fk_targets[source] = (target_table, targets[index])

        samples = self._sqlite_sample_values(engine, table_name)
        row_count = self._sqlite_row_count(engine, table_name)

        columns: list[ColumnInfo] = []
        for col_raw in columns_raw:
            col_name = col_raw["name"]
            columns.append(ColumnInfo(
                name=col_name,
                data_type=str(col_raw.get("type", "")),
                nullable=col_raw.get("nullable", True),
                primary_key=col_name in pk_names,
                foreign_key=col_name in fk_columns,
                foreign_key_table=fk_targets.get(col_name, (None, None))[0],
                foreign_key_column=fk_targets.get(col_name, (None, None))[1],
                default=str(col_raw["default"]) if col_raw.get("default") else None,
                comment=None,  # sqlite 通常无 comment
                sample_values=samples.get(col_name, []),
                entropy_level=self._entropy_level(
                    engine, table_name, col_name, row_count),
            ))

        return TableInfo(
            name=table_name,
            columns=columns,
            primary_keys=pk_names,
            row_count=row_count,
            comment=None,
        )

    def _entropy_level(self, engine, table_name: str, col_name: str,
                       row_count: Optional[int]) -> str:
        """COUNT(DISTINCT col)/row_count：<0.1 low；0.1~0.7 medium；>0.7 high。"""
        if not row_count:
            return "medium"
        try:
            with engine.connect() as conn:
                distinct = conn.execute(text(
                    f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table_name}" '
                    f'WHERE "{col_name}" IS NOT NULL')).fetchone()[0]
                if not distinct:
                    return "medium"
                ratio = distinct / row_count
                if ratio < 0.1:
                    return "low"
                if ratio > 0.7:
                    return "high"
                return "medium"
        except Exception:
            return "medium"

    def _sqlite_sample_values(self, engine, table_name: str) -> dict[str, list]:
        """采样每列 distinct 值（供合成数据接地用）。"""
        samples: dict[str, list] = {}
        limit = self.sample_values_per_column
        try:
            with engine.connect() as conn:
                cols = [row[1] for row in conn.execute(
                    text(f'PRAGMA table_info("{table_name}")'))]
                for col in cols:
                    try:
                        rows = conn.execute(text(
                            f'SELECT DISTINCT "{col}" FROM "{table_name}" '
                            f'WHERE "{col}" IS NOT NULL LIMIT {limit}')).fetchall()
                        values = [r[0] for r in rows]
                        if values:
                            samples[col] = [
                                str(v)[:40] if isinstance(v, str) else v
                                for v in values
                            ]
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("采样 %s 值失败: %s", table_name, e)
        return samples

    def _sqlite_row_count(self, engine, table_name: str) -> Optional[int]:
        try:
            with engine.connect() as conn:
                return int(conn.execute(
                    text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar())
        except Exception:
            return None

    # ---------- MySQL ----------

    def _read_mysql_schema(self, engine, source: DatabaseSource) -> SchemaMetadata:
        database = source.database
        tables: list[TableInfo] = []
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT TABLE_NAME, TABLE_COMMENT FROM INFORMATION_SCHEMA.TABLES "
                "WHERE TABLE_SCHEMA = :db ORDER BY TABLE_NAME"),
                {"db": database}).fetchall()
            for table_name, table_comment in rows:
                col_rows = conn.execute(text(
                    "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY, "
                    "COLUMN_COMMENT FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :tb "
                    "ORDER BY ORDINAL_POSITION"),
                    {"db": database, "tb": table_name}).fetchall()
                columns = [
                    ColumnInfo(
                        name=r[0],
                        data_type=r[1],
                        nullable=(r[2] or "YES") == "YES",
                        primary_key=(r[3] == "PRI"),
                        foreign_key=False,  # MySQL FK 检测沿用旧逻辑的保守策略
                        comment=r[4] or None,
                        sample_values=[],
                    )
                    for r in col_rows
                ]
                tables.append(TableInfo(
                    name=table_name,
                    columns=columns,
                    primary_keys=[c.name for c in columns if c.primary_key],
                    comment=table_comment or None,
                ))
        return SchemaMetadata(database_name=database, tables=tables)

    # ---------- 只读执行 ----------

    def execute_readonly(
        self, context: DatabaseContext, sql: str,
        limit: Optional[int] = None,
        row_cap: Optional[int] = None,
        keep_rows: Optional[int] = None,
    ) -> dict[str, Any]:
        """只读执行 SELECT（带 SQLite 超时闸门）。

        limit：>0 且 SQL 未写 LIMIT 时追加 LIMIT（对探针方便；答案落盘执行传 None）。
        row_cap：取回上限，逐行计数到该值即停止取回，返回 rows_capped=True。
        不修改 SQL 文本，只限制客户端取回量——SQL 语义保持完整。
        keep_rows：留存的行数上限（其余行只计数不留存，内存占用与结果规模无关）。
        返回受控摘要：{success, row_count, columns, data, rows_capped, sql}。
        """
        engine = self._require_engine(context)
        sql_clean = self._prepare_sql(sql, limit)
        is_sqlite = engine.dialect.name == "sqlite"
        try:
            with engine.connect() as conn:
                if is_sqlite and self.sql_timeout > 0:
                    raw_conn = conn.connection.dbapi_connection
                    deadline = time.monotonic() + self.sql_timeout

                    def _handler():
                        return 1 if time.monotonic() > deadline else 0

                    raw_conn.set_progress_handler(_handler, 20000)
                try:
                    result = conn.execute(text(sql_clean))
                    return self._process_result(result, sql_clean, row_cap,
                                                keep_rows)
                finally:
                    if is_sqlite:
                        try:
                            conn.connection.dbapi_connection.set_progress_handler(None, 0)
                        except Exception:  # noqa: BLE001
                            pass
        except OperationalError as e:
            msg = str(e)
            if "interrupted" in msg.lower():
                logger.warning("⏱️  SQL 超时（>%.0fs）已中断，判为不可执行",
                               self.sql_timeout)
                raise SQLExecutionError(
                    sql_clean, f"query timeout >{self.sql_timeout:.0f}s",
                    "QueryTimeout") from e
            logger.error("❌ SQL 执行失败: %s - %s", sql_clean, e)
            raise SQLExecutionError(sql_clean, msg, "SQLAlchemyError") from e
        except SQLAlchemyError as e:
            logger.error("❌ SQL 执行失败: %s - %s", sql_clean, e)
            raise SQLExecutionError(sql_clean, str(e), "SQLAlchemyError") from e
        except Exception as e:
            logger.error("❌ 查询执行异常: %s - %s", sql_clean, e)
            raise SQLExecutionError(sql_clean, str(e), type(e).__name__) from e

    @staticmethod
    def _prepare_sql(sql: str, limit: Optional[int]) -> str:
        """规范化语句；仅当 limit 为正整数且 SQL 未写 LIMIT 时追加。"""
        cleaned = sql.strip().rstrip(";")
        if limit and "LIMIT" not in cleaned.upper():
            cleaned = f"{cleaned} LIMIT {limit}"
        return cleaned

    @staticmethod
    def _process_result(result, sql_clean: str,
                        row_cap: Optional[int] = None,
                        keep_rows: Optional[int] = None) -> dict[str, Any]:
        """受控结果摘要：逐行流式取回，内存占用与结果规模无关。

        只留存前 keep_rows 行，计数到 row_cap 即停（rows_capped=True）。
        大结果集（数十万行）不会整体进内存。
        """
        if not result.returns_rows:
            return {"success": True, "row_count": 0, "columns": [], "data": [],
                    "rows_capped": False, "sql": sql_clean}
        columns = list(result.keys())
        keep = keep_rows if keep_rows is not None else DEFAULT_RESULT_ROWS_LIMIT
        data: list[dict] = []
        count = 0
        capped = False
        for row in result:
            count += 1
            if len(data) < keep:
                data.append({columns[i]: _normalize_value(row[i])
                             for i in range(len(columns))})
            if row_cap is not None and count >= row_cap:
                capped = True
                break
        return {"success": True, "row_count": count, "columns": columns,
                "data": data, "rows_capped": capped, "sql": sql_clean}


def _normalize_value(value) -> Any:
    """行值标准化（datetime/float/bytes），保证 JSON 可序列化。"""
    try:
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if hasattr(value, "__float__"):
            return float(value)
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except Exception:
                return str(value)
        return value
    except Exception:
        return str(value) if value is not None else None
