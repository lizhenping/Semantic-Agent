"""确定性校验（app/domain/verification.py）

论文 §III.E Diagnose 的程序化部分（B 类数据库反馈 + 结构检查）：
- check_columns:    s 引用的列是否存在（K1）
- check_joins:      JOIN 是否符合 K6 关系（保守规则防误报）
- check_aggregation: AVG/SUM 参数是否符合 K3 类型（Fig.1 反例）
- check_execution:  s 在数据库 I 上能否执行

与旧 core/knowledge_store.py 的 check_* 行为一致；风险修复：
解析失败显式产出 SYNTAX_ERROR，不再静默当作"无结构错误"。
空结果不算执行失败（语义正确的问题可以合法返回 0 行）。
"""

import logging
from typing import Optional

from app.contracts.database import DatabaseContext
from app.contracts.diagnosis import (
    Diagnosis,
    ErrorLocation,
    ErrorType,
)
from app.contracts.execution import (
    DEFAULT_RESULT_ROWS_LIMIT,
    SQLExecutionResult,
    apply_result_rows,
)
from app.contracts.knowledge import FieldCategory, KnowledgeSnapshot
from app.contracts.parsing import ParsedSql
from app.ports.database import DatabaseGateway, SQLExecutionError

logger = logging.getLogger(__name__)


class DeterministicVerifier:
    """结构/执行确定性校验（先于 LLM 语义审查运行）。"""

    def __init__(self, gateway: Optional[DatabaseGateway] = None,
                 context: Optional[DatabaseContext] = None,
                 rows_limit: int = DEFAULT_RESULT_ROWS_LIMIT,
                 row_cap: Optional[int] = None,
                 profiles: Optional[list] = None):
        self.gateway = gateway
        self.context = context
        # 代执行时回填结果行的采样上限（进 result_rows）
        self.rows_limit = rows_limit
        # 答案执行的取回上限（不改写 SQL；命中即 rows_capped=True）
        self.row_cap = row_cap or None
        # 实测 JOIN 画像（JOIN_UNJOINABLE / JOIN_UNNORMALIZED_KEY 判据）
        self.profiles = list(profiles or [])

    def check(self, snapshot: KnowledgeSnapshot, parsed: ParsedSql,
              feedback: SQLExecutionResult) -> list[Diagnosis]:
        """执行全部确定性检查 → list[Diagnosis]。"""
        errors: list[Diagnosis] = []
        errors.extend(self.check_columns(snapshot, parsed))
        errors.extend(self.check_joins(snapshot, parsed))
        errors.extend(self.check_aggregation(snapshot, parsed))
        errors.extend(self.check_value_membership(snapshot, parsed))
        errors.extend(self.check_execution(feedback))
        return errors

    # ---------- 列存在性（K1） ----------

    @staticmethod
    def check_columns(snapshot: KnowledgeSnapshot,
                      parsed: ParsedSql) -> list[Diagnosis]:
        schema = snapshot.k1_schema
        errors: list[Diagnosis] = []
        for table, column in parsed.columns:
            if column == "*":
                continue
            col_info = schema.get_column(table, column)
            if col_info is None:
                # SQL 可能没限定表名：任意表含该列则不算列不存在
                found = any(
                    schema.get_column(t, column) is not None
                    for t in schema.all_table_names()
                )
                if not found:
                    errors.append(Diagnosis(
                        type=ErrorType.COLUMN_NOT_FOUND,
                        location=ErrorLocation(column=column, table=table),
                        detail=f"列 {table}.{column} 在数据库中不存在",
                    ))
                    continue
            # K4 缺少该列语义不报错：知识覆盖不全不等于 SQL 有错
            # （II-1 字段角色误用由 LLM 语义审查判定）
        return errors

    # ---------- JOIN vs K6 + 实测画像 ----------

    def check_joins(self, snapshot: KnowledgeSnapshot,
                    parsed: ParsedSql) -> list[Diagnosis]:
        fks = snapshot.foreign_keys()
        valid_join_cols: set[tuple] = set()
        known_pairs: set[frozenset] = set()
        for fk in fks:
            valid_join_cols.add((fk.source_table, fk.source_column,
                                 fk.target_table, fk.target_column))
            valid_join_cols.add((fk.target_table, fk.target_column,
                                 fk.source_table, fk.source_column))
            known_pairs.add(frozenset((fk.source_table, fk.target_table)))
        # 实测画像也构成"该表对被覆盖"的依据（无 PK/FK 声明的库靠它）
        prof_by_pair: dict[frozenset, list] = {}
        for prof in (self.profiles or []):
            pair = frozenset((prof.source_table, prof.target_table))
            prof_by_pair.setdefault(pair, []).append(prof)
            known_pairs.add(pair)

        errors: list[Diagnosis] = []
        for join in parsed.joins:
            pair = frozenset((join.left_table, join.right_table))
            # 画像判定：该表对在数据上连不上 / 需 TRIM
            errors.extend(self._profile_diagnoses(join, prof_by_pair.get(pair)))
            if pair not in known_pairs:
                continue   # 既无 K6 也无实测画像覆盖 → 不判错（避免误报）
            forward = (join.left_table, join.left_column,
                       join.right_table, join.right_column)
            reverse = (join.right_table, join.right_column,
                       join.left_table, join.left_column)
            if forward in valid_join_cols or reverse in valid_join_cols:
                continue
            if self._join_matches_profile(forward, prof_by_pair.get(pair)):
                continue   # 实测画像认可该连接列
            errors.append(Diagnosis(
                type=ErrorType.JOIN_INVALID,
                location=ErrorLocation(
                    clause="JOIN",
                    table=f"{join.left_table} JOIN {join.right_table}",
                ),
                detail=(f"JOIN {join.left_table}.{join.left_column} = "
                        f"{join.right_table}.{join.right_column} 与已知的"
                        f"该表对连接列不一致（K6/实测画像）"),
            ))
        return errors

    @staticmethod
    def _join_matches_profile(forward: tuple, profiles) -> bool:
        for prof in (profiles or []):
            if (forward[0], forward[1], forward[2], forward[3]) == (
                    prof.source_table, prof.source_column,
                    prof.target_table, prof.target_column):
                return True
            if (forward[2], forward[3], forward[0], forward[1]) == (
                    prof.source_table, prof.source_column,
                    prof.target_table, prof.target_column):
                return True
        return False

    @staticmethod
    def _profile_diagnoses(join, profiles) -> list[Diagnosis]:
        """画像级 JOIN 诊断：不可连接 / 需 TRIM 归一化。"""
        out: list[Diagnosis] = []
        for prof in (profiles or []):
            same_key = (
                (join.left_column or "").lower() == prof.source_column.lower()
                and (join.right_column or "").lower() == prof.target_column.lower()
            ) or (
                (join.left_column or "").lower() == prof.target_column.lower()
                and (join.right_column or "").lower() == prof.source_column.lower()
            )
            if not same_key:
                continue
            table = f"{join.left_table} JOIN {join.right_table}"
            if not prof.joinable:
                out.append(Diagnosis(
                    type=ErrorType.JOIN_UNJOINABLE,
                    location=ErrorLocation(clause="JOIN", table=table),
                    detail=(f"实测 {prof.source_table}.{prof.source_column} ⋈ "
                            f"{prof.target_table}.{prof.target_column} 0 命中，"
                            f"该连接在数据上不成立"),
                ))
            elif prof.needs_trim:
                out.append(Diagnosis(
                    type=ErrorType.JOIN_UNNORMALIZED_KEY,
                    location=ErrorLocation(clause="JOIN", table=table),
                    detail=(f"实测键值含前导/尾随空格：原生抽样 "
                            f"{prof.sample_size} 个键值全部连不上，"
                            f"TRIM 归一化后可连上 {prof.trim_hits} 个"),
                ))
        return out

    # ---------- 聚合 vs K3（Fig.1 反例） ----------

    @staticmethod
    def check_aggregation(snapshot: KnowledgeSnapshot,
                          parsed: ParsedSql) -> list[Diagnosis]:
        errors: list[Diagnosis] = []
        numeric_only_funcs = {"SUM", "AVG"}  # COUNT/MIN/MAX 宽松
        descriptions = {
            FieldCategory.IDENTIFIER: "标识符，不可聚合",
            FieldCategory.MEASURE: "度量值，可聚合",
            FieldCategory.DIMENSION: "维度，可分组",
            FieldCategory.DATETIME: "日期时间",
            FieldCategory.TEXT: "文本",
            FieldCategory.BOOLEAN: "布尔",
            FieldCategory.OTHER: "其他",
        }
        for agg in parsed.aggregates:
            if agg.func not in numeric_only_funcs:
                continue
            # 未限定表名的聚合列：无歧义时解析归属表（论文 Fig.1 的
            # AVG(CDSCode) 常不带表前缀，不解析则 K3 查不到而漏检）
            if not agg.table:
                candidates = {t for t, c in parsed.columns if c == agg.column}
                if len(candidates) == 1:
                    agg = agg.model_copy(update={"table": next(iter(candidates))})
            field_type = snapshot.get_field_type(agg.table, agg.column)
            if field_type is None:
                continue  # K3 未分类，无法判断
            if field_type != FieldCategory.MEASURE:
                errors.append(Diagnosis(
                    type=ErrorType.AGGREGATION_TYPE_MISMATCH,
                    location=ErrorLocation(
                        clause=agg.clause or "SELECT",
                        column=agg.column,
                        table=agg.table,
                    ),
                    detail=(f"{agg.func}({agg.table}.{agg.column}) 要求 measure 类型，"
                            f"实际为 {field_type.value}（{descriptions.get(field_type, '')}）"),
                    evidence_ref=f"K3:{agg.table}.{agg.column}",
                ))
        return errors

    # ---------- 执行性（B 类数据库反馈） ----------

    _MEMBERSHIP_ENUM_MAX = 30   # 可穷举枚举列的 distinct 上限（与生成侧值域接地一致）

    def check_value_membership(self, snapshot: KnowledgeSnapshot,
                               parsed: ParsedSql) -> list[Diagnosis]:
        """谓词值成员检查（论文消融 C：rule-based checks 的值域部分）。

        - 枚举列（distinct ≤ 阈值）：等值/IN 字面值必须属于全量实测取值
        - 数值列：等值字面值必须落在实测 [min, max]；ISO 日期文本按字典序比较
        探针查询失败或网关不可用时跳过（辅助信号，不阻塞、不误报）。
        """
        errors: list[Diagnosis] = []
        if not parsed.predicates or self.gateway is None or self.context is None:
            return errors
        types: dict[tuple[str, str], str] = {}
        for table in snapshot.k1_schema.tables:
            for col in table.columns:
                types[(table.name.lower(), col.name.lower())] = \
                    (col.data_type or "").lower()
        enum_cache: dict[tuple[str, str], Optional[set]] = {}
        range_cache: dict[tuple[str, str], Optional[tuple]] = {}

        for p in parsed.predicates:
            if not p.table or not p.column or not p.literal:
                continue
            dtype = types.get((p.table.lower(), p.column.lower()))
            if dtype is None:   # 列不存在由 check_columns 负责
                continue
            key = (p.table, p.column)
            is_numeric = any(k in dtype for k in (
                "int", "real", "float", "double", "numeric", "decimal"))
            is_temporal = "date" in dtype or "time" in dtype
            try:
                if is_numeric or is_temporal:
                    if key not in range_cache:
                        lo = self._probe_scalar(
                            f'SELECT MIN("{p.column}") AS v FROM "{p.table}"')
                        hi = self._probe_scalar(
                            f'SELECT MAX("{p.column}") AS v FROM "{p.table}"')
                        range_cache[key] = ((lo, hi)
                                            if lo is not None and hi is not None
                                            else None)
                    bounds = range_cache[key]
                    if bounds is None:
                        continue
                    lo, hi = bounds
                    if is_temporal and p.is_string:
                        # ISO 日期文本与 min/max 文本做字典序比较
                        ok = str(lo) <= p.literal <= str(hi)
                    elif not p.is_string:
                        try:
                            value = float(p.literal)
                        except ValueError:
                            continue
                        ok = float(lo) <= value <= float(hi)
                    else:
                        continue
                    if not ok:
                        errors.append(Diagnosis(
                            type=ErrorType.VALUE_NOT_IN_DOMAIN,
                            location=ErrorLocation(clause="WHERE",
                                                   column=p.column,
                                                   table=p.table),
                            detail=(f"谓词值 {p.literal} 超出 "
                                    f"{p.table}.{p.column} 实测范围 "
                                    f"[{lo}, {hi}]"),
                            artifact="s",
                            violated_condition=(
                                f"谓词字面值应落在 "
                                f"{p.table}.{p.column} 实测值域内"),
                        ))
                else:
                    if key not in enum_cache:
                        n = self._probe_scalar(
                            f'SELECT COUNT(DISTINCT "{p.column}") AS v '
                            f'FROM "{p.table}"')
                        if n is None or int(n) > self._MEMBERSHIP_ENUM_MAX:
                            enum_cache[key] = None
                        else:
                            ret = self.gateway.execute_readonly(
                                self.context,
                                f'SELECT DISTINCT "{p.column}" AS v '
                                f'FROM "{p.table}" WHERE "{p.column}" '
                                f'IS NOT NULL '
                                f'LIMIT {self._MEMBERSHIP_ENUM_MAX}',
                                limit=None)
                            rows = (ret.get("data") or []
                                    if ret.get("success") else [])
                            enum_cache[key] = {
                                str(next(iter(r.values()), ""))
                                for r in rows
                                if next(iter(r.values()), None) is not None}
                    values = enum_cache[key]
                    if not values:
                        continue
                    if p.literal not in values:
                        errors.append(Diagnosis(
                            type=ErrorType.VALUE_NOT_IN_DOMAIN,
                            location=ErrorLocation(clause="WHERE",
                                                   column=p.column,
                                                   table=p.table),
                            detail=(f"谓词值 '{p.literal}' 不在 "
                                    f"{p.table}.{p.column} 实测枚举内"
                                    f"（共 {len(values)} 个取值）"),
                            artifact="s",
                            violated_condition=(
                                f"枚举列谓词字面值应取自 "
                                f"{p.table}.{p.column} 的真实取值"),
                        ))
            except Exception:   # 探针失败：跳过该项，不阻塞校验
                continue
        return errors

    def _probe_scalar(self, sql: str):
        """单值探针（走网关只读执行）；失败抛错由调用方跳过。"""
        ret = self.gateway.execute_readonly(self.context, sql, limit=None)
        if not ret.get("success"):
            raise RuntimeError(ret.get("error", "probe query failed"))
        data = ret.get("data") or []
        if not data:
            return None
        return next(iter(data[0].values()), None)

    def check_execution(self, feedback: SQLExecutionResult) -> list[Diagnosis]:
        """执行失败是错误；空结果不是（可合法返回 0 行）。"""
        errors: list[Diagnosis] = []
        if feedback.executed:
            if feedback.execution_success is False:
                errors.append(Diagnosis(
                    type=ErrorType.EXECUTION_FAILED,
                    location=ErrorLocation(clause="EXECUTION"),
                    detail=feedback.execution_error or "SQL 执行失败",
                ))
            return errors

        # 尚未执行则现执行一次（Phase 2 跳过执行的情形）
        if self.gateway is None or self.context is None:
            return errors
        try:
            ret = self.gateway.execute_readonly(self.context, feedback.sql,
                                                limit=None,
                                                row_cap=self.row_cap,
                                                keep_rows=self.rows_limit)
            if not ret.get("success"):
                errors.append(Diagnosis(
                    type=ErrorType.EXECUTION_FAILED,
                    location=ErrorLocation(clause="EXECUTION"),
                    detail=str(ret.get("error", "执行失败")),
                ))
            else:
                feedback.executed = True
                feedback.execution_success = True
                feedback.result_count = ret.get("row_count", 0)
                apply_result_rows(feedback, ret, self.rows_limit)
        except SQLExecutionError as e:
            feedback.executed = True
            feedback.execution_success = False
            feedback.execution_error = e.detail
            errors.append(Diagnosis(
                type=ErrorType.EXECUTION_FAILED,
                location=ErrorLocation(clause="EXECUTION"),
                detail=e.detail,
            ))
        except Exception as e:  # noqa: BLE001
            feedback.executed = True
            feedback.execution_success = False
            feedback.execution_error = str(e)
            errors.append(Diagnosis(
                type=ErrorType.EXECUTION_FAILED,
                location=ErrorLocation(clause="EXECUTION"),
                detail=str(e),
            ))
        return errors

    def execute_updated(self, sql: str, dialect: str = "sqlite") -> SQLExecutionResult:
        """执行新 SQL 并回填反馈（修正阶段用）。"""
        from app.contracts.execution import SQLExecutionResult
        result = SQLExecutionResult(sql=sql, dialect=dialect,
                                    corrected_by_reflection=True)
        if self.gateway is None or self.context is None:
            return result
        try:
            ret = self.gateway.execute_readonly(self.context, sql, limit=None,
                                                row_cap=self.row_cap,
                                                keep_rows=self.rows_limit)
            if ret.get("success"):
                result.executed = True
                result.execution_success = True
                result.result_count = ret.get("row_count", 0)
                apply_result_rows(result, ret, self.rows_limit)
            else:
                result.executed = True
                result.execution_success = False
                result.execution_error = str(ret.get("error", "执行失败"))
        except SQLExecutionError as e:
            result.executed = True
            result.execution_success = False
            result.execution_error = e.detail
        except Exception as e:  # noqa: BLE001
            result.executed = True
            result.execution_success = False
            result.execution_error = str(e)
        return result
