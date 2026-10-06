"""候选生成（app/domain/candidate_generation.py）

论文 §III.D：先生成 (q, r)，再生成 s：
    (q, r) = f_que(σ, ℓ, K)
    s      = f_sql(q, r, σ, ℓ, K)

移植自旧 tools/synthesis/{question_synth,sql_synth}.py：
- 场景驱动三层循环（scenarios × sub_scenarios × complexity）+ 随机补齐
- 值域接地（2026-09 修复：枚举=全量、数值/日期=范围、高基数=样本+格式）
- 表-列引用校验、SQL 后处理、执行反馈
- 论文要求的 sql_strategy 组装

环境开关 SEMANTICSQL_SQL_NO_R 改为构造参数 include_rationale_hints
（domain 不读环境变量，由 application 传入）。
"""

import logging
import random
import re
import threading
import uuid
from typing import Optional

from app.contracts.candidate import (
    CandidateSample,
    ColumnOperation,
    Question,
    Rationale,
    SQLStrategy,
    TableSelection,
)
from app.contracts.database import DatabaseContext
from app.contracts.execution import (
    DEFAULT_RESULT_ROWS_LIMIT,
    apply_result_rows,
)
from app.contracts.knowledge import (
    KnowledgeSnapshot,
    SchemaMetadata,
)
from app.ports.database import DatabaseGateway, SQLExecutionError
from app.ports.llm import ChatGateway
from app.ports.prompting import PromptRenderer

logger = logging.getLogger(__name__)

# 值域接地参数（与旧 question_synth 一致）
_ENUM_MAX_DISTINCT = 30   # 全量枚举列的 distinct 上限
_DOMAIN_BUDGET = 300      # 单列值域文本预算（字符；套话已移到提示词全局规则）
# 稀有值标注：行数低于该比例或绝对值的枚举值附带 "(行数)"，
# 让"某类记录只有几十行"这类量级前提可见（防阈值不可达）
_RARE_VALUE_RATIO = 0.05
_RARE_VALUE_ABS = 100
# 分位数探测：每快照最多探测的数值列数（控 Phase 1 扫描成本）
_QUANTILE_MAX_COLUMNS = 80
# 超大表阈值：超过该行数跳过全列扫描（COUNT(DISTINCT)/分位数/GROUP BY 成本不可控）
_LARGE_TABLE_ROWS = 2_000_000
_QUANTILE_MIN_ROWS = 500   # 行数太少时分位数无意义
_QUANTILE_MIN_DISTINCT = 10
# 时间列名提示词：命中才做 strftime 可用性探测（LIMIT 5，成本可忽略）
_TIME_NAME_HINTS = ("date", "time", "_at", "ts", "stamp", "day", "month", "year")
# 日期形态（K1 采样值）：'2017-09-10' / '2017-09-10 09:50:00+03' / '2104-01'
_DATE_SHAPE = re.compile(r"^\d{4}-\d{2}(-\d{2})?([ T]\d{2}:\d{2}.*)?$")


class UnusableDatabaseError(RuntimeError):
    """库不可用于合成（如非空表不足）：整库跳过，不消耗生成算力。"""


def _fmt(v) -> str:
    """值域文本里的数值紧凑化（大整数/浮点不铺满提示词）。"""
    if v is None:
        return "?"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.6g}"
    return str(v)


class ValueDomainScanner:
    """实测值域扫描（V：让生成提示词看到真实数据，论文消融 V）。

    - 空表（行数 0）：显式标注"空表，禁止提问"（不再静默跳过）
    - 文本低基数（distinct ≤ 30）：全量合法值；稀有值附行数（防阈值不可达）
    - 数值列：min~max + 均值 + p50/p90/p99（阈值可达性可见）
    - 时间列（或列名像时间的列）：strftime 可用性探测，不可用时给出替代写法
    - 高基数字符串：样本 + "只能使用真实出现的字面值"
    网关不可用/扫描失败时返回空 dict（退回样本值模式，不阻塞生成）。
    """

    def __init__(self, gateway, context):
        self.gateway = gateway
        self.context = context
        self._quantile_budget = _QUANTILE_MAX_COLUMNS

    def build(self, snapshot: KnowledgeSnapshot, schema: SchemaMetadata
              ) -> dict[tuple[str, str], str]:
        domains: dict[tuple[str, str], str] = {}
        if self.gateway is None or self.context is None:
            return domains
        try:
            for table in schema.tables:
                n_rows = table.row_count or 0
                if not n_rows:
                    n_rows = self._scalar(
                        f'SELECT COUNT(*) AS v FROM "{table.name}"')
                if not n_rows:
                    # 空表：显式标注（生成侧据此禁用该表；不再静默跳过）
                    for col in table.columns:
                        domains[(table.name, col.name)] = \
                            "空表(0 行)：本表无数据，禁止围绕本表提问"
                    continue
                large = n_rows > _LARGE_TABLE_ROWS
                for col in table.columns:
                    t = (col.data_type or "").lower()
                    is_numeric = any(k in t for k in (
                        "int", "real", "float", "double", "numeric", "decimal"))
                    is_temporal = "date" in t or "time" in t
                    # 时间列识别：列名像时间（用于触发探测）或 K1 采样值本身
                    # 是日期形态（如 '2017-09-10 09:50:00+03' 存成 NUMERIC）
                    name_hint = self._looks_temporal(col.name)
                    date_shaped = self._samples_look_temporal(col.sample_values)
                    time_hint = name_hint or date_shaped
                    if large:
                        # 超大表（千万行级）：不做全列扫描（COUNT(DISTINCT)/
                        # 分位数/GROUP BY 成本不可控），只给时间列警告与提示
                        domains[(table.name, col.name)] = \
                            self._describe_large_table(
                                table.name, col.name, n_rows, is_temporal,
                                time_hint)
                        continue
                    distinct_n = self._scalar(
                        f'SELECT COUNT(DISTINCT "{col.name}") AS v '
                        f'FROM "{table.name}"')
                    if not distinct_n:
                        continue
                    try:
                        text = self._describe_column(
                            table.name, col.name, n_rows, distinct_n,
                            is_numeric, is_temporal, time_hint, date_shaped)
                        if text:
                            domains[(table.name, col.name)] = text
                    except Exception:
                        continue
            # 预算截断
            for k, v in list(domains.items()):
                if len(v) > _DOMAIN_BUDGET:
                    domains[k] = v[:_DOMAIN_BUDGET] + " ..."
        except Exception as e:
            logger.warning("值域接地信息构建失败，退回样本值模式: %s", e)
            return {}
        return domains

    # ---------- 单列描述 ----------

    def _describe_large_table(self, table: str, column: str, n_rows: int,
                              is_temporal: bool, time_hint: bool) -> str:
        """超大表（> _LARGE_TABLE_ROWS）的廉价描述：不做全列扫描。"""
        warn = self._time_format_warning(table, column, False, is_temporal,
                                         time_hint)
        note = (f"超大表（{_fmt(n_rows)} 行）：未做全列扫描（值域/分位数略），"
                f"接地只能依赖列名与观测示例值，不要写依赖具体取值分布的阈值")
        return f"{warn}；{note}" if warn else note

    def _describe_column(self, table: str, column: str, n_rows: int,
                         distinct_n: int, is_numeric: bool,
                         is_temporal: bool, time_hint: bool,
                         date_shaped: bool) -> str:
        warn = self._time_format_warning(table, column, is_numeric,
                                         is_temporal, time_hint)
        parts: list[str] = []
        if warn:
            parts.append(warn)

        # 时间形态列（真日期列 / 日期形态文本 / strftime 不可用的时间文本）：
        # 只给日期形态范围——让"问了一个没有数据的年份"这类前提错误可见
        temporal_like = bool(warn) or is_temporal or (date_shaped and not is_numeric)
        if temporal_like:
            lo, hi = self._temporal_range(table, column)
            parts.append(f"{_fmt(lo)} ~ {_fmt(hi)}（{_fmt(distinct_n)} 值）")
            return "；".join(parts)

        if is_numeric:
            row = self._row(
                f'SELECT MIN("{column}"), MAX("{column}"), AVG("{column}") '
                f'FROM "{table}"')
            lo, hi, avg = ((row + (None, None, None))[:3] if row
                           else (None, None, None))
            parts.append(f"{_fmt(lo)} ~ {_fmt(hi)}（{_fmt(distinct_n)} 值）")
            if avg is not None:
                q = self._quantiles(table, column, n_rows, distinct_n)
                if q:
                    p50, p90, p99 = q
                    parts.append(
                        f"均值 {_fmt(avg)}；p50/p90/p99 = "
                        f"{_fmt(p50)}/{_fmt(p90)}/{_fmt(p99)}"
                        f"（阈值需落在此区间内）")
                else:
                    parts.append(f"均值 {_fmt(avg)}")
            return "；".join(parts)

        if distinct_n <= _ENUM_MAX_DISTINCT:
            counts = self._value_counts(table, column, _ENUM_MAX_DISTINCT)
            if not counts:
                return "；".join(parts) if parts else ""
            # 稀有值行数**单独列出**：合法值清单必须保持"可直接逐字引用"的纯净格式
            # （行数若写在值旁边，模型会把 "Delayed(41)" 当字面值抄进问题）
            rare_cut = max(1, min(_RARE_VALUE_ABS, int(n_rows * _RARE_VALUE_RATIO)))
            values = [str(v) for v, _ in counts]
            rare = [f"{v}={n}" for v, n in counts if n < rare_cut]
            text = f"合法值({_fmt(distinct_n)}): " + "|".join(values)
            if rare:
                text += "；该列稀有值行数（阈值不可超过这些数）: " + ", ".join(rare)
            parts.append(text)
            return "；".join(parts)

        sample = self._sample_literals(table, column, 3)
        parts.append(f"高基数({_fmt(distinct_n)} 值)样本: {sample}")
        return "；".join(parts)

    def _temporal_range(self, table: str, column: str) -> tuple:
        """时间列的日期形态范围（排除 '\\N' 等占位符与格式异常值）。"""
        row = self._row(
            f'SELECT MIN("{column}"), MAX("{column}") FROM "{table}" '
            f"WHERE \"{column}\" GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]*'")
        return (row + (None, None))[:2] if row else (None, None)

    def _quantiles(self, table: str, column: str, n_rows: int,
                   distinct_n: int) -> Optional[tuple]:
        """p50/p90/p99（ORDER BY + OFFSET，逐列一次三查，带全局预算）。"""
        if (n_rows < _QUANTILE_MIN_ROWS or distinct_n < _QUANTILE_MIN_DISTINCT
                or self._quantile_budget <= 0):
            return None
        self._quantile_budget -= 1
        out = []
        for p in (0.5, 0.9, 0.99):
            offset = max(0, min(n_rows - 1, int(n_rows * p)))
            v = self._scalar(
                f'SELECT "{column}" FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL '
                f'ORDER BY "{column}" LIMIT 1 OFFSET {offset}')
            if v is None:
                return None
            out.append(v)
        return tuple(out)

    def _time_format_warning(self, table: str, column: str, is_numeric: bool,
                             is_temporal: bool, time_hint: bool) -> str:
        """时间列格式探测：strftime 不可用时返回替代写法提示。

        命中条件：列名/类型像时间，且前 5 个非空值 strftime('%Y') 全为 NULL
        （如 '2017-09-10 09:50:00+03' 这类带时区后缀的文本）。
        """
        if not (is_temporal or time_hint):
            return ""
        try:
            ret = self.gateway.execute_readonly(
                self.context,
                f'SELECT strftime(\'%Y\', "{column}") AS v FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL LIMIT 5',
                limit=None)
        except Exception:  # noqa: BLE001
            return ""
        data = (ret or {}).get("data") or []
        if not data:
            return ""
        if any(row.get("v") is not None for row in data):
            return ""   # strftime 可用
        return ("⚠ 该时间列实际为带时区后缀/非常规格式的文本，strftime() 返回 NULL，"
                "禁止用 strftime/date()，请用 substr() 或字符串前缀比较")

    @staticmethod
    def _looks_temporal(name: str) -> bool:
        low = (name or "").lower()
        return any(h in low for h in _TIME_NAME_HINTS)

    @staticmethod
    def _samples_look_temporal(samples) -> bool:
        """K1 采样值是否为日期形态（类型可能标成 NUMERIC/TEXT）。"""
        for v in (samples or [])[:5]:
            s = str(v)
            if _DATE_SHAPE.match(s):
                return True
        return False

    def _scalar(self, sql: str):
        """单值查询（走网关只读执行，带超时闸门）。"""
        ret = self.gateway.execute_readonly(self.context, sql, limit=None)
        if not ret.get("success"):
            raise RuntimeError(ret.get("error", "query failed"))
        data = ret.get("data") or []
        if not data:
            return None
        return next(iter(data[0].values()), None)

    def _row(self, sql: str) -> Optional[tuple]:
        """单行多列查询（返回首个数据行的值元组）。"""
        ret = self.gateway.execute_readonly(self.context, sql, limit=None)
        if not ret.get("success"):
            raise RuntimeError(ret.get("error", "query failed"))
        data = ret.get("data") or []
        if not data:
            return None
        return tuple(data[0].values())

    def _value_counts(self, table: str, column: str,
                      limit: int) -> list[tuple]:
        """枚举列的 (值, 行数) 清单（一条 GROUP BY 查询，兼作全量枚举）。"""
        ret = self.gateway.execute_readonly(
            self.context,
            f'SELECT "{column}" AS v, COUNT(*) AS n FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL '
            f'GROUP BY "{column}" ORDER BY n DESC LIMIT {limit}',
            limit=None)
        data = (ret or {}).get("data") or []
        out: list[tuple] = []
        for row in data:
            v = row.get("v")
            if v is None or str(v) == "":
                continue
            out.append((str(v), int(row.get("n") or 0)))
        return out

    def _sample_literals(self, table: str, column: str, limit: int) -> str:
        ret = self.gateway.execute_readonly(
            self.context,
            f'SELECT DISTINCT "{column}" AS v FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL LIMIT {limit}',
            limit=None)
        data = ret.get("data") or []
        vals = []
        for row in data:
            v = next(iter(row.values()), None)
            if v is not None and str(v) != "":
                vals.append(str(v))
        return ", ".join(vals)


class QuestionGenerator:
    """q 生成：场景驱动的问题合成（f_que 产出 q 与场景元数据）。"""

    def __init__(
        self,
        chat: ChatGateway,
        prompts: PromptRenderer,
        catalog,  # ScenarioCatalog（adapters 提供，鸭子类型）
        gateway: Optional[DatabaseGateway] = None,
        context: Optional[DatabaseContext] = None,
        rng: Optional[random.Random] = None,
        domain_scan_timeout: float = 30.0,
        workers: int = 20,
        table_cap: int = 10,
    ):
        self.chat = chat
        self.prompts = prompts
        self.catalog = catalog
        self.gateway = gateway
        self.context = context
        self.rng = rng or random.Random()
        self.domain_scan_timeout = domain_scan_timeout
        # 问题级并发度（每个问题一次独立 LLM 调用）
        self.workers = max(1, int(workers))
        # 表格子集裁剪：每问题只带 N 张表（0 = 不裁剪）
        self.table_cap = max(0, int(table_cap))

    def generate(self, snapshot: KnowledgeSnapshot, count: int) -> list[Question]:
        """合成 count 个自然语言问题（场景三层循环 + 随机补齐）。

        每个问题的 LLM 调用相互独立：workers>1 时线程池并发生成，
        结果按场景顺序保持；不足 count 时随机补齐（带尝试上限防死循环）。
        """
        schema = snapshot.k1_schema
        if not schema.tables:
            raise RuntimeError("K1 schema 为空，请先执行知识构建")

        tables_data = self._build_tables_context(snapshot, schema)
        # 空表硬规则：row_count==0 的表（K1 已实测）不进任何问题的前提；
        # 非空表不足 2 张的库不具备提问空间（如 sdss 6 表中 5 表为空）→ 整库跳过
        total_tables = len(tables_data)
        tables_data = [t for t in tables_data if t.get("row_count") != 0]
        if len(tables_data) < 2:
            raise UnusableDatabaseError(
                f"非空表不足（{len(tables_data)}/{total_tables} 张表有数据），"
                f"跳过该库以避免生成围绕空表的问题")
        dropped = total_tables - len(tables_data)
        if dropped:
            logger.info("空表过滤：%d/%d 张表行数为 0，已移出问题前提",
                        dropped, total_tables)
        er_data = self._build_er_context(snapshot)
        combos = list(self.catalog.ordered_combos())

        questions: list[Question] = []
        if self.workers > 1 and count > 1 and combos:
            # 第一轮：按场景顺序并发生成前 count 个
            questions = self._generate_batch_parallel(
                snapshot, combos[:count], tables_data, er_data)
            remaining = combos[count:]
            # 第二轮：随机补齐（并发，带尝试上限）
            attempts = 0
            while len(questions) < count and attempts < count * 3:
                need = count - len(questions)
                batch = [self.catalog.random_combo() for _ in range(need * 2)]
                batch = [c for c in batch if c is not None]
                attempts += len(batch)
                if not batch:
                    break
                questions.extend(self._generate_batch_parallel(
                    snapshot, batch, tables_data, er_data))
            return questions[:count]

        for combo in combos:
            q = self._generate_one(snapshot, combo, tables_data, er_data)
            if q is not None:
                questions.append(q)
                logger.info("生成 %d/%d: %s/%s/%s", len(questions), count,
                            combo["main_key"], combo["sub_key"], combo["complexity"])
                if len(questions) >= count:
                    return questions

        # 第二轮：不够 count 时随机采样补齐（与旧循环一致）
        while len(questions) < count:
            combo = self.catalog.random_combo()
            if combo is None:
                break
            q = self._generate_one(snapshot, combo, tables_data, er_data)
            if q is not None:
                questions.append(q)
        return questions

    def _generate_batch_parallel(self, snapshot: KnowledgeSnapshot,
                                 combos: list[dict], tables_data: list[dict],
                                 er_data: dict) -> list[Question]:
        """并发生成一批问题，按 combos 顺序返回成功项。"""
        from concurrent.futures import ThreadPoolExecutor

        def _one(i: int, combo: dict) -> Optional[Question]:
            q = self._generate_one(snapshot, combo, tables_data, er_data)
            if q is not None:
                logger.info("生成 %d/%d: %s/%s/%s", i, len(combos),
                            combo["main_key"], combo["sub_key"], combo["complexity"])
            return q

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = list(pool.map(lambda pair: _one(*pair),
                                    list(enumerate(combos, 1))))
        return [q for q in results if q is not None]

    # ---------- 单问题生成 ----------

    def _generate_one(self, snapshot: KnowledgeSnapshot, combo: dict,
                      tables_data: list[dict], er_data: dict) -> Optional[Question]:
        # 表格子集裁剪：大库每问题随机抽 table_cap 张表（q/s 同子集保证一致）
        tables_used = tables_data
        if self.table_cap and len(tables_data) > self.table_cap:
            tables_used = self.rng.sample(tables_data, self.table_cap)
        template_data = {
            "main_scenario": {
                "name": combo["main"].get("name", ""),
                "description": combo["main"].get("description", ""),
            },
            "sub_scenario": {
                "name": combo["sub"].get("name", ""),
                "focus_areas": combo["sub"].get("focus_areas", []),
            },
            "complexity": combo["complexity"],
            "complexity_config": self.catalog.complexity_spec(combo["complexity"]),
            "use_case": combo["use_case"],
            "tables": tables_used,
            "er_analysis": er_data,
        }
        try:
            prompt = self.prompts.render(
                "generation/question_generation.j2", **template_data)
            result = self.chat.generate_json(prompt)
        except Exception as e:
            logger.warning("LLM 生成失败 (%s/%s): %s",
                           combo["complexity"], combo["use_case_name"], e)
            return None
        q = self._to_question(snapshot, result, combo)
        if q is not None:
            q.context_tables = [str(t.get("name", "")) for t in tables_used]
        return q

    def _to_question(self, snapshot: KnowledgeSnapshot, result: dict,
                     combo: dict) -> Optional[Question]:
        q_data = result.get("generated_question", {}) or result
        text = (q_data.get("question_text") or q_data.get("text") or "").strip()
        if not text:
            return None

        # 表-列引用校验（K1 schema，防幻觉引用）
        tables_used = self._extract_tables_used(result)
        columns_used = self._extract_columns_used(result)
        if not self._validate_refs(snapshot, tables_used, columns_used):
            logger.debug("表-列引用校验失败，丢弃该问题")
            return None

        from app.contracts.scenario import ComplexityLevel, GenerationMetadata
        level = ComplexityLevel.from_name(combo["complexity"]).to_int()
        metadata = GenerationMetadata(
            main_scenario=combo["main"].get("name", ""),
            sub_scenario=combo["sub"].get("name", ""),
            complexity_level=level,
            use_case=combo["use_case_name"],
        )
        return Question(
            question_id=f"q-{uuid.uuid4().hex[:12]}",
            text=text,
            question_focus=q_data.get("question_focus", ""),
            business_rules=self._extract_business_rules(q_data),
            metadata=metadata,
        )

    # ---------- 上下文构建（K1/K4/K5 聚合） ----------

    def _build_tables_context(self, snapshot: KnowledgeSnapshot,
                              schema: SchemaMetadata) -> list[dict]:
        """聚合 K1/K4/K5 + 实测值域（2026-09 接地修复）成模板期望结构。"""
        value_domains = self._build_value_domains(snapshot, schema)
        tables: list[dict] = []
        for table in schema.tables:
            ts = snapshot.get_table_semantic(table.name)
            desc = ts.description if ts else ""
            cols = []
            for col in table.columns:
                cs = snapshot.get_column(table.name, col.name)
                col_desc = cs.description if cs else ""
                cols.append({
                    "name": col.name,
                    "type": col.data_type,
                    "comment": col_desc or (col.comment or ""),
                    "description": col_desc or (col.comment or ""),
                    "classification": {"category": self._classify_type(col.data_type)},
                    "full_reference": f"{table.name}.{col.name}",
                    "sample_values": col.sample_values[:5],
                    "value_domain": value_domains.get((table.name, col.name), ""),
                })
            tables.append({
                "name": table.name,
                "description": desc or (table.comment or ""),
                # 实测行数（K1）：0 表示空表（生成侧据此过滤），None 表示未采集
                "row_count": table.row_count,
                "columns": cols,
            })
        tables.sort(key=lambda t: t["name"])
        return tables

    # ---------- 实测值域（接地修复：与生成侧 OBSERVED 块同源） ----------

    def _build_value_domains(self, snapshot: KnowledgeSnapshot,
                             schema: SchemaMetadata) -> dict[tuple[str, str], str]:
        """实测值域（共享 ValueDomainScanner，按快照缓存避免逐题重扫）。"""
        if not hasattr(self, "_vd_cache"):
            self._vd_cache: dict[str, dict] = {}
        key = snapshot.snapshot_id
        if key not in self._vd_cache:
            self._vd_cache[key] = ValueDomainScanner(
                self.gateway, self.context).build(snapshot, schema)
        return self._vd_cache[key]

    # ---------- ER 上下文（K6） ----------

    @staticmethod
    def _build_er_context(snapshot: KnowledgeSnapshot) -> dict:
        """K6 关系 → 问题提示词的三段式（键名与模板严格对齐）。

        - physical：带具体列的关系（可执行的 JOIN 键，如 a.col → b.col）
        - logical：表级关系（{tables, pattern}）
        - conceptual：关系语义（{entity1, entity2, relationship}）

        旧实现产出 from/to/relationship，与模板期望的键名不一致，
        导致即使 K6 非空、问题提示词里的 ER 块也是空的（渲染为空白）。
        """
        er_data: dict[str, list] = {"physical": [], "logical": [],
                                    "conceptual": []}
        for rel in snapshot.k6_relations:
            if rel.source_column and rel.target_column:
                er_data["physical"].append({
                    "from": f"{rel.source_table}.{rel.source_column}",
                    "to": f"{rel.target_table}.{rel.target_column}",
                })
            er_data["logical"].append({
                "tables": [rel.source_table, rel.target_table],
                "pattern": rel.relationship_type,
            })
            er_data["conceptual"].append({
                "entity1": rel.source_table,
                "entity2": rel.target_table,
                "relationship": rel.reason or rel.relationship_type,
            })
        return er_data

    @staticmethod
    def _classify_type(data_type: str) -> str:
        t = (data_type or "").lower()
        if any(k in t for k in ("int", "decimal", "float", "double", "numeric")):
            return "numeric"
        if "date" in t or "time" in t:
            return "date"
        if any(k in t for k in ("varchar", "text", "char")):
            return "text"
        return "other"

    # ---------- 校验 ----------

    @staticmethod
    def _extract_tables_used(result: dict) -> list[str]:
        ta = result.get("table_analysis", {}) or {}
        raw = result.get("tables_used") or ta.get("tables_used") or []
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                name = item.get("table_name") or item.get("name") or ""
                if name:
                    out.append(name)
        return out

    @staticmethod
    def _extract_columns_used(result: dict) -> list[str]:
        ca = result.get("column_analysis", {}) or {}
        raw = result.get("columns_used") or ca.get("columns_used") or []
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                ref = item.get("column_full_name") or item.get("name") or ""
                if ref:
                    out.append(ref)
        return out

    @staticmethod
    def _validate_refs(snapshot: KnowledgeSnapshot, tables_used: list,
                       columns_used: list) -> bool:
        schema = snapshot.k1_schema
        all_tables = set(schema.all_table_names())
        for t in tables_used:
            if not isinstance(t, str) or not t:
                continue
            if t not in all_tables:
                return False
        for ref in columns_used:
            if not isinstance(ref, str) or "." not in ref:
                continue
            tname, cname = ref.split(".", 1)
            if schema.get_column(tname, cname) is None:
                return False
        return True

    @staticmethod
    def _extract_business_rules(q_data: dict) -> list[dict]:
        rules = q_data.get("business_rules", [])
        if not isinstance(rules, list):
            return []
        out = []
        for r in rules:
            if isinstance(r, str):
                out.append({"rule": r})
            elif isinstance(r, dict):
                out.append(r)
        return out


class SQLGenerator:
    """(s, r) 生成：基于 q + K 生成 SQL 和结构化推理链（f_sql）。"""

    def __init__(
        self,
        chat: ChatGateway,
        prompts: PromptRenderer,
        gateway: Optional[DatabaseGateway] = None,
        context: Optional[DatabaseContext] = None,
        dialect: str = "sqlite",
        include_rationale_hints: bool = True,
        rows_limit: int = DEFAULT_RESULT_ROWS_LIMIT,
        row_cap: Optional[int] = None,
        profiles: Optional[list] = None,
    ):
        self.chat = chat
        self.prompts = prompts
        self.gateway = gateway
        self.context = context
        self.dialect = dialect
        # 消融开关（旧 SEMANTICSQL_SQL_NO_R）：剥离推理链提示，
        # 仅保留问题文本 + schema + 外键
        self.include_rationale_hints = include_rationale_hints
        # 执行结果行采样上限（进 SQLExecutionResult.result_rows）
        self.rows_limit = rows_limit
        # 答案执行的取回上限（不改写 SQL；命中即 rows_capped=True）
        self.row_cap = row_cap or None
        # 实测 JOIN 画像（覆盖警告/脏键提示/不可连接禁令）
        self.profiles = list(profiles or [])

    def generate(self, snapshot: KnowledgeSnapshot, question: Question,
                 execute: bool = True) -> CandidateSample:
        """对单个 Question 生成 SQL + Rationale → CandidateSample。"""
        schema = snapshot.k1_schema
        # 1. 聚合 K1/K4/K5/K6 上下文
        context = self._build_context(snapshot, schema, question)
        # 2. 渲染 prompt + LLM 生成
        sql, rationale_raw = self._generate_sql_with_llm(question, context)
        # 3. 后处理 SQL
        sql = self._postprocess_sql(sql)
        # 4. 执行反馈（可选）
        sql_result = self._maybe_execute(sql, execute)
        # 5. 组装 Rationale
        rationale = self._build_rationale(question, rationale_raw, context)

        sample = CandidateSample(
            question=question, sql_result=sql_result, rationale=rationale)
        sample.provenance.database_id = (self.context.database_id
                                         if self.context else "")
        sample.provenance.dialect = self.dialect
        sample.provenance.knowledge_snapshot_id = snapshot.snapshot_id
        sample.provenance.knowledge_content_digest = snapshot.content_digest()
        sample.provenance.model_id = getattr(self.chat, "model_id", "")
        return sample

    # ---------- 上下文聚合 ----------

    def _build_context(self, snapshot: KnowledgeSnapshot,
                       schema: SchemaMetadata, question: Question) -> dict:
        domains = self._value_domains(snapshot, schema)
        # 表格子集：q 生成时看到的表集合（记录在 question.context_tables），
        # s 必须在同子集上生成，否则 q-s 语义断裂
        allowed = {t.lower() for t in (question.context_tables or [])}
        tables: dict[str, dict] = {}
        for table in schema.tables:
            if allowed and table.name.lower() not in allowed:
                continue
            ts = snapshot.get_table_semantic(table.name)
            t_desc = ts.description if ts else ""
            cols = []
            for col in table.columns:
                cs = snapshot.get_column(table.name, col.name)
                c_desc = cs.description if cs else ""
                cols.append({
                    "name": col.name,
                    "type": col.data_type,
                    "comment": c_desc or (col.comment or ""),
                    "is_primary": col.primary_key,
                    "is_foreign": col.foreign_key,
                    "is_nullable": col.nullable,
                    "sample_values": col.sample_values[:5],
                    "value_domain": domains.get((table.name, col.name), ""),
                })
            tables[table.name] = {
                "name": table.name,
                "comment": t_desc or (table.comment or ""),
                "columns": cols,
            }

        foreign_keys = self._extract_foreign_keys(snapshot, schema)
        question_focus = (question.question_focus if self.include_rationale_hints
                          else "")
        business_rules = (question.business_rules if self.include_rationale_hints
                          else [])
        return {
            "schema_info": {"tables": tables},
            "foreign_keys": foreign_keys,
            "question_data": {
                "question_text": question.text,
                "question_focus": question_focus,
                "business_rules": business_rules,
            },
        }

    def _value_domains(self, snapshot: KnowledgeSnapshot,
                       schema: SchemaMetadata) -> dict[tuple[str, str], str]:
        """实测值域（V：SQL 生成提示词同样看到真实数据；按快照缓存）。

        并发生成时首个样本触发扫描——加锁防止同快照被并发重复扫库。
        """
        if not hasattr(self, "_vd_cache"):
            self._vd_cache: dict[str, dict] = {}
            self._vd_lock = threading.Lock()
        key = snapshot.snapshot_id
        if key not in self._vd_cache:
            with self._vd_lock:
                if key not in self._vd_cache:
                    self._vd_cache[key] = ValueDomainScanner(
                        self.gateway, self.context).build(snapshot, schema)
        return self._vd_cache[key]

    @staticmethod
    def _extract_foreign_keys(snapshot: KnowledgeSnapshot,
                              schema: SchemaMetadata) -> list[dict]:
        """K6 外键（优先），回退 *_id 列名启发式（与旧行为一致）。"""
        fks: list[dict] = []
        for rel in snapshot.k6_relations:
            fks.append({
                "from_table": rel.source_table,
                "to_table": rel.target_table,
                "column_name": rel.source_column or "",
                "from": (f"{rel.source_table}.{rel.source_column}"
                         if rel.source_column else rel.source_table),
                "to": (f"{rel.target_table}.{rel.target_column}"
                       if rel.target_column else rel.target_table),
                "relationship_type": rel.relationship_type,
            })
        if fks:
            return fks

        table_names = set(schema.all_table_names())
        for table in schema.tables:
            for col in table.columns:
                ref = None
                if col.name.endswith("_id"):
                    ref = col.name[:-3]
                elif col.name.endswith("id") and len(col.name) > 2:
                    ref = col.name[:-2]
                if ref and ref in table_names:
                    fks.append({
                        "from_table": table.name,
                        "to_table": ref,
                        "column_name": col.name,
                        "from": f"{table.name}.{col.name}",
                        "to": f"{ref}.id",
                        "relationship_type": "many_to_one",
                    })
        return fks

    # ---------- LLM 生成 ----------

    def _generate_sql_with_llm(self, question: Question,
                               context: dict) -> tuple[str, dict]:
        enhanced = self._build_enhanced_context(context, self.profiles)
        q_data = context["question_data"]
        prompt = self.prompts.render(
            "tools/sql_generation.j2",
            question=q_data["question_text"],
            context=enhanced,
            dialect=self.dialect,
            question_focus=q_data.get("question_focus", ""),
            expected_output="",  # Phase 2 不预知答案
            business_rules=q_data.get("business_rules", []),
            join_relationships=context.get("foreign_keys", []),
        )
        response = self.chat.generate(prompt)
        return self._parse_sql_and_rationale(response)

    @staticmethod
    def _parse_sql_and_rationale(response: str) -> tuple[str, dict]:
        """从 LLM 响应解析 (sql, rationale)；解析失败退化为只提取 SQL。"""
        import json
        try:
            start = response.find("{")
            end = response.rfind("}")
            if start >= 0 and end > start:
                parsed = json.loads(response[start:end + 1])
                sql = parsed.get("sql", "")
                rationale = parsed.get("rationale", {})
                if not sql:
                    sql = SQLGenerator._extract_sql_from_response(response)
                if not isinstance(rationale, dict):
                    rationale = {}
                return sql, rationale
        except (json.JSONDecodeError, ValueError):
            pass
        return SQLGenerator._extract_sql_from_response(response), {}

    @staticmethod
    def _build_enhanced_context(context: dict,
                                profiles: Optional[list] = None) -> str:
        parts: list[str] = []
        tables = context.get("schema_info", {}).get("tables", {})
        foreign_keys = context.get("foreign_keys", [])
        if tables:
            parts.append("数据库结构：")
            for table_name, t_info in tables.items():
                parts.append(f"\n表: {table_name}")
                if t_info.get("comment"):
                    parts.append(f"  说明: {t_info['comment']}")
                parts.append("  列:")
                for col in t_info.get("columns", [])[:15]:
                    line = f"    - {col.get('name')} ({col.get('type')})"
                    if col.get("comment"):
                        line += f" -- {col['comment']}"
                    if col.get("is_primary"):
                        line += " [PK]"
                    if col.get("is_foreign"):
                        line += " [FK]"
                    if col.get("sample_values"):
                        observed = ", ".join(str(v) for v in col["sample_values"])
                        line += f" [观测示例值: {observed}]"
                    parts.append(line)
                    if col.get("value_domain"):
                        parts.append(f"      📊 实测值域: {col['value_domain']}")
        if foreign_keys:
            parts.append("\n外键关系:")
            for rel in foreign_keys[:10]:
                parts.append(f"  - {rel.get('from', '')} → {rel.get('to', '')}")
        if profiles:
            relevant = SQLGenerator._relevant_profiles(profiles, tables.keys())
            if relevant:
                parts.append("\n⚠ 实测连接覆盖（Phase 1 探测，务必遵守）:")
                for prof in relevant:
                    parts.append(f"  - {prof.label()}")
        return "\n".join(parts)

    @staticmethod
    def _relevant_profiles(profiles: list, table_names,
                           limit: int = 6) -> list:
        """带进提示词的画像条目：只保留"有信息量"的，按严重度取前 N 条。

        优先级：不可连接（禁止使用）> 需 TRIM（脏键）> 低覆盖。
        完全连通（100% 且无需 TRIM）的条目零信息量，不占提示词配额——
        否则 6 个名额会被无害连接占满，真正要提示的脏键警告反而被挤掉。
        """
        names = {t.lower() for t in table_names}
        related = [p for p in profiles
                   if p.source_table.lower() in names
                   and p.target_table.lower() in names]

        def severity(p) -> int:
            if not p.joinable:
                return 0        # 不可连接
            if p.needs_trim:
                return 1        # 脏键
            if p.hit_rate < 0.5:
                return 2        # 低覆盖
            if p.hit_rate < 1.0:
                return 3        # 部分覆盖
            return 4            # 完全连通：无信息量

        related.sort(key=severity)
        return [p for p in related if severity(p) < 4][:limit]

    # ---------- SQL 后处理 ----------

    @staticmethod
    def _extract_sql_from_response(response: str) -> str:
        m = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
        if m:
            content = m.group(1).strip()
            if any(k in content.upper() for k in
                   ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE")):
                return content
        return response.strip()

    @staticmethod
    def _postprocess_sql(sql: str) -> str:
        sql = re.sub(r"```sql\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"```\s*", "", sql)
        sql = re.sub(r"\s+", " ", sql).strip()
        if not sql.endswith(";"):
            sql += ";"
        return sql

    # ---------- 执行反馈 ----------

    def _maybe_execute(self, sql: str, execute: bool):
        from app.contracts.execution import SQLExecutionResult
        result = SQLExecutionResult(sql=sql, dialect=self.dialect)
        if not execute or self.gateway is None or self.context is None:
            result.executed = False
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
                result.execution_error = ret.get("error", "execution failed")
        except SQLExecutionError as e:
            result.executed = True
            result.execution_success = False
            result.execution_error = e.detail
            logger.warning("SQL 执行失败: %s", e.detail)
        except Exception as e:
            result.executed = True
            result.execution_success = False
            result.execution_error = str(e)
            logger.warning("SQL 执行失败: %s", e)
        return result

    # ---------- Rationale 组装 ----------

    @staticmethod
    def _build_rationale(question: Question, rationale_raw: dict,
                         context: dict) -> Rationale:
        """组装结构化推理链 r（论文 Fig.training_data_format）。

        σ=(C,T), ℓ 直接沿用问题合成阶段的元数据。
        """
        q_data = context["question_data"]
        metadata = question.metadata

        ts_raw = rationale_raw.get("table_selection", {}) or {}
        table_selection = TableSelection(
            tables_used=ts_raw.get("tables_used", []),
            reasoning=ts_raw.get("reasoning", ""),
        )

        cs_raw = rationale_raw.get("column_selection", {}) or {}
        columns_list = (cs_raw.get("columns_used", [])
                        if isinstance(cs_raw, dict) else cs_raw)
        column_selection: list[ColumnOperation] = []
        for col in columns_list:
            if isinstance(col, dict):
                column_selection.append(ColumnOperation(
                    name=col.get("name", ""),
                    type=col.get("type", ""),
                    operation=col.get("operation", ""),
                    purpose=col.get("purpose", ""),
                ))

        strategy_raw = rationale_raw.get("sql_strategy", {}) or {}
        sql_strategy = SQLStrategy(
            operations=strategy_raw.get("operations", []),
            approach=strategy_raw.get("approach", ""),
            no_need=strategy_raw.get("no_need", []),
        )

        return Rationale(
            focus=(rationale_raw.get("focus", "")
                   or q_data.get("question_focus", question.question_focus)),
            metadata=metadata,
            table_selection=table_selection,
            column_selection=column_selection,
            sql_strategy=sql_strategy,
            expected_output=rationale_raw.get("expected_output", ""),
        )
