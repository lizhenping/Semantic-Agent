"""实测数据画像（app/domain/data_profile.py）

Phase 1 知识构建之后的一次性旁路探测，回答"这些表之间真的连得上吗"：

- JOIN 命中率（原生 / TRIM 归一化）：脏键（前导/尾随空格）与"键完全不匹配"可见
- 目标键唯一性 → 基数（N:1 / N:N）

产物是 JoinProfile 列表，由适配器写成快照目录的 profile.jsonl 侧车文件
（不进入 K1-K6，也不进 content_digest——画像随数据变化，不改变知识身份）。

动机（量产实测）：spider2/Db-IMDB 的 M_Cast.PID 带前导空格 → Person⋈M_Cast
原生 0 命中（TRIM 后 86,850 行）；部分 benchmark 的 SQLite 无 PK/FK 声明 →
K6 为空 → JOIN 校验静默失效（K6 空库错误率 37.0% vs 非空 22.8%）。
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from app.contracts.knowledge import (
    JoinProfile,
    KnowledgeSnapshot,
    SchemaMetadata,
)

logger = logging.getLogger(__name__)

# 每个候选表对抽样的源键值个数（探测成本 = 1~2 次目标表扫描，与样本数无关）
_PROBE_SAMPLE = 16
# 默认候选表对上限（每库），控制 Phase 1 之后的探测成本
_DEFAULT_MAX_PAIRS = 40
# 超大表阈值：参与键对的表超过该行数则跳过（单次扫描成本不可控，
# 如 bird/language_corpus 1.57 亿行）
_MAX_TABLE_ROWS = 2_000_000
# 单库探测时间预算（秒）：超预算即停止该库剩余键对，已探测的结果照常写入
_DEFAULT_DB_BUDGET_S = 90.0


def _literal(value) -> str:
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


class DataProfiler:
    """候选 JOIN 键的实测画像（构造注入网关/上下文，遵守 domain 层约束）。"""

    def __init__(self, gateway, context, max_pairs: int = _DEFAULT_MAX_PAIRS,
                 db_budget_s: float = _DEFAULT_DB_BUDGET_S):
        self.gateway = gateway
        self.context = context
        self.max_pairs = max(1, int(max_pairs))
        self.db_budget_s = float(db_budget_s)
        self._distinct_cache: dict[tuple[str, str], Optional[int]] = {}

    # ---------- 候选表对 ----------

    @staticmethod
    def candidate_pairs(snapshot: KnowledgeSnapshot,
                        schema: SchemaMetadata) -> list[tuple[str, str, str, str]]:
        """候选 (源表, 源列, 目标表, 目标列)：K6 ∪ K1 外键 ∪ 共享列名 ∪ 表名主干。

        - 共享列名（如 case_id / PID 被多张表持有）：以"枢纽表"（该列为 PK
          或表名与列名最接近者）为中心向其余持有表配对，避免组合爆炸
        - 表名主干（如 project_members.project → projects 的 PK）：经典 FK 推断
        """
        pairs: list[tuple[str, str, str, str]] = []
        seen: set[tuple[str, str, str, str]] = set()

        def _add(a: str, ac: str, b: str, bc: str) -> None:
            if not all((a, ac, b, bc)) or a == b:
                return
            key = (a, ac, b, bc)
            if key in seen:
                return
            seen.add(key)
            pairs.append(key)

        for rel in snapshot.k6_relations:
            if rel.source_column and rel.target_column:
                _add(rel.source_table, rel.source_column,
                     rel.target_table, rel.target_column)

        pk_of = {t.name: list(t.primary_keys or []) for t in schema.tables}
        column_types: dict[tuple[str, str], str] = {}
        by_name: dict[str, list[tuple[str, str]]] = {}
        for table in schema.tables:
            for col in table.columns:
                column_types[(table.name, col.name)] = (col.data_type or "").lower()
                if col.foreign_key and col.foreign_key_table and col.foreign_key_column:
                    _add(table.name, col.name,
                         col.foreign_key_table, col.foreign_key_column)
                by_name.setdefault(col.name.lower(), []).append(
                    (table.name, col.name))
                # 列名指向另一张表（主干匹配）→ 与该表 PK 配对
                for cand in _stem_tables(col.name, schema):
                    for pk in pk_of.get(cand, [])[:1]:
                        if cand != table.name:
                            _add(table.name, col.name, cand, pk)

        for name, holders in by_name.items():
            if len(holders) < 2 or len(holders) > _MAX_NAME_HOLDERS:
                continue
            if not (name.endswith("id") or name.endswith("_id")
                    or name in _COMMON_JOIN_NAMES):
                continue
            if len(holders) <= _ALL_PAIRS_MAX_HOLDERS:
                # 小组：全配对（真实缺陷常出现在非枢纽那一侧，
                # 如 Db-IMDB 的 M_Cast.PID ⋈ Person.PID）
                combos = [(holders[i], holders[j])
                          for i in range(len(holders))
                          for j in range(i + 1, len(holders))]
            else:
                hub = DataProfiler._hub_holder(name, holders, pk_of)
                combos = [(hub, h) for h in holders if h != hub]
            for a, b in combos:
                if column_types.get(a) != column_types.get(b):
                    continue   # 同名不同类型多半不是同一个键
                _add(a[0], a[1], b[0], b[1])
        return pairs

    @staticmethod
    def _hub_holder(name: str, holders: list[tuple[str, str]],
                    pk_of: dict[str, list[str]]) -> tuple[str, str]:
        """共享列名的"枢纽表"：该列为 PK 者优先，其次表名与列名主干最接近者。"""
        for holder in holders:
            if holder[1].lower() in {p.lower() for p in pk_of.get(holder[0], [])}:
                return holder
        for holder in holders:
            stem = name[:-3] if name.endswith("_id") else (
                name[:-2] if name.endswith("id") else name)
            if stem and (stem in holder[0].lower()
                         or holder[0].lower().rstrip("s") == stem):
                return holder
        return holders[0]

    # ---------- 探测 ----------

    def build(self, snapshot: KnowledgeSnapshot,
              schema: SchemaMetadata) -> list[JoinProfile]:
        if self.gateway is None or self.context is None:
            return []
        rows = {t.name: (t.row_count or 0) for t in schema.tables}
        profiles: list[JoinProfile] = []
        started = time.monotonic()
        for src_t, src_c, tgt_t, tgt_c in self.candidate_pairs(snapshot, schema):
            if len(profiles) >= self.max_pairs:
                break
            if time.monotonic() - started > self.db_budget_s:
                logger.info("画像探测超时预算（%.0fs），本库已完成 %d 条",
                            self.db_budget_s, len(profiles))
                break
            # 超大表跳过：单次扫描成本不可控
            if any(rows.get(t, 0) > _MAX_TABLE_ROWS for t in (src_t, tgt_t)):
                logger.debug("跳过超大表键对 %s→%s", src_t, tgt_t)
                continue
            try:
                prof = self._probe(src_t, src_c, tgt_t, tgt_c, rows)
            except Exception as e:  # noqa: BLE001
                logger.debug("画像探测失败 %s.%s→%s.%s: %s",
                             src_t, src_c, tgt_t, tgt_c, e)
                continue
            if prof is not None:
                profiles.append(prof)
        return profiles

    def _probe(self, src_t: str, src_c: str, tgt_t: str, tgt_c: str,
               rows: dict[str, int]) -> Optional[JoinProfile]:
        """抽样探测：源列取 N 个去重值，**一次 IN 查询**判定这些值在目标列是否存在。

        成本 = 每个候选表对 1~2 次目标表扫描（原生 + 必要时 TRIM），
        与表规模线性、与样本数无关——早先"逐值探测"在大库上会退化成
        12 次全表扫描（甚至每次都撞 15s 超时）。
        """
        sample = self._sample_values(src_t, src_c, _PROBE_SAMPLE)
        if not sample:
            return None
        source_rows = rows.get(src_t, 0) or self._scalar(
            f'SELECT COUNT(*) FROM "{src_t}"') or 0
        native = self._membership(tgt_t, tgt_c, sample, trim=False)
        trim = None
        if native == 0:
            # 原生全连不上：可能是脏键，也可能是真的不可连接 → 再测 TRIM
            trim = self._membership(tgt_t, tgt_c, sample, trim=True)
        prof = JoinProfile(
            source_table=src_t, source_column=src_c,
            target_table=tgt_t, target_column=tgt_c,
            source_rows=int(source_rows), sample_size=len(sample),
            sample_hits=native, trim_hits=trim,
        )
        if prof.joinable:
            distinct = self._distinct_count(tgt_t, tgt_c)
            if distinct is not None:
                prof.distinct_target = distinct
                target_rows = rows.get(tgt_t, 0) or self._scalar(
                    f'SELECT COUNT(*) FROM "{tgt_t}"') or 0
                prof.target_unique = bool(target_rows) and distinct >= target_rows
        return prof

    def _sample_values(self, table: str, column: str, limit: int) -> list:
        ret = self.gateway.execute_readonly(
            self.context,
            f'SELECT DISTINCT "{column}" AS v FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL LIMIT {limit}',
            limit=None)
        data = (ret or {}).get("data") or []
        out = []
        for row in data:
            v = next(iter(row.values()), None)
            if v is not None and str(v) != "":
                out.append(v)
        return out

    def _membership(self, table: str, column: str, values: list,
                    trim: bool) -> int:
        """样本值在目标列的存在性（一次 IN 查询，返回命中的去重值个数）。"""
        col = f'TRIM("{column}")' if trim else f'"{column}"'
        lits = ", ".join(
            (f"TRIM({_literal(v)})" if trim else _literal(v)) for v in values)
        sql = (f'SELECT DISTINCT {col} AS c FROM "{table}" '
               f'WHERE {col} IN ({lits}) LIMIT {len(values) + 5}')
        ret = self.gateway.execute_readonly(self.context, sql, limit=None)
        rows = (ret or {}).get("data") or []
        return len(rows)

    def _distinct_count(self, table: str, column: str) -> Optional[int]:
        key = (table, column)
        if key not in self._distinct_cache:
            self._distinct_cache[key] = self._scalar(
                f'SELECT COUNT(DISTINCT "{column}") AS v FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL')
        v = self._distinct_cache[key]
        return int(v) if v is not None else None

    def _scalar(self, sql: str):
        ret = self.gateway.execute_readonly(self.context, sql, limit=None)
        if not ret.get("success"):
            raise RuntimeError(ret.get("error", "probe failed"))
        data = ret.get("data") or []
        if not data:
            return None
        return next(iter(data[0].values()), None)


# 同名同类型列做候选时的持有人数上限（防宽库同名列爆炸；
# spider2/Db-IMDB 的 PID 被 4 张表持有，需要 >= 4）
_MAX_NAME_HOLDERS = 8
# 持有人不超过该数时做全配对（缺陷常在非枢纽一侧）；超过则枢纽↔其余
_ALL_PAIRS_MAX_HOLDERS = 4


def _stem_tables(column: str, schema: SchemaMetadata) -> list[str]:
    """列名的主干像哪张表（project → projects / case → case_ids）。"""
    stem = (column or "").lower()
    if stem.endswith("_id"):
        stem = stem[:-3]
    elif stem.endswith("id") and len(stem) > 2:
        stem = stem[:-2]
    if len(stem) < 3:
        return []
    out = []
    for table in schema.tables:
        low = table.name.lower()
        if low == stem or low.rstrip("s") == stem or low.startswith(stem + "_"):
            out.append(table.name)
    return out
# 除 *id 外，常见 JOIN 列名
_COMMON_JOIN_NAMES = frozenset({
    "code", "key", "no", "num", "number", "ref", "reference", "uuid",
    "case_id", "project", "name_id",
})
