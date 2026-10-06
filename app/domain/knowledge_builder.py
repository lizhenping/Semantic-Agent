"""知识构建器（app/domain/knowledge_builder.py）

论文 §III.C：K = K1..K6 固定依赖顺序构建（doc/02 §2.2）：

    K1 Schema（直接数据库事实，无 LLM）
     |--> K3 Field Classification --> K4 Column Semantics --> K5 Table Semantics
     |--> K2 Domain Knowledge
     +--> K6 Cross-table Relations（消费 K1-K5）

首版保持 K1→K2→K3→K4→K5→K6 的执行顺序。每个阶段显式输入输出，
不再通过共享可变 kbase 穿透；产物统一由 KnowledgeSnapshot 发布。

逻辑逐字移植自旧 tools/analysis/{schema_extraction,domain_analysis,
field_analysis,column_analysis,table_analysis,er_analysis}.py，
保持 Prompt 输入形状与 LLM 调用次数不变（行为冻结基线）。
"""

import logging
import re
from collections import Counter
from typing import Optional

from app.contracts.database import DatabaseContext
from app.contracts.knowledge import (
    ColumnInfo,
    ColumnSemantics,
    CrossTableRelation,
    DomainKnowledge,
    FieldCategory,
    FieldClassification,
    KnowledgeSnapshot,
    SchemaMetadata,
    TableInfo,
    TableSemantics,
)
from app.ports.database import DatabaseGateway
from app.ports.knowledge import KnowledgeStore
from app.ports.llm import ChatGateway
from app.ports.prompting import PromptRenderer

logger = logging.getLogger(__name__)

_CATEGORY_MAP = {c.value: c for c in FieldCategory}

# DDL 超过该长度时降采样（与旧行为一致，防超 LLM token 上限）
_DDL_MAX_CHARS = 60000

# K6 中 LLM ER 候选关系条数上限（防实体多属性时关系爆炸）
_ER_RELATION_CAP = 40


def _clean_think(text: str) -> str:
    """剥离 <think>/<thinking> 标签（Qwen3 thinking 残留）。"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()


class KnowledgeBuilder:
    """K1..K6 知识构建器：DatabaseContext -> KnowledgeSnapshot。"""

    def __init__(
        self,
        gateway: DatabaseGateway,
        chat: ChatGateway,
        prompts: PromptRenderer,
        builder_version: str = "1.0.0",
    ):
        self.gateway = gateway
        self.chat = chat
        self.prompts = prompts
        self.builder_version = builder_version

    # ============================================================
    # 主入口
    # ============================================================

    def build(self, context: DatabaseContext,
              store: Optional[KnowledgeStore] = None) -> KnowledgeSnapshot:
        """构建 K1..K6 并发布不可变 KnowledgeSnapshot。

        store 非空时，各层产物同时写入库（供旧目录布局回归对比）；
        快照本身是唯一发布边界。
        """
        # ---- K1: Schema（直接数据库事实，无 LLM）----
        schema = self.gateway.read_schema(context)
        if store:
            store.save("schema", schema.model_dump())
        logger.info("K1 schema: %d 表", len(schema.tables))

        # ---- K2: Domain（依赖 K1）----
        domain = self._build_domain(context, schema)
        if store:
            store.save("domain", domain.model_dump())
        logger.info("K2 domain: %s", domain.domain_type)

        # ---- K3: Field Classification（依赖 K1，读取 K2 上下文）----
        field_types = self._build_field_types(context, schema, domain)
        if store:
            store.save("field", [f.model_dump() for f in field_types])
        logger.info("K3 field_types: %d 字段", len(field_types))

        # ---- K4: Column Semantics（依赖 K1 + K3）----
        columns = self._build_columns(context, schema, domain, field_types)
        if store:
            store.save("column", [c.model_dump() for c in columns])
        logger.info("K4 column_semantics: %d 列", len(columns))

        # ---- K5: Table Semantics（依赖 K1 + K3 + K4）----
        tables = self._build_tables(context, schema, domain, field_types, columns)
        if store:
            store.save("table", [t.model_dump() for t in tables])
        logger.info("K5 table_semantics: %d 表", len(tables))

        # ---- K6: Cross-table Relations（依赖 K1~K5）----
        relations = self._build_relations(context, schema, domain, columns, tables)
        if store:
            store.save("er", [r.model_dump() for r in relations])
        logger.info("K6 relations: %d 关系", len(relations))

        snapshot = KnowledgeSnapshot(
            # 快照 ID 需可安全作为目录名：database_id 中的 "/" 换成 "-"
            snapshot_id=f"{context.database_id.replace('/', '-')}-{_utc_stamp()}",
            database_id=context.database_id,
            dialect=context.dialect.value,
            builder_version=self.builder_version,
            model_id=getattr(self.chat, "model_id", ""),
            prompt_versions=self.prompts.versions(),
            k1_schema=schema,
            k2_domain=domain,
            k3_field_types=field_types,
            k4_column_semantics=columns,
            k5_table_semantics=tables,
            k6_relations=relations,
        )
        context.knowledge_snapshot_id = snapshot.snapshot_id
        logger.info("✅ 知识快照发布: %s (digest=%s)",
                    snapshot.snapshot_id, snapshot.content_digest())
        return snapshot

    # ============================================================
    # K2: Domain Analysis（移植 domain_analysis.py）
    # ============================================================

    def _build_domain(self, context: DatabaseContext,
                      schema: SchemaMetadata) -> DomainKnowledge:
        if not schema.tables:
            raise RuntimeError("K1 schema 为空，无法执行 K2 域分析")

        ddl_content = self._format_schema_to_ddl(schema)
        prompt = self.prompts.render("analysis/domain_analysis.j2",
                                     schema_ddl=ddl_content)
        response = self.chat.generate_json(prompt)
        validated = self._validate_domain_response(response)
        return DomainKnowledge(
            domain_type=validated.get("domain_type", "未知领域"),
            description="; ".join(validated.get("business_rules", [])[:2]),
            business_concepts=validated.get("key_entities", []),
            naming_patterns=validated.get("special_fields", []),
        )

    @staticmethod
    def _format_schema_to_ddl(schema: SchemaMetadata) -> str:
        ddl_lines: list[str] = []
        for table in schema.tables:
            if not table.columns:
                continue
            ddl_lines.append(f"CREATE TABLE `{table.name}` (")
            column_defs: list[str] = []
            primary_keys: list[str] = []
            for col in table.columns:
                col_def = f"  `{col.name}` {col.data_type or 'VARCHAR(255)'}"
                if not col.nullable:
                    col_def += " NOT NULL"
                if col.comment:
                    col_def += f" COMMENT '{col.comment}'"
                column_defs.append(col_def)
                if col.primary_key:
                    primary_keys.append(col.name)
            if primary_keys:
                column_defs.append(f"  PRIMARY KEY (`{'`, `'.join(primary_keys)}`)")
            ddl_lines.append(",\n".join(column_defs))
            ddl_lines.append(");")
            ddl_lines.append("")
        ddl = "\n".join(ddl_lines)
        if len(ddl) > _DDL_MAX_CHARS:
            ddl = KnowledgeBuilder._optimize_ddl_for_llm(ddl)
        return ddl

    @staticmethod
    def _optimize_ddl_for_llm(ddl_content: str) -> str:
        keep_keywords = (
            "create table", "primary key", "foreign key",
            "not null", "unique", "id", "name", "status",
        )
        important = [
            line for line in ddl_content.split("\n")
            if line.strip() and any(k in line.lower() for k in keep_keywords)
        ]
        return "\n".join(important)

    @staticmethod
    def _validate_domain_response(data: dict) -> dict:
        required = (
            "domain_type", "business_problems", "solution_approaches",
            "key_entities", "business_rules", "special_fields",
        )
        validated: dict = {}
        for field in required:
            if field not in data:
                validated[field] = "" if field == "domain_type" else []
                continue
            if field == "domain_type":
                validated[field] = str(data[field]).strip()
            else:
                value = data[field]
                if isinstance(value, list):
                    validated[field] = [
                        str(item).strip() for item in value
                        if item and str(item).strip()
                    ]
                else:
                    validated[field] = []
        return validated

    # ============================================================
    # K3: Field Classification（移植 field_analysis.py）
    # ============================================================

    def _build_field_types(self, context: DatabaseContext, schema: SchemaMetadata,
                           domain: DomainKnowledge) -> list[FieldClassification]:
        classifications: list[FieldClassification] = []
        for table in schema.tables:
            for col in table.columns:
                cls = self._classify_field(context, table.name, col, domain)
                if cls is not None:
                    classifications.append(cls)
        return classifications

    def _classify_field(self, context: DatabaseContext, table_name: str,
                        col: ColumnInfo,
                        domain: DomainKnowledge) -> Optional[FieldClassification]:
        """对单列做 LLM 分类（失败跳过，不阻断构建）。"""
        try:
            prompt = self.prompts.render(
                "tools/field_analysis.j2",
                database_name=context.database_id,
                table_name=table_name,
                field_name=f"{table_name}.{col.name}",
                data_type=col.data_type,
                sample_values=col.sample_values,
                entropy_level=col.entropy_level,
                is_primary=col.primary_key,
                is_foreign=col.foreign_key,
                is_nullable=col.nullable,
                domain_type=domain.domain_type,
                domain_description=domain.description,
                business_desc="",  # K4 尚未构建（与旧行为一致）
            )
            response = self.chat.generate_json(prompt)
            category_str = str(response.get("category", "other")).lower()
            category = _CATEGORY_MAP.get(category_str, FieldCategory.OTHER)
            reasoning = str(response.get("category_desc", "")).strip()
            return FieldClassification(
                table_name=table_name,
                column_name=col.name,
                category=category,
                confidence=0.8,
                reasoning=reasoning,
                data_type=col.data_type,
            )
        except Exception as e:
            logger.warning("字段 %s.%s 分类失败: %s", table_name, col.name, e)
            return None

    # ============================================================
    # K4: Column Semantics（移植 column_analysis.py）
    # ============================================================

    def _build_columns(self, context: DatabaseContext, schema: SchemaMetadata,
                       domain: DomainKnowledge,
                       field_types: list[FieldClassification]) -> list[ColumnSemantics]:
        table_ddls = {t.name: self._format_table_ddl(t) for t in schema.tables}
        field_type_index = {f.field_key: f.category for f in field_types}

        semantics: list[ColumnSemantics] = []
        for table in schema.tables:
            for col in table.columns:
                desc = self._describe_column(
                    context, table, col, table_ddls, field_type_index, domain)
                if desc is not None:
                    semantics.append(desc)
        return semantics

    @staticmethod
    def _format_table_ddl(table: TableInfo) -> str:
        lines = [f"CREATE TABLE `{table.name}` ("]
        col_defs = []
        for col in table.columns:
            d = f"  `{col.name}` {col.data_type or 'VARCHAR(255)'}"
            if not col.nullable:
                d += " NOT NULL"
            if col.primary_key:
                d += " PRIMARY KEY"
            if col.comment:
                d += f" COMMENT '{col.comment}'"
            col_defs.append(d)
        lines.append(",\n".join(col_defs))
        lines.append(");")
        return "\n".join(lines)

    def _describe_column(self, context: DatabaseContext, table: TableInfo,
                         col: ColumnInfo, table_ddls: dict[str, str],
                         field_type_index: dict[str, FieldCategory],
                         domain: DomainKnowledge) -> Optional[ColumnSemantics]:
        try:
            field_key = f"{table.name}.{col.name}"
            category = field_type_index.get(field_key)
            prompt = self.prompts.render(
                "tools/column_description.j2",
                table_name=table.name,
                column_name=col.name,
                database_name=context.database_id,
                database_domain=domain.description or "general domain",
                table_ddl=table_ddls.get(table.name, ""),
                column_type=col.data_type,
                is_nullable=col.nullable,
                is_primary_key=col.primary_key,
                is_foreign_key=col.foreign_key,
                column_examples=col.sample_values,
                field_category=category.value if category else "other",
                entropy_info={"level": col.entropy_level},
                dim_or_meas=category.value if category in (
                    FieldCategory.MEASURE, FieldCategory.DIMENSION) else "dimension",
                field_importance="high" if (col.primary_key or col.foreign_key) else (
                    "medium" if category in (FieldCategory.MEASURE, FieldCategory.DIMENSION)
                    else "low"),
            )
            response = self.chat.generate(prompt)
            description = self._parse_description_response(response)
            if not description:
                return None
            return ColumnSemantics(
                table_name=table.name,
                column_name=col.name,
                description=description,
                confidence=0.8,
                source="generated",
                field_category=category,
                data_type=col.data_type,
                is_nullable=col.nullable,
                is_primary_key=col.primary_key,
            )
        except Exception as e:
            logger.warning("列 %s.%s 描述生成失败: %s", table.name, col.name, e)
            return None

    @staticmethod
    def _parse_description_response(response: str) -> Optional[str]:
        text = _clean_think(response)
        if len(text) < 2:
            return None
        return text

    # ============================================================
    # K5: Table Semantics（移植 table_analysis.py）
    # ============================================================

    def _build_tables(self, context: DatabaseContext, schema: SchemaMetadata,
                      domain: DomainKnowledge,
                      field_types: list[FieldClassification],
                      columns: list[ColumnSemantics]) -> list[TableSemantics]:
        field_type_index = {f.field_key: f.category for f in field_types}
        col_descs = {c.field_key: c.description for c in columns}

        semantics: list[TableSemantics] = []
        for table in schema.tables:
            desc = self._describe_table(context, table, field_type_index,
                                        col_descs, domain)
            if desc is not None:
                semantics.append(desc)
        return semantics

    def _describe_table(self, context: DatabaseContext, table: TableInfo,
                        field_type_index: dict[str, FieldCategory],
                        col_descs: dict[str, str],
                        domain: DomainKnowledge) -> Optional[TableSemantics]:
        try:
            columns = [
                self._column_view(c, table.name, field_type_index, col_descs)
                for c in table.columns
            ]
            category_stats = self._analyze_field_category_distribution(columns)
            representative_samples = self._extract_representative_samples(columns)
            business_pattern = self._infer_table_business_pattern(category_stats)
            entropy_stats = self._compute_entropy_stats(table)
            entropy_guidance = self._get_entropy_guidance(entropy_stats)
            table_ddl = self._build_table_ddl_with_comments(table.name, columns)

            pk_count = sum(1 for c in table.columns if c.primary_key)
            fk_count = sum(1 for c in table.columns if c.foreign_key)

            prompt = self.prompts.render(
                "tools/table_description.j2",
                table_name=table.name,
                database_name=context.database_id,
                database_domain=domain.description or "general domain",
                table_schema_with_comments_ddl=table_ddl,
                total_columns=len(columns),
                field_category_stats=category_stats,
                entropy_stats=entropy_stats,
                entropy_guidance=entropy_guidance,
                representative_samples=representative_samples,
                table_business_pattern=business_pattern,
                primary_key_count=pk_count,
                foreign_key_count=fk_count,
            )
            response = self.chat.generate(prompt)
            description = _clean_think(response)
            if len(description) < 2:
                return None

            return TableSemantics(
                table_name=table.name,
                description=description,
                business_type=self._infer_business_type(business_pattern),
                confidence=0.8,
                key_columns=[c["column_name"] for c in columns if c.get("is_primary")],
            )
        except Exception as e:
            logger.warning("表 %s 描述生成失败: %s", table.name, e)
            return None

    @staticmethod
    def _column_view(col: ColumnInfo, table_name: str,
                     field_type_index: dict[str, FieldCategory],
                     col_descs: dict[str, str]) -> dict:
        field_key = f"{table_name}.{col.name}"
        category = field_type_index.get(field_key)
        return {
            "column_name": col.name,
            "data_type": col.data_type,
            "is_primary": col.primary_key,
            "is_foreign": col.foreign_key,
            "is_nullable": col.nullable,
            "category": category.value if category else "other",
            "category_desc": category.value if category else "other",
            "ai_business_desc": col_descs.get(field_key, "") or (col.comment or ""),
            "sample_values": col.sample_values,
        }

    @staticmethod
    def _analyze_field_category_distribution(columns: list[dict]) -> dict:
        counts: Counter = Counter()
        examples: dict[str, list[str]] = {}
        for col in columns:
            cat = col.get("category", "other")
            counts[cat] += 1
            examples.setdefault(cat, [])
            if len(examples[cat]) < 3:
                examples[cat].append(col.get("column_name", ""))
        total = len(columns) or 1
        stats = {}
        for cat, n in counts.items():
            stats[cat] = {
                "category_desc": cat,
                "count": n,
                "percentage": round(n / total * 100, 1),
                "example_fields": examples[cat],
            }
        return stats

    @staticmethod
    def _extract_representative_samples(columns: list[dict]) -> list[dict]:
        samples = []
        for col in columns:
            values = col.get("sample_values", [])
            if not values:
                continue
            formatted = []
            for v in values[:3]:
                if v is None:
                    continue
                s = str(v)
                if len(s) > 20:
                    s = s[:17] + "..."
                formatted.append(s)
            if formatted:
                samples.append({
                    "field_name": col.get("column_name", ""),
                    "category_desc": col.get("category_desc", "other"),
                    "samples": ", ".join(formatted),
                })
        return samples

    @staticmethod
    def _compute_entropy_stats(table: TableInfo) -> dict:
        total = len(table.columns) or 1
        counts = {"low": 0, "medium": 0, "high": 0}
        for col in table.columns:
            level = getattr(col, "entropy_level", "medium")
            counts[level if level in counts else "medium"] += 1
        return {
            "low_count": counts["low"],
            "medium_count": counts["medium"],
            "high_count": counts["high"],
            "low_percentage": round(counts["low"] / total * 100, 1),
            "medium_percentage": round(counts["medium"] / total * 100, 1),
            "high_percentage": round(counts["high"] / total * 100, 1),
        }

    @staticmethod
    def _get_entropy_guidance(entropy_stats: dict) -> str:
        low = entropy_stats.get("low_percentage", 0)
        high = entropy_stats.get("high_percentage", 0)
        medium = entropy_stats.get("medium_percentage", 0)
        if low > 60:
            return "数据重复度高，主要为状态、类型、等级等枚举类信息"
        if high > 40:
            return "数据多样性强，包含大量标识符、名称、金额等独特值"
        if medium > 50:
            return "数据分散适中，平衡了分类信息和个性化数据"
        return "数据特征混合，包含多种类型的业务信息"

    @staticmethod
    def _infer_table_business_pattern(category_stats: dict) -> str:
        ident_count = category_stats.get("identifier", {}).get("count", 0)
        dim_count = category_stats.get("dimension", {}).get("count", 0)
        measure_count = category_stats.get("measure", {}).get("count", 0)
        if dim_count >= 3 and ident_count == 0:
            return "字典码表类型 - 主要存储枚举和分类信息"
        if ident_count >= 2:
            return "主数据表类型 - 存储核心业务实体信息"
        if measure_count > 0 and dim_count >= 3:
            return "事实表类型 - 记录业务交易和度量数据"
        if dim_count >= 3:
            return "维度表类型 - 提供业务分析的分类维度"
        return "通用业务表 - 支持日常业务操作"

    @staticmethod
    def _infer_business_type(business_pattern: str) -> str:
        if "字典码表" in business_pattern or "枚举" in business_pattern:
            return "config"
        if "主数据表" in business_pattern or "核心业务实体" in business_pattern:
            return "entity"
        if "事实表" in business_pattern:
            return "data_table"
        if "维度表" in business_pattern:
            return "dimension"
        return "data_table"

    @staticmethod
    def _build_table_ddl_with_comments(table_name: str, columns: list[dict]) -> str:
        lines = [f"CREATE TABLE `{table_name}` ("]
        col_defs = []
        for col in columns:
            d = f"  `{col['column_name']}` {col.get('data_type') or 'VARCHAR(255)'}"
            if not col.get("is_nullable", True):
                d += " NOT NULL"
            if col.get("is_primary"):
                d += " PRIMARY KEY"
            comment_parts = []
            ai_desc = col.get("ai_business_desc", "")
            if ai_desc:
                comment_parts.append(ai_desc)
            cat_desc = col.get("category_desc", "")
            if cat_desc:
                comment_parts.append(f"[{cat_desc}]")
            values = col.get("sample_values", [])
            if values:
                formatted = []
                for v in values[:3]:
                    if v is None:
                        continue
                    s = str(v)
                    if len(s) > 15:
                        s = s[:12] + "..."
                    formatted.append(s)
                if formatted:
                    comment_parts.append(f"[样本:{','.join(formatted)}]")
            if comment_parts:
                d += f" COMMENT '{' '.join(comment_parts)}'"
            col_defs.append(d)
        lines.append(",\n".join(col_defs))
        lines.append(");")
        return "\n".join(lines)

    # ============================================================
    # K6: Cross-table Relations（移植 er_analysis.py）
    # ============================================================

    def _build_relations(self, context: DatabaseContext, schema: SchemaMetadata,
                         domain: DomainKnowledge,
                         columns: list[ColumnSemantics],
                         table_semantics: list[TableSemantics]) -> list[CrossTableRelation]:
        # 聚合 K1~K5 上下文（与旧 _gather_database_context 一致）
        col_descs = {c.field_key: c.description for c in columns}
        t_descs = {t.table_name: t.description for t in table_semantics}
        tables: dict[str, dict] = {}
        for table in schema.tables:
            tables[table.name] = {
                "name": table.name,
                "description": t_descs.get(table.name, "") or (table.comment or ""),
                "columns": [],
            }
            for col in table.columns:
                tables[table.name]["columns"].append({
                    "name": col.name,
                    "data_type": col.data_type or "unknown",
                    "comment": col_descs.get(f"{table.name}.{col.name}", "")
                               or (col.comment or ""),
                    "is_primary_key": col.primary_key,
                    "is_foreign": col.foreign_key,
                    "sample_values": col.sample_values,
                })

        # LLM 概念 ER 分析：实体属性带具体 (表,列)，因此可落成候选可执行关系
        er_result = self._perform_er_analysis(context, tables, domain, schema)

        # 可执行 K6：K1 声明外键 + 保守命名推断 + LLM ER 候选（按顺序去重，
        # 声明外键优先）；无 PK/FK 声明的库（spider2 多为无约束 DDL）因此不再空 K6
        relations = self._to_cross_table_relations(schema, tables)
        merged = self._merge_er_relations(relations, er_result, tables)
        if len(merged) > len(relations):
            logger.info("K6: ER 分析补充 %d 条候选关系（共 %d）",
                        len(merged) - len(relations), len(merged))
        return merged

    @staticmethod
    def _merge_er_relations(relations: list[CrossTableRelation],
                            er_result: dict,
                            tables: dict) -> list[CrossTableRelation]:
        """把 LLM ER 的实体间关系落成候选 JOIN 关系（不覆盖声明外键）。

        实体属性里带具体 (table, column)，因此关系可还原为列级 JOIN 候选；
        confidence 0.5、reason 标注来源，供 check_joins 的"覆盖"判据与提示词使用。
        """
        merged = list(relations)
        seen = {(r.source_table, r.source_column, r.target_table, r.target_column)
                for r in merged}
        er = (er_result or {}).get("er_relation") or {}
        entities = {e.get("name"): e for e in (er.get("entities") or [])
                    if e.get("name")}
        if not entities:
            return merged

        def _cols(entity: dict) -> list[tuple[str, str]]:
            out: list[tuple[str, str]] = []
            for attr in entity.get("attributes", []):
                t, c = attr.get("table"), attr.get("column")
                if not t or not c or t not in tables:
                    continue
                if c not in {x.get("name") for x in tables[t].get("columns", [])}:
                    continue
                out.append((t, c))
            return out

        type_map = {"ONE_TO_ONE": "many_to_one",
                    "MANY_TO_ONE": "many_to_one",
                    "ONE_TO_MANY": "one_to_many",
                    "MANY_TO_MANY": "many_to_many"}

        added = 0
        for rel in er.get("inter_entity_relations", []):
            if added >= _ER_RELATION_CAP:
                break
            a = entities.get(rel.get("from_entity"))
            b = entities.get(rel.get("to_entity"))
            if not a or not b:
                continue
            ca, cb = _cols(a), _cols(b)
            if not ca or not cb:
                continue
            for (t1, c1), (t2, c2) in KnowledgeBuilder._pair_er_columns(ca, cb):
                if t1 == t2 and c1 == c2:
                    continue
                key = (t1, c1, t2, c2)
                if key in seen:
                    continue
                seen.add(key)
                rtype = type_map.get(str(rel.get("relation_type", "")).upper(),
                                     "many_to_one")
                meaning = str(rel.get("relation_name")
                              or rel.get("business_meaning") or "")[:60]
                merged.append(CrossTableRelation(
                    source_table=t1, source_column=c1,
                    target_table=t2, target_column=c2,
                    relationship_type=rtype, confidence=0.5,
                    reason=f"LLM ER relation: {meaning}" if meaning else "LLM ER relation",
                ))
                added += 1
                if added >= _ER_RELATION_CAP:
                    break
        return merged

    @staticmethod
    def _pair_er_columns(ca: list[tuple], cb: list[tuple]) -> list[tuple]:
        """实体对之间的列配对：同名优先，其次 *_id，最后首列（最多 3 对）。"""
        same = [(x, y) for x in ca for y in cb
                if x[1].lower() == y[1].lower()]
        if same:
            return same[:3]
        ids_a = [x for x in ca if x[1].lower().endswith("id")]
        ids_b = [y for y in cb if y[1].lower().endswith("id")]
        if ids_a and ids_b:
            return [(ids_a[0], ids_b[0])]
        return [(ca[0], cb[0])]

    @staticmethod
    def _to_cross_table_relations(schema: SchemaMetadata,
                                  tables: dict) -> list[CrossTableRelation]:
        """可执行 K6 关系（按可靠性排序）：
        1. K1 声明外键（confidence 1.0）
        2. 保守命名推断 *_id/*id（confidence 0.5，要求目标列是主键）
        LLM 概念关系由 _merge_er_relations 另行补充（实体属性带列，可还原）。
        """

    def _perform_er_analysis(self, context: DatabaseContext, tables: dict,
                             domain: DomainKnowledge, schema: SchemaMetadata) -> dict:
        prompt_data = {
            "formatted_schema": self._format_schema_with_descriptions(tables),
            "fk_info": self._format_foreign_key_info(tables),
            "tables": tables,
            "database_name": context.database_id,
        }
        prompt = self.prompts.render("tools/er_analysis_conceptual.j2", **prompt_data)
        response = self.chat.generate_json(prompt)
        return self._validate_er_analysis(response, tables)

    @staticmethod
    def _format_schema_with_descriptions(tables: dict) -> str:
        lines = ["数据库表结构（包含业务注释）："]
        for table_name, table_info in tables.items():
            lines.append(f"\n表: {table_name}")
            if table_info.get("description"):
                lines.append(f"  注释: {table_info['description']}")
            lines.append("  列:")
            for column in table_info["columns"]:
                info = f"    - {column['name']} ({column.get('data_type', 'unknown')})"
                if column.get("is_primary_key"):
                    info += " [主键]"
                if column.get("is_foreign"):
                    info += " [外键]"
                if column.get("comment"):
                    info += f" -- {column['comment']}"
                lines.append(info)
        return "\n".join(lines)

    @staticmethod
    def _format_foreign_key_info(tables: dict) -> str:
        lines = ["外键关系信息："]
        fk_relations = []
        for table_name, table_info in tables.items():
            for column in table_info.get("columns", []):
                col_name = column.get("name", "")
                ref_table = None
                if col_name.endswith("_id"):
                    ref_table = col_name[:-3]
                elif col_name.endswith("id") and len(col_name) > 2:
                    ref_table = col_name[:-2]
                if ref_table and (ref_table in tables or f"{ref_table}s" in tables):
                    target = ref_table if ref_table in tables else f"{ref_table}s"
                    fk_relations.append(
                        f"  {table_name}.{col_name} -> {target}.id (推测外键)")
        if fk_relations:
            lines.extend(fk_relations)
        else:
            lines.append("  未发现明显的外键关系")
        return "\n".join(lines)

    @staticmethod
    def _validate_er_analysis(er_analysis: dict, tables: dict) -> dict:
        if "business_domain" not in er_analysis:
            raise RuntimeError("K6: LLM 响应缺少 business_domain 字段")
        if "er_relation" not in er_analysis:
            raise RuntimeError("K6: LLM 响应缺少 er_relation 字段")

        er_relation = er_analysis["er_relation"]
        er_relation.setdefault("entities", [])
        er_relation.setdefault("inter_entity_relations", [])

        table_columns = {
            t: {c["name"] for c in info["columns"]} for t, info in tables.items()
        }
        valid_entities = []
        for entity in er_relation["entities"]:
            if "name" not in entity:
                continue
            valid_attrs = []
            for attr in entity.get("attributes", []):
                if not all(k in attr for k in ("column", "table")):
                    continue
                if attr["table"] not in table_columns:
                    continue
                if attr["column"] not in table_columns[attr["table"]]:
                    continue
                valid_attrs.append(attr)
            entity["attributes"] = valid_attrs
            if valid_attrs:
                valid_entities.append(entity)
        er_relation["entities"] = valid_entities

        entity_names = {e["name"] for e in valid_entities}
        valid_relations = []
        for rel in er_relation["inter_entity_relations"]:
            if not all(k in rel for k in ("from_entity", "to_entity", "relation_type")):
                continue
            if rel["from_entity"] not in entity_names or rel["to_entity"] not in entity_names:
                continue
            valid_relations.append(rel)
        er_relation["inter_entity_relations"] = valid_relations
        return er_analysis

    @staticmethod
    def _to_cross_table_relations(schema: SchemaMetadata,
                                  tables: dict) -> list[CrossTableRelation]:
        """可执行 K6 关系（按可靠性排序）：
        1. K1 声明外键（confidence 1.0）
        2. 保守命名推断 *_id/*id（confidence 0.5，要求目标列是主键）
        LLM 概念关系无具体列，不能作为可执行 JOIN 约束（与旧实现一致）。
        """
        relations: list[CrossTableRelation] = []
        seen: set[tuple] = set()
        table_names_lower = {t.lower(): t for t in tables}

        # 1. K1 声明外键
        for table in schema.tables:
            for col in table.columns:
                if not (col.foreign_key and col.foreign_key_table
                        and col.foreign_key_column):
                    continue
                if table.name.lower() == col.foreign_key_table.lower():
                    continue
                key = (table.name, col.name,
                       col.foreign_key_table, col.foreign_key_column)
                if key in seen:
                    continue
                seen.add(key)
                relations.append(CrossTableRelation(
                    source_table=table.name,
                    source_column=col.name,
                    target_table=col.foreign_key_table,
                    target_column=col.foreign_key_column,
                    relationship_type="many_to_one",
                    confidence=1.0,
                    reason="SQLite declared foreign key",
                ))

        # 2. 保守命名推断
        primary_keys = {
            table.name.lower(): {c.lower() for c in table.primary_keys}
            for table in schema.tables
        }
        for table_name, table_info in tables.items():
            for column in table_info.get("columns", []):
                col_name = column.get("name", "")
                tgt = KnowledgeBuilder._find_fk_target(col_name, table_names_lower)
                if not tgt or table_name.lower() == tgt[0].lower():
                    continue
                target_pks = primary_keys.get(tgt[0].lower(), set())
                if tgt[1].lower() not in target_pks:
                    continue
                key = (table_name, col_name, tgt[0], tgt[1])
                if key in seen:
                    continue
                seen.add(key)
                relations.append(CrossTableRelation(
                    source_table=table_name,
                    source_column=col_name,
                    target_table=tgt[0],
                    target_column=tgt[1],
                    relationship_type="many_to_one",
                    confidence=0.5,
                    reason="conservative naming inference",
                ))
        return relations

    @staticmethod
    def _find_fk_target(col_name: str, table_names_lower: dict) -> Optional[tuple]:
        low = col_name.lower()
        ref = None
        if low.endswith("_id"):
            ref = low[:-3]
        elif low.endswith("id") and len(low) > 2:
            ref = low[:-2]
        if not ref:
            return None
        if ref in table_names_lower:
            return (table_names_lower[ref], col_name)
        if ref + "s" in table_names_lower:
            return (table_names_lower[ref + "s"], col_name)
        if ref.rstrip("s") in table_names_lower:
            return (table_names_lower[ref.rstrip("s")], col_name)
        return None


def _utc_stamp() -> str:
    import time
    return time.strftime("%Y%m%d-%H%M%S")
