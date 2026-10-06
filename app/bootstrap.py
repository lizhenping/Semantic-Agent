"""装配根（app/bootstrap.py）

唯一的依赖组装点：CLI / 测试都从这里取完整装配的服务束。
不在此之外创建 LLM 客户端、数据库连接或渲染器（doc/02 §1）。

PipelineBundle 把单库运行所需的服务收拢在一起：
synthesis 与 reflection 共享同一知识快照、同一数据库上下文和同一
RunRecordStore（生成与反思必须使用同一知识快照）。
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.adapters.database_sqlalchemy import SQLAlchemyDatabaseGateway
from app.adapters.llm_openai import ChatOpenAIGateway, FakeChatGateway
from app.adapters.prompt_jinja import JinjaPromptRenderer
from app.adapters.scenario_catalog import YamlScenarioCatalog
from app.adapters.storage_jsonl import (
    JsonlKnowledgeStore,
    JsonlRunRecordStore,
    KnowledgeSnapshotRepository,
)
from app.adapters.sqlglot_parser import SqlglotParser
from app.application.analysis_service import AnalysisService
from app.application.batch_service import BatchService
from app.application.reflection_service import ReflectionService
from app.application.synthesis_service import SynthesisService
from app.contracts.database import DatabaseContext, DatabaseDialect, DatabaseSource
from app.contracts.knowledge import KnowledgeSnapshot
from app.domain.candidate_generation import QuestionGenerator, SQLGenerator
from app.domain.correction import Corrector
from app.domain.data_profile import DataProfiler
from app.domain.diagnosis import SemanticReviewer
from app.domain.evidence import EvidenceRetriever
from app.domain.knowledge_builder import KnowledgeBuilder
from app.domain.verification import DeterministicVerifier
from app.ports.llm import ChatGateway
from app.settings import Settings


@dataclass
class PipelineBundle:
    """单数据库运行束：快照 + 上下文 + 生成/反思服务 + 会话产物存储。"""

    snapshot: KnowledgeSnapshot
    context: DatabaseContext
    synthesis: SynthesisService
    reflection: ReflectionService
    run_store: JsonlRunRecordStore


class Bootstrap:
    """应用装配根。"""

    def __init__(self, settings: Optional[Settings] = None,
                 project_root: Optional[str] = None):
        self.settings = settings or Settings()
        self.root = Path(project_root or Path(__file__).parent.parent)
        self._gateway: Optional[SQLAlchemyDatabaseGateway] = None

    # ---------- 基础设施 ----------

    @property
    def gateway(self) -> SQLAlchemyDatabaseGateway:
        if self._gateway is None:
            self._gateway = SQLAlchemyDatabaseGateway(
                sql_timeout=self.settings.sql_timeout)
        return self._gateway

    def chat(self) -> ChatGateway:
        if self.settings.llm_fake:
            return FakeChatGateway()
        return ChatOpenAIGateway(
            model=self.settings.llm_model,
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            timeout=self.settings.llm_timeout,
            max_retries=self.settings.llm_max_retries,
            enable_thinking=self.settings.llm_enable_thinking,
        )

    def prompts(self) -> JinjaPromptRenderer:
        d = Path(self.settings.templates_dir)
        if not d.exists():
            d = self.root / self.settings.templates_dir
        return JinjaPromptRenderer(template_dir=str(d))

    def scenario_catalog(self, seed: Optional[int] = None) -> YamlScenarioCatalog:
        d = Path(self.settings.configs_dir)
        if not d.exists():
            d = self.root / self.settings.configs_dir
        return YamlScenarioCatalog(config_dir=str(d),
                                   rng=random.Random(seed))

    def snapshot_repository(self) -> KnowledgeSnapshotRepository:
        return KnowledgeSnapshotRepository(self._abs(self.settings.knowledge_root))

    def run_store(self, benchmark: str, database: str,
                  session_id: Optional[str] = None) -> JsonlRunRecordStore:
        return JsonlRunRecordStore(
            self._abs(self.settings.runs_root), benchmark, database,
            session_id=session_id)

    def legacy_store(self, benchmark: str, database: str) -> JsonlKnowledgeStore:
        """旧布局 K 层存储（history/<benchmark>/<database>，回归对比用）。"""
        return JsonlKnowledgeStore(
            self._abs(self.settings.runs_root) / "_legacy" / benchmark / database)

    def _abs(self, p: str | Path) -> Path:
        path = Path(p)
        return path if path.is_absolute() else self.root / path

    # ---------- 数据库来源 ----------

    def sqlite_source(self, sqlite_path: str, database_name: Optional[str] = None,
                      benchmark: str = "") -> DatabaseSource:
        name = database_name or Path(sqlite_path).stem
        return DatabaseSource(
            database_id=f"{benchmark}/{name}" if benchmark else name,
            dialect=DatabaseDialect.SQLITE,
            benchmark=benchmark,
            sqlite_path=str(sqlite_path),
        )

    # ---------- 用例服务 ----------

    def analysis_service(self, benchmark: str = "",
                         database: str = "") -> AnalysisService:
        chat = self.chat()
        builder = KnowledgeBuilder(
            gateway=self.gateway, chat=chat, prompts=self.prompts(),
            builder_version=self.settings.builder_version)
        legacy = (self.legacy_store(benchmark, database)
                  if benchmark and database else None)
        return AnalysisService(builder, self.snapshot_repository(),
                               legacy_store=legacy,
                               profiler=DataProfiler(self.gateway, None))

    def _profiles(self, benchmark: str, database: str,
                  snapshot: KnowledgeSnapshot) -> list:
        """加载侧车实测画像（缺失/无库身份 → 空，功能自动降级）。"""
        if not (benchmark and database):
            return []
        return self.analysis_service(benchmark, database).load_profile(
            benchmark, database, snapshot.snapshot_id)

    def _verifier(self, context: DatabaseContext,
                  profiles: Optional[list] = None) -> DeterministicVerifier:
        return DeterministicVerifier(gateway=self.gateway, context=context,
                                     rows_limit=self.settings.result_rows_limit,
                                     row_cap=self.settings.execution_row_cap,
                                     profiles=profiles)

    def synthesis_service(self, context: DatabaseContext,
                          snapshot: KnowledgeSnapshot,
                          benchmark: str = "", database: str = "",
                          seed: Optional[int] = None) -> SynthesisService:
        chat = self.chat()
        prompts = self.prompts()
        catalog = self.scenario_catalog(seed=seed)
        profiles = self._profiles(benchmark, database, snapshot)
        q_gen = QuestionGenerator(
            chat=chat, prompts=prompts, catalog=catalog,
            gateway=self.gateway, context=context,
            domain_scan_timeout=self.settings.domain_scan_timeout,
            workers=self.settings.synthesis_workers,
            table_cap=self.settings.context_table_cap)
        s_gen = SQLGenerator(
            chat=chat, prompts=prompts,
            gateway=self.gateway, context=context,
            dialect=context.dialect.value,
            include_rationale_hints=self.settings.include_rationale_hints,
            rows_limit=self.settings.result_rows_limit,
            row_cap=self.settings.execution_row_cap,
            profiles=profiles)
        run_store = self.run_store(benchmark, database)
        legacy = self.legacy_store(benchmark, database)
        return SynthesisService(snapshot, context, q_gen, s_gen,
                                run_store=run_store, legacy_store=legacy,
                                workers=self.settings.synthesis_workers,
                                empty_result_retries=self.settings.empty_result_retries)

    def reflection_service(self, context: DatabaseContext,
                           snapshot: KnowledgeSnapshot,
                           benchmark: str = "", database: str = "",
                           session_id: Optional[str] = None) -> ReflectionService:
        chat = self.chat()
        prompts = self.prompts()
        return ReflectionService(
            snapshot=snapshot, context=context,
            verifier=self._verifier(
                context, profiles=self._profiles(benchmark, database, snapshot)),
            reviewer=SemanticReviewer(chat=chat, prompts=prompts,
                                      enabled=self.settings.enable_llm_review),
            retriever=EvidenceRetriever(),
            corrector=Corrector(chat=chat, prompts=prompts,
                                gateway=self.gateway, context=context,
                                dialect=context.dialect.value,
                                rows_limit=self.settings.result_rows_limit,
                                row_cap=self.settings.execution_row_cap),
            parser=SqlglotParser(dialect=context.dialect.value),
            run_store=self.run_store(benchmark, database,
                                     session_id=session_id),
            workers=self.settings.synthesis_workers,
        )

    def pipeline_bundle(self, context: DatabaseContext,
                        snapshot: KnowledgeSnapshot,
                        benchmark: str = "", database: str = "",
                        seed: Optional[int] = None,
                        session_id: Optional[str] = None) -> PipelineBundle:
        """生成与反思共享同一快照/上下文/会话存储。"""
        synthesis = self.synthesis_service(context, snapshot, benchmark,
                                           database, seed=seed)
        # reflection 复用 synthesis 的会话存储（同一 session_id）
        reflection = self.reflection_service(
            context, snapshot, benchmark, database,
            session_id=synthesis.run_store.session_id)
        run_store = synthesis.run_store
        # 会话目录自包含 K 层（设计 §6：runs/<...>/<session>/knowledge/）
        run_store.write_knowledge_layer("k1_schema", [snapshot.k1_schema.model_dump()])
        run_store.write_knowledge_layer("k2_domain", [snapshot.k2_domain.model_dump()])
        run_store.write_knowledge_layer("k3_field",
                                        [f.model_dump() for f in snapshot.k3_field_types])
        run_store.write_knowledge_layer("k4_column",
                                        [c.model_dump() for c in snapshot.k4_column_semantics])
        run_store.write_knowledge_layer("k5_table",
                                        [t.model_dump() for t in snapshot.k5_table_semantics])
        run_store.write_knowledge_layer("k6_relations",
                                        [r.model_dump() for r in snapshot.k6_relations])
        run_store.write_manifest({
            "code_version": self.settings.code_version,
            "database": context.source.redacted_params(),
            "schema_digest": context.snapshot.schema_digest,
            "knowledge_snapshot": {
                "snapshot_id": snapshot.snapshot_id,
                "content_digest": snapshot.content_digest(),
                "model_id": snapshot.model_id,
                "builder_version": snapshot.builder_version,
                "prompt_versions": snapshot.prompt_versions,
            },
            "reflection": {
                "max_correction_iterations": self.settings.max_correction_iterations,
                "llm_review_enabled": self.settings.enable_llm_review,
            },
        })
        return PipelineBundle(snapshot=snapshot, context=context,
                              synthesis=synthesis, reflection=reflection,
                              run_store=run_store)

    def batch_service(self) -> BatchService:
        from app.adapters.dataset_split import list_train_databases

        def pipeline_factory(benchmark: str, db_name: str,
                             sqlite_path: str) -> PipelineBundle:
            source = self.sqlite_source(sqlite_path, db_name, benchmark)
            context = self.gateway.open(source)
            analysis = self.analysis_service(benchmark, db_name)
            snapshot = analysis.load_latest(benchmark, db_name)
            if snapshot is None:
                snapshot = analysis.analyze(context)
                analysis.publish(context, snapshot, benchmark=benchmark)
            return self.pipeline_bundle(context, snapshot, benchmark, db_name)

        return BatchService(
            self.settings, self.gateway, pipeline_factory,
            train_db_lister=lambda b: list_train_databases(b, self.settings.data_root))
