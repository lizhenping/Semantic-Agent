"""知识构建用例（app/application/analysis_service.py）

DatabaseContext -> KnowledgeSnapshot（论文 Phase 1 / DA）。
快照持久化到 knowledge/snapshots/<benchmark>/<database>/<snapshot_id>/。
知识不可用时任务级失败（不得生成没有数据库身份的样本）。
"""

import logging
from pathlib import Path
from typing import Optional

from app.contracts.database import DatabaseContext
from app.contracts.knowledge import JoinProfile, KnowledgeSnapshot
from app.domain.data_profile import DataProfiler
from app.domain.knowledge_builder import KnowledgeBuilder
from app.ports.knowledge import KnowledgeStore

logger = logging.getLogger(__name__)


class AnalysisService:
    """单数据库知识构建用例。"""

    def __init__(self, builder: KnowledgeBuilder, snapshot_repository,
                 legacy_store: Optional[KnowledgeStore] = None,
                 profiler: Optional[DataProfiler] = None):
        self.builder = builder
        self.snapshot_repository = snapshot_repository  # KnowledgeSnapshotRepository
        self.legacy_store = legacy_store                # 可选：旧布局 K 层 JSONL
        self.profiler = profiler                        # 可选：实测 JOIN 画像

    def analyze(self, context: DatabaseContext) -> KnowledgeSnapshot:
        """构建 K1..K6 并发布快照。"""
        return self.builder.build(context, store=self.legacy_store)

    def publish(self, context: DatabaseContext,
                snapshot: KnowledgeSnapshot, benchmark: str = "") -> Path:
        """快照落盘到 knowledge/snapshots/，返回快照目录。"""
        bench = benchmark or context.database_id.split("/")[0]
        out = self.snapshot_repository.save(snapshot)
        logger.info("知识快照已发布: %s", out)
        return out

    def build_profile(self, context: DatabaseContext,
                      snapshot: KnowledgeSnapshot) -> list[JoinProfile]:
        """实测 JOIN 画像（Phase 1 之后的一次性旁路探测）并写侧车文件。

        探测失败不影响知识构建：返回空列表，消费方自动降级。
        """
        if self.profiler is None:
            return []
        bench = context.database_id.split("/")[0]
        db = context.database_id.split("/")[-1]
        try:
            profiles = self.profiler.build(snapshot, snapshot.k1_schema)
        except Exception as e:  # noqa: BLE001
            logger.warning("数据画像探测失败（跳过，不影响生成）: %s", e)
            return []
        if profiles:
            self.snapshot_repository.save_profile(
                bench, db, snapshot.snapshot_id,
                [p.model_dump() for p in profiles])
            logger.info("数据画像已写入: %s/%s（%d 条 JOIN 候选）",
                        bench, db, len(profiles))
        return profiles

    def load_profile(self, benchmark: str, database: str,
                     snapshot_id: Optional[str] = None) -> list[JoinProfile]:
        """读侧车画像（缺失 → 空列表，功能自动降级）。"""
        try:
            raw = self.snapshot_repository.load_profile(benchmark, database,
                                                        snapshot_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("画像读取失败: %s", e)
            return []
        out: list[JoinProfile] = []
        for rec in raw:
            try:
                out.append(JoinProfile(**rec))
            except Exception:  # noqa: BLE001
                continue
        return out

    def load_latest(self, benchmark: str, database: str) -> Optional[KnowledgeSnapshot]:
        """加载最新快照（--reuse-knowledge 用）。"""
        return self.snapshot_repository.load_latest(benchmark, database)
