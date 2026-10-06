"""反思用例（app/application/reflection_service.py）

论文 §III.E Eq.4 循环（Phase 3 / DT）：
    E^(k) = Diagnose(τ^(k), K, D)
    (Ẽ^(k), H^(k)) = Retrieve(τ^(k), B^(k), E^(k), K)
    (s^(k+1), r^(k+1)) = Correct(τ^(k), B^(k), Ẽ^(k), H^(k), K)
    重复直到无诊断或达 max_iters

状态机（doc/02 §3）：
    GENERATED -> DATABASE_CHECKED -> DIAGNOSED -> EVIDENCE_REVIEWED
      -> ACCEPTED / REJECTED(question invalid) / CORRECTED -> 复检
      -> UNRESOLVED（预算耗尽仍有诊断）

移植自旧 core/pipeline.py 的 run_diagnosis/_eq4_loop；行为变化：
- 解析失败显式产出 SYNTAX_ERROR（旧实现静默当"无结构错误"）
- Correct 消费复核后诊断（retain_all_supported 策略下与旧行为等价）
- 编排异常写成可审计拒绝记录（不静默丢样本）
"""

import logging
import threading

from app.contracts.candidate import CandidateSample
from app.contracts.database import DatabaseContext
from app.contracts.diagnosis import (
    DiagnosisIteration,
    Diagnosis,
    ErrorLocation,
    ErrorType,
    EvidenceBundle,
    ReflectionTrace,
    SampleDecision,
)
from app.contracts.knowledge import KnowledgeSnapshot
from app.contracts.execution import AdmissionDecision, is_execution_timeout
from app.domain.admission import (
    decide,
    has_review_failure,
    is_question_invalid,
)
from app.ports.parsing import SqlParseError
from app.ports.records import RunRecordStore

logger = logging.getLogger(__name__)


class ReflectionService:
    """Diagnose -> Retrieve -> Correct 反思循环和准入。"""

    def __init__(
        self,
        snapshot: KnowledgeSnapshot,
        context: DatabaseContext,
        verifier,      # domain.verification.DeterministicVerifier
        reviewer,      # domain.diagnosis.SemanticReviewer
        retriever,     # domain.evidence.EvidenceRetriever
        corrector,     # domain.correction.Corrector
        parser,        # ports.parsing.SqlParser
        run_store: RunRecordStore | None = None,
        workers: int = 10,
    ):
        self.snapshot = snapshot
        self.context = context
        self.verifier = verifier
        self.reviewer = reviewer
        self.retriever = retriever
        self.corrector = corrector
        self.parser = parser
        self.run_store = run_store
        # 候选间并发度（Eq.4 循环相互独立；落盘在全部完成后单线程执行）
        self.workers = max(1, int(workers))
        # 逐条落盘锁：每条候选裁决完成立即追加（崩溃不丢已完成工作）
        self._persist_lock = threading.Lock()

    # ============================================================
    # 批量入口
    # ============================================================

    def refine(self, samples: list[CandidateSample],
               max_iters: int = 3) -> tuple[list[CandidateSample], list[ReflectionTrace]]:
        """对每个候选跑 Eq.4 循环。

        单候选内串行（保证 s^k 顺序清楚）；候选之间相互独立，
        workers>1 时线程池并发反思，结果保持输入顺序，落盘在全部
        完成后单线程进行（JSONL 追加写不做跨线程并发）。
        """
        if self.workers > 1 and len(samples) > 1:
            refined, traces = self._refine_parallel(samples, max_iters)
        else:
            refined, traces = self._refine_serial(samples, max_iters)

        ok = sum(1 for s in refined if s.sql_result.execution_success)
        accepted = sum(1 for s in refined
                       if s.rationale.admission_decision == "accepted")
        logger.info("✅ 反思完成：%d/%d 准入（%d 条 SQL 执行通过）",
                    accepted, len(refined), ok)
        return refined, traces

    def _refine_serial(self, samples, max_iters):
        refined: list[CandidateSample] = []
        traces: list[ReflectionTrace] = []
        for i, sample in enumerate(samples, 1):
            try:
                result, trace = self._eq4_loop(sample, max_iters)
                refined.append(result)
                traces.append(trace)
                self._persist_one(result, trace)   # 裁决完成立即落盘
                logger.info("  [%d/%d] %s: %s %d 轮修正",
                            i, len(samples), sample.question_id,
                            "✅" if result.sql_result.execution_success else "⚠️",
                            len(result.rationale.correction_history))
            except Exception as e:  # noqa: BLE001
                logger.warning("  [%d/%d] %s 反思异常: %s",
                               i, len(samples), sample.question_id, e)
                failed, failed_trace = (self._failed_sample(sample, str(e)),
                                        self._failed_trace(sample, max_iters, str(e)))
                refined.append(failed)
                traces.append(failed_trace)
                self._persist_one(failed, failed_trace)
        return refined, traces

    def _refine_parallel(self, samples, max_iters):
        from concurrent.futures import ThreadPoolExecutor

        def _one(i: int, sample: CandidateSample):
            try:
                result, trace = self._eq4_loop(sample, max_iters)
                self._persist_one(result, trace)   # 裁决完成立即落盘
                logger.info("  [%d/%d] %s: %s %d 轮修正",
                            i, len(samples), sample.question_id,
                            "✅" if result.sql_result.execution_success else "⚠️",
                            len(result.rationale.correction_history))
                return result, trace
            except Exception as e:  # noqa: BLE001
                logger.warning("  [%d/%d] %s 反思异常: %s",
                               i, len(samples), sample.question_id, e)
                failed, failed_trace = (self._failed_sample(sample, str(e)),
                                        self._failed_trace(sample, max_iters, str(e)))
                self._persist_one(failed, failed_trace)
                return failed, failed_trace

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(_one, i, s)
                       for i, s in enumerate(samples, 1)]
            pairs = [f.result() for f in futures]
        refined = [p[0] for p in pairs]
        traces = [p[1] for p in pairs]
        return refined, traces

    def _persist_one(self, sample: CandidateSample,
                     trace: ReflectionTrace) -> None:
        """单条候选落盘（reflection/candidates.final/training.accepted），线程安全。"""
        with self._persist_lock:
            if self.run_store is None:
                return
            self.run_store.append_reflection(
                [trace.model_dump(mode="json")])
            self.run_store.append_candidates([{
                "question_id": sample.question_id,
                "decision": trace.decision.value,
                "decision_reason": trace.decision_reason,
                "final_execution_success": trace.final_execution_success,
                "diagnosis_iterations": len(trace.iterations),
                "correction_rounds": len(sample.rationale.correction_history),
            }], stage="final")
            # training.accepted 只收 accepted（实现侧再校验）
            if trace.decision.value == "accepted":
                self.run_store.append_training([sample.to_training_record()])

    # ============================================================
    # 单候选 Eq.4 循环
    # ============================================================

    def _eq4_loop(self, sample: CandidateSample,
                  max_iters: int) -> tuple[CandidateSample, ReflectionTrace]:
        current = sample
        trace = ReflectionTrace(
            question_id=sample.question_id,
            original_question=sample.question.text,
            original_sql=sample.sql_result.sql,
            max_correction_iterations=max_iters,
        )
        terminal: list[Diagnosis] = []
        timed_out = False

        for k in range(1, max_iters + 1):
            # 用户规则：SQL 执行超时（settings.sql_timeout，默认30s）→
            # 放弃整个问题的合成，不进入修正循环（慢查询不会被 LLM 改写救回来）
            if is_execution_timeout(current.sql_result.execution_error):
                timed_out = True
                terminal = [Diagnosis(
                    type=ErrorType.EXECUTION_FAILED,
                    location=ErrorLocation(clause="EXECUTION"),
                    detail=current.sql_result.execution_error or "execution timeout",
                    artifact="s")]
                trace.iterations.append(DiagnosisIteration(
                    iteration=k, question=current.question.text,
                    sql_before=current.sql_result.sql,
                    rationale_focus=current.rationale.focus,
                    errors=terminal,
                    execution_success_after=False,
                    execution_error_after=current.sql_result.execution_error,
                    result_count_after=current.sql_result.result_count,
                    action="rejected_execution_timeout"))
                break

            errors = self._diagnose(current)
            if not errors:
                # 收敛：无诊断（论文 "until no errors are detected"）
                terminal = []
                trace.iterations.append(DiagnosisIteration(
                    iteration=k, question=current.question.text,
                    sql_before=current.sql_result.sql,
                    rationale_focus=current.rationale.focus, errors=[],
                    execution_success_after=current.sql_result.execution_success,
                    execution_error_after=current.sql_result.execution_error,
                    result_count_after=current.sql_result.result_count,
                    action="verified"))
                break

            # 审查失败不是可修复的语义发现（不等于"没有检测到错误"）
            if has_review_failure(errors):
                terminal = errors
                trace.iterations.append(DiagnosisIteration(
                    iteration=k, question=current.question.text,
                    sql_before=current.sql_result.sql,
                    rationale_focus=current.rationale.focus, errors=errors,
                    execution_success_after=current.sql_result.execution_success,
                    execution_error_after=current.sql_result.execution_error,
                    result_count_after=current.sql_result.result_count,
                    action="semantic_review_failed"))
                break

            # 问题无效不可修复：q 在反思中固定 → 直接拒绝
            if is_question_invalid(errors):
                terminal = [e for e in errors if (e.artifact or "") == "q"]
                trace.iterations.append(DiagnosisIteration(
                    iteration=k, question=current.question.text,
                    sql_before=current.sql_result.sql,
                    rationale_focus=current.rationale.focus, errors=terminal,
                    execution_success_after=current.sql_result.execution_success,
                    execution_error_after=current.sql_result.execution_error,
                    result_count_after=current.sql_result.result_count,
                    action="rejected_question_invalid"))
                break

            # Φ = Retrieve(E, K)：取证路由 + 诊断复核
            evidence = self.retriever.retrieve(errors, self.snapshot)
            reviewed = self.retriever.review(
                errors, current.sql_result, evidence)
            # (s', r') = Correct(...)：q 固定，只修 s/r
            before = current
            current = self.corrector.correct(
                current, reviewed.retained, evidence, iteration=k)
            trace.iterations.append(DiagnosisIteration(
                iteration=k, question=before.question.text,
                sql_before=before.sql_result.sql,
                rationale_focus=before.rationale.focus,
                errors=errors, evidence=evidence,
                sql_after=current.sql_result.sql,
                execution_success_after=current.sql_result.execution_success,
                execution_error_after=current.sql_result.execution_error,
                result_count_after=current.sql_result.result_count,
                action="corrected"))
            terminal = errors
        else:
            # 修正预算耗尽：必须对最终版本再跑一次完整验证，
            # 不能把"已修正"误写成"已通过"
            terminal = self._diagnose(current)
            trace.iterations.append(DiagnosisIteration(
                iteration=max_iters + 1, question=current.question.text,
                sql_before=current.sql_result.sql,
                rationale_focus=current.rationale.focus,
                errors=terminal,
                execution_success_after=current.sql_result.execution_success,
                execution_error_after=current.sql_result.execution_error,
                result_count_after=current.sql_result.result_count,
                action="max_iterations_reached"))

        # 终态裁决（三互斥终态）；超时样本固定 rejected/execution_timeout
        # 结果形态闸门：空结果 / 取回上限命中 / 问"所有"却被 LIMIT 截断
        decision, reason = decide(
            current.sql_result.execution_success, terminal,
            result_count=current.sql_result.result_count,
            rows_capped=current.sql_result.rows_capped,
            sql=current.sql_result.sql,
            question_text=current.question.text)
        if timed_out:
            decision, reason = AdmissionDecision.REJECTED, "execution_timeout"
        trace.final_question = current.question.text
        trace.final_sql = current.sql_result.sql
        trace.final_execution_success = current.sql_result.execution_success
        trace.final_execution_error = current.sql_result.execution_error
        trace.final_result_count = current.sql_result.result_count
        trace.decision = decision
        trace.decision_reason = reason
        current.rationale.admission_decision = decision.value
        current.rationale.admission_reason = reason
        return current, trace

    # ============================================================
    # 一次完整验证（程序化 + LLM 语义审查）
    # ============================================================

    def _diagnose(self, sample: CandidateSample) -> list[Diagnosis]:
        sql = sample.sql_result.sql
        if not sql:
            return [Diagnosis(type=ErrorType.SYNTAX_ERROR, detail="SQL 为空")]

        # 程序化校验先运行，为语义审查提供结构事实
        try:
            parsed = self.parser.parse(sql)
        except SqlParseError as e:
            # 风险修复 5：解析失败显式报错，绝不当作"无问题"
            return [Diagnosis(type=ErrorType.SYNTAX_ERROR,
                              detail=f"SQL 解析失败: {e.detail}")]

        errors = self.verifier.check(self.snapshot, parsed, sample.sql_result)

        # 不可执行 → B 类失败，不混入七类语义审查（与旧行为一致）
        if any(e.type == ErrorType.EXECUTION_FAILED for e in errors):
            return errors

        errors.extend(self.reviewer.review(sample, self.snapshot, parsed))
        return errors

    # ============================================================
    # 失败样本的可审计记录
    # ============================================================

    @staticmethod
    def _failed_sample(sample: CandidateSample, detail: str) -> CandidateSample:
        s = sample.model_copy(deep=True)
        s.rationale.admission_decision = "rejected"
        s.rationale.admission_reason = "reflection_pipeline_exception"
        return s

    @staticmethod
    def _failed_trace(sample: CandidateSample, max_iters: int,
                      detail: str) -> ReflectionTrace:
        return ReflectionTrace(
            question_id=sample.question_id,
            original_question=sample.question.text,
            original_sql=sample.sql_result.sql,
            max_correction_iterations=max_iters,
            final_question=sample.question.text,
            final_sql=sample.sql_result.sql,
            final_execution_success=sample.sql_result.execution_success,
            final_execution_error=sample.sql_result.execution_error,
            decision=SampleDecision.REJECTED,
            decision_reason=f"reflection_pipeline_exception: {detail}",
        )
