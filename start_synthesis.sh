#!/usr/bin/env bash
# ============================================================
# SemanticSQL 正式合成脚本（10 万条准入目标）
#
# 用法（Git Bash / 终端里）：
#   bash start_synthesis.sh
#
# 断点续跑：任何中断（关机/断网/隧道断）之后，重新执行同一条命令即可，
#   已完成样本数的库自动跳过，只补差额。可反复执行，直到全部达标。
#
# 监控进度（另开一个终端）：
#   tail -f runs/_review/synthesis.log
#   # 或统计已准入条数：
#   find runs -name training.accepted.jsonl -exec cat {} + | wc -l
# ============================================================
set -u
PY="D:/学术/tkde/code/semanticsql-agent/.venv-fixed/Scripts/python.exe"
cd "D:/学术/tkde/semantic-agent" || exit 1
mkdir -p runs/_review
W=40   # 并发队列深度（后端已实测支持）

echo "==== [$(date '+%F %T')] Phase 1 预构建检查（267 库已全部就绪，此步秒过）===="
for b in ehrsql science_benchmark spider2 bird spider; do
  "$PY" cli.py build-knowledge -b "$b" -w 20 | tail -1
done

echo "==== [$(date '+%F %T')] 开始合成：目标 137,200 生成 → 约 10 万准入 ===="
# 顺序：小库先行（尽早拿到完整 benchmark 产物），大库在后
"$PY" cli.py run-benchmark -b science_benchmark --total 6900  -w $W
"$PY" cli.py run-benchmark -b ehrsql         --total 6900  -w $W
"$PY" cli.py run-benchmark -b spider2        --total 27400 -w $W
"$PY" cli.py run-benchmark -b bird           --total 41000 -w $W
"$PY" cli.py run-benchmark -b spider         --total 55000 -w $W

echo "==== [$(date '+%F %T')] 全部 benchmark 合成结束 ===="
echo "准入总量：$(find runs -name training.accepted.jsonl -exec cat {} + 2>/dev/null | wc -l)"
