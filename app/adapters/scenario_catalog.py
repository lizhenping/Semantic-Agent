"""场景目录适配器（app/adapters/scenario_catalog.py）

加载 configs/{scenarios,operation_mapping,complexity}.yaml，
按 (主场景 × 子场景 × 复杂度) 枚举生成组合（论文 σ=(c,a), ℓ）。

移植自旧 tools/synthesis/question_synth.py 的三层循环枚举与
use_case 解析逻辑（含随机选择规则），保持与 question_generation.j2
模板变量形状一致。文件读取只发生在这里；domain 层拿到的已是对象。
"""

import random
from pathlib import Path
from typing import Optional

import yaml

# YAML 配置里的元数据键（不是真实场景）
META_KEYS = {"scenario_types", "total_scenarios", "total_sub_scenarios"}
# 复杂度级别顺序（YAML 名）
COMPLEXITY_ORDER = ["simple", "moderate", "complex", "expert"]


class YamlScenarioCatalog:
    """场景 × 子场景 × 复杂度 × 用例 的组合目录。"""

    def __init__(self, config_dir: Optional[str] = None,
                 rng: Optional[random.Random] = None):
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "configs"
        self.config_dir = Path(config_dir)
        self.scenarios = self._load_yaml("scenarios.yaml")
        self.operation_mapping = self._load_yaml("operation_mapping.yaml")
        self.complexity_config = self._load_yaml("complexity.yaml")
        self.rng = rng or random.Random()

    def _load_yaml(self, filename: str) -> dict:
        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ---------- 组合枚举 ----------

    def main_scenario_names(self) -> list[str]:
        return [k for k in self.scenarios.get("scenarios", {}) if k not in META_KEYS]

    def ordered_combos(self) -> list[dict]:
        """第一轮穷举：按 YAML 顺序的所有合法组合。

        每个组合携带 question_generation.j2 期望的原始形状：
        {main_key, main, sub_key, sub, complexity, use_case_name, use_case}
        """
        combos: list[dict] = []
        for main_key in self.main_scenario_names():
            main_data = self.scenarios["scenarios"][main_key]
            for sub_key, sub_data in main_data.get("sub_scenarios", {}).items():
                for complexity in COMPLEXITY_ORDER:
                    resolved = self._resolve_use_case(main_key, sub_key, complexity)
                    if resolved is None:
                        continue
                    use_case_name, use_case = resolved
                    combos.append({
                        "main_key": main_key,
                        "main": main_data,
                        "sub_key": sub_key,
                        "sub": sub_data,
                        "complexity": complexity,
                        "use_case_name": use_case_name,
                        "use_case": use_case,
                    })
        return combos

    def random_combo(self) -> Optional[dict]:
        """第二轮随机补齐：随机 (主场景 × 子场景 × 复杂度)。"""
        valid_mains = self.main_scenario_names()
        if not valid_mains:
            return None
        main_key = self.rng.choice(valid_mains)
        main_data = self.scenarios["scenarios"][main_key]
        sub_keys = list(main_data.get("sub_scenarios", {}).keys())
        if not sub_keys:
            return None
        sub_key = self.rng.choice(sub_keys)
        sub_data = main_data["sub_scenarios"][sub_key]
        complexity = self.rng.choice(COMPLEXITY_ORDER)
        resolved = self._resolve_use_case(main_key, sub_key, complexity)
        if resolved is None:
            return None
        use_case_name, use_case = resolved
        return {
            "main_key": main_key,
            "main": main_data,
            "sub_key": sub_key,
            "sub": sub_data,
            "complexity": complexity,
            "use_case_name": use_case_name,
            "use_case": use_case,
        }

    def _resolve_use_case(self, main_key: str, sub_key: str,
                          complexity: str) -> Optional[tuple[str, dict]]:
        """解析 (场景, 子场景, 复杂度) 对应的 use_case（含权重桶随机）。"""
        bucket = (
            self.operation_mapping.get("scenario_use_case_mapping", {})
            .get(main_key, {})
            .get(sub_key, {})
            .get(complexity)
        )
        if not bucket:
            return None
        names: list[str] = []
        for item in bucket:
            names.extend(item.keys())
        if not names:
            return None
        name = self.rng.choice(names)
        use_case = self.operation_mapping.get("use_case_operations", {}).get(
            name, self.operation_mapping.get("use_case_operations", {}).get(
                "data_viewing", {}))
        return name, use_case

    def complexity_spec(self, name: str) -> dict:
        """复杂度级别的配置（论文 L1..L4 的结构描述）。"""
        return self.complexity_config.get("complexity_levels", {}).get(name, {})
