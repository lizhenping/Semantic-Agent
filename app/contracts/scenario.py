"""生成场景契约（app/contracts/scenario.py）

论文 §III.D：场景 σ=(c,a)，c 是数据库支持的域上下文，a 是分析任务；
结构规格 ℓ ∈ {L1,L2,L3,L4} 指导 SQL 构造，与事后 SQL 复杂度分类无关
（doc/01-conceptual-model.md §3.5：场景是生成约束，不是样本本身）。
"""

from enum import Enum

from pydantic import BaseModel, Field


class ComplexityLevel(str, Enum):
    """论文 ℓ ∈ {L1..L4}（configs/complexity.yaml 的 simple..expert）。"""

    L1 = "L1"   # 单表过滤与聚合
    L2 = "L2"   # 多表 JOIN
    L3 = "L3"   # 嵌套子查询与窗口函数
    L4 = "L4"   # CTE 与模块化组合

    @classmethod
    def from_name(cls, name: str) -> "ComplexityLevel":
        """YAML 复杂度名（simple/moderate/complex/expert）→ 级别。"""
        mapping = {
            "simple": cls.L1,
            "moderate": cls.L2,
            "complex": cls.L3,
            "expert": cls.L4,
        }
        return mapping.get(str(name).lower(), cls.L1)

    def to_int(self) -> int:
        """序列化用整数 1..4（对齐既有产物 complexity_level 字段）。"""
        return int(self.value[1])


class GenerationMetadata(BaseModel):
    """生成场景元数据 σ=(C,T), ℓ（写入 think.metadata）。"""

    main_scenario: str = ""        # C 域上下文
    sub_scenario: str = ""         # T 分析任务类型
    complexity_level: int = 1      # ℓ ∈ {1,2,3,4} = L1..L4
    use_case: str = ""             # 操作用例


class GenerationScenario(BaseModel):
    """一次候选生成的任务意图（生成约束）。"""

    main_scenario: str = ""            # c：域上下文名
    main_scenario_description: str = ""
    sub_scenario: str = ""             # a：分析任务名
    focus_areas: list[str] = Field(default_factory=list)
    complexity: ComplexityLevel = ComplexityLevel.L1
    use_case: str = ""                 # 操作用例名
    use_case_spec: dict = Field(default_factory=dict)  # 用例的操作配置

    def to_metadata(self) -> GenerationMetadata:
        return GenerationMetadata(
            main_scenario=self.main_scenario,
            sub_scenario=self.sub_scenario,
            complexity_level=self.complexity.to_int(),
            use_case=self.use_case,
        )
