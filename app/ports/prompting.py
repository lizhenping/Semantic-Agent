"""Prompt 端口（app/ports/prompting.py）

PromptRenderer：模板渲染 + 模板版本指纹。
Prompt 快照纳入知识/运行版本摘要（doc/03-physical-model.md §1 迁移表）。
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class PromptRenderer(Protocol):
    """提示词渲染协议。"""

    def render(self, template_path: str, **variables) -> str:
        """渲染相对模板目录的模板（如 'generation/question_generation.j2'）。"""
        ...

    def template_version(self, template_path: str) -> str:
        """模板内容指纹（sha1 前 12 位），纳入 KnowledgeSnapshot/manifest。"""
        ...

    def versions(self) -> dict[str, str]:
        """已知模板的版本表（PromptManifest）。"""
        ...
