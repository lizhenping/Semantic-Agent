"""Jinja2 Prompt 适配器（app/adapters/prompt_jinja.py）

移植自旧 prompts/template_manager.py（render_template 行为一致），
增加模板版本指纹：渲染过的模板登记到 usage registry，
供 KnowledgeSnapshot.prompt_versions / manifest 引用。
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

logger = logging.getLogger(__name__)


class JinjaPromptRenderer:
    """Jinja2 模板渲染器 + 模板版本指纹。"""

    def __init__(self, template_dir: Optional[str] = None):
        if template_dir is None:
            template_dir = Path(__file__).parent.parent.parent / "prompts" / "templates"
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._version_cache: dict[str, str] = {}
        self._used: dict[str, str] = {}

    def render(self, template_path: str, **variables) -> str:
        template = self.env.get_template(template_path)
        self._used[template_path] = self.template_version(template_path)
        return template.render(**variables)

    def template_version(self, template_path: str) -> str:
        """模板内容 sha1 前 12 位（文件不变则指纹不变）。"""
        if template_path not in self._version_cache:
            path = self.template_dir / template_path
            try:
                content = path.read_bytes()
            except OSError:
                return "unknown"
            self._version_cache[template_path] = hashlib.sha1(content).hexdigest()[:12]
        return self._version_cache[template_path]

    def versions(self) -> dict[str, str]:
        """本次运行已渲染模板的版本表（PromptManifest）。"""
        return dict(self._used)
