"""LLM 适配器（app/adapters/llm_openai.py）

移植自旧 infra/llm.py（行为保持）：
- ChatOpenAI（vLLM / OpenAI 兼容端点）同步实现
- <think> 标签剥离（Qwen3 thinking 模式残留）
- JSON 解析兼容 fenced 输出与前后夹杂的说明文本
- FakeChatGateway 单测用

增加（doc/03-physical-model.md §1 迁移表）：调用元数据与异常分类，
空响应/坏 JSON 抛 ports.llm 的具体异常，供上层区分"无错误"与"审查失败"。
"""

import json
import logging
import re
import uuid
from typing import Optional

from app.ports.llm import EmptyResponseError, InvalidJSONError

logger = logging.getLogger(__name__)


class ChatOpenAIGateway:
    """ChatOpenAI 实现（vLLM/SSH 隧道场景的显式超时与重试保留）。"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.1,
        max_tokens: int = 8000,
        timeout: int = 1200,
        max_retries: int = 2,
        enable_thinking: bool = True,
    ):
        from langchain_openai import ChatOpenAI

        # 思考模式关闭：Qwen3 系列在 vLLM 上经 chat_template_kwargs 控制，
        # 请求体里直接省掉推理过程；开启时不传该字段（云端 API 会拒绝未知参数）。
        extra_body = None
        if not enable_thinking:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        # 显式超时：隧道断线时连接仍可建立但数据不流动，必须设读超时。
        # langchain-openai 1.6 会对 httpx.Timeout 再包一层导致
        # "unhashable type" 报错，因此必须传标量秒数。
        self._llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=float(timeout),
            max_retries=max_retries,
            extra_body=extra_body,
        )
        self.model_id = model
        self.request_count = 0

    def generate(self, prompt: str) -> str:
        self.request_count += 1
        request_id = uuid.uuid4().hex[:12]
        response = self._llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        if "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        logger.debug("llm response request_id=%s model=%s len=%d",
                     request_id, self.model_id, len(content))
        return content.strip()

    def generate_json(self, prompt: str) -> dict:
        text = self.generate(prompt).strip()
        if not text:
            raise EmptyResponseError("LLM returned an empty response")

        # 优先 fenced JSON，但不要求模型使用代码围栏
        if "```" in text:
            fenced = text.split("```", 2)
            if len(fenced) == 3:
                text = fenced[1].strip()
                if text.lower().startswith("json"):
                    text = text[4:].lstrip()

        # 解码第一个完整 JSON 值，忽略前后夹杂的说明文字
        decoder = json.JSONDecoder()
        candidates = [text]
        first_object = text.find("{")
        if first_object > 0:
            candidates.append(text[first_object:])
        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                value, _ = decoder.raw_decode(candidate.lstrip())
                if not isinstance(value, dict):
                    raise ValueError("LLM JSON response must be an object")
                return value
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
        raise InvalidJSONError(f"Invalid JSON response: {last_error}") from last_error


class FakeChatGateway:
    """假 LLM（单测用，不连真服务）。按序返回预置响应。"""

    def __init__(self, responses: Optional[list[str]] = None):
        self._responses = responses or []
        self._call_count = 0
        self.calls: list[str] = []
        self.model_id = "fake-model"

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return '{"status": "fake"}'

    def generate_json(self, prompt: str) -> dict:
        try:
            return json.loads(self.generate(prompt))
        except json.JSONDecodeError:
            return {"status": "fake"}
