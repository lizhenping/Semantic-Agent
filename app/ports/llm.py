"""LLM 端口（app/ports/llm.py）

工具/领域服务只依赖 ChatGateway，不创建客户端、不读环境变量。
测试注入 Fake 实现；调用元数据（request_id/model/attempt）由实现负责。
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ChatGateway(Protocol):
    """LLM 网关协议（同步接口）。"""

    model_id: str

    def generate(self, prompt: str) -> str:
        """生成文本"""
        ...

    def generate_json(self, prompt: str) -> dict:
        """生成 JSON（返回解析后的 dict；解析失败抛 ValueError）"""
        ...


class LLMError(RuntimeError):
    """LLM 调用失败（网络/超时/格式）。技术重试由 adapter 负责，
    不得变成业务反思轮次。"""


class EmptyResponseError(LLMError):
    """模型返回空响应"""


class InvalidJSONError(LLMError):
    """模型响应不是合法 JSON 对象"""
