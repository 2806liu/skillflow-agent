"""所有工具的基类和共享同步锁。"""

import threading
from abc import ABC, abstractmethod

# File-mutating tools (write_file/edit_file) serialize on this, so a parallel
# batch can't interleave read-modify-write on the same file and lose an edit.
FILE_MUTATION_LOCK = threading.Lock()


class Tool(ABC):
    """最小工具接口；新增能力时继承本类并实现 execute。"""

    name: str
    description: str
    parameters: dict  # JSON Schema for the function args

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具并返回文本结果。"""
        ...

    def schema(self) -> dict:
        """生成 OpenAI function calling 所需的 JSON Schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
