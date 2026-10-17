"""新图的 Skill Registry：启动时校验并固定内容，加载时不读取模型指定路径。"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from .contracts import Contract, SkillDocument


SKILL_ROOT = Path(__file__).resolve().parents[1] / "skills"
COMPLETION_CONDITIONS = frozenset({"order_verified", "policy_assessed", "inventory_checked", "discount_assessed",
                                  "price_assessed", "incident_assessed", "logistics_verified"})


class SkillError(Exception):
    """技能发现、版本或配置错误使用稳定错误码，不回显本机路径。"""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


class SkillManifest(Contract):
    """新图使用独立 Manifest，原四个 legacy Manifest 不变。"""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    description: str = Field(min_length=1, max_length=600)
    version: str = Field(min_length=1, max_length=40)
    input_schema: Literal["AgentTask"] = "AgentTask"
    output_schema: Literal["ToolObservation"] = "ToolObservation"
    allowed_tools: list[str] = Field(min_length=1, max_length=20)
    instructions_path: str
    completion_conditions: list[str] = Field(min_length=1, max_length=10)
    write_policy: Literal["disabled_until_s4", "confirmed_only"] = "disabled_until_s4"


class FileSkillRegistry:
    """只加载仓库配置路径，返回深拷贝；模型与调用方不能修改进程内固定内容。"""

    def __init__(self, known_tools: set[str], root: Path = SKILL_ROOT, s4: bool = False, s5: bool = False):
        self.root = root.resolve()
        project = Path(__file__).resolve().parents[2]
        if not self.root.is_relative_to(project):
            raise SkillError("SKILL_PATH_FORBIDDEN")
        self._documents, self._manifests = {}, {}
        try:
            paths = sorted((self.root / "agent").glob("*.json"))
            if not paths:
                raise SkillError("SKILL_CONFIG_INVALID")
            for path in paths:
                # 先解析符号链接并核验边界，再读配置，避免越界文件已被读取后才拒绝。
                if not path.resolve().is_relative_to(self.root):
                    raise SkillError("SKILL_PATH_FORBIDDEN")
                manifest = SkillManifest.model_validate_json(path.read_text(encoding="utf-8"))
                if (manifest.name in self._documents or len(set(manifest.allowed_tools)) != len(manifest.allowed_tools)
                        or not set(manifest.allowed_tools) <= known_tools
                        or not set(manifest.completion_conditions) <= COMPLETION_CONDITIONS):
                    raise SkillError("SKILL_CONFIG_INVALID")
                if s4:
                    # S4 使用独立规范和版本，不改写运行中的 S3 规范，也不接受模型提供路径。
                    manifest = manifest.model_copy(update={"version":manifest.version+"-s4.1",
                        "write_policy":"confirmed_only", "instructions_path":"../instructions_s4/"+manifest.name+".md",
                        # 价保/优惠原 S3 只声明只读工具；S4 的独立 Manifest 视图才加入写入口。
                        "allowed_tools":list(dict.fromkeys(manifest.allowed_tools+(['create_ticket_draft','submit_ticket']
                            if manifest.name in {'coupon_dispute','price_adjustment'} else [])))})
                if s5 and manifest.name == 'logistics_tracking':
                    # 只替换 S5 的物流规范，不改变旧 S3/S4 的内容锁和待恢复任务。
                    manifest = manifest.model_copy(update={'version': '1.0-s5.1',
                        'description': '通过项目 B 自有 MCP 服务查询本人订单配送物流',
                        'instructions_path': '../instructions_s5/logistics_tracking.md'})
                instructions_path = (path.parent / manifest.instructions_path).resolve()
                if Path(manifest.instructions_path).is_absolute() or not instructions_path.is_relative_to(self.root):
                    raise SkillError("SKILL_PATH_FORBIDDEN")
                if not path.resolve().is_relative_to(self.root) or instructions_path.suffix != ".md":
                    raise SkillError("SKILL_PATH_FORBIDDEN")
                instructions = instructions_path.read_text(encoding="utf-8")
                if not instructions.strip() or len(instructions) > 24000:
                    raise SkillError("SKILL_CONFIG_INVALID")
                canonical = json.dumps(manifest.model_dump(), ensure_ascii=False, sort_keys=True) + "\n" + instructions
                digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                self._documents[manifest.name] = SkillDocument(
                    name=manifest.name, version=manifest.version, instructions=instructions,
                    tool_names=manifest.allowed_tools, content_hash=digest,
                    completion_conditions=manifest.completion_conditions)
                self._manifests[manifest.name] = manifest
        except SkillError:
            raise
        except Exception as exc:
            raise SkillError("SKILL_CONFIG_INVALID") from exc

    async def list_catalog(self) -> list[dict[str, str]]:
        """目录只返回名称、说明、版本和哈希，SOP 在 load 时才返回。"""
        return [{"name": name, "description": self._manifests[name].description,
                 "version": doc.version, "content_hash": doc.content_hash}
                for name, doc in sorted(self._documents.items())]

    async def load(self, name: str, version: str | None = None) -> SkillDocument:
        """版本不一致明确拒绝；同版本内容跨进程核验由 get_pinned 的哈希完成。"""
        if name not in self._documents:
            raise SkillError("UNKNOWN_SKILL")
        doc = self._documents[name]
        if version is not None and version != doc.version:
            raise SkillError("SKILL_VERSION_INCOMPATIBLE")
        return doc.model_copy(deep=True)

    async def get_pinned(self, name: str, version: str, content_hash: str) -> SkillDocument:
        """相同版本号但内容变化也视为不兼容，不能静默替换规范。"""
        doc = await self.load(name, version)
        if doc.content_hash != content_hash:
            raise SkillError("SKILL_VERSION_INCOMPATIBLE")
        return doc
