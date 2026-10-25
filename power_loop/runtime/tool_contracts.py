"""Host-declared tool capabilities; no tool-name or handler identity policy.

This is a trusted Python host extension point, not model-supplied permission.
Declarations describe implemented lifecycle behavior; they cannot create it.
"""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

ROLES = ("main", "workflow", "workflow_node", "subagent")


class ToolContractError(ValueError):
    pass


@dataclass(frozen=True)
class ToolDispatchPolicy:
    native_admitted: bool = False
    control_only: bool = False

    def __post_init__(self):
        if (
            type(self.native_admitted) is not bool
            or type(self.control_only) is not bool
            or self.control_only
            and not self.native_admitted
        ):
            raise ToolContractError("invalid_tool_dispatch_policy")


@dataclass(frozen=True)
class ToolRuntimeContract:
    key: str
    allowed_roles: tuple[str, ...] = ROLES
    capabilities: tuple[str, ...] = ()
    background_allowed: bool = True
    result_container: bool = False

    def __post_init__(self):
        if (
            type(self.background_allowed) is not bool
            or type(self.result_container) is not bool
            or type(self.key) is not str
            or not 0 < len(self.key) <= 128
            or type(self.allowed_roles) is not tuple
            or not self.allowed_roles
            or any(role not in ROLES for role in self.allowed_roles)
            or len(set(self.allowed_roles)) != len(self.allowed_roles)
            or type(self.capabilities) is not tuple
            or any(type(c) is not str or not 0 < len(c) <= 128 for c in self.capabilities)
            or len(set(self.capabilities)) != len(self.capabilities)
        ):
            raise ToolContractError("invalid_tool_runtime_contract")


@dataclass(frozen=True)
class ToolContextPresentation:
    """Pure views of an existing call; never execution or lifecycle authority."""
    keep_in_send: bool = False
    archive_arguments: str | None = None
    archive_label: str | None = None

    def __post_init__(self):
        if (type(self.keep_in_send) is not bool
                or self.archive_label is not None and type(self.archive_label) is not str
                or self.archive_arguments is not None and type(self.archive_arguments) is not str):
            raise ToolContractError("invalid_tool_context_presentation")
        if self.archive_arguments is not None:
            try:
                valid = type(json.loads(self.archive_arguments)) is dict
            except ValueError:
                valid = False
            if not valid:
                raise ToolContractError("invalid_tool_context_arguments")


class RuntimeTool(ABC):
    @property
    @abstractmethod
    def runtime_contract(self) -> ToolRuntimeContract:
        """Stable host implementation identity, roles and advertised capabilities."""
        raise NotImplementedError()

    def dispatch_policy(self, arguments, *, runtime=None) -> ToolDispatchPolicy:
        """Classify this call; default grants neither native nor control admission."""
        return ToolDispatchPolicy()

    def format_result(self, result, max_chars):
        """Optional structure-preserving result view; None uses normal formatting."""
        return None

    def context_spec(self, spec):
        """Optional model-facing schema when tool-context preparation is enabled."""
        return spec

    def context_presentation(self, source) -> ToolContextPresentation:
        """Describe archival views and in-send preservation for this original call."""
        return ToolContextPresentation()

    def truncate_context(self, source, policy, *, raw_arguments):
        """Pure argument/result preview; None preserves the existing pair."""
        from .tool_truncation import truncate_pair
        return truncate_pair(raw_arguments, source.result, policy, source.recall)

    def pending_call_spec(self, arguments, *, runtime=None, definition=None):
        """Pure recovery description; None is unsupported, never not-dispatched."""
        return None

    def background_hint(self, definition, *, scheduler_name):
        """Optional usage for submitting a target to this implemented scheduler."""
        return None

    async def result_sources(self, source, *, store, registry):
        """A declared result container resolves original targets; never starts work."""
        return []

    @abstractmethod
    async def __call__(self, **arguments):
        raise NotImplementedError()


@dataclass(frozen=True)
class FunctionRuntimeTool(RuntimeTool):
    handler: object
    contract: ToolRuntimeContract
    policy: object = None
    formatter: object = None

    def __post_init__(self):
        if (
            not callable(self.handler)
            or type(self.contract) is not ToolRuntimeContract
            or any(value is not None and not callable(value) for value in (self.policy, self.formatter))
        ):
            raise ToolContractError("invalid_runtime_tool_implementation")

    @property
    def runtime_contract(self):
        return self.contract

    def dispatch_policy(self, arguments, *, runtime=None):
        return self.policy(arguments) if self.policy is not None else ToolDispatchPolicy()

    def format_result(self, result, max_chars):
        return self.formatter(result, max_chars) if self.formatter is not None else None

    async def __call__(self, **arguments):
        value = self.handler(**arguments)
        return await value if inspect.isawaitable(value) else value


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _implementation(handler):
    try:
        actual = inspect.unwrap(handler)
    except ValueError:
        raise ToolContractError("invalid_runtime_tool_wrapper") from None
    return actual if isinstance(actual, RuntimeTool) else None


def snapshot(handler, definition):
    implementation = _implementation(handler)
    if implementation is None:
        return None
    contract = implementation.runtime_contract
    if type(contract) is not ToolRuntimeContract:
        raise ToolContractError("invalid_tool_runtime_contract")
    return (
        _json(asdict(contract)),
        _json(
            {
                "definition": definition.to_openai_tool(),
                "required": definition.required_params,
                "interrupt": definition.interrupt,
                "usage_hint": definition.usage_hint,
                "background_hint": definition.background_hint,
                "async_capable": definition.async_capable
                if type(definition.async_capable) is bool
                else sorted(definition.async_capable),
            }
        ),
        implementation,
    )


def resolve(registered):
    """Only a capability frozen at actual registration can be used later."""
    if registered is None or getattr(registered, "runtime_binding", None) is None:
        return None
    current = snapshot(registered.handler, registered.definition)
    if (
        current is None
        or current[:2] != registered.runtime_binding[:2]
        or current[2] is not registered.runtime_binding[2]
    ):
        raise ToolContractError("tool_runtime_binding_changed")
    return _implementation(registered.handler)


def validate_registration(registry, name, expected):
    """Recheck after admission awaits, immediately before invoking the handler."""
    if registry.get(name) is not expected:
        raise ToolContractError("tool_runtime_binding_changed")
    resolve(expected)


def policy(registered, arguments, *, runtime=None):
    implementation = resolve(registered)
    if implementation is None:
        return ToolDispatchPolicy()
    result = implementation.dispatch_policy(arguments, runtime=runtime)
    if type(result) is not ToolDispatchPolicy:
        raise ToolContractError("invalid_tool_dispatch_policy")
    return result


def capability_tools(registry, capability, *, role=None):
    if registry is None:
        return []
    return [
        registered
        for name in registry.names()
        if (registered := registry.get(name)) is not None
        and (implementation := resolve(registered)) is not None
        and capability in implementation.runtime_contract.capabilities
        and (role is None or role in implementation.runtime_contract.allowed_roles)
    ]


def has_capability(registered, capability):
    implementation = resolve(registered)
    return implementation is not None and capability in implementation.runtime_contract.capabilities


def format_result(registered, result, max_chars):
    implementation = resolve(registered)
    value = implementation.format_result(result, max_chars) if implementation is not None else None
    if value is not None and type(value) is not str:
        raise ToolContractError("invalid_tool_result_view")
    return value


def background_allowed(registered):
    implementation = resolve(registered)
    return implementation is None or implementation.runtime_contract.background_allowed


def result_container(registered):
    implementation = resolve(registered)
    return implementation is not None and implementation.runtime_contract.result_container


def context_spec(registered, spec):
    implementation = resolve(registered)
    value = implementation.context_spec(spec) if implementation is not None else spec
    if (type(value) is not dict or value.get("type") != "function"
            or type(value.get("function")) is not dict
            or value["function"].get("name") != registered.definition.name):
        raise ToolContractError("invalid_tool_context_spec")
    return value


def context_presentation(registered, source):
    implementation = resolve(registered)
    value = implementation.context_presentation(source) if implementation is not None else ToolContextPresentation()
    if type(value) is not ToolContextPresentation:
        raise ToolContractError("invalid_tool_context_presentation")
    return value


def truncate_context(registered, source, policy, *, raw_arguments):
    implementation = resolve(registered)
    if implementation is None:
        from .tool_truncation import truncate_pair
        value = truncate_pair(raw_arguments, source.result, policy, source.recall)
    else:
        value = implementation.truncate_context(source, policy, raw_arguments=raw_arguments)
    if value is not None and (type(value) is not tuple or len(value) != 2
                              or any(type(part) is not str for part in value)):
        raise ToolContractError("invalid_tool_context_truncation")
    return value
