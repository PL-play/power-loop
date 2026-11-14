# Managed tool recovery

The prepare/execute/recover adapter remains the extension point for externally managed effects. Runtime state uses the session and original MessageRef/tool_call_id. Recover reads the original external result or reports unknown; it does not execute the action again. Normal permissions and host billing still apply.

See [Runtime 12](runtime-12.md) and [tool contracts](runtime-tool-contracts.md).
