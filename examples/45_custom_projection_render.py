"""45 · 自定义投影渲染 / Custom projection render (ProjectionRenderConfig + subclass)

What you learn / 你将学到
--------------------------
- 投影模式下,已结束的 send 存成 ``pl_project_messages`` 行;``render()`` 再把这些行变成喂给
  LLM 的消息文本。这一步现在**可定制**——不必复制整个 render。/ In projection mode finished
  sends are stored as ``pl_project_messages`` rows; ``render()`` turns them into the LLM message
  text. That step is now CUSTOMIZABLE — without copy-pasting the whole render.
- **配置路径**:传 ``ProjectionRenderConfig`` 调标签 / 分隔符 / 开关(纯标量,可从 JSON / 管理台
  下发,随时改了重渲染来对比)。/ **Config path**: pass a ``ProjectionRenderConfig`` to tune tags /
  separators / toggles (pure scalars — so the same config can come from JSON / an admin UI).
- **子类路径**:重写 ``render_project_row``(或 user / compact)其中**一个形状**即可,其余沿用
  内置。/ **Subclass path**: override exactly ONE shape — ``render_project_row`` (or user/compact);
  the rest keep the built-in render.

实际接入 / Wiring into a loop
    AgentLoopConfig(representation=ProjectedRepresentation(render_config=cfg), ...)
  完整的投影 + 折叠回路见 example 40。/ for the full projection+fold loop see example 40.

Run / 运行
----------
    python examples/45_custom_projection_render.py
"""

from __future__ import annotations

from power_loop import ProjectedRepresentation, ProjectionRenderConfig
from power_loop.runtime.store.types import ProjectMessageRow


def pmrow(send_index: int, kind: str, content: dict, **kw: object) -> ProjectMessageRow:
    """A stored projection row, shaped as ProjectedRepresentation.project_send writes it (example 40)."""
    base: dict = dict(
        session_id="demo", send_index=send_index, kind=kind, content=content, rendered_text=None,
        source_seq_lo=None, source_seq_hi=None, compact_from_send=None, compact_to_send=None,
        projector_version=1, token_estimate=None, created_at=0,
    )
    base.update(kw)
    return ProjectMessageRow(**base)


# Two finished sends' worth of stored projection rows + one folded compact row.
ROWS = [
    pmrow(1, "user", {"input": ["搜索今天的天气 / look up today's weather"]}),
    pmrow(1, "project", {
        "tools": [{"name": "web_search", "result": "晴 22°C / sunny 22°C"}],
        "final_text": "今天晴,22 度。/ sunny today, 22°C.",
    }),
    pmrow(3, "compact", {"summary": "更早的寒暄 / earlier small talk"},
          compact_from_send=1, compact_to_send=2),
]


def show(title: str, rep: ProjectedRepresentation) -> None:
    print(f"\n── {title} ──")
    for m in rep.render(ROWS):
        print(f"[{m['role']}] {m['content']}")


def main() -> None:
    # 1) Default — the built-in format, unchanged.
    show("默认 / default", ProjectedRepresentation())

    # 2) Config path — retune tags + separators and drop the private final_text. Every field is a
    #    plain scalar, so this exact dict could just as well arrive from JSON / an admin UI.
    cfg = ProjectionRenderConfig.from_dict({
        "user_tag": "👤#{n} ", "project_tag": "🤖#{n} ",
        "tools_header": "调用/calls: ", "include_final_text": False,
        "fold_note": "〔已折叠 {range},recall_send(N) 展开〕",
    })
    show("配置路径 / config path", ProjectedRepresentation(render_config=cfg))

    # 3) Subclass path — override ONE shape; the user / compact rows keep the built-in render.
    class TerseRender(ProjectedRepresentation):
        def render_project_row(self, r: ProjectMessageRow) -> dict:
            tools = (r.content or {}).get("tools") or []
            names = ", ".join(t.get("name", "?") for t in tools) or "—"
            return {"role": "assistant", "content": f"#{r.send_index} did: {names}"}

    show("子类路径 / subclass path", TerseRender())


if __name__ == "__main__":
    main()
