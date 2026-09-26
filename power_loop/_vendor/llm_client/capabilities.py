"""Model capabilities — **declared, never guessed**.

History (why this file looks the way it does): capabilities used to be *inferred* from the
model NAME via a table of ~15 vendor regexes, with an env-var escape hatch
(``POWER_LOOP_SUPPORTS_*``) for models the table didn't recognise. Both are gone, because
both were wrong in the same way — they made the library decide, silently, whether your
model could see an image:

* **Name-guessing fails open-endedly.** Any model outside the table (a new release, a
  vendor's experimental endpoint, a proxy that renames models) was judged
  ``supports_image_input=False``. Real case: ``deepseek-v4-flash-vision-exp`` — a model
  that demonstrably accepts ``image_url`` — was classified blind, and every image sent to
  it was silently replaced with the sentence "the current model does not support image
  input". The model then answered from the filename and the caller saw a plausible reply.
  Nothing errored. That is the worst possible failure mode: a green light over a
  capability that never ran.
* **Env vars are the wrong scope.** ``POWER_LOOP_SUPPORTS_IMAGE_INPUT`` is process-wide. A
  host running many agent definitions against different models in one process cannot say
  "this one sees images, that one doesn't" — it can only lie for all of them at once.

So: capabilities are **configuration on the LLM config object** (hence per-loop /
per-definition, see :class:`power_loop.runtime.provider.LLMProviderConfig`), every field is
tri-state, and nothing is inferred. Undeclared is treated exactly like unsupported — most
configured models cannot see, so that is the only safe default.

What "unsupported" does to an image is an EXPLICIT degradation, never a silent one: the
renderer (``multimodal.py``) replaces it with a placeholder that says the model did NOT see
it and carries the host's recall coordinate, so the model cannot answer as if it had looked,
and the session keeps working (history full of images must not brick a model switch).
Callers that would rather fail use :meth:`ModelCapabilities.require_image_input`.

A declaration belongs to ONE model (:meth:`ModelCapabilities.for_model`): a request for a
different model name — a sub-agent or workflow leaf overriding ``model`` on the parent's
client — gets nothing declared, instead of silently inheriting what the parent can do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ModelCapabilityError(RuntimeError):
    """Raised when a request needs a capability the model has not DECLARED.

    Covers both "declared unsupported" and "never declared". Both are caller bugs, and both
    must be loud: the alternative — quietly dropping the image and sending the text — yields
    an answer that looks fine and is unfounded.
    """


@dataclass(frozen=True)
class ModelCapabilities:
    """What a model accepts. Every capability field is **tri-state**:

    ``True``
        Declared supported. Used natively.
    ``False``
        Declared unsupported. Input that needs it is degraded explicitly (see module doc).
    ``None`` (the default)
        **Undeclared** — nothing guesses on your behalf; handled like ``False``. It stays a
        separate value so errors and logs can say "not declared" rather than "declared no".

    ``model`` names the model the declaration is FOR (see :meth:`for_model`).
    """

    model: str = ""
    #: Accepts images inline in a chat message (as a base64 ``data:`` URL — the only image
    #: transport this library implements).
    supports_image_input: bool | None = None
    #: Longest edge, in pixels, of an image sent to this model. ``None`` = send as-is.
    #:
    #: This is the ONE knob that actually reduces cost: image tokens are billed by PIXEL
    #: DIMENSIONS, not by bytes — a 787 KB noise PNG and a 1.8 KB flat-colour PNG of the same
    #: size cost exactly the same (measured). Re-encoding at lower JPEG quality saves bandwidth
    #: and not one token; halving the longest edge saves ~43%. Enforced at the single render
    #: point, so every path in (a fresh send, a recalled image, anything a host builds) is
    #: covered by construction.
    max_image_edge: int | None = None
    #: Accepts ``response_format={"type": "json_schema", ...}`` natively. Undeclared, a json_schema
    #: request is sent WITHOUT it and the schema goes into the system prompt as a hard instruction
    #: (``LLMRequest.with_structured_fallback``) — many OpenAI-compatible endpoints reject the
    #: native form outright (DeepSeek: 400 "This response_format type is unavailable now").
    supports_json_schema: bool | None = None

    def for_model(self, model: str | None) -> ModelCapabilities:
        """The declaration that applies to a request for ``model``.

        A client serves one configured model, but a request may name another (a sub-agent or
        workflow leaf with its own ``model`` on the parent's client). What the parent's model
        can do says nothing about that one, so it gets an all-undeclared instance. Same (or
        no) model name → ``self``."""
        if not model or not self.model or model == self.model:
            return self
        return ModelCapabilities(model=model)

    @property
    def sees_images(self) -> bool:
        """Images reach this model natively (declared). Anything else gets a placeholder."""
        return self.supports_image_input is True

    def require_image_input(self, *, what: str) -> None:
        """Raise unless image input is DECLARED supported. ``what`` names the offending
        attachment so the error points at a file, not just at a config field."""
        if self.supports_image_input is True:
            return
        model = self.model or "<unnamed model>"
        if self.supports_image_input is False:
            reason = f"model {model!r} is declared NOT to support image input"
        else:
            reason = (
                f"model {model!r} has not declared image support "
                "(capabilities are declared, never inferred from the model name)"
            )
        raise ModelCapabilityError(
            f"Cannot send {what}: {reason}. Either declare it — "
            "LLMProviderConfig(..., capabilities={'supports_image_input': True}) — "
            "or stop sending images to this model. (This is the strict check; the renderer "
            "itself degrades an image to an explicit 'you did not see this' placeholder.)"
        )


def coerce_capabilities(value: Any, *, model: str = "") -> ModelCapabilities:
    """Build a :class:`ModelCapabilities` from config (a dict, an instance, or ``None``).

    ``None`` / ``{}`` yields an all-undeclared instance — which is exactly right: a caller
    that declared nothing gets a model treated as text-only, with explicit placeholders.
    """
    if isinstance(value, ModelCapabilities):
        if value.model or not model:
            return value
        return ModelCapabilities(
            model=model,
            supports_image_input=value.supports_image_input,
            max_image_edge=value.max_image_edge,
            supports_json_schema=value.supports_json_schema,
        )
    fields: dict[str, Any] = dict(value or {})
    unknown = set(fields) - {"model", "supports_image_input", "max_image_edge", "supports_json_schema"}
    if unknown:
        # A typo'd or retired key (supports_tools, supports_stream, api_family, provider,
        # supports_pdf_input_* — all removed as dead config) must not read as "declared".
        raise ValueError(
            f"Unknown model capability key(s): {sorted(unknown)}. "
            "Supported keys: 'supports_image_input', 'max_image_edge', 'supports_json_schema'."
        )
    fields.setdefault("model", model)
    return ModelCapabilities(**fields)
