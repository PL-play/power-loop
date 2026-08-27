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
tri-state, and asking for a capability that was not declared **raises**. The library never
infers, never falls back, never downgrades your input behind your back.
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
        Declared unsupported. Sending input that needs it raises.
    ``None`` (the default)
        **Undeclared** — *not* a synonym for ``False``. Nothing guesses on your behalf;
        input that needs the capability raises, and the error tells you where to declare it.

    ``model`` is carried only so errors can name the model that lacks the declaration.
    """

    model: str = ""
    #: Accepts images inline in a chat message (as a base64 ``data:`` URL — the only image
    #: transport this library implements).
    supports_image_input: bool | None = None

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
            "or stop sending images to this model. It is NOT downgraded to text: an answer "
            "produced without the image would look valid and be unfounded."
        )


def coerce_capabilities(value: Any, *, model: str = "") -> ModelCapabilities:
    """Build a :class:`ModelCapabilities` from config (a dict, an instance, or ``None``).

    ``None`` / ``{}`` yields an all-undeclared instance — which is exactly right: a caller
    that declared nothing gets a model that can do nothing beyond plain text, loudly.
    """
    if isinstance(value, ModelCapabilities):
        return value if value.model or not model else ModelCapabilities(
            model=model, supports_image_input=value.supports_image_input
        )
    fields: dict[str, Any] = dict(value or {})
    unknown = set(fields) - {"model", "supports_image_input"}
    if unknown:
        # A typo'd or retired key (supports_tools, supports_stream, api_family, provider,
        # supports_pdf_input_* — all removed as dead config) must not read as "declared".
        raise ValueError(
            f"Unknown model capability key(s): {sorted(unknown)}. "
            "Supported keys: 'supports_image_input'."
        )
    fields.setdefault("model", model)
    return ModelCapabilities(**fields)
