"""Reasoning-effort range guardrail registration and initialization."""

from typing import TYPE_CHECKING

from litellm.types.guardrails import GuardrailEventHooks, SupportedGuardrailIntegrations

from .reasoning_effort_range import ReasoningEffortRangeGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> ReasoningEffortRangeGuardrail:
    """Create and register a reasoning-effort range guardrail callback."""
    import litellm

    if litellm_params.model is None or litellm_params.max_effort is None:
        raise ValueError("model and max_effort are required")
    if litellm_params.mode != GuardrailEventHooks.pre_call:
        raise ValueError("reasoning_effort_range only supports pre_call mode")

    callback = ReasoningEffortRangeGuardrail(
        guardrail_name=guardrail["guardrail_name"],
        event_hook=GuardrailEventHooks.pre_call,
        default_on=litellm_params.default_on or False,
        model=litellm_params.model,
        min_effort=litellm_params.min_effort or "none",
        max_effort=litellm_params.max_effort,
        default_effort=litellm_params.default_effort,
    )
    litellm.logging_callback_manager.add_litellm_callback(  # pyright: ignore[reportUnknownMemberType] -- callback manager accepts CustomLogger subclasses
        callback
    )
    return callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.REASONING_EFFORT_RANGE.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.REASONING_EFFORT_RANGE.value: ReasoningEffortRangeGuardrail,
}
