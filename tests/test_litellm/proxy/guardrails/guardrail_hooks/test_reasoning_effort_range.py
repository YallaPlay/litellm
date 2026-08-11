from unittest.mock import patch

import pytest

from litellm.proxy.guardrails.guardrail_hooks.reasoning_effort_range import (
    initialize_guardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.reasoning_effort_range.reasoning_effort_range import (
    ReasoningEffortRangeGuardrail,
)
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.proxy.guardrails.guardrail_hooks.reasoning_effort_range import (
    ReasoningEffortRangeGuardrailConfigModel,
)


def make_guardrail(**overrides) -> ReasoningEffortRangeGuardrail:
    params = {
        "guardrail_name": "sol-medium",
        "event_hook": GuardrailEventHooks.pre_call,
        "default_on": False,
        "model": "gpt-5.6-sol",
        "min_effort": "none",
        "max_effort": "medium",
        "default_effort": "medium",
    }
    params.update(overrides)
    return ReasoningEffortRangeGuardrail(**params)


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium"])
@pytest.mark.asyncio
async def test_allows_effort_inside_range(effort: str) -> None:
    guardrail = make_guardrail()
    data = {"model": "gpt-5.6-sol", "reasoning_effort": effort}

    result = await guardrail.async_pre_call_hook(None, None, data, "acompletion")

    assert result["reasoning_effort"] == effort


@pytest.mark.parametrize("effort", ["high", "xhigh", "max", "default"])
@pytest.mark.asyncio
async def test_clamps_effort_above_range(effort: str) -> None:
    guardrail = make_guardrail()
    data = {"model": "gpt-5.6-sol", "reasoning_effort": effort}

    result = await guardrail.async_pre_call_hook(None, None, data, "acompletion")

    assert result["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_clamps_effort_below_range() -> None:
    guardrail = make_guardrail(min_effort="low")
    data = {"model": "gpt-5.6-sol", "reasoning_effort": "none"}

    result = await guardrail.async_pre_call_hook(None, None, data, "acompletion")

    assert result["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_clamps_to_boundary_not_default() -> None:
    guardrail = make_guardrail(min_effort="low", max_effort="high", default_effort="low")
    data = {"model": "gpt-5.6-sol", "reasoning_effort": "xhigh"}

    result = await guardrail.async_pre_call_hook(None, None, data, "acompletion")

    assert result["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_default_alias_uses_configured_default() -> None:
    guardrail = make_guardrail(min_effort="low", max_effort="max", default_effort="low")
    data = {"model": "gpt-5.6-sol", "reasoning_effort": "default"}

    result = await guardrail.async_pre_call_hook(None, None, data, "acompletion")

    assert result["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_defaults_omitted_chat_effort() -> None:
    guardrail = make_guardrail()
    data = {"model": "gpt-5.6-sol"}

    result = await guardrail.async_pre_call_hook(None, None, data, "acompletion")

    assert result["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_handles_structured_chat_and_responses_effort() -> None:
    guardrail = make_guardrail()
    structured = {
        "model": "gpt-5.6-sol",
        "reasoning_effort": {"effort": "low", "summary": "detailed"},
    }
    responses = {"model": "gpt-5.6-sol", "reasoning": {"effort": "medium"}}

    structured_result = await guardrail.async_pre_call_hook(None, None, structured, "acompletion")
    responses_result = await guardrail.async_pre_call_hook(None, None, responses, "aresponses")

    assert structured_result["reasoning_effort"] == {"effort": "low", "summary": "detailed"}
    assert responses_result["reasoning"] == {"effort": "medium"}


@pytest.mark.asyncio
async def test_clamps_structured_chat_and_responses_effort() -> None:
    guardrail = make_guardrail()
    structured = {
        "model": "gpt-5.6-sol",
        "reasoning_effort": {"effort": "high", "summary": "detailed"},
    }
    responses = {"model": "gpt-5.6-sol", "reasoning": {"effort": "xhigh"}}

    structured_result = await guardrail.async_pre_call_hook(None, None, structured, "acompletion")
    responses_result = await guardrail.async_pre_call_hook(None, None, responses, "aresponses")

    assert structured_result["reasoning_effort"] == {"effort": "medium", "summary": "detailed"}
    assert responses_result["reasoning"] == {"effort": "medium"}


@pytest.mark.asyncio
async def test_clamps_anthropic_budget_and_adaptive_effort_above_range() -> None:
    guardrail = make_guardrail()
    budget = {
        "model": "gpt-5.6-sol",
        "thinking": {"type": "enabled", "budget_tokens": 4096},
    }
    adaptive = {
        "model": "gpt-5.6-sol",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "xhigh"},
    }

    budget_result = await guardrail.async_pre_call_hook(None, None, budget, "anthropic_messages")
    adaptive_result = await guardrail.async_pre_call_hook(None, None, adaptive, "anthropic_messages")

    assert budget_result["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "output_config" not in budget_result
    assert adaptive_result["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "output_config" not in adaptive_result


@pytest.mark.asyncio
async def test_defaults_omitted_anthropic_effort() -> None:
    guardrail = make_guardrail()
    data = {"model": "gpt-5.6-sol", "output_config": {"format": {"type": "json_schema"}}}

    result = await guardrail.async_pre_call_hook(None, None, data, "anthropic_messages")

    assert result["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert result["output_config"] == {"format": {"type": "json_schema"}}


@pytest.mark.asyncio
async def test_defaults_adaptive_thinking_without_explicit_effort() -> None:
    guardrail: ReasoningEffortRangeGuardrail = make_guardrail()
    data = {"model": "gpt-5.6-sol", "thinking": {"type": "adaptive"}}

    result = await guardrail.async_pre_call_hook(None, None, data, "anthropic_messages")

    assert result["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "output_config" not in result


@pytest.mark.asyncio
async def test_defaults_passthrough_adaptive_thinking_without_explicit_effort() -> None:
    guardrail: ReasoningEffortRangeGuardrail = make_guardrail()
    data = {"model": "gpt-5.6-sol", "thinking": {"type": "adaptive"}}

    result = await guardrail.async_pre_call_hook(None, None, data, "pass_through_endpoint")

    assert result["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "output_config" not in result


@pytest.mark.asyncio
async def test_maps_adaptive_minimal_effort_to_anthropic_low() -> None:
    guardrail: ReasoningEffortRangeGuardrail = make_guardrail()
    data = {
        "model": "gpt-5.6-sol",
        "reasoning_effort": "minimal",
        "thinking": {"type": "adaptive"},
    }

    result = await guardrail.async_pre_call_hook(None, None, data, "completion")

    assert result["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert "output_config" not in result


@pytest.mark.asyncio
async def test_adaptive_none_disables_anthropic_thinking() -> None:
    guardrail: ReasoningEffortRangeGuardrail = make_guardrail()
    data = {
        "model": "gpt-5.6-sol",
        "reasoning_effort": "none",
        "thinking": {"type": "adaptive"},
    }

    result = await guardrail.async_pre_call_hook(None, None, data, "completion")

    assert result["thinking"] == {"type": "disabled"}
    assert "output_config" not in result


@pytest.mark.asyncio
async def test_maps_adaptive_minimal_to_closest_native_effort() -> None:
    guardrail: ReasoningEffortRangeGuardrail = make_guardrail(max_effort="minimal", default_effort="minimal")

    result = await guardrail.async_pre_call_hook(
        None,
        None,
        {
            "model": "gpt-5.6-sol",
            "reasoning_effort": "minimal",
            "thinking": {"type": "adaptive"},
        },
        "completion",
    )

    assert result["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert "output_config" not in result


@pytest.mark.parametrize("effort", ["none", "minimal"])
@pytest.mark.asyncio
async def test_rejects_unsupported_native_output_config_effort(effort: str) -> None:
    guardrail: ReasoningEffortRangeGuardrail = make_guardrail()

    with pytest.raises(Exception):
        await guardrail.async_pre_call_hook(
            None,
            None,
            {
                "model": "gpt-5.6-sol",
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
            },
            "completion",
        )


@pytest.mark.asyncio
async def test_rejects_passthrough_null_adaptive_effort() -> None:
    guardrail: ReasoningEffortRangeGuardrail = make_guardrail()

    with pytest.raises(Exception):
        await guardrail.async_pre_call_hook(
            None,
            None,
            {
                "model": "gpt-5.6-sol",
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": None},
            },
            "pass_through_endpoint",
        )


@pytest.mark.asyncio
async def test_materializes_scalar_effort_for_adaptive_thinking() -> None:
    guardrail: ReasoningEffortRangeGuardrail = make_guardrail()
    data = {
        "model": "gpt-5.6-sol",
        "reasoning_effort": "low",
        "thinking": {"type": "adaptive"},
    }

    result = await guardrail.async_pre_call_hook(None, None, data, "anthropic_messages")

    assert result["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert "output_config" not in result


@pytest.mark.asyncio
async def test_clamps_xhigh_legacy_budget_to_high_ceiling() -> None:
    guardrail = make_guardrail(max_effort="high", default_effort="high")

    result = await guardrail.async_pre_call_hook(
        None,
        None,
        {
            "model": "gpt-5.6-sol",
            "thinking": {"type": "enabled", "budget_tokens": 8192},
        },
        "anthropic_messages",
    )

    assert result["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert "output_config" not in result


@pytest.mark.asyncio
async def test_preserves_in_range_legacy_budget() -> None:
    guardrail = make_guardrail(max_effort="high", default_effort="medium")
    data = {
        "model": "gpt-5.6-sol",
        "thinking": {"type": "enabled", "budget_tokens": 4096},
    }

    result = await guardrail.async_pre_call_hook(None, None, data, "anthropic_messages")

    assert result["thinking"] == {"type": "enabled", "budget_tokens": 4096}


@pytest.mark.asyncio
async def test_clamped_budget_stays_bounded_on_nonadaptive_anthropic_model() -> None:
    from litellm.utils import get_optional_params

    model = "claude-sonnet-4-5-20250929"
    guardrail = make_guardrail(model=model, max_effort="low", default_effort="low")
    data = {
        "model": model,
        "thinking": {"type": "enabled", "budget_tokens": 8192},
    }

    result = await guardrail.async_pre_call_hook(None, None, data, "anthropic_messages")
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider="anthropic",
        thinking=result["thinking"],
    )

    assert result["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert optional_params["thinking"] == {"type": "enabled", "budget_tokens": 1024}


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_effort", [None, "low"])
async def test_rejects_null_adaptive_output_effort(reasoning_effort: str | None) -> None:
    guardrail: ReasoningEffortRangeGuardrail = make_guardrail()
    data = {
        "model": "gpt-5.6-sol",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": None},
    }
    if reasoning_effort is not None:
        data["reasoning_effort"] = reasoning_effort

    with pytest.raises(Exception, match="reasoning effort must be one of"):
        await guardrail.async_pre_call_hook(None, None, data, "anthropic_messages")


@pytest.mark.asyncio
@pytest.mark.parametrize("output_effort", [None, "high"])
async def test_rejects_enabled_thinking_output_effort_bypass(output_effort: str | None) -> None:
    guardrail = make_guardrail()
    data = {
        "model": "gpt-5.6-sol",
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "output_config": {"effort": output_effort},
    }

    with pytest.raises(Exception):
        await guardrail.async_pre_call_hook(None, None, data, "anthropic_messages")


@pytest.mark.asyncio
async def test_accepts_matching_enabled_thinking_output_effort() -> None:
    guardrail = make_guardrail()
    data = {
        "model": "gpt-5.6-sol",
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "output_config": {"effort": "low"},
    }

    result = await guardrail.async_pre_call_hook(None, None, data, "anthropic_messages")

    assert result["output_config"] == {"effort": "low"}


@pytest.mark.asyncio
async def test_fails_closed_on_conflicting_or_malformed_values() -> None:
    guardrail = make_guardrail()

    with pytest.raises(Exception, match="conflicting reasoning effort values"):
        await guardrail.async_pre_call_hook(
            None,
            None,
            {
                "model": "gpt-5.6-sol",
                "reasoning_effort": "low",
                "reasoning": {"effort": "medium"},
            },
            "aresponses",
        )

    with pytest.raises(Exception, match="thinking.budget_tokens"):
        await guardrail.async_pre_call_hook(
            None,
            None,
            {
                "model": "gpt-5.6-sol",
                "thinking": {"type": "enabled", "budget_tokens": True},
            },
            "anthropic_messages",
        )


@pytest.mark.asyncio
async def test_rejects_conflicting_values_even_when_both_would_clamp_to_same_boundary() -> None:
    guardrail = make_guardrail()
    data = {
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "reasoning": {"effort": "xhigh"},
    }

    with pytest.raises(Exception, match="conflicting reasoning effort values"):
        await guardrail.async_pre_call_hook(None, None, data, "aresponses")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [None, {}, {"effort": None}, True, 3, [], "bogus"],
)
async def test_rejects_malformed_chat_effort(value: object) -> None:
    guardrail = make_guardrail()

    with pytest.raises(Exception):
        await guardrail.async_pre_call_hook(
            None,
            None,
            {"model": "gpt-5.6-sol", "reasoning_effort": value},
            "acompletion",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "thinking",
    [None, {}, {"budget_tokens": 1024}, {"type": "enabled"}],
)
async def test_rejects_malformed_thinking(thinking: object) -> None:
    guardrail = make_guardrail()

    with pytest.raises(Exception):
        await guardrail.async_pre_call_hook(
            None,
            None,
            {"model": "gpt-5.6-sol", "thinking": thinking},
            "anthropic_messages",
        )


@pytest.mark.asyncio
async def test_fails_closed_when_attached_to_another_model() -> None:
    guardrail = make_guardrail()

    with pytest.raises(Exception, match="configured for model"):
        await guardrail.async_pre_call_hook(
            None,
            None,
            {"model": "gpt-5.6-terra", "reasoning_effort": "medium"},
            "acompletion",
        )


def test_config_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="min_effort must not exceed max_effort"):
        ReasoningEffortRangeGuardrailConfigModel(
            model="gpt-5.6-sol",
            min_effort="high",
            max_effort="medium",
        )


def test_initializer_registers_db_compatible_callback() -> None:
    params = LitellmParams(
        guardrail="reasoning_effort_range",
        mode="pre_call",
        default_on=False,
        model="gpt-5.6-sol",
        min_effort="none",
        max_effort="medium",
        default_effort="medium",
    )
    guardrail = {
        "guardrail_id": "test-id",
        "guardrail_name": "sol-medium",
        "litellm_params": params,
    }

    with patch("litellm.logging_callback_manager.add_litellm_callback") as add_callback:
        callback = initialize_guardrail(params, guardrail)  # type: ignore[arg-type]

    assert isinstance(callback, ReasoningEffortRangeGuardrail)
    assert callback.model == "gpt-5.6-sol"
    add_callback.assert_called_once_with(callback)


def test_provider_exposes_admin_ui_config_model() -> None:
    from litellm.proxy.guardrails.guardrail_endpoints import _get_fields_from_model

    config_model = ReasoningEffortRangeGuardrail.get_config_model()

    assert config_model is ReasoningEffortRangeGuardrailConfigModel
    assert config_model.ui_friendly_name() == "Reasoning Effort Range"
    assert config_model.model_fields["model"].is_required()
    assert config_model.model_fields["max_effort"].is_required()
    fields = _get_fields_from_model(config_model)
    assert fields["max_effort"]["type"] == "select"
    assert fields["max_effort"]["options"] == [
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]


def test_admin_update_revalidates_and_replaces_range() -> None:
    guardrail = make_guardrail()
    updated = LitellmParams(
        guardrail="reasoning_effort_range",
        mode="pre_call",
        default_on=False,
        model="gpt-5.6-terra",
        min_effort="low",
        max_effort="high",
        default_effort="medium",
    )

    guardrail.update_in_memory_litellm_params(updated)

    assert guardrail.model == "gpt-5.6-terra"
    assert guardrail.min_effort == "low"
    assert guardrail.max_effort == "high"
    assert guardrail.default_effort == "medium"
