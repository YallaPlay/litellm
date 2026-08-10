"""Guardrail that enforces an inclusive reasoning-effort range per model."""

from typing import TYPE_CHECKING, Dict, List, Mapping, NoReturn, Optional, Type, cast

from litellm.caching import DualCache
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.reasoning_effort_utils import (
    reasoning_effort_from_thinking_budget,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.reasoning_effort_range import (
    EFFORT_RANK,
    ReasoningEffort,
    ReasoningEffortRangeGuardrailConfigModel,
)
from litellm.types.utils import CallTypesLiteral

if TYPE_CHECKING:
    from pydantic import BaseModel

    from litellm.types.guardrails import LitellmParams
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

REQUEST_EFFORT_RANK: Dict[str, int] = {
    **EFFORT_RANK,
    "default": max(EFFORT_RANK.values()) + 1,
}
EFFORT_NAMES = ", ".join(REQUEST_EFFORT_RANK)
ANTHROPIC_MESSAGE_CALL_TYPES = frozenset({"anthropic_messages", "aanthropic_messages"})
RESPONSES_CALL_TYPES = frozenset({"responses", "aresponses", "_aresponses_websocket"})


class ReasoningEffortRangeGuardrail(CustomGuardrail):
    """Enforce a configured reasoning-effort range on one exact model name."""

    def __init__(
        self,
        guardrail_name: str,
        event_hook: GuardrailEventHooks,
        model: str,
        max_effort: ReasoningEffort,
        min_effort: ReasoningEffort = "none",
        default_effort: Optional[ReasoningEffort] = None,
        default_on: bool = False,
    ) -> None:
        super().__init__(  # pyright: ignore[reportUnknownMemberType] -- base accepts provider kwargs
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
        )
        self._configure(
            model=model,
            min_effort=min_effort,
            max_effort=max_effort,
            default_effort=default_effort,
        )

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel[BaseModel]"]]:
        return ReasoningEffortRangeGuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> List[GuardrailEventHooks]:
        return [GuardrailEventHooks.pre_call]

    def _configure(
        self,
        model: str,
        min_effort: ReasoningEffort,
        max_effort: ReasoningEffort,
        default_effort: Optional[ReasoningEffort],
    ) -> None:
        config = ReasoningEffortRangeGuardrailConfigModel(
            model=model,
            min_effort=min_effort,
            max_effort=max_effort,
            default_effort=default_effort,
        )
        self.model = config.model
        self.min_effort = config.min_effort
        self.max_effort = config.max_effort
        self.default_effort = config.default_effort or config.max_effort

    def update_in_memory_litellm_params(self, litellm_params: "LitellmParams") -> None:
        super().update_in_memory_litellm_params(litellm_params)
        if litellm_params.model is None or litellm_params.max_effort is None:
            raise ValueError("model and max_effort are required")
        self._configure(
            model=litellm_params.model,
            min_effort=litellm_params.min_effort or "none",
            max_effort=litellm_params.max_effort,
            default_effort=litellm_params.default_effort,
        )

    def _raise(self, message: str) -> NoReturn:
        raise GuardrailRaisedException(
            guardrail_name=self.guardrail_name or "reasoning_effort_range",
            message=message,
            should_wrap_with_default_message=False,
        )

    def _append_effort(self, values: List[str], value: object, field: str) -> None:
        if isinstance(value, dict):
            structured_value = cast(Mapping[str, object], value)
            if "effort" not in structured_value:
                self._raise(f"{field}.effort is required")
            value = structured_value["effort"]
        if not isinstance(value, str) or value not in REQUEST_EFFORT_RANK:
            self._raise(f"reasoning effort must be one of {EFFORT_NAMES}")
        values.append(value)

    def _thinking_effort(self, data: Dict[str, object]) -> Optional[str]:
        if "thinking" not in data:
            return None
        thinking_value = data["thinking"]
        if not isinstance(thinking_value, dict):
            self._raise("thinking must be an object")
        thinking = cast(Mapping[str, object], thinking_value)

        thinking_type = thinking.get("type")
        if thinking_type == "disabled":
            return "none"
        if thinking_type == "enabled":
            if "budget_tokens" not in thinking:
                self._raise("thinking.budget_tokens is required when thinking is enabled")
            budget_tokens = thinking["budget_tokens"]
            if isinstance(budget_tokens, bool) or not isinstance(budget_tokens, int) or budget_tokens < 0:
                self._raise("thinking.budget_tokens must be a non-negative integer")
            return reasoning_effort_from_thinking_budget(budget_tokens)
        if thinking_type == "adaptive":
            return None
        self._raise("thinking.type must be disabled, enabled, or adaptive")

    def _output_config_effort(self, data: Dict[str, object]) -> Optional[str]:
        if "output_config" not in data:
            return None
        output_config_value = data["output_config"]
        if not isinstance(output_config_value, dict):
            self._raise("output_config must be an object")
        output_config = cast(Mapping[str, object], output_config_value)
        if "effort" not in output_config:
            return None
        effort = output_config["effort"]
        if not isinstance(effort, str) or effort not in REQUEST_EFFORT_RANK:
            self._raise(f"reasoning effort must be one of {EFFORT_NAMES}")
        return effort

    def _requested_effort(self, data: Dict[str, object], call_type: CallTypesLiteral) -> Optional[str]:
        values: List[str] = []
        if "reasoning_effort" in data:
            self._append_effort(values, data["reasoning_effort"], "reasoning_effort")

        if "reasoning" in data:
            reasoning_value = data["reasoning"]
            if not isinstance(reasoning_value, dict):
                self._raise("reasoning must be an object")
            reasoning = cast(Mapping[str, object], reasoning_value)
            if "effort" in reasoning:
                self._append_effort(values, reasoning["effort"], "reasoning")

        thinking_effort = self._thinking_effort(data)
        if thinking_effort is not None:
            values.append(thinking_effort)

        if call_type in ANTHROPIC_MESSAGE_CALL_TYPES:
            output_config_effort = self._output_config_effort(data)
            if output_config_effort is not None:
                values.append(output_config_effort)

        if not values:
            return None
        if len(set(values)) != 1:
            self._raise("conflicting reasoning effort values")
        return values[0]

    def _set_default_effort(self, data: Dict[str, object], call_type: CallTypesLiteral) -> None:
        if call_type in ANTHROPIC_MESSAGE_CALL_TYPES:
            data["thinking"] = {"type": "adaptive"}
            output_config_value = data.get("output_config")
            if output_config_value is None:
                output_config: Dict[str, object] = {}
                data["output_config"] = output_config
            elif isinstance(output_config_value, dict):
                output_config = cast(Dict[str, object], output_config_value)
            else:
                self._raise("output_config must be an object")
            output_config["effort"] = self.default_effort
            return

        reasoning_effort = data.get("reasoning_effort")
        if isinstance(reasoning_effort, dict):
            cast(Dict[str, object], reasoning_effort)["effort"] = self.default_effort
            return

        reasoning_value = data.get("reasoning")
        if isinstance(reasoning_value, dict) or call_type in RESPONSES_CALL_TYPES:
            if reasoning_value is None:
                reasoning: Dict[str, object] = {}
                data["reasoning"] = reasoning
            elif isinstance(reasoning_value, dict):
                reasoning = cast(Dict[str, object], reasoning_value)
            else:
                self._raise("reasoning must be an object")
            reasoning["effort"] = self.default_effort
            return

        data["reasoning_effort"] = self.default_effort

    def _set_adaptive_effort(
        self,
        data: Dict[str, object],
        call_type: CallTypesLiteral,
        requested_effort: str,
    ) -> None:
        """Make an allowed effort explicit when adaptive thinking would otherwise choose it."""
        if call_type not in ANTHROPIC_MESSAGE_CALL_TYPES:
            return
        thinking_value = data.get("thinking")
        if not isinstance(thinking_value, dict):
            return
        thinking = cast(Mapping[str, object], thinking_value)
        if thinking.get("type") != "adaptive":
            return
        output_config_value = data.get("output_config")
        if output_config_value is None:
            output_config: Dict[str, object] = {}
            data["output_config"] = output_config
        elif isinstance(output_config_value, dict):
            output_config = cast(Dict[str, object], output_config_value)
        else:
            self._raise("output_config must be an object")
        output_config.setdefault("effort", requested_effort)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Dict[str, object],
        call_type: CallTypesLiteral,
    ) -> Dict[str, object]:
        request_model = data.get("model")
        if request_model != self.model:
            self._raise(f"guardrail is configured for model '{self.model}', not '{request_model}'")

        requested_effort = self._requested_effort(data, call_type)
        if requested_effort is None:
            self._set_default_effort(data, call_type)
            return data

        rank = REQUEST_EFFORT_RANK[requested_effort]
        if rank < EFFORT_RANK[self.min_effort]:
            self._raise(f"reasoning effort '{requested_effort}' is not allowed; minimum allowed is {self.min_effort}")
        if rank > EFFORT_RANK[self.max_effort]:
            self._raise(f"reasoning effort '{requested_effort}' is not allowed; maximum allowed is {self.max_effort}")
        self._set_adaptive_effort(data, call_type, requested_effort)
        return data
