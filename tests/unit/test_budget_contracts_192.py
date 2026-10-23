"""Strict host metadata boundaries; no network or business database."""

from dataclasses import FrozenInstanceError

import pytest

from power_loop.runtime.budget_contracts import (
    BudgetAck,
    BudgetConfig,
    BudgetGrant,
    BudgetPlan,
    BudgetSubmission,
    LocalBudgetScope,
)
from power_loop.runtime.jobs import JobError

from .test_model_budgets_192 import Host

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "limits",
    [{}, {"max_calls": True}, {"max_tokens": -1}, {"max_inflight": 2**63}, {"max_calls": 1.1}, {"max_calls": "1"}],
)
def test_local_caps_are_explicit_exact_bounded_numbers(limits):
    with pytest.raises(JobError, match="invalid_local_budget_scope"):
        LocalBudgetScope("scope", **limits)


@pytest.mark.parametrize("key", ["", " ", "a\nb", "a\x00b", "a\u200bb", "a" * 256, False])
def test_authority_keys_are_printable_and_not_model_supplied_free_text(key):
    with pytest.raises(JobError, match="invalid_local_budget_scope"):
        LocalBudgetScope(key, max_calls=1)


@pytest.mark.parametrize(
    "change",
    [
        {"timeout_s": True},
        {"timeout_s": float("inf")},
        {"timeout_s": 0.01},
        {"max_pending": 0},
        {"max_pending": True},
        {"max_pending": 10001},
        {"provider": object()},
        {"provider_key": ""},
    ],
)
def test_bad_provider_or_delivery_bounds_fail_before_any_admission(change):
    values = dict(provider_key="configured-host", scopes=(LocalBudgetScope("s", max_calls=1),), provider=Host())
    values.update(change)
    with pytest.raises(JobError, match="invalid_budget_config"):
        BudgetConfig(**values)


def test_policy_manifest_is_stable_sorted_and_excludes_credentials_and_object_identity():
    provider = Host()
    config = BudgetConfig(
        "stable-binding", (LocalBudgetScope("z", max_tokens=0), LocalBudgetScope("a", max_calls=1)), provider
    )
    assert [s["key"] for s in config.manifest()["scopes"]] == ["a", "z"]
    assert config.manifest() == BudgetConfig("stable-binding", tuple(reversed(config.scopes)), Host()).manifest()
    assert config.manifest()["max_pending"] == 100
    with pytest.raises(FrozenInstanceError):
        config.provider_key = "new allowance"


@pytest.mark.parametrize("value", [True, -1, 2**63, 1.5, None])
def test_token_bound_is_not_a_float_or_unknown_zero(value):
    with pytest.raises(JobError, match="invalid_budget_token_bound"):
        BudgetPlan(value, {})


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BudgetGrant("id", "a" * 64, True, "reserved"),
        lambda: BudgetGrant("id", "a" * 64, 0, "reserved"),
        lambda: BudgetGrant("id", "a" * 64, 1, "completed"),
        lambda: BudgetAck("id", True, "a" * 64),
        lambda: BudgetAck("id", 1, "bad"),
        lambda: BudgetSubmission("id", 0, {}),
    ],
)
def test_receipts_have_typed_identity_state_and_revision(factory):
    with pytest.raises(JobError, match="invalid_budget_"):
        factory()


def test_plan_submission_and_grant_detach_nested_host_objects():
    original = {"nested": {"count": 1}}
    values = [
        BudgetPlan(10, original).request,
        BudgetSubmission("a", 1, original).request,
        BudgetGrant("a", "a" * 64, 1, "reserved", original).receipt,
    ]
    original["nested"]["count"] = 99
    assert all(v["nested"]["count"] == 1 for v in values)


@pytest.mark.parametrize(
    "body", [{"a": [0] * 257}, {"a": "x" * 65536}, {str(i): i for i in range(129)}, {1: "non-string key"}]
)
def test_opaque_metadata_is_size_bounded_before_persistence(body):
    with pytest.raises(JobError, match="invalid_budget_payload"):
        BudgetPlan(10, body)


@pytest.mark.parametrize(
    "values",
    [
        {"binding": "session"},
        {"binding": True},
        {"binding": "root", "owner_kinds": ("workflow",)},
        {"binding": "owner", "owner_kinds": []},
        {"binding": "owner", "owner_kinds": ()},
        {"binding": "owner", "owner_kinds": ("invalid",)},
        {"binding": "owner", "owner_kinds": ("workflow", "workflow")},
    ],
)
def test_scope_templates_cannot_be_freeform_model_selected_bindings(values):
    with pytest.raises(JobError, match="invalid_local_budget_scope"):
        LocalBudgetScope("s", max_calls=1, **values)


def test_authority_manifest_keeps_original_v1_shape_and_tree_mode_is_explicit():
    old = BudgetConfig("binding", (LocalBudgetScope("s", max_calls=1),), Host())
    assert old.manifest() == {
        "version": 1,
        "provider_key": "binding",
        "max_pending": 100,
        "scopes": [{"key": "s", "max_calls": 1, "max_tokens": None, "max_inflight": None}],
    }
    tree = BudgetConfig(
        "binding",
        (LocalBudgetScope("s", max_calls=1, binding="owner", owner_kinds=("workflow_node", "workflow")),),
        Host(),
    )
    assert tree.manifest()["version"] == 2
    assert tree.manifest()["scopes"][0]["owner_kinds"] == ["workflow", "workflow_node"]
    assert (
        tree.manifest()
        == BudgetConfig(
            "binding",
            (LocalBudgetScope("s", max_calls=1, binding="owner", owner_kinds=("workflow", "workflow_node")),),
            Host(),
        ).manifest()
    )
