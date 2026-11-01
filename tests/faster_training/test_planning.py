"""CPU tests of the per-micro-batch recompute planner (OV2_OPT_MEM_BUDGET_GIB) and of the cross-rank balancing
assignment (OV2_OPT_BALANCE).

    PYTHONPATH=aiak_megatron:. python -m pytest tests/faster_training -q
"""
import random

import pytest

from aiak_training_llm.models.custom.common import activation_memory

LAYERS = {'lm': 36, 'vision': 24}       # 4B quick-start model


def test_plan_light_sequence_needs_no_recompute():
    p = activation_memory.plan(T=8.19, P=0.0, budget=33, layers=LAYERS, qkv=True)
    assert not (p['act_lm'] or p['act_vision'] or p['fc1_vision'])
    assert p['fc1_lm'] == p['full_vision'] == p['full_lm'] == 0
    assert p['est_gib'] <= 33


@pytest.mark.parametrize('patches_k', [5.0, 12.0, 20.0, 27.8, 32.388])
def test_plan_stays_within_budget(patches_k):
    p = activation_memory.plan(T=8.19, P=patches_k, budget=33, layers=LAYERS, qkv=True)
    assert p['est_gib'] <= 33 + 1e-9
    assert 0 <= p['fc1_lm'] <= LAYERS['lm'] and 0 <= p['full_vision'] <= LAYERS['vision']
    assert 0 <= p['full_lm'] <= LAYERS['lm']


def test_plan_is_monotone_in_vision_patches():
    def cost(p):
        return (p['act_lm'], p['act_vision'], p['fc1_vision'], p['fc1_lm'], p['full_vision'], p['full_lm'])
    plans = [activation_memory.plan(T=8.19, P=k / 2, budget=33, layers=LAYERS, qkv=True) for k in range(0, 66)]
    for a, b in zip(plans, plans[1:]):
        assert all(x <= y for x, y in zip(cost(a), cost(b)))


def test_plan_without_qkv_recompute_needs_more():
    a = activation_memory.plan(T=8.19, P=20.0, budget=33, layers=LAYERS, qkv=True)
    b = activation_memory.plan(T=8.19, P=20.0, budget=33, layers=LAYERS, qkv=False)
    assert b['est_gib'] <= 33 and a['est_gib'] <= 33
    assert b['fc1_lm'] >= a['fc1_lm'] and b['full_vision'] >= a['full_vision']


def test_balance_assignment_is_a_balanced_permutation():
    balanced_iterator = pytest.importorskip('aiak_training_llm.train.balanced_iterator')
    rng = random.Random(0)
    for world, per_rank in ((4, 4), (8, 2), (2, 8)):
        for _ in range(20):
            costs = [(rng.choice([275.0, rng.uniform(275, 2500)]), r, j)
                     for r in range(world) for j in range(per_rank)]
            plan, loads, _ = balanced_iterator.assign(costs, world, per_rank)
            assert [len(x) for x in plan] == [per_rank] * world
            taken = sorted(x for rank in plan for x in rank)
            assert taken == sorted((r, j) for _, r, j in costs)        # every sequence exactly once
            by = {(r, j): c for c, r, j in costs}
            assert sorted(sum(by[x] for x in rank) for rank in plan) == pytest.approx(sorted(loads))
            before = max(sum(c for c, r, _ in costs if r == k) for k in range(world))
            assert max(loads) <= before + 1e-9
            assert plan == balanced_iterator.assign(list(reversed(costs)), world, per_rank)[0]   # deterministic
