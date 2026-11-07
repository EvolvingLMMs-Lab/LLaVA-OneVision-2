"""CPU tests of the per-micro-batch recompute planner (OV2_OPT_MEM_BUDGET_GIB) and of the cross-rank balancing
assignment (OV2_OPT_BALANCE).

    PYTHONPATH=aiak_megatron:. python -m pytest tests/faster_training -q
"""
import itertools
import random

import pytest
import torch
from megatron.core import parallel_state

from aiak_training_llm.models.custom.common import activation_memory

LAYERS = {'lm': 36, 'vision': 24}       # 4B quick-start model


@pytest.mark.parametrize('pp, planned', [(1, True), (2, False), (4, False)])
def test_planner_stays_off_with_pipeline_parallelism(monkeypatch, capsys, pp, planned):
    # a pipeline stage may hold no vision encoder, which the planner cannot handle
    calls = []
    monkeypatch.setattr(activation_memory, '_BUDGET', 33.0)
    monkeypatch.setattr(activation_memory, '_LAYERS', dict(LAYERS))
    monkeypatch.setattr(activation_memory, 'install_planner', calls.append)
    monkeypatch.setattr(parallel_state, 'get_pipeline_model_parallel_world_size', lambda: pp)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    activation_memory.mark_model('model')
    assert bool(calls) == planned
    assert ('OV2_OPT_MEM_BUDGET_GIB is off' in capsys.readouterr().out) != planned


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
    for world, per_rank in ((4, 4), (8, 2), (2, 8), (48, 2)):
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


def _stay(plan):
    return sum(1 for rank, seqs in enumerate(plan) for src, _ in seqs if src == rank)


def test_balance_matching_is_optimal():
    balanced_iterator = pytest.importorskip('aiak_training_llm.train.balanced_iterator')
    rng = random.Random(1)
    for world, per_rank in ((2, 8), (4, 4), (6, 2), (7, 3)):
        for _ in range(20):
            costs = [(rng.choice([275.0, rng.uniform(275, 2500)]), r, j)
                     for r in range(world) for j in range(per_rank)]
            plan, _, perm = balanced_iterator.assign(costs, world, per_rank)
            groups = [plan[r] for r in perm]
            best = max(sum(1 for g in range(world) for src, _ in groups[g] if src == p[g])
                       for p in itertools.permutations(range(world)))
            assert sorted(perm) == list(range(world)) and _stay(plan) == best


def test_balance_one_sequence_per_rank_moves_nothing():
    balanced_iterator = pytest.importorskip('aiak_training_llm.train.balanced_iterator')
    rng = random.Random(2)
    costs = [(rng.uniform(275, 2500), r, 0) for r in range(48)]
    plan, loads, _ = balanced_iterator.assign(costs, 48, 1)
    assert _stay(plan) == 48 and max(loads) == max(c for c, _, _ in costs)


@pytest.mark.parametrize('dp_size, job_size, micro_batches, nodes, cross_node, off', [
    (8, 8, 2, 1, False, False),         # the quick-start layout
    (4, 4, 4, 1, False, False),
    (16, 16, 2, 2, True, False),        # several nodes, asked for
    (16, 16, 2, 2, False, True),        # several nodes, not asked for
    (48, 48, 1, 6, True, True),         # one sequence per rank: nothing to balance, even when asked for
    (8, 8, 1, 1, False, True),
    (4, 8, 4, 1, False, True),          # TP, PP or CP above 1
    (1, 1, 16, 1, False, True),
])
def test_balance_startup_problem(dp_size, job_size, micro_batches, nodes, cross_node, off):
    balanced_iterator = pytest.importorskip('aiak_training_llm.train.balanced_iterator')
    problem = balanced_iterator.startup_problem(dp_size, job_size, micro_batches, nodes, cross_node)
    assert (problem is not None) == off
