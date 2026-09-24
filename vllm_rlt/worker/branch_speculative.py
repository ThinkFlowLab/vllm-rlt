"""Optional single-fallback branch speculation on top of fixed-depth decoding.

This is an experimental extension of :class:`SpeculativeRunner`. It is
greedy-only and synchronous-eager-only, and it opens at most one alternate
suffix per request per round. The shallow top-1/top-2 full-softmax margin gates
the fork; primary and live alternate rows share the same shallow-draft and
deep-verification batches, so the extra path costs work rather than serial
rounds. The alternate KV shares every complete committed page before the
current token and copies only the private stem. Deep verification injects the
per-layer stem copy through an explicit cache view, so the actual cache still
owns every write and liveness check. When the target confirms the pre-selected
alternate candidate, the branch is swapped into the root allocation and freed.

Correctness is the goal here; the prototype's throughput numbers are historical
and are not reproduced or claimed. BF16 exact agreement with the plain runner is
not guaranteed because branch rows change batch shapes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_rlt.worker.speculative import (
    SpeculativeResult,
    SpeculativeRunner,
    SpeculativeStats,
)

_ALT_SUFFIX = "::alt"


@dataclass
class BranchSpeculativeStats(SpeculativeStats):
    """Speculative counters plus branch bookkeeping."""

    fork_rounds: int = 0
    alternate_selected: int = 0
    primary_rejected_at_fork: int = 0
    primary_rejected_elsewhere: int = 0
    alt_drafted: int = 0
    alt_accepted: int = 0
    rollback_rounds: int = 0


class _VerificationCacheView:
    """Delegate real writes, then copy each fork's deep stem for this layer.

    ``_prepare_batch`` is always called on the actual cache, so the prepared
    batch still owns the actual cache and every owner/liveness check in the
    actual ``_write_prepared``/``_attend_prepared`` continues to run. The view
    only injects the alternate-stem copy between the write and the attention of
    each layer.
    """

    def __init__(self, runner: BranchSpeculativeRunner, actual, fork_plans):
        self._runner = runner
        self._actual = actual
        # (root request id, alternate request id, token_start, fork offset)
        self._fork_plans = tuple(fork_plans)
        self.depth = None

    def _write_prepared(self, layer, batch, k, v):
        self._actual._write_prepared(layer, batch, k, v)
        for rid, alt_id, p0, fork in self._fork_plans:
            self._runner._copy_deep_stem(rid, alt_id, p0, fork, layer, self.depth)

    def _attend_prepared(self, layer, batch, q):
        return self._actual._attend_prepared(layer, batch, q)


class _Plan:
    """Per-request mutable branch state for one execute call."""

    __slots__ = (
        "item",
        "request",
        "rid",
        "alt_id",
        "p0",
        "token_count",
        "k_drafts",
        "primary_cand",
        "primary_state",
        "alt_cand",
        "alt_state",
        "fork",
        "alt_live",
        "verify_start",
        "alt_verify_start",
    )

    def __init__(self, item):
        self.item = item
        self.request = item.request
        self.rid = item.request.request_id
        self.alt_id: str | None = None
        self.p0 = item.token_start
        self.token_count = item.token_count
        self.k_drafts = item.token_count - 1
        self.primary_cand: list[int] = []
        self.primary_state: list[torch.Tensor] = []
        self.alt_cand: dict[int, int] = {}
        self.alt_state: dict[int, torch.Tensor] = {}
        self.fork: int | None = None
        self.alt_live = False
        self.verify_start = 0
        self.alt_verify_start: int | None = None


class BranchSpeculativeRunner(SpeculativeRunner):
    """One bounded top-2 fallback per request and round, greedily verified."""

    def __init__(self, model, cache, config):
        super().__init__(model, cache, config)
        self.margin = config.fallback_margin
        self.stats = BranchSpeculativeStats()

    # ------------------------------------------------------------------ #
    # core call: always prepare on the actual cache, optionally run through
    # the verification view that copies alternate stems after each write.
    # ------------------------------------------------------------------ #
    def _core(self, hidden, ids, positions, depth, *, packed=False, cache=None):
        batch = self.cache._prepare_batch(
            ids,
            [depth] * len(ids),
            positions,
            packed_prefill=packed and getattr(self.cache.attention, "generation", None) == 4,
        )
        hidden, _ = self.model.recurrent_prepared(
            hidden,
            batch,
            self.cache if cache is None else cache,
            compute_gate=False,
        )
        return hidden

    def _gate_decision(self, logits, offset):
        """Return ``(trigger, second_best_token)`` using the full softmax margin."""
        probs = torch.softmax(logits.float(), dim=-1)
        primary = int(logits.argmax())
        ranked = torch.topk(probs, 2).indices.tolist()
        second = next(token for token in ranked if token != primary)
        margin = float(probs[primary] - probs[second])
        return margin <= self.margin, second

    # ------------------------------------------------------------------ #
    # temporary KV branch
    # ------------------------------------------------------------------ #
    def _new_alt_id(self, rid: str) -> str:
        """Return a cache ID that does not collide with a live allocation.

        Alternate IDs only need to be unique for the duration of one execute
        call; the branch is freed before the next scheduling round, so checking
        the current allocations is sufficient.
        """
        base = f"{rid}{_ALT_SUFFIX}"
        candidate = base
        counter = 0
        while candidate in self.cache._allocations:
            counter += 1
            candidate = f"{base}::{counter}"
        return candidate

    def _create_alt(self, rid: str, alt_id: str, p0: int) -> bool:
        """Open the alternate allocation, or return False if capacity is short.

        The child mirrors the parent's current page count so the commit swap
        keeps the same block-table shape. Capacity is checked against the truly
        free list first: ``_claim`` can evict cached prefix entries, and a fork
        must not sacrifice prefix reuse or abort an admitted request.
        """
        cache = self.cache
        block_size = cache.block_size
        parent = cache._get_allocation(rid)
        shared_pages = p0 // block_size
        prefix = tuple(
            tuple(parent.block_tables[d][page] for d in range(cache.storage_depths))
            for page in range(shared_pages)
        )
        pages = len(parent.block_tables[0])
        initial_tokens = min(parent.max_tokens, pages * block_size)
        fresh = (pages - shared_pages) * cache.storage_depths
        if fresh > len(cache._free_blocks):
            return False
        return cache.allocate(
            alt_id, parent.max_tokens, initial_tokens=initial_tokens, prefix=prefix
        )

    def _copy_positions(self, rid, alt_id, positions, depths, layers=None):
        """Copy the contiguous private stem using views, without GPU index tensors."""
        if not positions:
            return
        cache = self.cache
        parent, child = cache._get_allocation(rid), cache._get_allocation(alt_id)
        layer_range = range(cache.num_layers) if layers is None else layers
        selected = slice(None) if layers is None else layers[0]
        begin, end, size = positions[0], positions[-1] + 1, cache.block_size
        for depth in depths:
            plane = cache._plane(depth)
            cursor = begin
            while cursor < end:
                page = cursor // size
                stop = min(end, (page + 1) * size)
                left, right = cursor % size, cursor % size + stop - cursor
                src, dst = parent.block_tables[plane][page], child.block_tables[plane][page]
                if src != dst:
                    cache.key_cache[dst, selected, left:right].copy_(
                        cache.key_cache[src, selected, left:right]
                    )
                    cache.value_cache[dst, selected, left:right].copy_(
                        cache.value_cache[src, selected, left:right]
                    )
                cursor = stop
            for layer in layer_range:
                written = child.written[plane][layer]
                for position in positions:
                    written.add(position)

    def _stem_range(self, p0: int, fork: int):
        block_size = self.cache.block_size
        page_start = (p0 // block_size) * block_size
        return list(range(page_start, p0 + fork + 1))

    def _copy_shallow_stem(self, rid, alt_id, p0, fork):
        # The primary has finished shallow-drafting the shared stem; copy it once
        # before the combined child drafting starts. No per-layer hook is needed.
        self._copy_positions(
            rid, alt_id, self._stem_range(p0, fork), depths=range(self.config.draft_loops)
        )

    def _copy_deep_stem(self, rid, alt_id, p0, fork, layer, depth):
        self._copy_positions(
            rid, alt_id, self._stem_range(p0, fork), depths=[depth], layers=[layer]
        )

    def _commit_alt(self, rid: str, alt_id: str) -> None:
        """Swap the branch's private suffix into the root allocation, then free it.

        ``max_tokens`` and the current page count match, so swapping the tables
        and written planes is an exact commit.
        """
        root = self.cache._allocations[rid]
        child = self.cache._allocations[alt_id]
        root.block_tables, child.block_tables = child.block_tables, root.block_tables
        root.written, child.written = child.written, root.written
        self.cache.free(alt_id)

    # ------------------------------------------------------------------ #
    # execution
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def execute(self, batch):
        return self._execute_batched(batch.items)

    def _execute_batched(self, items):
        plans = [_Plan(item) for item in items]
        try:
            self._draft_shallow(plans, self.config.draft_loops)
            return self._verify_and_commit(plans, self.config.draft_loops, self.config.target_loops)
        finally:
            for plan in plans:
                if plan.alt_live and plan.alt_id in self.cache._allocations:
                    self.cache.free(plan.alt_id)
                    plan.alt_live = False

    # ------------------------------------------------------------------ #
    # shallow drafting: one packed call per depth and offset
    # ------------------------------------------------------------------ #
    def _draft_shallow(self, plans, draft_loops):
        model, device = self.model, self.device
        max_count = max((plan.token_count for plan in plans), default=0)
        for offset in range(max_count):
            active = [plan for plan in plans if offset < plan.token_count]
            tokens: list[int] = []
            ids: list[str] = []
            positions: list[int] = []
            row_refs: list[tuple[_Plan, int]] = []
            for plan in active:
                token = (
                    plan.request.input_token_id if offset == 0 else plan.primary_cand[offset - 1]
                )
                tokens.append(token)
                ids.append(plan.rid)
                positions.append(plan.p0 + offset)
                row_refs.append((plan, 0))
                if plan.fork is not None:
                    tokens.append(plan.alt_cand[offset - 1])
                    ids.append(plan.alt_id)
                    positions.append(plan.p0 + offset)
                    row_refs.append((plan, 1))

            hidden = model.prelude(torch.tensor(tokens, device=device, dtype=torch.long))
            for depth in range(draft_loops):
                hidden = self._core(hidden, ids, positions, depth)

            needs_coda = [
                row for row, (plan, _) in enumerate(row_refs) if offset + 1 < plan.token_count
            ]
            if not needs_coda:
                coda_logits = None
            else:
                # Most offsets use every active row; avoid an extra device gather.
                coda_hidden = hidden if len(needs_coda) == len(row_refs) else hidden[needs_coda]
                coda_logits = model.coda(coda_hidden)
            coda_row = {row: index for index, row in enumerate(needs_coda)}

            for row, (plan, kind) in enumerate(row_refs):
                if kind == 0:
                    plan.primary_state.append(hidden[row])
                else:
                    plan.alt_state[offset] = hidden[row]

            for row, (plan, kind) in enumerate(row_refs):
                if offset + 1 >= plan.token_count:
                    continue
                logits = coda_logits[coda_row[row]]
                if kind == 1:
                    plan.alt_cand[offset] = int(logits.argmax())
                    continue
                plan.primary_cand.append(int(logits.argmax()))
                if plan.fork is None and offset < plan.k_drafts:
                    trigger, second = self._gate_decision(logits, offset)
                    if trigger:
                        alt_id = self._new_alt_id(plan.rid)
                        if self._create_alt(plan.rid, alt_id, plan.p0):
                            plan.alt_id = alt_id
                            plan.fork = offset
                            plan.alt_cand[offset] = second
                            plan.alt_live = True
                            self._copy_shallow_stem(plan.rid, alt_id, plan.p0, offset)
                            self.stats.fork_rounds += 1

    # ------------------------------------------------------------------ #
    # deep verification: primary chains + alternate suffixes in one pack
    # ------------------------------------------------------------------ #
    def _verify_and_commit(self, plans, draft_loops, target_loops):
        model = self.model
        states: list[torch.Tensor] = []
        verify_ids: list[str] = []
        verify_positions: list[int] = []
        fork_plans: list[tuple[str, str, int, int]] = []
        cursor = 0
        for plan in plans:
            plan.verify_start = cursor
            for offset in range(plan.token_count):
                states.append(plan.primary_state[offset])
                verify_ids.append(plan.rid)
                verify_positions.append(plan.p0 + offset)
            cursor += plan.token_count
            if plan.fork is not None:
                fork_plans.append((plan.rid, plan.alt_id, plan.p0, plan.fork))
                plan.alt_verify_start = cursor
                for offset in range(plan.fork + 1, plan.token_count):
                    states.append(plan.alt_state[offset])
                    verify_ids.append(plan.alt_id)
                    verify_positions.append(plan.p0 + offset)
                    cursor += 1

        hidden = torch.stack(states)
        if fork_plans:
            view = _VerificationCacheView(self, self.cache, fork_plans)
            for depth in range(draft_loops, target_loops):
                view.depth = depth
                hidden = self._core(
                    hidden, verify_ids, verify_positions, depth, packed=True, cache=view
                )
        else:
            for depth in range(draft_loops, target_loops):
                hidden = self._core(hidden, verify_ids, verify_positions, depth, packed=True)

        self.stats.verified_rows += len(verify_ids)
        logits = model.coda(hidden)
        results = []
        for plan in plans:
            primary_rows = logits[plan.verify_start : plan.verify_start + plan.token_count]
            if plan.fork is not None:
                alt_count = plan.token_count - (plan.fork + 1)
                alt_rows = logits[plan.alt_verify_start : plan.alt_verify_start + alt_count]
            else:
                alt_rows = None
            results.append(self._verify_plan(plan, primary_rows, alt_rows))
        return results

    def _verify_plan(self, plan, primary_rows, alt_rows):
        token_count = plan.token_count
        k_drafts = plan.k_drafts
        fork = plan.fork
        emitted: list[int] = []
        accepted = 0
        if fork is None:
            draft_count = k_drafts
            reject = None
            for offset in range(k_drafts):
                token = int(primary_rows[offset].argmax())
                if token == plan.primary_cand[offset]:
                    emitted.append(token)
                    accepted += 1
                else:
                    reject = offset
                    emitted.append(token)
                    break
            if reject is None:
                emitted.append(int(primary_rows[token_count - 1].argmax()))
        else:
            draft_count = k_drafts + (k_drafts - fork)
            self.stats.alt_drafted += k_drafts - fork
            reject = None
            reject_token = None
            for offset in range(k_drafts):
                token = int(primary_rows[offset].argmax())
                if token == plan.primary_cand[offset]:
                    emitted.append(token)
                    accepted += 1
                else:
                    reject = offset
                    reject_token = token
                    break
            if reject is None:
                # Primary accepted in full: ordinary bonus, branch discarded.
                emitted.append(int(primary_rows[token_count - 1].argmax()))
            elif reject == fork and reject_token == plan.alt_cand[fork]:
                # The first rejection is exactly the gated fork and the target
                # confirms the pre-selected second-best token: follow the child.
                emitted.append(reject_token)
                accepted += 1
                alt_reject = None
                for offset in range(fork + 1, k_drafts):
                    token = int(alt_rows[offset - (fork + 1)].argmax())
                    if token == plan.alt_cand[offset]:
                        emitted.append(token)
                        accepted += 1
                    else:
                        alt_reject = offset
                        emitted.append(token)
                        break
                if alt_reject is None:
                    emitted.append(int(alt_rows[token_count - 1 - (fork + 1)].argmax()))
                self._commit_alt(plan.rid, plan.alt_id)
                plan.alt_live = False
                self.stats.alternate_selected += 1
                self.stats.alt_accepted += accepted - fork
                self.stats.primary_rejected_at_fork += 1
            else:
                # Primary rejected somewhere else: keep the ordinary correction.
                emitted.append(reject_token)
                self.stats.primary_rejected_elsewhere += 1

        if accepted < k_drafts:
            self.stats.rollback_rounds += 1
        self.stats.rounds += 1
        self.stats.drafted_tokens += draft_count
        return SpeculativeResult(emitted, accepted, draft_count)
