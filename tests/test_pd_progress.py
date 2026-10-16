"""Ready handoffs must progress before potentially blocking compute submission."""

from collections import deque
from unittest.mock import Mock

import pytest

from tests.test_stage_engine import make_engine
from tests.test_transfer_connector import make_connector, reserve


@pytest.fixture
def prefill_connector():
    engine = make_engine(role="prefill", atomic=True)
    connector = make_connector(engine)
    handoff = reserve(connector, "handoff", receiving=False)
    next_transfer = reserve(connector, "next", receiving=False)
    engine.suspend_transfer(handoff.request)
    ready = Mock()
    ready.query.return_value = True
    handoff.chunks = deque((ready, i, [], [], 1024) for i in range(3))
    handoff.next_sequence = 3
    handoff.compute_done = handoff.compute_released = True
    handoff.last_event = ready
    connector.transport.submit.return_value = True
    connector.notify.reset_mock()
    yield connector, handoff, next_transfer
    engine.close()


def test_ready_handoff_drains_before_compute_even_with_transfer_backpressure(prefill_connector):
    connector, handoff, next_transfer = prefill_connector
    connector.transport.submit.return_value = False
    connector.progress()
    assert len(handoff.chunks) == 3
    assert connector.engine._schedule() is None

    connector.transport.submit.return_value = True
    for remaining in (2, 1):
        connector.progress()
        assert len(handoff.chunks) == remaining
        assert connector.engine._schedule() is None
        connector.notify.assert_not_called()

    connector.progress()
    assert not handoff.chunks
    connector.notify.assert_called_once_with("commit", tid="handoff", count=3)
    batch = connector.engine._schedule()
    assert batch is not None
    assert [item.request for item in batch.requests] == [next_transfer.request]


def test_unready_handoff_does_not_prevent_other_prefill(prefill_connector):
    connector, handoff, next_transfer = prefill_connector
    handoff.last_event.query.return_value = False
    batch = connector.engine._schedule()
    assert batch is not None
    assert [item.request for item in batch.requests] == [next_transfer.request]
    assert len(handoff.chunks) == 3
