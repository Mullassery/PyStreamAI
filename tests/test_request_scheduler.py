"""Tests for pystreamai.request_scheduler - priority queue and quotas."""

from pystreamai.request_scheduler import (
    RequestScheduler,
    RequestPriority,
    ScheduledRequest,
    FairShareScheduler,
    DeadlineScheduler,
)


def make_request(request_id="r1", priority=RequestPriority.NORMAL.value, model_id="m1", **kwargs):
    return ScheduledRequest(priority=priority, request_id=request_id, model_id=model_id, **kwargs)


class TestRequestScheduler:
    def test_dequeue_returns_highest_priority_first(self):
        scheduler = RequestScheduler()
        scheduler.enqueue(make_request("low", priority=RequestPriority.LOW.value))
        scheduler.enqueue(make_request("critical", priority=RequestPriority.CRITICAL.value))
        scheduler.enqueue(make_request("normal", priority=RequestPriority.NORMAL.value))

        assert scheduler.dequeue().request_id == "critical"
        assert scheduler.dequeue().request_id == "normal"
        assert scheduler.dequeue().request_id == "low"

    def test_dequeue_on_empty_queue_returns_none(self):
        assert RequestScheduler().dequeue() is None

    def test_enqueue_rejects_when_queue_full(self):
        scheduler = RequestScheduler(max_queue_size=1)
        assert scheduler.enqueue(make_request("a")) is True
        assert scheduler.enqueue(make_request("b")) is False

    def test_peek_does_not_remove_request(self):
        scheduler = RequestScheduler()
        scheduler.enqueue(make_request("a"))
        assert scheduler.peek().request_id == "a"
        assert scheduler.size() == 1

    def test_expired_requests_are_skipped_on_dequeue(self):
        scheduler = RequestScheduler()
        scheduler.enqueue(make_request("expired", timeout_ms=-1))
        scheduler.enqueue(make_request("fresh", timeout_ms=30000))

        result = scheduler.dequeue()
        assert result.request_id == "fresh"

    def test_user_quota_blocks_after_limit(self):
        # Regression test: enqueue() used to only bump request_counts by
        # model_id, so the per-user counter _check_user_quota() reads was
        # never populated and quotas were unenforceable. Fixed to also
        # track a "user:<id>" counter.
        scheduler = RequestScheduler()
        scheduler.set_user_quota("bob", requests_per_minute=1)

        assert scheduler.enqueue(make_request("r1", user_id="bob")) is True
        assert scheduler.enqueue(make_request("r2", user_id="bob")) is False

    def test_user_quota_is_independent_per_user(self):
        scheduler = RequestScheduler()
        scheduler.set_user_quota("bob", requests_per_minute=1)
        scheduler.enqueue(make_request("r1", user_id="bob"))

        assert scheduler.enqueue(make_request("r2", user_id="alice")) is True

    def test_get_queue_stats_counts_by_model_and_priority(self):
        scheduler = RequestScheduler()
        scheduler.enqueue(make_request("a", model_id="m1", priority=RequestPriority.HIGH.value))
        scheduler.enqueue(make_request("b", model_id="m2", priority=RequestPriority.HIGH.value))

        stats = scheduler.get_queue_stats()
        assert stats["queue_size"] == 2
        assert stats["by_model"]["m1"] == 1
        assert stats["by_priority"]["HIGH"] == 2


class TestFairShareScheduler:
    def test_dequeue_fair_returns_request_with_model_name(self):
        scheduler = FairShareScheduler(models=["m1", "m2"])
        scheduler.enqueue(make_request("r1", model_id="m1"))

        result = scheduler.dequeue_fair()
        assert result is not None
        request, model = result
        assert request.request_id == "r1"
        assert model == "m1"

    def test_enqueue_rejects_unknown_model(self):
        scheduler = FairShareScheduler(models=["m1"])
        assert scheduler.enqueue(make_request("r1", model_id="unknown")) is False

    def test_refill_tokens_increases_available_tokens(self):
        scheduler = FairShareScheduler(models=["m1"])
        scheduler.model_tokens["m1"] = 0.0
        scheduler.refill_tokens()
        assert scheduler.model_tokens["m1"] == scheduler.token_rate


class TestDeadlineScheduler:
    def test_dequeue_skips_expired_and_returns_valid(self):
        scheduler = DeadlineScheduler()
        scheduler.enqueue(make_request("expired", timeout_ms=-1))
        scheduler.enqueue(make_request("valid", timeout_ms=30000))

        result = scheduler.dequeue()
        assert result.request_id == "valid"

    def test_size_reflects_queue_length(self):
        scheduler = DeadlineScheduler()
        scheduler.enqueue(make_request("a"))
        scheduler.enqueue(make_request("b"))
        assert scheduler.size() == 2
