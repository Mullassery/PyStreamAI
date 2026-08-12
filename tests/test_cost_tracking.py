"""Tests for pystreamai.cost_tracking - GPU/token pricing and budget management."""

import pytest

from pystreamai.cost_tracking import CostMetric, Pricing, CostTracker, SpendManager, CostOptimizer


def make_metric(model_id="m1", gpu_type="A100", latency_ms=100.0, batch_size=1):
    return CostMetric(
        request_id="r1",
        model_id=model_id,
        gpu_type=gpu_type,
        latency_ms=latency_ms,
        batch_size=batch_size,
        input_tokens=100,
        output_tokens=50,
    )


class TestPricing:
    def test_gpu_cost_per_ms_is_positive_for_known_gpu(self):
        pricing = Pricing()
        assert pricing.get_gpu_cost_per_ms("A100") > 0

    def test_unknown_gpu_falls_back_to_default_rate(self):
        pricing = Pricing()
        assert pricing.get_gpu_cost_per_ms("UNKNOWN_GPU") == pytest.approx(1.0 / (3600 * 1000))

    def test_token_cost_zero_for_unknown_model(self):
        pricing = Pricing()
        assert pricing.get_token_cost(1000, 1000, model="not-a-real-model") == 0.0

    def test_token_cost_scales_with_tokens(self):
        pricing = Pricing()
        small = pricing.get_token_cost(1000, 0, model="gpt-4")
        large = pricing.get_token_cost(2000, 0, model="gpt-4")
        assert large == pytest.approx(small * 2)


class TestCostMetric:
    def test_calculate_cost_divides_by_batch_size(self):
        pricing = Pricing()
        single = make_metric(batch_size=1).calculate_cost(pricing)
        batched = make_metric(batch_size=4).calculate_cost(pricing)
        # Same total compute, spread over 4x the samples -> per-sample cost drops
        assert batched < single

    def test_calculate_cost_is_positive(self):
        cost = make_metric().calculate_cost(Pricing())
        assert cost > 0


class TestCostTracker:
    def test_record_inference_updates_totals(self):
        tracker = CostTracker()
        cost = tracker.record_inference(make_metric())

        assert cost > 0
        assert tracker.get_total_cost() == pytest.approx(cost)
        assert tracker.get_model_cost("m1") == pytest.approx(cost)

    def test_get_daily_cost_defaults_to_today(self):
        tracker = CostTracker()
        tracker.record_inference(make_metric())
        assert tracker.get_daily_cost() > 0

    def test_cost_breakdown_reports_avg_per_request(self):
        tracker = CostTracker()
        tracker.record_inference(make_metric())
        tracker.record_inference(make_metric())

        breakdown = tracker.get_cost_breakdown()
        assert breakdown["num_requests"] == 2
        assert breakdown["avg_cost_per_request"] == pytest.approx(
            breakdown["total_cost_usd"] / 2
        )

    def test_cost_breakdown_avg_is_zero_when_no_requests(self):
        breakdown = CostTracker().get_cost_breakdown()
        assert breakdown["avg_cost_per_request"] == 0


class TestSpendManager:
    def test_allows_requests_under_budget(self):
        tracker = CostTracker()
        manager = SpendManager(tracker, monthly_budget=1000.0)
        assert manager.should_allow_request("m1") is True

    def test_blocks_requests_once_budget_exceeded(self):
        tracker = CostTracker()
        manager = SpendManager(tracker, monthly_budget=0.0000001)
        tracker.record_inference(make_metric())

        assert manager.should_allow_request("m1") is False
        assert manager.blocked is True
        # Once blocked, stays blocked even if a later check would pass
        assert manager.should_allow_request("m1") is False

    def test_budget_status_reports_percent_used(self):
        tracker = CostTracker()
        manager = SpendManager(tracker, monthly_budget=1000.0)
        tracker.record_inference(make_metric())

        status = manager.get_budget_status()
        assert status["monthly_budget_usd"] == 1000.0
        assert 0 <= status["percent_used"] < 1

    def test_reset_monthly_budget_clears_state(self):
        tracker = CostTracker()
        manager = SpendManager(tracker, monthly_budget=0.0000001)
        tracker.record_inference(make_metric())
        manager.should_allow_request("m1")
        assert manager.blocked is True

        manager.reset_monthly_budget()

        assert manager.blocked is False
        assert tracker.get_total_cost() == 0


class TestCostOptimizer:
    def test_recommends_gpu_downgrade_after_many_h100_requests(self):
        tracker = CostTracker()
        for _ in range(11):
            tracker.record_inference(make_metric(gpu_type="H100"))

        recs = CostOptimizer(tracker).get_recommendations()
        types = {r["type"] for r in recs}
        assert "gpu_downgrade" in types

    def test_recommends_batching_for_small_batches(self):
        tracker = CostTracker()
        tracker.record_inference(make_metric(batch_size=1))

        recs = CostOptimizer(tracker).get_recommendations()
        types = {r["type"] for r in recs}
        assert "batching" in types

    def test_recommendations_are_empty_for_a_tracker_with_no_data(self):
        # Regression test: get_recommendations() used to divide by
        # len(self.tracker.metrics) unconditionally and raised
        # ZeroDivisionError for a fresh tracker. Fixed to return [] instead.
        tracker = CostTracker()
        assert CostOptimizer(tracker).get_recommendations() == []
