import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.evict_policy import QoSAwareStrategy
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestQoSHiCacheAdmission(unittest.TestCase):
    def _cache(self, *, enabled=True):
        cache = object.__new__(HiRadixCache)
        cache.cache_controller = MagicMock()
        cache.cache_controller.write_policy = "write_back"
        cache.write_through_threshold = 2
        cache.enable_qos_aware_prefix_cache = enabled
        cache.eviction_strategy = QoSAwareStrategy()
        cache.schedule_low_priority_values_first = False
        cache.qos_hicache_recompute_time_per_token = 0.1
        cache.qos_hicache_transfer_time_per_token = 0.05
        cache.qos_hicache_write_time_per_token = 0.0
        cache.qos_hicache_cost_ewma_alpha = 0.2
        cache.qos_hicache_auto_calibrate = True
        cache.qos_hicache_recompute_calibration_samples = 0
        cache.qos_hicache_max_eviction_steps = 64
        cache.ongoing_load_back_stats = {}
        cache.pending_host_source_releases = set()
        cache.page_size = 1
        return cache

    def _node(self, *, priority=1):
        node = TreeNode(priority=priority)
        node.key = RadixKey([1, 2, 3, 4])
        node.value = torch.tensor([1, 2, 3, 4])
        return node

    def test_exclusive_write_back_tracks_demand_without_proactive_copy(self):
        cache = self._cache()
        cache.write_backup = MagicMock(return_value=4)
        node = self._node()

        cache._inc_hit_count(node)
        cache._inc_hit_count(node)

        self.assertEqual(node.hit_count, 2)
        cache.write_backup.assert_not_called()

    def test_disabled_gate_preserves_original_selective_write(self):
        cache = self._cache(enabled=False)
        cache.cache_controller.write_policy = "write_through_selective"
        cache.write_backup = MagicMock(return_value=4)
        node = self._node()

        cache._inc_hit_count(node)
        cache._inc_hit_count(node)

        cache.write_backup.assert_called_once_with(node)

    def test_total_gate_does_not_change_write_through_semantics(self):
        cache = self._cache()
        cache.cache_controller.write_policy = "write_through_selective"
        cache.write_backup = MagicMock(return_value=4)
        node = self._node()

        cache._inc_hit_count(node)
        cache._inc_hit_count(node)

        self.assertFalse(cache._qos_exclusive_hicache_enabled())
        cache.write_backup.assert_called_once_with(node)

    def test_gpu_eviction_uses_selective_host_demotion(self):
        cache = self._cache()
        node = self._node()
        cache._make_eviction_heap = MagicMock(return_value=[((0.0, 0.0), node)])
        cache.write_backup = MagicMock(return_value=0)
        cache._drop_subtree_no_host = MagicMock(return_value=4)
        cache._promote_parent = MagicMock()

        self.assertEqual(cache._evict_write_back(4), 4)
        cache.write_backup.assert_called_once_with(
            node, write_back=True, selective_admission=True
        )

    def test_rejects_single_use_gpu_prefix_demotion(self):
        cache = self._cache()
        cache.cache_controller.mem_pool_host.available_size.return_value = 4
        node = self._node()
        node.hit_count = 1

        self.assertFalse(cache._prepare_selective_host_space(node))

    def test_host_victim_budget_failure_is_atomic(self):
        cache = self._cache()
        cache.root_node = TreeNode()
        cache.evictable_host_leaves = set()
        cache._record_remove_event = MagicMock()
        cache._update_host_leaf_status = MagicMock()
        cache.cache_controller.evict_host.side_effect = len
        for token, priority in ((1, 1), (2, 2)):
            node = TreeNode(priority=priority)
            node.key = RadixKey([token] * 4)
            node.value = None
            node.host_value = torch.tensor([token] * 4)
            node.parent = cache.root_node
            cache.root_node.children[
                node.key.child_key(cache.page_size)
            ] = node
            cache.evictable_host_leaves.add(node)
        cache.qos_hicache_max_eviction_steps = 1

        evicted = cache.evict_host(
            8, max_priority=10, priority_fn=lambda node: node.priority
        )

        self.assertEqual(evicted, 0)
        self.assertEqual(len(cache.root_node.children), 2)
        cache.cache_controller.evict_host.assert_not_called()

    def test_host_victims_commit_after_sufficient_preselection(self):
        cache = self._cache()
        cache.root_node = TreeNode()
        cache.evictable_host_leaves = set()
        cache._record_remove_event = MagicMock()
        cache._update_host_leaf_status = MagicMock()
        cache.cache_controller.evict_host.side_effect = len
        for token, priority in ((1, 1), (2, 2)):
            node = TreeNode(priority=priority)
            node.key = RadixKey([token] * 4)
            node.value = None
            node.host_value = torch.tensor([token] * 4)
            node.parent = cache.root_node
            cache.root_node.children[
                node.key.child_key(cache.page_size)
            ] = node
            cache.evictable_host_leaves.add(node)
        cache.qos_hicache_max_eviction_steps = 2

        evicted = cache.evict_host(
            8, max_priority=10, priority_fn=lambda node: node.priority
        )

        self.assertEqual(evicted, 8)
        self.assertEqual(len(cache.root_node.children), 0)
        self.assertEqual(cache.cache_controller.evict_host.call_count, 2)

    def test_rejects_zero_benefit_even_when_host_has_space(self):
        cache = self._cache()
        cache.qos_hicache_transfer_time_per_token = 0.2
        cache.cache_controller.mem_pool_host.available_size.return_value = 4
        cache.evict_host = MagicMock(return_value=0)
        node = self._node()
        node.hit_count = 2

        self.assertFalse(cache._prepare_selective_host_space(node))
        cache.evict_host.assert_not_called()

    def test_admits_when_host_has_free_space(self):
        cache = self._cache()
        cache.cache_controller.mem_pool_host.available_size.return_value = 4
        cache.evict_host = MagicMock(return_value=0)
        node = self._node()
        node.hit_count = 2

        self.assertTrue(cache._prepare_selective_host_space(node))
        cache.evict_host.assert_not_called()

    def test_replaces_only_nodes_colder_than_candidate(self):
        cache = self._cache()
        cache.cache_controller.mem_pool_host.available_size.return_value = 0
        cache.evict_host = MagicMock(return_value=4)
        node = self._node(priority=3)
        node.hit_count = 2
        candidate_priority = cache._get_host_admission_priority(node)

        self.assertTrue(cache._prepare_selective_host_space(node))
        cache.evict_host.assert_called_once()
        call = cache.evict_host.call_args
        self.assertEqual(call.args[0], 4)
        self.assertEqual(call.kwargs["max_priority"], candidate_priority)
        self.assertEqual(
            call.kwargs["priority_fn"], cache._get_host_admission_priority
        )

    def test_rejects_when_colder_nodes_cannot_free_enough_space(self):
        cache = self._cache()
        cache.cache_controller.mem_pool_host.available_size.return_value = 0
        cache.evict_host = MagicMock(return_value=2)
        node = self._node()
        node.hit_count = 2

        self.assertFalse(cache._prepare_selective_host_space(node))
        cache.evict_host.assert_called_once()

    def test_host_value_uses_logical_matches_and_net_benefit(self):
        cache = self._cache()
        node = self._node(priority=3)
        node.hit_count = 2
        node.host_match_count = 3
        node.host_load_count = 1
        node.host_transfer_time_per_token = 0.02

        value, _ = cache._get_host_admission_priority(node)

        self.assertAlmostEqual(value, 5 * (0.1 - 0.02) * 3)

    def test_host_value_is_zero_when_transfer_is_slower(self):
        cache = self._cache()
        node = self._node(priority=3)
        node.hit_count = 10
        node.host_load_count = 1
        node.host_transfer_time_per_token = 0.2

        value, _ = cache._get_host_admission_priority(node)

        self.assertEqual(value, 0.0)

    def test_load_back_is_skipped_when_recompute_is_faster(self):
        cache = self._cache()
        node = self._node()
        node.host_value = torch.tensor([1, 2, 3, 4])
        node.host_load_count = 1
        node.host_transfer_time_per_token = 0.2

        self.assertFalse(cache._should_load_back([node]))

    def test_cold_load_back_uses_global_cost_estimate(self):
        cache = self._cache()
        node = self._node()
        node.host_value = torch.tensor([1, 2, 3, 4])

        self.assertTrue(cache._should_load_back([node]))

    def test_completed_load_updates_node_and_global_cost(self):
        cache = self._cache()
        node = self._node()

        cache._record_host_load([node], num_tokens=4, duration=0.4)

        self.assertEqual(node.host_load_count, 1)
        self.assertAlmostEqual(node.host_transfer_time_per_token, 0.1)
        self.assertAlmostEqual(cache.qos_hicache_transfer_time_per_token, 0.06)

    def test_recompute_calibration_averages_first_three_real_prefills(self):
        cache = self._cache()

        cache.record_recompute_calibration(10, 0.5)
        cache.record_recompute_calibration(10, 1.0)
        cache.record_recompute_calibration(10, 1.5)
        cache.record_recompute_calibration(10, 10.0)

        self.assertEqual(cache.qos_hicache_recompute_calibration_samples, 3)
        self.assertAlmostEqual(cache.qos_hicache_recompute_time_per_token, 0.1)

    def test_host_logical_match_counts_without_load_back(self):
        cache = self._cache()
        cache.root_node = TreeNode()
        node = self._node()
        node.value = None
        node.host_value = torch.tensor([1, 2, 3, 4])
        node.parent = cache.root_node
        key = RadixKey([1, 2, 3, 4])
        cache.root_node.children[key.child_key(cache.page_size)] = node

        cache._match_prefix_helper(
            cache.root_node, key, update_cache_stats=True
        )
        cache._match_prefix_helper(
            cache.root_node, key, update_cache_stats=False
        )

        self.assertEqual(node.host_match_count, 1)
        self.assertEqual(node.host_load_count, 0)

    def test_recompute_insert_releases_host_source_copy(self):
        cache = self._cache()
        cache.root_node = TreeNode()
        cache.evictable_size_ = 0
        cache.is_eagle = False
        cache.enable_storage = False
        cache.enable_kv_cache_events = False
        cache._record_remove_event = MagicMock()
        cache._update_leaf_status = MagicMock()
        cache._update_host_leaf_status = MagicMock()
        node = self._node()
        node.value = None
        node.host_value = torch.tensor([5, 6, 7, 8])
        node.parent = cache.root_node
        key = RadixKey([1, 2, 3, 4])
        cache.root_node.children[key.child_key(cache.page_size)] = node

        cache.insert(
            InsertParams(
                key=key,
                value=torch.tensor([10, 11, 12, 13]),
                priority=1,
            )
        )

        self.assertIsNotNone(node.value)
        self.assertIsNone(node.host_value)
        cache.cache_controller.evict_host.assert_called_once()

    def test_promotion_defers_referenced_host_source_release(self):
        cache = self._cache()
        cache.root_node = TreeNode()
        cache._record_remove_event = MagicMock()
        cache._update_host_leaf_status = MagicMock()
        node = self._node()
        node.parent = cache.root_node
        node.host_value = torch.tensor([5, 6, 7, 8])
        node.protect_host()

        cache._drop_host_copy(node)
        self.assertIsNotNone(node.host_value)
        self.assertIn(node, cache.pending_host_source_releases)

        node.release_host()
        cache._release_pending_host_copies()
        self.assertIsNone(node.host_value)
        self.assertNotIn(node, cache.pending_host_source_releases)

    def test_promotion_releases_host_source_copy(self):
        cache = self._cache()
        cache.root_node = TreeNode()
        cache._record_remove_event = MagicMock()
        cache._update_host_leaf_status = MagicMock()
        node = self._node()
        node.parent = cache.root_node
        node.host_value = torch.tensor([5, 6, 7, 8])

        cache._drop_host_copy(node)

        self.assertIsNone(node.host_value)
        cache.cache_controller.evict_host.assert_called_once()

    def test_rejected_candidate_does_not_start_host_write(self):
        cache = self._cache()
        cache.root_node = TreeNode()
        node = self._node()
        node.parent = cache.root_node
        cache._prepare_selective_host_space = MagicMock(return_value=False)

        self.assertEqual(
            cache.write_backup(node, selective_admission=True), 0
        )
        cache.cache_controller.write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
