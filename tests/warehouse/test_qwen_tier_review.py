import unittest

from ticvla.warehouse.dataset_tools.qwen_tier_review import VARIANTS, summarize_rows


class QwenTierReviewTests(unittest.TestCase):
    def test_failed_responses_cost_money_and_recursive_tails_stay_in_denominator(self):
        rows=[dict(status='format_valid_unreviewed',response_sha256='a',elapsed_seconds=2.,cost_native_estimate=.01),
              dict(status='failed',response_sha256='b',elapsed_seconds=4.,cost_native_estimate=.02),
              dict(status='not_run_previous_proposal_failed')]
        r=summarize_rows(rows,4)
        self.assertEqual(r['responses'],2)
        self.assertEqual(r['rule_pass_over_selected'],.25)
        self.assertEqual(r['not_run'],2)
        self.assertAlmostEqual(r['historical_and_new_usage_cost_cny'],.03)
        self.assertIsNone(r['semantic_accuracy'])

    def test_unsupported_completion_is_not_format_pass_or_human_approval(self):
        r=summarize_rows([dict(status='rule_rejected_unreviewed',response_sha256='x',
                              elapsed_seconds=1.,cost_native_estimate=.01)],1)
        self.assertEqual(r['rule_pass'],0)
        self.assertEqual(r['rule_rejected'],1)
        self.assertEqual(r['human_approved'],0)

    def test_no_gpt_jobs_and_thinking_controls_are_explicit(self):
        self.assertTrue(all(v[0].startswith('qwen') for v in VARIANTS.values()))
        self.assertEqual(VARIANTS['plus-thinking-2048'][1:3],(True,2048))
        self.assertEqual(VARIANTS['max-no-thinking'][1:3],(False,None))


if __name__=='__main__':unittest.main()
