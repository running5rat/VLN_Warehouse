import copy
import unittest
from ticvla.warehouse.dataset_tools.oracle_review import (
    FIELDS, estimate_cost, format_diagnostics, token_counts, validate_reviews)


class OracleReviewTests(unittest.TestCase):
    def test_cost_weights_actual_cohorts_and_includes_failed_responses(self):
        rows=[dict(source_image_id='r2r-1:0',cost_native_estimate=.1,status='failed'),
              dict(source_image_id='r2r-2:0',cost_native_estimate=.3,status='format_valid_unreviewed'),
              dict(source_image_id='rxr-1:0',cost_native_estimate=.4),
              dict(source_image_id='manual-1:0',cost_native_estimate=99),
              dict(source_image_id='r2r-3:0',status='not_run_previous_proposal_failed')]
        cost=estimate_cost(rows,{'r2r':120,'rxr':80})
        self.assertAlmostEqual(cost['one_attempt_per_frame'],56)
        self.assertAlmostEqual(cost['review_rework_20_percent_scenario'],67.2)
        self.assertIsNone(cost['cost_per_human_approved_label'])
        self.assertIsNone(estimate_cost(rows,{'missing':1})['one_attempt_per_frame'])

    def test_reasoning_is_part_of_output_not_added_twice(self):
        counts=token_counts(dict(prompt_tokens=100,completion_tokens=80,
                                completion_tokens_details=dict(reasoning_tokens=60)))
        self.assertEqual(counts['output'],80)
        self.assertEqual(counts['nonreasoning_residual'],20)
        self.assertIsNone(counts['visible_reported'])
        self.assertIsNone(token_counts({})['output'])

    def test_length_failures_are_not_reported_as_invalid_json(self):
        import json
        d=format_diagnostics(json.dumps(dict(local='A'*91,route='DONE',landmark=[])))
        self.assertTrue(d['json_valid'])
        self.assertEqual(d['violations'],['local_exceeds_90_characters'])
        self.assertFalse(format_diagnostics('not JSON')['json_valid'])

    def test_review_is_bound_to_source_response_and_report(self):
        template=dict(report_id='report',reviewer='',reviews=[dict(key='m|source',response_sha256='hash',
            fields={f:'pending' for f in FIELDS},overall='pending')])
        validate_reviews(template,template)
        for change in [lambda d:d.update(report_id='other'),
                       lambda d:d['reviews'][0].update(response_sha256='changed'),
                       lambda d:d['reviews'].append(copy.deepcopy(d['reviews'][0])),
                       lambda d:d['reviews'][0].update(overall='pass')]:
            data=copy.deepcopy(template);change(data)
            with self.assertRaises(ValueError):validate_reviews(data,template)
        data=copy.deepcopy(template);data['reviewer']='human'
        data['reviews'][0].update(overall='pass',fields={f:'pass' for f in FIELDS})
        validate_reviews(data,template)


if __name__=='__main__':unittest.main()
