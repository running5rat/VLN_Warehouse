import unittest
from ticvla.warehouse.dataset_tools.r2r_batch import select_tasks, remaining_tasks


class R2RBatchTests(unittest.TestCase):
    def test_remaining_keeps_multiple_tasks_per_building_but_excludes_prior_runs_and_comparison(self):
        import tempfile
        import json
        from pathlib import Path
        rows=[dict(uid=u,source='r2r',role='train',scene_id='same') for u in ['new2','old','new1']]
        rows += [dict(uid='heldout',source='r2r',role='oracle_comparison'),dict(uid='rxr',source='rxr',role='train')]
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'pilot.json').write_text(json.dumps({'episodes':[{'uid':'old'}]}))
            self.assertEqual([r['uid'] for r in remaining_tasks(rows,[tmp])],['new1','new2'])
            with self.assertRaises(ValueError):remaining_tasks(rows+[rows[0]],[tmp])

    def test_selection_is_reproducible_distinct_and_train_only(self):
        rows=[dict(uid=f'r2r-{s}-{i}',source='r2r',role='train',scene_id=f's{s}') for s in range(20) for i in range(2)]
        rows += [dict(uid='excluded',source='r2r',role='oracle_comparison',scene_id='comparison'),
                 dict(uid='rxr',source='rxr',role='train',scene_id='rxr')]
        a=select_tasks(rows)
        self.assertEqual(a,select_tasks(list(reversed(rows))))
        self.assertEqual(len(a),12)
        self.assertEqual(len({r['scene_id'] for r in a}),12)
        self.assertTrue(all(r['role']=='train' and r['source']=='r2r' for r in a))
        self.assertNotEqual(a,select_tasks(rows,seed=43))

    def test_insufficient_buildings_does_not_replace_or_duplicate(self):
        rows=[dict(uid='a',source='r2r',role='train',scene_id='s')]
        with self.assertRaises(ValueError):select_tasks(rows,2)
        with self.assertRaises(ValueError):select_tasks(rows,0)

    def test_batch_stop_preserves_unexecuted_denominator_without_api_or_map_writes(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch
        from ticvla.warehouse.dataset_tools.core import write_json
        from ticvla.warehouse.dataset_tools.semantic_loop import run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prepared=root/'prepared';prepared.mkdir()
            write_json(prepared/'manifest.json',dict(complete=True,records=[dict(folder=str(i)) for i in range(2)]))
            config=root/'config.json';write_json(config,dict(model='fixture'))
            store=MagicMock();store.landmark_view.return_value={'landmarks':[]};store.summary.return_value={}
            def load(_,entry):
                return {'observation':{}},{'source':{'image_id':'frame-'+entry['folder']}},{},None
            with patch('ticvla.warehouse.dataset_tools.semantic_loop.load_credentials'), \
                 patch('ticvla.warehouse.dataset_tools.semantic_loop.MapStore',return_value=store), \
                 patch('ticvla.warehouse.dataset_tools.semantic_loop.load_frame',side_effect=load), \
                 patch('ticvla.warehouse.dataset_tools.semantic_loop.send_request') as api:
                result=run(prepared,config,root/'out','unused',stop_requested=lambda:True)
            api.assert_not_called();store.add_observation.assert_not_called()
            self.assertEqual(len(result['samples']),2)
            self.assertTrue(all(r['status']=='not_run_batch_stop' for r in result['samples']))
            self.assertEqual(result['actual_api_calls'],0)
            self.assertTrue(result['fatal_stop'])

if __name__=='__main__':unittest.main()
