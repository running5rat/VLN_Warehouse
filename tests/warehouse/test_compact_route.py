import copy
import json
import unittest

import numpy as np

from ticvla.warehouse.dataset_tools.core import tagged_text
from ticvla.warehouse.navigation.core import NavigationState
from ticvla.warehouse.navigation.route_protocol import parse_route, compact_route
from ticvla.warehouse.structured_io import parse_structured, response_grammar


class CompactRouteTests(unittest.TestCase):
    def fixture(self):
        nav = NavigationState('t', instruction='Reach the extinguisher and stop.')
        obs = dict(task={'task_id':'t'}, route_revision=0, source_monotonic=1.,
                   T_odom_body=np.eye(4).tolist(), odom_epoch='e', map_version='1',
                   rgb_history=[dict(image_id='f')], route_schema='compact-v2')
        result = dict(valid=True, inference_mode='TEXT', parsed=dict(
            route='Reach the extinguisher and stop.', local='unknown', landmark=[], anchors=[]),
            z_sem=np.zeros((16,512)), reference_xy=np.zeros((32,2)),
            reference_mask=np.zeros(32,bool), confidence=.5)
        return nav, obs, result

    def test_compact_grammar_and_no_quoted_route(self):
        for route in ('Reach the aisle end, then turn left.', 'DONE'):
            text=tagged_text(['sign'],'unknown',route,[])
            parser=response_grammar(['f'], categories_only=True, compact_route=True)
            for char in text:
                self.assertIn(char,parser.get_allowed_characters())
                parser=parser.add_character(char)
            self.assertTrue(parser.can_end())
            parsed=parse_structured(text,['f'],compact_route=True)
            self.assertEqual(parsed['route'],route)
            self.assertIn('<Route>'+route+'</Route>',text)
        for invalid in ('', '{}', '"DONE"', 'a'*601, 'Go <now>', {'status':'active'}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_route(invalid,require_compact=True)

    def test_recursive_route_without_subgoal_and_invalid_reference(self):
        nav,obs,result=self.fixture()
        instruction=nav.state.contract.instruction
        packet=nav.commit(result,obs,1,1.1,1.2,1.3,10)
        self.assertFalse(packet.reference_usable)
        self.assertEqual(nav.state.route,result['parsed']['route'])
        self.assertIsNone(nav.facts.data['subgoal'])
        obs['route_revision']=1
        result['parsed']['route']='Stop beside the extinguisher.'
        nav.commit(result,obs,2,1.1,1.2,1.3,10)
        self.assertEqual(nav.state.route_revision,2)
        self.assertEqual(nav.state.route,result['parsed']['route'])
        self.assertEqual(nav.state.contract.instruction,instruction)
        restored=NavigationState.restore(json.loads(json.dumps(nav.checkpoint())))
        self.assertEqual(restored.state.route,nav.state.route)
        self.assertIsNone(restored.terminal)

    def test_DONE_stops_but_does_not_complete_without_facts(self):
        nav,obs,result=self.fixture();result['parsed']['route']='DONE'
        packet=nav.commit(result,obs,1,1.1,1.2,1.3,10)
        self.assertFalse(packet.reference_usable)
        self.assertIsNone(nav.terminal)
        self.assertTrue(nav.state.execution['completion_claim_rejected'])
        result['parsed']['route']='Continue toward the extinguisher.'
        nav.commit(result,dict(obs,route_revision=1),2,1.1,1.2,1.3,10)
        self.assertFalse(nav.state.execution['completion_claim_rejected'])
        self.assertEqual(nav.status,'active')

    def test_DONE_with_independent_task_arrival_and_stop_evidence(self):
        nav,obs,result=self.fixture();result['parsed']['route']='DONE'
        nav.facts.observe(np.eye(4),1.,'e',velocity=(0,0))
        for kind in ('target_visible','arrival_confirmed','observed_stopped'):
            nav.facts.event(kind,image_id='f')
        nav.commit(result,obs,1,1.1,1.2,1.3,10)
        self.assertEqual(nav.terminal,'completed')

    def test_reverse_diagnostic_stays_non_executable_without_action_enum(self):
        nav,obs,result=self.fixture()
        result['parsed']['route']='Move backward five metres.'
        result['reference_xy'][:,0]=np.linspace(0,-5,32)
        result['reference_mask'][:]=True
        packet=nav.commit(result,obs,1,1.1,1.2,1.3,10)
        self.assertEqual(packet.motion_frame_or_direction,'backward')
        self.assertFalse(packet.reference_usable)

    def test_legacy_adapter_preserves_waiting_instead_of_inventing_active(self):
        route=dict(remaining_instruction='Continue.', current_subgoal={'action':'wait'},status='waiting')
        before=copy.deepcopy(route)
        with self.assertRaises(ValueError):compact_route(route)
        self.assertEqual(route,before)
        route['status']='completed'
        self.assertEqual(compact_route(route),'DONE')


if __name__ == '__main__':
    unittest.main()
