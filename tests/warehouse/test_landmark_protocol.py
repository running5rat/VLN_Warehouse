import copy
import json
import unittest

from ticvla.warehouse.landmark_protocol import bind_landmarks, current_source_id
from ticvla.warehouse.structured_io import parse_structured, response_grammar


def response(items):
    return ('<landmark>' + json.dumps(items) + '</landmark><local>unknown</local>'
            '<Route>Continue.</Route><waypoint>[]</waypoint><REFLINE>')


class CompactLandmarkTests(unittest.TestCase):
    def test_current_binding_normalization_and_deduplication(self):
        text = response([' Sign ', 'SIGN', 'fire   extinguisher'])
        parsed = parse_structured(text, ['history', 'current'], current_image_id='current')
        self.assertEqual(parsed['landmark'], [
            dict(image_id='current', category='sign'),
            dict(image_id='current', category='fire extinguisher')])
        self.assertIn('SIGN', text)  # Original output remains available for auditing.

    def test_no_current_source_guessing(self):
        for current in (None, 'wrong'):
            with self.assertRaisesRegex(ValueError, 'current source'):
                parse_structured(response(['sign']), ['old', 'new'], current_image_id=current)

    def test_empty_and_legacy_history(self):
        self.assertEqual(parse_structured(response([]), ['old', 'new'])['landmark'], [])
        old = [dict(image_id='old', category='sign')]
        self.assertEqual(parse_structured(response(old), ['old', 'new'],
                         current_image_id='new')['landmark'], old)
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            parse_structured(response(old), ['new'], current_image_id='new')

    def test_invalid_categories_and_mixed_schema(self):
        for items in ([''], [' '], ['x'*41], [3], ['sign', {'category':'door'}],
                      ['sign', 'door', 'box', 'rack'], ['sign/door']):
            with self.subTest(items=items), self.assertRaises(ValueError):
                bind_landmarks(items, ['current'], 'current')

    def test_grammar_only_emits_categories(self):
        def accepts(text):
            parser = response_grammar(['current'], categories_only=True)
            for char in text:
                if char not in parser.get_allowed_characters():
                    return False
                parser = parser.add_character(char)
            return parser.can_end()
        for items in ([], ['sign'], ['sign', 'fire extinguisher']):
            self.assertTrue(accepts(response(items)))
        self.assertFalse(accepts(response([dict(image_id='current', category='sign')])))
        self.assertFalse(accepts(response(['x'*41])))
        self.assertFalse(accepts(response(['sign'])[:-3]))

    def test_current_source_uses_metadata_not_history_order(self):
        obs = dict(t_obs=2., rgb_history=[
            dict(image_id='current', t_obs=2., path='current.png'),
            dict(image_id='history', t_obs=1., path='history.png')])
        self.assertEqual(current_source_id(obs), 'current')
        wrong = copy.deepcopy(obs); wrong['image_id'] = 'history'
        with self.assertRaises(ValueError):
            current_source_id(wrong)
        obs['rgb_history'].append(dict(image_id='other', t_obs=2., path='other.png'))
        with self.assertRaises(ValueError):
            current_source_id(obs)


if __name__ == '__main__':
    unittest.main()
