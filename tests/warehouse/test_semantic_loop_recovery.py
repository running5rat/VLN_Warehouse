import json
from unittest.mock import patch
import pytest
from ticvla.warehouse.dataset_tools.semantic_loop import recovery_prefix,recovery_request_key


def test_prefix_preserves_failed_frames_and_authorizes_only_unknown(tmp_path):
    cfg={'base_url':'https://example.com','api':'chat_completions'}
    key=recovery_request_key({'messages':[]},cfg)
    cfg['authorized_request_recovery']={'request_keys':[key]}
    for name in ['0','1','2']:(tmp_path/name).mkdir()
    for name in ['0','1']:(tmp_path/name/'response.json').write_text('{}')
    (tmp_path/'2/sample.json').write_text('{}')
    rows=[{'folder':'0','status':'format_valid_unreviewed'}, {'folder':'1','status':'failed','failure_stage':'response_validation'}, {'folder':'2','status':'request_started'}]
    manifest={'records':[{'folder':str(i)} for i in range(3)]}
    with patch('ticvla.warehouse.dataset_tools.semantic_loop.build_request',return_value=({'messages':[]},{})):
        prefix,pending=recovery_prefix(tmp_path,{'samples':rows},cfg,manifest)
        assert prefix==rows[:2] and pending==key
        cfg['authorized_request_recovery']['request_keys']=[]
        with pytest.raises(ValueError,match='not explicitly authorized'):recovery_prefix(tmp_path,{'samples':rows},cfg,manifest)
        (tmp_path/'2/response.json').write_text('{}')
        with pytest.raises(ValueError,match='Saved but unprocessed'):recovery_prefix(tmp_path,{'samples':rows},cfg,manifest)
    rows[1]['failure_stage']='map_commit'
    with pytest.raises(ValueError,match='Partial map commit'):recovery_prefix(tmp_path,{'samples':rows},cfg,manifest)
