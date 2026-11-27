import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
from ticvla.warehouse.perception.realtime import CORE, FixedSegmenter, load_vocabulary
from ticvla.warehouse.schema import file_sha256


class IndoorVocabularyTests(unittest.TestCase):
    def test_independent_names_and_hash(self):
        warehouse=load_vocabulary()
        indoor=load_vocabulary('configs/dataset_preparation/r2r_indoor.v1.json')
        self.assertEqual(warehouse['categories'],list(CORE))
        self.assertEqual(warehouse['aliases']['shelf'],'storage rack')
        self.assertNotIn('shelf',indoor['aliases'])
        self.assertEqual(len(indoor['categories']),26)
        self.assertEqual(load_vocabulary(indoor),indoor)
        with self.assertRaisesRegex(ValueError,'hash'):
            load_vocabulary(dict(indoor,sha256='wrong'))
        for categories in ([],['door','door'],[' Door'],['Door']):
            with self.assertRaises(ValueError):load_vocabulary(dict(version='test',categories=categories))
        with self.assertRaises(ValueError):
            load_vocabulary(dict(version='test',categories=['door'],aliases={'shelf':'storage rack'}))

    def test_pytorch_receives_exact_indoor_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rgb.png';Image.new('RGB',(16,12)).save(path)
            model=MagicMock();model.provenance={}
            model.segment_instances.return_value=dict(instances=[],possibly_truncated=False)
            with patch('ticvla.warehouse.mapping.segmentation.YOLOESegmenter',return_value=model) as init:
                segmenter=FixedSegmenter('fixture.pt',vocabulary='configs/dataset_preparation/r2r_indoor.v1.json')
                source=dict(rgb=str(path),rgb_hash=file_sha256(path),image_id='e:0')
                result=segmenter.predict(source)
                init.assert_called_once_with('fixture.pt','cuda:0',cache_classes=True,half=True)
                self.assertEqual(model.segment_instances.call_args.args[1],segmenter.categories)
                self.assertEqual(result['provenance']['vocabulary']['version'],'r2r-indoor-v1')

    def test_wrong_engine_vocabulary_rejected_before_loading(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            weights=Path(tmp)/'model.engine';weights.write_bytes(b'fixture')
            Path(str(weights)+'.json').write_text(json.dumps(dict(categories=list(CORE),sha256=file_sha256(weights))))
            with patch('ultralytics.YOLO') as model:
                with self.assertRaisesRegex(ValueError,'vocabulary'):
                    FixedSegmenter(weights,vocabulary='configs/dataset_preparation/r2r_indoor.v1.json')
                model.assert_not_called()

    def test_posthoc_done_flag_does_not_modify_response(self):
        from ticvla.warehouse.dataset_tools.r2r_pilot_review import review_flags
        answer={'local':'Near the doorway.','route':'Wait here. DONE'}
        sample={'observation':{'execution':{'events':[]}}}
        self.assertEqual(review_flags(answer,sample),['DONE_TOKEN_WITHOUT_RECORDED_COMPLETION_EVIDENCE'])
        self.assertEqual(answer['route'],'Wait here. DONE')
        sample['observation']['execution']['events']=[{'kind':e} for e in ['target_visible','stopped','arrival_confirmed']]
        self.assertEqual(review_flags(answer,sample),[])

    def test_review_reproduces_actual_jpeg_bytes(self):
        import hashlib,io
        from ticvla.warehouse.dataset_tools.r2r_pilot_review import export_actual_image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'rgb.png';im=Image.new('RGB',(17,12),'red');im.save(source)
            buf=io.BytesIO();im.save(buf,format='JPEG',quality=95,subsampling=0)
            item={'path':str(source),'sha256':file_sha256(source),'encoding':'jpeg_quality_95_444_original_resolution',
                  'transmitted_sha256':hashlib.sha256(buf.getvalue()).hexdigest()}
            dest=export_actual_image(item,root/'actual')
            self.assertEqual(dest.read_bytes(),buf.getvalue())
            item['transmitted_sha256']='wrong'
            with self.assertRaises(ValueError):export_actual_image(item,root/'bad')
