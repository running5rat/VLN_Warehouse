import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ticvla.warehouse.mapping.postprocess import build_view, LandmarkViewCache
from ticvla.warehouse.mapping import MapStore


def evidence(frame, text, x=30, width=50, camera_x=0, confidence=.99):
    pose = np.eye(4); pose[0, 3] = camera_x
    return dict(image_id=str(frame), signature=text, camera_pose=pose.tolist(),
                texts=[dict(text=text, normalized=text, confidence=confidence, status="bound",
                            polygon=[[x, 20], [x+width, 20], [x+width, 40], [x, 40]])])


def landmark(uid="one", category="sign", readings=None):
    readings = readings or []
    return dict(landmark_uid=uid, submap_id="epoch", category=category, status="tentative",
                static_confirmed=False, identity_status="conflict", identity_text=None,
                position=[0, 0, 0], observation_ids=[r["image_id"] for r in readings],
                text_evidence=readings)


def view(record):
    return build_view([record], lambda _: (200, 100))["landmarks"][0]


class MapPostprocessTests(unittest.TestCase):
    def test_substring_preserves_parent_without_confirming(self):
        r=landmark(readings=[evidence(1,"A-123"),evidence(2,"123",x=0)])
        before=copy.deepcopy(r);v=view(r)
        self.assertEqual(v["identity_text"],"A-123")
        self.assertEqual(v["identity_status"],"candidate")
        self.assertEqual(v["resolved_identity"]["full_support_frames"],["1"])
        self.assertEqual(v["resolved_identity"]["compatible_partial_frames"],["2"])
        self.assertEqual(r,before)

    def test_substring_does_not_require_clipping_or_prefix_suffix(self):
        for x in (40,145):
            v=view(landmark(readings=[evidence(1,"A-123"),evidence(2,"12",x=x)]))
            self.assertEqual(v["identity_status"],"candidate")
            self.assertEqual(v["identity_text"],"A-123")
            self.assertEqual(v['resolved_identity']['ignored_substrings'],['12'])

    def test_true_conflict_never_majority_corrected(self):
        r=landmark(readings=[evidence(i,"A-123",camera_x=i) for i in range(4)]+[evidence(9,"A-128")])
        result=view(r)
        self.assertEqual(result["identity_status"],"conflict")
        self.assertIsNone(result['identity_text'])
        self.assertEqual(result['resolved_identity']['registration_status'],'rejected')

    def test_fragments_never_invent_full_reading(self):
        r=landmark(readings=[evidence(1,"A-",x=150),evidence(2,"123",x=0)])
        self.assertIsNone(view(r)["identity_text"])

    def test_distinct_full_frames_and_views_required(self):
        rows=[evidence(i,"A-123",camera_x=i) for i in range(3)]
        self.assertEqual(view(landmark(readings=rows))["identity_status"],"identity_confirmed")
        self.assertEqual(view(landmark(readings=[rows[0]]*3))["identity_status"],"candidate")
        self.assertEqual(view(landmark(readings=[evidence(i,"A-123") for i in range(3)]))["identity_status"],"candidate")
        v=view(landmark(readings=[rows[0],evidence(2,"123",x=0),evidence(3,"123",x=0)]))
        self.assertEqual(v["identity_status"],"candidate")
        self.assertFalse(v["static_confirmed"])

    def test_low_confidence_does_not_confirm(self):
        self.assertEqual(view(landmark(readings=[evidence(i,"A-123",camera_x=i,confidence=.3) for i in range(3)]))["identity_status"],"unknown")

    def test_adjacent_regions_join_but_remote_text_is_not_serial(self):
        e=evidence(1,"72",x=20,width=20)
        e["texts"]+=evidence(1,"/5R",x=45,width=30)["texts"]
        self.assertEqual(view(landmark(readings=[e]))["identity_text"],"72/5R")
        e["texts"][1]=evidence(1,"/5R",x=150,width=30)["texts"][0]
        self.assertEqual(view(landmark(readings=[e]))["identity_text"],"72 | /5R")

    def test_repeated_structures_and_direct_printed_carriers(self):
        a=view(landmark(category="storage rack"))
        self.assertEqual(a["landmark_view"]["decision"],"hide")
        b=view(landmark(category="storage rack",readings=[evidence(1,"A-123")]))
        self.assertTrue(b["landmark_view"]["visible"])
        self.assertEqual(b["landmark_view"]["decision"],"review")
        self.assertTrue(view(landmark())["landmark_view"]["visible"])

    def test_generic_text_is_attribute_and_not_identity_anchor(self):
        v=view(landmark(readings=[evidence(1,"CAUTION")]))
        self.assertEqual(v["landmark_view"]["decision"],"hide")
        self.assertEqual(v["identity_text"],"CAUTION")
        self.assertFalse(v["static_confirmed"])

    def test_duplicate_text_keeps_separate_instances(self):
        rows=[evidence(i,"A-123",camera_x=i) for i in range(3)]
        a,b=landmark("one",readings=rows),landmark("two",readings=rows)
        b["position"]=[10,0,0]
        vs=build_view([a,b])["landmarks"]
        self.assertEqual(len(vs),2)
        self.assertTrue(all(v["identity_status"]=="duplicate_text" for v in vs))
        self.assertEqual(vs[1]["position"],[10,0,0])
        a["duplicate_text"]=True
        single=view(a)
        self.assertTrue(single["raw_duplicate_text"])
        self.assertFalse(single["duplicate_text"])
        self.assertTrue(a["duplicate_text"])

    def test_missing_image_dimensions_do_not_prevent_substring_policy(self):
        r=landmark(readings=[evidence(1,"A-123"),evidence(2,"123",x=0)])
        self.assertEqual(build_view([r])["landmarks"][0]["identity_text"],"A-123")

    def test_latest_failed_signs_now_choose_parent(self):
        for parent,short in [('72/4R','/4R'),('72/5R','2/5R')]:
            self.assertEqual(view(landmark(readings=[evidence(1,parent),evidence(2,short)]))['identity_text'],parent)

    def test_containment_chain_order_independent_and_incomparable_roots_rejected(self):
        import itertools
        for texts in itertools.permutations(['2','2/5','72/5R']):
            result=view(landmark(readings=[evidence(i,t) for i,t in enumerate(texts)]))
            self.assertEqual(result['identity_text'],'72/5R')
        result=view(landmark(readings=[evidence(i,t) for i,t in enumerate(['/5R','72/5R','173/5R'])]))
        self.assertIsNone(result['identity_text'])
        self.assertEqual(result['resolved_identity']['registration_status'],'rejected')
        self.assertEqual(result['resolved_identity']['ignored_substrings'],['/5R'])

    def test_cache_incremental_and_restart_do_not_mutate_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);rgb=p/"rgb.png";Image.new("RGB",(200,100)).save(rgb)
            store=MapStore(p/"map")
            row=landmark(readings=[evidence(1,"A-123")])
            store._put("observations","1",dict(image_id="1",rgb=str(rgb)))
            store._put("landmarks","one",row);store.db.commit()
            v=store.landmark_view();self.assertEqual(v,store.landmark_view())
            row["text_evidence"].append(evidence(2,"123",x=0))
            store._put("observations","2",dict(image_id="2",rgb=str(rgb)))
            store._put("landmarks","one",row);store.db.commit()
            updated=store.landmark_view()
            self.assertEqual(updated["landmarks"][0]["resolved_identity"]["compatible_partial_frames"],["2"])
            self.assertEqual(store.get("landmarks","one"),row)
            store.db.close()
            again=MapStore(p/"map")
            self.assertEqual(updated,again.landmark_view());again.db.close()

    def test_hidden_landmark_does_not_erase_geometric_occupancy(self):
        from DynaNav.warehouse_loop.live_mapping import LiveBEV
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);bev=LiveBEV()
            bev.add(np.array([[1,1,1],[2,2,1.5]]),np.eye(4))
            original_cells=set(bev.cells)
            bev.render([],p/'geometry.png')
            bev.render([view(landmark(category='storage rack'))],p/'filtered.png')
            np.testing.assert_array_equal(np.array(Image.open(p/'geometry.png')),np.array(Image.open(p/'filtered.png')))
            self.assertEqual(bev.cells,original_cells)


if __name__ == "__main__":
    unittest.main()
