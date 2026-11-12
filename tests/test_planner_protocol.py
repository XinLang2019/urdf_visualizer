"""Protocol boundaries and sparse NumPy-assignment compatibility."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planner_protocol import DEFAULTS,validate_config

class PlannerProtocolTests(unittest.TestCase):
    def test_defaults_are_valid_and_not_mutated(self):
        before=copy.deepcopy(DEFAULTS);cfg=validate_config({})
        self.assertEqual(sum(round(t/cfg['dt']) for t in cfg['phases'].values()),125)
        cfg['phases']['flight']=1
        self.assertEqual(DEFAULTS,before)
    def test_limits_cannot_expand_motor_rating_or_run_unbounded(self):
        for value in [{'torqueScale':1.01},{'friction':0},{'gravity':float('nan')},{'maxIterations':1.5},{'rounds':True},{'wallTime':0},{'dt':.01},{'meshCollision':'false'},{'lockLandingJoints':0},{'phases':{'stand':0}},{'unknown':1}]:
            with self.subTest(value=value),self.assertRaises(ValueError):validate_config(value)
    def test_grid_quantization_and_standing_require_real_contact_nodes(self):
        cfg=validate_config({'motion':'stand','dt':.1,'phases':{'stand':.2,'crouch':0,'load':0,'flight':0,'landing':0,'settle':.2},'wallTime':30})
        self.assertEqual(sum(cfg['phases'].values()),.4)
        self.assertIsInstance(cfg['wallTime'],float)
        with self.assertRaises(ValueError):validate_config({'motion':'stand'})
        with self.assertRaises(ValueError):validate_config({'waypointEnabled':True,'phases':{'flight':0}})
    def test_sparse_blocks_match_numpy_including_duplicates(self):
        import numpy as np
        from planner_sparse import SparseJacobian
        pairs=[(r,c) for r in range(4) for c in range(5)]+[(4,c) for c in range(3)]+[(0,0)]
        rows,cols=zip(*pairs);s=SparseJacobian(rows,cols,(5,5));d=np.zeros((5,5))
        for key,value in [((slice(0,2),slice(1,4)),np.arange(6).reshape(2,3)),((2,slice(0,5)),np.arange(5).reshape(1,5)),((slice(0,5),1),np.arange(5)),((3,4),7)]:s[key]=value;d[key]=value
        np.testing.assert_array_equal(s.result(),d[rows,cols])
        s.clear();np.testing.assert_array_equal(s.result(),np.zeros(len(pairs)))
        with self.assertRaises(ValueError):s[4,4]=1

if __name__=='__main__':unittest.main()
