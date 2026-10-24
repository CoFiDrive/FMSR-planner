"""Ordered sampling keeps curved labels on their original forward polyline."""
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'3rd_party/carla_garage/team_code'),str(ROOT/'3rd_party/CARLA_0915/PythonAPI/carla')]
from route_geometry import sample_route_on_polyline,smooth_route,prepare_route


class OrderedSamplingTests(unittest.TestCase):
    def test_straight_sampling_and_end_extrapolation(self):
        actual=sample_route_on_polyline([[.5,0.],[1.,0.],[2.,0.]],10)
        np.testing.assert_allclose(actual,np.column_stack((2.5+np.arange(10),np.zeros(10))),atol=1e-8)

    def test_short_first_segment_curve_does_not_fold(self):
        route=np.array([[.73,-.87],[1.43,-1.88],[2.15,-2.85],[2.9,-3.7],[3.7,-4.4],[4.5,-5.0]])
        legacy=smooth_route(prepare_route(route,20),20)[:6]
        self.assertTrue((np.diff(legacy[:,0])<0).any())
        sampled=sample_route_on_polyline(route,4)
        self.assertTrue((np.diff(sampled[:,0])>0).all())
        self.assertTrue((np.diff(sampled[:,1])<0).all())
        np.testing.assert_allclose(np.linalg.norm(sampled[0]),2.5,atol=1e-8)
        np.testing.assert_allclose(np.linalg.norm(np.diff(sampled,axis=0),axis=1),1.,atol=1e-8)
        points=np.vstack((np.zeros((1,2)),route))
        for p in sampled:
            distances=[]
            for a,b in zip(points,points[1:]):
                t=np.clip((p-a)@(b-a)/((b-a)@(b-a)),0.,1.)
                distances.append(np.linalg.norm(p-(a+t*(b-a))))
            self.assertLess(min(distances),1e-8)

    def test_invalid_and_degenerate_inputs_fail(self):
        for route,count in [([[0.,0.]],10),([[float('nan'),0.]],10),([[1.,0.]],0)]:
            with self.assertRaises(ValueError):sample_route_on_polyline(route,count)


if __name__=='__main__':unittest.main()
