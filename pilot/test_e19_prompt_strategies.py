"""Numerical contract checks: score-map averaging and paired IoU aggregation."""
import unittest
import numpy as np
from run_e19_prompt_strategies import condition_maps, confusion, counts, ious, paired_intervals, load_prompts, STUDY

class E19Contracts(unittest.TestCase):
    def test_average_before_argmax(self):
        # Two paraphrases narrowly favor class 0, but the third strongly favors 1.
        # Majority voting would choose 0; arithmetic probability averaging chooses 1.
        maps=[np.array([[.8,.2]]),np.array([[.51,.49]]),np.array([[.51,.49]]),np.array([[.01,.99]])]
        a,b,c=condition_maps(maps)
        self.assertEqual(int(c.argmax(-1)[0]),1)
        np.testing.assert_allclose(c.sum(-1),1)
        np.testing.assert_array_equal(a,maps[0]);np.testing.assert_array_equal(b,maps[1])

    def test_global_iou_is_not_mean_object_iou(self):
        cm1=confusion([0],[0]);cm2=confusion([0]*9,[18]*9)
        self.assertAlmostEqual(float(ious(counts(cm1+cm2))[0]),.1)
        self.assertNotAlmostEqual(float(ious(counts(cm1+cm2))[0]),.5)

    def test_none_competitor(self):
        cm=confusion([18,18,0],[0,18,18]);tp,fp,fn=counts(cm)[0]
        self.assertEqual((tp,fp,fn),(0,1,1))
        self.assertTrue(np.isnan(ious(counts(cm))[1]))

    def test_paired_identical_maps_have_zero_interval(self):
        ct=counts(confusion([0,18,0],[0,18,18]))
        obj=np.stack([np.stack([ct,ct,ct]) for _ in range(4)])
        for interval in paired_intervals(obj,1,40).values():
            np.testing.assert_array_equal(interval,[0,0])

    def test_prompt_slot_alignment(self):
        _,v=load_prompts(STUDY/'prompts.json')
        self.assertEqual(len(v),4)
        self.assertTrue(all(len(words)==19 and words[-1]=='none' for words in v))
        self.assertEqual(v[0][15],'press')
        self.assertEqual(v[0][10],'displaY')

if __name__=='__main__':unittest.main()
