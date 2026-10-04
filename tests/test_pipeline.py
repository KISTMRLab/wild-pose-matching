import numpy as np
from wild_pose_matching.pipeline import augment_projected,sixgrams,cluster_latents
from wild_pose_matching.pipeline import build_rules,retrieve
def test_utilities():
    x=np.arange(30,dtype=np.float32).reshape(5,6)
    y=augment_projected(x,0,1,np.random.default_rng(1)); assert y.shape==x.shape
    assert sixgrams("one two three four five six seven")==["one two three four five six","seven"]
    labels,centers=cluster_latents(np.array([[1.,0.],[.9,.1],[0.,1.],[.1,.9]]),2); assert len(set(labels))==2 and centers.shape==(2,2)

def test_paired_latent_matching_and_cluster_sampling():
    z=np.eye(2,dtype=np.float32)
    rules=build_rules(z,["open hand","point right"],z,z,["open","point"],np.array([0,1]))
    assert [r["gesture_id"] for r in rules]==["open","point"]
    chosen=retrieve("point right",rules,lambda _:z[1:2],{0:["open"],1:["point"]},seed=3)
    assert chosen[0]["gesture_id"]=="point" and chosen[0]["cluster_id"]==1
