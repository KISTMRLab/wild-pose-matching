import numpy as np
from wild_pose_matching.pipeline import augment_projected,sixgrams,cluster_latents
def test_utilities():
    x=np.arange(30,dtype=np.float32).reshape(5,6)
    y=augment_projected(x,0,1,np.random.default_rng(1)); assert y.shape==x.shape
    assert sixgrams("one two three four five six seven")==["one two three four five six","seven"]
    labels,centers=cluster_latents(np.array([[1.,0.],[.9,.1],[0.,1.],[.1,.9]]),2); assert len(set(labels))==2 and centers.shape==(2,2)

