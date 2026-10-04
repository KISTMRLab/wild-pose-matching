import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from example_demo import query


def test_authored_example_uses_cluster_retrieval():
    result=query("wild","point to the result",{"seed":["3"]})
    assert result["slots"][0]["gesture_id"]=="point_right"
    assert result["trace"][0]["cluster_id"] in (0,1)
    assert "no trained weights" in result["data_label"]
