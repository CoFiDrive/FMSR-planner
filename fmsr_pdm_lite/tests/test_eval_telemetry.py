import torch
from test_paper_bridge import planner_fixture
from scripts.pdm_lite.planner_telemetry import GenerationCapture


def test_paper_bridge_logging_preserves_seeded_sample(tmp_path):
    torch.set_num_threads(2)
    head=planner_fixture(tmp_path).eval()
    query,bev=torch.randn(1,11,256)*.1,torch.randn(1,4,4,4)*.1
    torch.manual_seed(123)
    expected,_=head.sample(query,bev,steps=4)
    torch.manual_seed(123)
    capture=GenerationCapture(20,20)
    head._eval_capture=capture
    capture.scene(head,query,bev,(4,4))
    actual,_=head.sample(query,bev,steps=4)
    capture.finish()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert capture.report()['nfe']==4
    assert len(capture.trace)==4
