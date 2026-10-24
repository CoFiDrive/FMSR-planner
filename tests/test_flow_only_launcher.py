"""Launch argument regression: stale recovery exports cannot change fresh FM."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/train_fmsr_pdm_lite_flow_only.sh'
OFFICIAL='/media/vcdc/T71/tfpp_pretrained/pretrained_models/all_towns/model_0030_2.pth'


class FlowOnlyLauncherTests(unittest.TestCase):
    def run_launcher(self,**overrides):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);calls=root/'calls.jsonl';fake=root/'fake_python'
            fake.write_text('#!/usr/bin/env python3\nimport os,json,sys\nfrom pathlib import Path\nkeys=["TFPP_CHECKPOINT","COSINE_T0","RECOVERY_DATA_ROOT","CONTINUE_EPOCH","FLOW_CONTRACT_ENABLE"]\nwith Path(os.environ["FLOW_TEST_CALLS"]).open("a") as f:f.write(json.dumps({"argv":sys.argv[1:],"env":{k:os.environ.get(k) for k in keys}})+"\\n")\n')
            fake.chmod(0o755)
            env=dict(os.environ,PYTHON_BIN=str(fake),FLOW_TEST_CALLS=str(calls),TRAIN_LOG_ROOT=str(root/'runs'),RUN_ID='test_run',
                TFPP_CHECKPOINT='/stale/tie_cr_fd/model_0002.pth',RECOVERY_DATA_ROOT='/missing/old_recovery',
                RECOVERY_SAMPLING_PLAN='/missing/old_sampling',CONTINUE_EPOCH='3',COSINE_T0='20',
                FLOW_CONTRACT_ENABLE='1',EPOCHS='10',PREFLIGHT_ONLY='0')
            for k in ('FLOW_ONLY_INIT_CHECKPOINT','FLOW_ONLY_COSINE_T0'):env.pop(k,None)
            env.update(overrides)
            proc=subprocess.run(['bash',str(SCRIPT)],cwd=ROOT,env=env,text=True,capture_output=True)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            return [json.loads(s) for s in calls.read_text().splitlines()]

    def test_fresh_launch_discards_stale_checkpoint_recovery_and_schedule(self):
        calls=self.run_launcher()
        for call in calls:
            self.assertEqual(call['env']['TFPP_CHECKPOINT'],OFFICIAL)
            self.assertEqual(call['env']['COSINE_T0'],'10')
            self.assertIsNone(call['env']['RECOVERY_DATA_ROOT'])
            self.assertEqual(call['env']['CONTINUE_EPOCH'],'0')
            self.assertEqual(call['env']['FLOW_CONTRACT_ENABLE'],'0')
        train=next(c['argv'] for c in calls if 'torch.distributed.run' in c['argv'])
        self.assertEqual(train[train.index('--fmsr_trajectory_head')+1],'flow_only')
        self.assertEqual(train[train.index('--load_file')+1],OFFICIAL)
        self.assertNotIn('--fmsr_recovery_root',train)
        self.assertEqual(train[train.index('--continue_epoch')+1],'0')

    def test_explicit_flow_only_overrides_work_in_preflight(self):
        calls=self.run_launcher(FLOW_ONLY_INIT_CHECKPOINT='/explicit/tfpp.pth',FLOW_ONLY_COSINE_T0='5',PREFLIGHT_ONLY='1')
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['env']['TFPP_CHECKPOINT'],'/explicit/tfpp.pth')
        self.assertEqual(calls[0]['env']['COSINE_T0'],'5')


if __name__=='__main__':unittest.main()
