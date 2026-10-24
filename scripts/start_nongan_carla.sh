#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${FM_PYTHON:-/home/vcdc/anaconda3/envs/fmsr/bin/python}" - "$ROOT" "${1:-localhost}" "${2:-2000}" <<'PY'
import os,pathlib,subprocess,sys,time
import carla
root=pathlib.Path(sys.argv[1]);host=sys.argv[2];port=int(sys.argv[3])
def connected():
    try:
        client=carla.Client(host,port);client.set_timeout(1.);client.get_server_version();return True
    except RuntimeError:return False
if connected():
    print('CARLA 已连接，复用现有服务；场景退出不会停止它。',flush=True);raise SystemExit(0)
if host not in ('localhost','127.0.0.1'):
    raise SystemExit('远程 CARLA 未连接，请先在服务器启动 CARLA。')
# Do not start a duplicate over a server that is still booting or stalled.
running=subprocess.run(['pgrep','-af','CarlaUE4-Linux-Shipping'],capture_output=True,text=True).stdout
if running:
    raise SystemExit('已有 CARLA 进程但 RPC 尚不可用，请检查该进程；未启动重复服务。')
carla_root=pathlib.Path(os.environ.get('CARLA_ROOT','/home/vcdc/CARLA_0.9.15'))
launcher=carla_root/'CarlaUE4.sh'
if not launcher.is_file():raise SystemExit('CARLA 启动文件不存在: '+str(launcher))
logs=root/'outputs/carla'/str(port);logs.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.setdefault('DISPLAY',':0')
auth=pathlib.Path('/run/user/{}/gdm/Xauthority'.format(os.getuid()))
if auth.is_file():env.setdefault('XAUTHORITY',str(auth))
with open(logs/'server.log','ab',buffering=0) as log:
    server=subprocess.Popen([str(launcher),'-nosound','-quality-level=Epic',
        '-carla-rpc-port='+str(port),'-windowed','-ResX=1280','-ResY=720'],
        cwd=str(carla_root),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,close_fds=True)
(logs/'server.pid').write_text(str(server.pid)+'\n')
print('已独立启动 CARLA，PID={}；日志={}；场景结束后保留服务。'.format(server.pid,logs/'server.log'),flush=True)
for unused in range(45):
    if server.poll() is not None:raise SystemExit('CARLA 启动失败，请查看 '+str(logs/'server.log'))
    if connected():print('CARLA RPC 就绪。',flush=True);break
    time.sleep(.5)
else:raise SystemExit('CARLA 尚未就绪，服务未被终止；请查看 '+str(logs/'server.log'))
PY
