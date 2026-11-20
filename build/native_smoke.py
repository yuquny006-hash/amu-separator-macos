"""Exercise the signed native launcher and verify its service exits with it."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

app = Path(sys.argv[1]).resolve()
report = Path(sys.argv[2]).resolve()
env = dict(os.environ, AMU_NATIVE_SMOKE_RESULT=str(report))
subprocess.run([str(app / 'Contents/MacOS/AmuMusic'), '--smoke-test'], env=env,
               timeout=120, check=True)
result = json.loads(report.read_text())
assert result['passed'], result
pid = result['service_pid']
assert pid > 0
for _ in range(30):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        break
    time.sleep(0.1)
else:
    raise AssertionError('Service process group still running after app quit')
print('Signed native launch, HTTP readiness and service shutdown passed')
