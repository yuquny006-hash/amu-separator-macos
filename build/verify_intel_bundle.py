"""Check every Mach-O in the Intel bundle for architecture and deployment floor."""
from pathlib import Path
import re
import subprocess
import sys

app = Path(sys.argv[1]).resolve()
magic = {bytes.fromhex(h) for h in ('cffaedfe', 'cefaedfe', 'feedfacf', 'feedface', 'cafebabe', 'bebafeca', 'cafebabf', 'bfbafeca')}
count = 0
for path in app.rglob('*'):
    if not path.is_file() or path.is_symlink():
        continue
    with path.open('rb') as stream:
        if stream.read(4) not in magic:
            continue
    subprocess.run(['lipo', str(path), '-verify_arch', 'x86_64'], check=True)
    load = subprocess.check_output(['otool', '-arch', 'x86_64', '-l', str(path)], text=True)
    for command in load.split('Load command'):
        if 'LC_BUILD_VERSION' in command:
            found = re.search(r'\bminos\s+(\d+(?:\.\d+){0,2})', command)
        elif 'LC_VERSION_MIN_MACOSX' in command:
            found = re.search(r'\bversion\s+(\d+(?:\.\d+){0,2})', command)
        else:
            continue
        if found is None:
            raise RuntimeError('Cannot read deployment floor: '+str(path))
        version = tuple(map(int, found[1].split('.')))
        if (version+(0, 0))[:3] > (14, 0, 0):
            raise RuntimeError(f'Requires macOS {found[1]}: {path}')
    dependencies = subprocess.check_output(['otool', '-arch', 'x86_64', '-L', str(path)], text=True)
    own_ids = subprocess.check_output(['otool', '-arch', 'x86_64', '-D', str(path)], text=True).splitlines()[1:]
    own_ids = {line.strip() for line in own_ids}
    for line in dependencies.splitlines()[1:]:
        dep = line.strip().split(' (compatibility version')[0]
        if dep.startswith('/') and not dep.startswith(('/usr/lib/', '/System/Library/')) and dep not in own_ids:
            raise RuntimeError(f'External non-system library {dep}: {path}')
    count += 1
assert count > 100, count
print(f'Intel bundle audit passed: {count} Mach-O files, macOS <=14.0, no external non-system libraries')
