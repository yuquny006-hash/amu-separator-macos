"""Fetch hash-pinned model assets from THIS repository's public release.

No credentials beyond the workflow's own GITHUB_TOKEN; no third-party weight
is re-hosted anywhere else. SHA256 of every分片 and every extracted file is
verified before the build is allowed to continue.
"""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import urllib.request
import urllib.parse


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new and urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            new.remove_header('Authorization')
        return new


opener = urllib.request.build_opener(SafeRedirect())
root = Path(__file__).resolve().parents[1]
meta = json.loads((root / 'build/assets.json').read_text(encoding='utf-8'))
headers = {'Authorization': 'Bearer ' + os.environ['GH_TOKEN'], 'User-Agent': 'Amu-build'}
url = 'https://api.github.com/repos/' + os.environ['GITHUB_REPOSITORY'] + '/releases/tags/' + meta['tag']
with opener.open(urllib.request.Request(url, headers=headers), timeout=90) as r:
    assets = {a['name']: a for a in json.load(r)['assets']}
archive = root / 'models.tar'
with archive.open('wb') as out:
    for part in meta['parts']:
        h = dict(headers, Accept='application/octet-stream')
        req = urllib.request.Request(assets[part['name']]['url'], headers=h)
        digest, size = hashlib.sha256(), 0
        with opener.open(req, timeout=300) as response:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
                out.write(chunk)
        assert size == part['size'] and digest.hexdigest() == part['sha256'], part['name']
        print('Verified', part['name'], flush=True)
with tarfile.open(archive) as tar:
    tar.extractall(root / 'app', filter='data')
manifest = json.loads((root / 'build/models-manifest.json').read_text(encoding='utf-8'))
for name, info in manifest.items():
    with (root / 'app' / name).open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == info['sha256'], name
archive.unlink()
print('All model hashes verified')
