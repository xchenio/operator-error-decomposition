"""Pinned public Burgers download; resume partial bytes and verify publisher SHA-256."""
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REV='97bf5bb5315b9a083b4f9fc03e5c84d35286d52c'
NAME='1D_Burgers_Sols_Nu0.001.hdf5'
URL=f'https://huggingface.co/datasets/pdebench/Burgers/resolve/{REV}/{NAME}'
SIZE=8232968312
SHA='afeeed1c40ce01d2ba5e1702f4f66b6bf95c65943eb0e4847691e9a10d1cb50d'


def main():
    folder=ROOT/'external_data/pdebench'
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/NAME
    part=folder/(NAME+'.part')
    state=folder/'download_status.json'
    def status(kind,**kw):
        info=dict(status=kind,url=URL,revision=REV,expected_bytes=SIZE,expected_sha256=SHA,
                  downloaded_bytes=part.stat().st_size if part.exists() else (target.stat().st_size if target.exists() else 0),
                  time_unix=time.time(),**kw)
        tmp=state.with_suffix('.tmp'); tmp.write_text(json.dumps(info,indent=2),encoding='utf-8'); tmp.replace(state)
        print(json.dumps(info),flush=True)
    failures=0
    while not target.exists() and (not part.exists() or part.stat().st_size<SIZE):
        offset=part.stat().st_size if part.exists() else 0
        status('downloading')
        req=urllib.request.Request(URL,headers={'Range':f'bytes={offset}-'} if offset else {})
        try:
            with urllib.request.urlopen(req,timeout=30) as response:
                if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                    raise RuntimeError('server did not honor resume range')
                with part.open('ab') as f:
                    last=time.monotonic()
                    while chunk:=response.read(4*1024*1024):
                        f.write(chunk)
                        if time.monotonic()-last>10:
                            f.flush(); os.fsync(f.fileno()); status('downloading'); last=time.monotonic()
                    f.flush(); os.fsync(f.fileno())
        except Exception as e:
            failures+=1; status('retrying',error=str(e),attempt=failures)
            if failures>=5:
                status('failed',error=str(e)); raise
            time.sleep(3)
    candidate=target if target.exists() else part
    if candidate.stat().st_size!=SIZE:
        status('failed',error='size mismatch'); raise RuntimeError('size mismatch')
    status('verifying')
    h=hashlib.sha256()
    with candidate.open('rb') as f:
        while chunk:=f.read(4*1024*1024): h.update(chunk)
    if h.hexdigest()!=SHA:
        status('failed',error='SHA-256 mismatch'); raise RuntimeError('SHA-256 mismatch')
    if candidate==part: part.replace(target)
    status('completed',sha256=h.hexdigest())


if __name__=='__main__': main()
