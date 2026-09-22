import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.public_benchmark import prepare

if __name__=='__main__':
    folder=ROOT/'external_data/pdebench/fixed_t1_v1'
    if (folder/'protocol.json').exists():
        raise SystemExit('Prepared data already exists; do not overwrite a locked split.')
    result=prepare(ROOT/'external_data/pdebench/1D_Burgers_Sols_Nu0.001.hdf5',folder)
    print('Prepared',result['grid_size'],'points at time',result['target_time'],flush=True)
