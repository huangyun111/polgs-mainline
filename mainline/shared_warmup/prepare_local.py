from pathlib import Path
import zipfile,hashlib,json,io,ast,shutil
import numpy as np
from PIL import Image
E=Path(__file__).resolve().parent;P=E.parents[1]
zpath=P.parent/'01_paperreserarch/reproductions/PolGS/offline_bundle_20261014/downloads/PolGS_plus_official_dataset.zip'
z=zipfile.ZipFile(zpath);prefix='dataset/PISR/StandingRabbit/'
files=[]
for n in z.namelist():
    if n.startswith(prefix) and not n.endswith('/'):
        b=z.read(n);files.append(dict(path=n[len(prefix):],bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
assert len([x for x in files if x['path'].startswith('image/')])==20
raw=zipfile.ZipFile(P.parent/'01_paperreserarch/reproductions/PISR/downloads/StandingRabbit.zip')
def pixelsha(b):return hashlib.sha256(np.asarray(Image.open(io.BytesIO(b)).convert('RGB')).tobytes()).hexdigest()
official_cams=np.load(io.BytesIO(z.read(prefix+'cameras.npz')))
original_cams=np.load(io.BytesIO(raw.read('StandingRabbit/cameras_sphere.npz')))
mapping=[]
for x in files:
    if x['path'].startswith('image/'):
        i=int(Path(x['path']).stem);errors=[float(abs(official_cams[f'world_mat_{i}']-original_cams[f'world_mat_{j}']).max()) for j in range(40)];j=int(np.argmin(errors));assert errors[j]==0 and errors.count(0)==1
        original=f'StandingRabbit/image/{j:03d}.png';assert original in raw.namelist()
        mapping.append(dict(official_id=i,source_id=j,original_image=original,sha256=x['sha256'],decoded_RGB_sha256=pixelsha(z.read(prefix+x['path'])),source_projection_max_error=errors[j],source_match='exact unique original world_mat;official processed image verified bytewise against author release,not asserted equal to raw PISR RGB'))
assert len(set(x['source_id'] for x in mapping))==20
(E/'RABBIT_OFFICIAL_MANIFEST.json').write_text(json.dumps(dict(official_zip=str(zpath),files=files,source_mapping=mapping),indent=2),encoding='utf-8')
for name in ['routing_core.py','density_state.py']:
    src=P/('GEOMETRY_UPDATE_STATS_2X2_20261109/evidence' if name=='routing_core.py' else 'FULLTRAIN_GEO_PAIRING_20261109/evidence')/name
    shutil.copy2(src,E/name)
for name in ['geo_renderer.py','slope_proxy.py','extension_loader.py','multiview_geometry.py','covisibility.py']:
    shutil.copy2(P/'PAIRING_AND_DENSITY_20261108/evidence/verified_launch/pairing/StandingRabbit/D'/name,E/name)
print('OFFICIAL_RABBIT_SOURCE_IDS',mapping)
