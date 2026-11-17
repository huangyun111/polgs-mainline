import importlib.util
from pathlib import Path
p=Path('/home/hy/gaussian_surfels_stage4_20261020/POLAR_PGSR_REPROJECTION_20261106/evidence/jit_geo_depth/polgs_geo_depth_v1.so')
spec=importlib.util.spec_from_file_location("polgs_geo_depth_v1",str(p))
extension=importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)
