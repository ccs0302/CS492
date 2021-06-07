import copy;
import math;
import open3d as o3d;
import point_cloud_utils as pcu;
import numpy as np;


 
pcd = o3d.geometry.PointCloud();

v, f = pcu.load_mesh_vf('observation_2.ply');

pointsize = v.size;
pointsize1 = pointsize
print(pointsize1);


pointsize8 = math.ceil((pointsize/3*0.7));
print(pointsize8);
#samples_3d = pcu.sample_mesh_lloyd(v, f, pointsize8);
idx = pcu.downsample_point_cloud_poisson_disk(v, pointsize8)

# Use the indices to get the sample positions and normals
v_sampled = v[idx]

pcd.points = o3d.utility.Vector3dVector(v_sampled);
o3d.io.write_point_cloud('gObservation_random_2_70_3.ply', pcd);







