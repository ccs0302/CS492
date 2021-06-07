import copy;
import math;
import open3d as o3d;
import point_cloud_utils as pcu;
import numpy as np;

pcd = o3d.geometry.PointCloud();

v, f = pcu.load_mesh_vf('bun180.ply');

pointsize = v.size;
pointsize1 = pointsize
print(pointsize1);
samples_3d = pcu.sample_mesh_lloyd(v, f, pointsize1);
pcd.points = o3d.utility.Vector3dVector(samples_3d);
o3d.io.write_point_cloud('gBun100.ply', pcd);

pointsize2 = math.ceil((pointsize*0.7));
print(pointsize2);
samples_3d = pcu.sample_mesh_lloyd(v, f, pointsize2);

pcd.points = o3d.utility.Vector3dVector(samples_3d);
o3d.io.write_point_cloud('gBun70.ply', pcd);

pointsize3 = math.ceil((pointsize*0.4));
print(pointsize3);
samples_3d = pcu.sample_mesh_lloyd(v, f, pointsize3);
pcd.points = o3d.utility.Vector3dVector(samples_3d);
o3d.io.write_point_cloud('gBun40.ply', pcd);

pointsize4 = math.ceil((pointsize*0.1));
print(pointsize4);
samples_3d = pcu.sample_mesh_lloyd(v, f, pointsize4);
pcd.points = o3d.utility.Vector3dVector(samples_3d);
o3d.io.write_point_cloud('gBun10.ply', pcd);


pointsize5 = math.ceil((pointsize/3));
print(pointsize5);
samples_3d = pcu.sample_mesh_lloyd(v, f, pointsize5);
pcd.points = o3d.utility.Vector3dVector(samples_3d);
o3d.io.write_point_cloud('gBun100_3.ply', pcd);

pointsize6 = math.ceil((pointsize/3*0.7));
print(pointsize6);
samples_3d = pcu.sample_mesh_lloyd(v, f, pointsize6);
pcd.points = o3d.utility.Vector3dVector(samples_3d);
o3d.io.write_point_cloud('gBun70_3.ply', pcd);

pointsize7 = math.ceil((pointsize/3*0.4));
print(pointsize7);
samples_3d = pcu.sample_mesh_lloyd(v, f, pointsize7);
pcd.points = o3d.utility.Vector3dVector(samples_3d);
o3d.io.write_point_cloud('gBun40_3.ply', pcd);

pointsize8 = math.ceil((pointsize/3*0.1));
print(pointsize8);
samples_3d = pcu.sample_mesh_lloyd(v, f, pointsize8);
pcd.points = o3d.utility.Vector3dVector(samples_3d);
o3d.io.write_point_cloud('gBun10_3.ply', pcd);





