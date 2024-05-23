import open3d as o3d

txt_path = '/home/chris/Code/PointClouds/data/FLIPscans/MortenPlateTop.txt'
#output_path = txt_path.split('.')[0]+'.ply'
output_path = '/home/chris/Code/PointClouds/data/ply/MortenPlateTop.ply'

print('Converting txt to ply...')
pcd = o3d.io.read_point_cloud(txt_path, format="xyz")

print(f'Saving in: {output_path}')
o3d.io.write_point_cloud(output_path, pcd)

print('Done')