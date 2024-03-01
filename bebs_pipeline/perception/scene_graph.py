import numpy as np
import scipy
import open3d as o3d
from .clip_utils import *
# from .utils import *
from .point_cloud_utils import *
from scalingup.utils.core import (
    PointCloud,
)
import sys 
sys.path.append("..")
from .object_detect import get_point_cloud_dict
from .state_detect import get_object_state, get_gt_object_state


# =========  Parameters for spatial relation heuristics ============
IN_CONTACT_DISTANCE = 0.1
CLOSE_DISTANCE = 0.4
INSIDE_THRESH = 0.5
ON_TOP_OF_THRESH = 0.7
NORM_THRESH_FRONT_BACK = 0.9
NORM_THRESH_UP_DOWN = 0.9
NORM_THRESH_LEFT_RIGHT = 0.8
OCCLUDE_RATIO_THRESH = 0.5
DEPTH_THRESH = 0.9
RELATION_EXC_OBJ_NAMES = ['drawer handle', 'catapult button']

def get_scene_graph(env):
    pcd_dict = get_point_cloud_dict(env)
    node_list = []

    scene_graph = SceneGraph()
    for name in pcd_dict.keys():
        node = Node(name, pcd=pcd_dict[name])
        node_list.append(node)

    for node in node_list:
        scene_graph.add_node(node, env.obs.images['front'].rgb, env)
    
    return str(scene_graph)

def get_pcd_dist(pts_A, pts_B):
    pcd_A = o3d.geometry.PointCloud()
    pcd_A.points = o3d.utility.Vector3dVector(pts_A)
    pcd_B = o3d.geometry.PointCloud()
    pcd_B.points = o3d.utility.Vector3dVector(pts_B)

    dists = pcd_A.compute_point_cloud_distance(pcd_B)
    try:
        dist = np.min(np.array(dists))
    except:
        dist = np.inf
    return dist

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, scipy.spatial.Delaunay):
        hull = scipy.spatial.Delaunay(hull)

    return hull.find_simplex(p)>=0

def is_inside(src_pts, target_pts, thresh=0.5):
    try:
        hull = scipy.spatial.ConvexHull(target_pts)
    except:
        return False
    # print("vertices of hull: ", np.array(hull.vertices).shape)
    hull_vertices = np.array([[0,0,0]])
    for v in hull.vertices:
        hull_vertices = np.vstack((hull_vertices, np.array([target_pts[v,0], target_pts[v,1], target_pts[v,2]])))
    hull_vertices = hull_vertices[1:]

    num_src_pts = len(src_pts)
    # Don't want threshold to be too large (specially with more objects, like 4, 0.9*thresh becomes too large)
    thresh_obj_particles = thresh * num_src_pts
    src_points_in_hull = in_hull(src_pts, hull_vertices)
    # print("src pts in target, thresh: ", src_points_in_hull.sum(), thresh_obj_particles)
    if src_points_in_hull.sum() > thresh_obj_particles:
        return True
    else:
        return False

def is_in_top_of(src_pos, target_box):
    upper_plane_z = np.max(target_box[:, 2])
    return src_pos[2] > upper_plane_z


class Node(object):
    def __init__(self, name, pcd):
        self.name = name
        self.pcd = torch.tensor(pcd.xyz_pts) # point cloud (px3)
        boxes3d_pts = o3d.utility.Vector3dVector(self.pcd)
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(boxes3d_pts)
        self.bbox = bbox # 3d bounding box
        self.pos = bbox.get_center()
        self.corner_pts = np.array(bbox.get_box_points())
        self.name_w_state = None

    def set_state(self, state):
        self.name_w_state = state

    def __str__(self):
        return self.get_name()

    def __hash__(self):
        return hash(self.get_name())

    def __eq__(self, other):
        return True if self.get_name() == other.get_name() else False

    def get_name(self):
        if self.name_w_state is not None:
            return self.name_w_state
        else:
            return self.name


class Edge(object):
    def __init__(self, start_node, end_node, edge_type="none"):
        self.start = start_node
        self.end = end_node
        self.edge_type = edge_type
    
    def __hash__(self):
        return hash((self.start, self.end, self.edge_type))

    def __eq__(self, other):
        if self.start == other.start and self.end == other.end and self.edge_type == other.edge_type:
            return True
        else:
            return False

    def __str__(self):
        return str(self.start) + " -> " + self.edge_type + " -> " + str(self.end)


class SceneGraph(object):
    """
    Create a spatial scene graph
    """
    def __init__(self):
        self.nodes = []
        self.total_nodes = []
        self.edges = {}

    def add_node(self, new_node, image, env):
        for node in self.nodes:
            self.add_edge(node, new_node)
            self.add_edge(new_node, node)
        self.nodes.append(new_node)
        # self.add_object_state(new_node, image, env, mode="gt")

    def add_edge(self, node, new_node):
        if new_node.name in RELATION_EXC_OBJ_NAMES or node.name in RELATION_EXC_OBJ_NAMES:
            return
        dist = get_pcd_dist(node.pcd, new_node.pcd)
        
        box_A_pts, box_B_pts = np.array(node.pcd), np.array(new_node.pcd)
        box_A, box_B = node.corner_pts, new_node.corner_pts
        pos_A, pos_B = node.pos, new_node.pos

        # IN CONTACT
        if dist < IN_CONTACT_DISTANCE:
            if is_inside(src_pts=box_B_pts, target_pts=box_A_pts, thresh=INSIDE_THRESH):
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "inside")
            elif is_inside(src_pts=box_A_pts, target_pts=box_B_pts, thresh=INSIDE_THRESH):
                self.edges[(node.name, new_node.name)] = Edge(node, new_node, "inside")
            elif is_in_top_of(src_pos=pos_B, target_box=box_A):
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on top of")
            elif is_in_top_of(src_pos=pos_A, target_box=box_B):
                self.edges[(node.name, new_node.name)] = Edge(node, new_node, "on top of")
    
    def add_object_state(self, node, image, env, mode="gt"):
        if mode == "gt":
            state = get_gt_object_state(node.name, env)
        elif mode == "clip":
            state = get_object_state(node.name, image)
        if state is not None:
            node.set_state(f"{node.name} ({state})")
        return node

    def __eq__(self, other):
        if (set(self.nodes) == set(other.nodes)) and (set(self.edges.values()) == set(other.edges.values())):
            return True
        else:
            return False

    def __str__(self):
        visited = []
        res = "  [Nodes]:\n"
        for node in set(self.nodes):
            res += "    "
            res += f'{node.get_name()} -- position: [{float(node.pos[0]):.2f}, {float(node.pos[1]):.2f}, {float(node.pos[2]):.2f}], x_range: [{float(node.bbox.min_bound[0]):.2f}, {float(node.bbox.max_bound[0]):.2f}], y_range: [{float(node.bbox.min_bound[1]):.2f}, {float(node.bbox.max_bound[1]):.2f}], z_range: [{float(node.bbox.min_bound[2]):.2f}, {float(node.bbox.max_bound[2]):.2f}]'
            res += "\n"
        res += "  [Edges]:\n"
        for edge_key, edge in self.edges.items():
            name_1, name_2 = edge_key
            edge_key_reversed = (name_2, name_1)
            if (edge_key not in visited and edge_key_reversed not in visited) or edge.edge_type in ['on top of', 'inside', 'occluding']:
                res += "    "
                res += str(edge)
                res += "\n"
            visited.append(edge_key)
        return res
