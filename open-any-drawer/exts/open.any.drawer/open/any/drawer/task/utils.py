import omni
from pxr import UsdGeom, Usd


def get_bounding_box(prim_path: str):
    """
    Get the bounding box of a prim
    """
    stage = omni.usd.get_context().get_stage()

    purposes = [UsdGeom.Tokens.default_]
    bboxcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    prim = stage.GetPrimAtPath(prim_path)
    bboxes = bboxcache.ComputeWorldBound(prim)
    # print("bboxes", bboxes)
    game_bboxes = [bboxes.ComputeAlignedRange().GetMin(),bboxes.ComputeAlignedRange().GetMax()]
    
    return game_bboxes

def bboxes_overlap(bboxes1, bboxes2):
    """
    To judge whether two bboxes overlap with each other
    bboxes: [min (vec3), max (vec3)]
    """
    return not ( bboxes1[0][0] > bboxes2[1][0] or  # left
                 bboxes1[1][0] < bboxes2[0][0] or # right
                 bboxes1[0][1] > bboxes2[1][1] or # bottom
                 bboxes1[1][1] < bboxes2[0][1] or # up
                 bboxes1[0][2] > bboxes2[1][2] or # front
                 bboxes1[1][2] < bboxes2[0][2]) # back 
                

def get_mesh_bboxes(self, keyword: str):
    stage = omni.usd.get_context().get_stage()
    prim_list = list(stage.TraverseAll())
    prim_list = [ item for item in prim_list if keyword in item.GetPath().pathString and item.GetTypeName() == 'Mesh' ]

    bboxes_list  = []
    for prim in prim_list:
        bboxes = get_bounding_box(prim.GetPath().pathString)
        bboxes_list.append(bboxes)

    return bboxes_list