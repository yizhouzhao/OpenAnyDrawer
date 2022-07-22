from pxr import Gf

def calcuate_rotation_axis(q, axis = 2, direction = 0):
    """
    Calculate quaternion axis (x,y,z) project on direction (x,y,z)
    q: [x,y,z,w]
    """ 
    mat = Gf.Matrix4f().SetRotate(Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
    return mat.GetRow(axis)[direction]