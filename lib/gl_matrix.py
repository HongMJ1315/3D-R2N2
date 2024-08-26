import glm
import math

PI = glm.pi()
def quaternion_rotate(axis, theta):
    axis = glm.normalize(axis)
    
    a = math.cos(theta / 2)
    b = -axis.x * math.sin(theta / 2)
    c = -axis.y * math.sin(theta / 2)
    d = -axis.z * math.sin(theta / 2)
    
    aa = a * a; bb = b * b; cc = c * c; dd = d * d
    bc = b * c; ad = a * d; ac = a * c; ab = a * b; bd = b * d; cd = c * d
    
    rot_mat = glm.mat3(
        glm.vec3(aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)),
        glm.vec3(2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)),
        glm.vec3(2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc)
    )
    
    return rot_mat

def quaternion_rotate_vec3(O, P, theta):
    t = theta * PI / 180
    
    OP = P - O
        
    OM = glm.vec3(0, 1, 0)
    
    axis = glm.cross(OP, OM)
    axis = glm.normalize(axis)
    
    R = quaternion_rotate(axis, t)
    
    return R * OP + O

def move_camera_up_down(O, P, theta):
    result = quaternion_rotate_vec3(O, P, theta)
    tresult = result - O
    tv = glm.vec3(0, 1, 0)
    
    cos_theta = glm.dot(tresult, tv) / (glm.length(tresult) * glm.length(tv))
    angle = math.acos(cos_theta) * 180 / PI
    
    if(angle < 1e-5 or angle > 180 - 1e-5):
        return result
    else:
        return result
    
def move_camera_left_right(O, P, theta):
    center_x = O.x; center_z = O.z
    
    tox = P.x - center_x; toz = P.z - center_z
    
    degree = theta * PI / 180
    
    tx = tox * math.cos(degree) - toz * math.sin(degree)
    tz = tox * math.sin(degree) + toz * math.cos(degree)
    
    return glm.vec3(center_x + tx, P.y, center_z + tz)
