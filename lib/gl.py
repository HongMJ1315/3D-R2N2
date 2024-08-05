# %%
import OpenGL.GL as GL
import OpenGL.GLUT as GLUT
import OpenGL.GLU as GLU
import glfw
from lib.binvox import read_binvox
import numpy as np
import asyncio


original_voxel = None
predicted_voxel = None

# %%
async def update_voxel(predicted, original):
    global original_voxel, predicted_voxel
    original_voxel = original
    predicted_voxel = predicted

# %%
def gl_init():
    if not glfw.init():
        print("Failed to initialize GLFW")
        return None
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4)
    window = glfw.create_window(800, 600, "Hello World", None, None)
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        return None
    glfw.make_context_current(window)
    GL.glClearColor(0.0, 0.0, 0.0, 1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)  # Add this line  
    return window

# %%
def check_gl_error():
    error = GL.glGetError()
    if error != GL.GL_NO_ERROR:
        print(f"GL Error: {error}")


# %%
def cube(color):
    GL.glColor3f(*color)
    GL.glBegin(GL.GL_QUADS)
    # Front face
    GL.glVertex3f(-0.5, -0.5, +0.5)
    GL.glVertex3f(+0.5, -0.5, +0.5)
    GL.glVertex3f(+0.5, +0.5, +0.5)
    GL.glVertex3f(-0.5, +0.5, +0.5)
    # Back face
    GL.glVertex3f(-0.5, -0.5, -0.5)
    GL.glVertex3f(+0.5, -0.5, -0.5)
    GL.glVertex3f(+0.5, +0.5, -0.5)
    GL.glVertex3f(-0.5, +0.5, -0.5)
    # Left face
    GL.glVertex3f(-0.5, -0.5, -0.5)
    GL.glVertex3f(-0.5, -0.5, +0.5)
    GL.glVertex3f(-0.5, +0.5, +0.5)
    GL.glVertex3f(-0.5, +0.5, -0.5)
    # Right face
    GL.glVertex3f(+0.5, -0.5, -0.5)
    GL.glVertex3f(+0.5, -0.5, +0.5)
    GL.glVertex3f(+0.5, +0.5, +0.5)
    GL.glVertex3f(+0.5, +0.5, -0.5)
    # Top face
    GL.glVertex3f(-0.5, +0.5, -0.5)
    GL.glVertex3f(-0.5, +0.5, +0.5)
    GL.glVertex3f(+0.5, +0.5, +0.5)
    GL.glVertex3f(+0.5, +0.5, -0.5)
    # Bottom face
    GL.glVertex3f(-0.5, -0.5, -0.5)
    GL.glVertex3f(-0.5, -0.5, +0.5)
    GL.glVertex3f(+0.5, -0.5, +0.5)
    GL.glVertex3f(+0.5, -0.5, -0.5)
    GL.glEnd()
    

# %%
def draw_model(voxel, color, loc):
    if(voxel is None):
        return
    for i in range(voxel.shape[0]):
        for j in range(voxel.shape[1]):
            for k in range(voxel.shape[2]):
                GL.glPushMatrix()
                # print(f"Checking voxel at {i, j, k} value {voxel[i, j, k]}")
                if (voxel[i, j, k] == True or voxel[i, j, k] >= 0.5):
                    GL.glTranslatef(i + loc[0], j + loc[1], k + loc[2])
                    # print(f"Drawing cube at {i + loc[0], j + loc[1], k + loc[2]}")
                    cube(color)
                GL.glPopMatrix()

def gl_main():
    global original_voxel, predicted_voxel
    window = gl_init()
    if not window:
        return
    while not glfw.window_should_close(window):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        # 設置投影矩陣
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(45, 800/600, 0.1, 100.0)
        
        # 設置模型視圖矩陣
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GLU.gluLookAt(50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        
        GL.glScalef(0.5, 0.5, 0.5)
        GL.glPushMatrix()
        GL.glTranslatef(0.0, 0.0, -40.0)
        draw_model(original_voxel, (0.0, 1.0, 0.0), (0, 0, 0))
        GL.glPopMatrix()
        GL.glPushMatrix()
        GL.glTranslatef(0.0, 0.0, 40.0)
        draw_model(predicted_voxel, (0.0, 0.0, 1.0), (0, 0, 0))
        GL.glPopMatrix()
                
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        check_gl_error()  # Check for errors at the end of each frame
    glfw.terminate()
    


