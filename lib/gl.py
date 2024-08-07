# %%
import OpenGL.GL as GL
import OpenGL.GLUT as GLUT
import OpenGL.GLU as GLU
import glfw
from lib.binvox import read_binvox
import numpy as np
import asyncio
import ctypes
import glm


original_voxel = None
predicted_voxel = None

# 定義頂點著色器
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;  
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

# 定義片段著色器
fragment_shader_source = """
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;
uniform bool isEdge;  // 新增的 uniform 變量

void main()
{
    if (isEdge) {
        FragColor = vec4(1.0, 1.0, 1.0, 1.0);  // 白色邊緣
    } else {
        // 原有的光照計算
        float ambientStrength = 0.5;
        vec3 ambient = ambientStrength * lightColor;
        
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;
        
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * lightColor;  
        
        float cutOff = cos(radians(12.5));
        float outerCutOff = cos(radians(17.5));
        float epsilon = cutOff - outerCutOff;
        float intensity = clamp((dot(lightDir, vec3(0.0, -1.0, 0.0)) - outerCutOff) / epsilon, 0.0, 1.0);
        
        vec3 result = (ambient + intensity * (diffuse + specular)) * objectColor;
        FragColor = vec4(result, 1.0);
    }
}
"""

# %%
async def update_voxel(predicted, original):
    global original_voxel, predicted_voxel
    original_voxel = original
    predicted_voxel = predicted

# %%
def create_shader_program(vertex_shader_source, fragment_shader_source):
    vertex_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
    GL.glShaderSource(vertex_shader, vertex_shader_source)
    GL.glCompileShader(vertex_shader)
    
    fragment_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
    GL.glShaderSource(fragment_shader, fragment_shader_source)
    GL.glCompileShader(fragment_shader)
    
    shader_program = GL.glCreateProgram()
    GL.glAttachShader(shader_program, vertex_shader)
    GL.glAttachShader(shader_program, fragment_shader)
    GL.glLinkProgram(shader_program)
    
    GL.glDeleteShader(vertex_shader)
    GL.glDeleteShader(fragment_shader)
    
    return shader_program

# %%
def gl_init():
    if not glfw.init():
        print("Failed to initialize GLFW")
        return None
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4)
    window = glfw.create_window(800, 600, "Voxel Viewer", None, None)
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        return None
    glfw.make_context_current(window)
    GL.glEnable(GL.GL_DEPTH_TEST)
    return window

# %%
def check_gl_error():
    error = GL.glGetError()
    if error != GL.GL_NO_ERROR:
        print(f"GL Error: {error}")
# %%
def setup_cube():
    vertices = np.array([
        -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
         0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
         0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
         0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
        -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
        -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,

        -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
         0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
         0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
         0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
        -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
        -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,

        -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,
        -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,
        -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
        -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
        -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,
        -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,

         0.5,  0.5,  0.5,  1.0,  0.0,  0.0,
         0.5,  0.5, -0.5,  1.0,  0.0,  0.0,
         0.5, -0.5, -0.5,  1.0,  0.0,  0.0,
         0.5, -0.5, -0.5,  1.0,  0.0,  0.0,
         0.5, -0.5,  0.5,  1.0,  0.0,  0.0,
         0.5,  0.5,  0.5,  1.0,  0.0,  0.0,

        -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
         0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
         0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
         0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
        -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
        -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,

        -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
         0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
         0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
         0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
        -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
        -0.5,  0.5, -0.5,  0.0,  1.0,  0.0
    ], dtype=np.float32)

    # 添加邊緣線的頂點
    edges = np.array([
        -0.5, -0.5, -0.5,  0.5, -0.5, -0.5,
        -0.5, -0.5,  0.5,  0.5, -0.5,  0.5,
        -0.5,  0.5, -0.5,  0.5,  0.5, -0.5,
        -0.5,  0.5,  0.5,  0.5,  0.5,  0.5,
        
        -0.5, -0.5, -0.5, -0.5,  0.5, -0.5,
         0.5, -0.5, -0.5,  0.5,  0.5, -0.5,
        -0.5, -0.5,  0.5, -0.5,  0.5,  0.5,
         0.5, -0.5,  0.5,  0.5,  0.5,  0.5,
        
        -0.5, -0.5, -0.5, -0.5, -0.5,  0.5,
         0.5, -0.5, -0.5,  0.5, -0.5,  0.5,
        -0.5,  0.5, -0.5, -0.5,  0.5,  0.5,
         0.5,  0.5, -0.5,  0.5,  0.5,  0.5
    ], dtype=np.float32)

    vao = GL.glGenVertexArrays(1)
    vbo = GL.glGenBuffers(2)  # 我們現在需要兩個 VBO

    GL.glBindVertexArray(vao)

    # 綁定和設置立方體頂點數據
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo[0])
    GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)

    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, ctypes.c_void_p(12))
    GL.glEnableVertexAttribArray(1)

    # 綁定和設置邊緣線頂點數據
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo[1])
    GL.glBufferData(GL.GL_ARRAY_BUFFER, edges.nbytes, edges, GL.GL_STATIC_DRAW)

    GL.glVertexAttribPointer(2, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(2)

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)

    return vao, vbo

# %%
def draw_model(shader_program, vao, vbo, voxel, model, color, loc):
    if voxel is None:
        return
    
    GL.glUseProgram(shader_program)
    
    GL.glUniform3f(GL.glGetUniformLocation(shader_program, "objectColor"), *color)
    isEdgeLoc = GL.glGetUniformLocation(shader_program, "isEdge")

    modelLoc = GL.glGetUniformLocation(shader_program, "model")
    for i in range(voxel.shape[0]):
        for j in range(voxel.shape[1]):
            for k in range(voxel.shape[2]):
                if voxel[i, j, k] == True or voxel[i, j, k] >= 0.5:
                    new_model = glm.translate(model, glm.vec3(i + loc[0], j + loc[1], k + loc[2]))
                    GL.glUniformMatrix4fv(modelLoc, 1, GL.GL_FALSE, glm.value_ptr(new_model))
                   
                     # 繪製立方體
                    GL.glUniform1i(isEdgeLoc, 0)
                    GL.glBindVertexArray(vao)
                    GL.glDrawArrays(GL.GL_TRIANGLES, 0, 36)
                    
                    # 繪製邊緣
                    GL.glUniform1i(isEdgeLoc, 1)
                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo[1])
                    GL.glVertexAttribPointer(2, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, ctypes.c_void_p(0))
                    GL.glEnableVertexAttribArray(2)
                    GL.glDrawArrays(GL.GL_LINES, 0, 24)
                    
                    
# %%
def gl_main():
    global original_voxel, predicted_voxel
    window = gl_init()
    if not window:
        return
    
    shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
    cube_vao, cube_vbo = setup_cube()
    
    while not glfw.window_should_close(window):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        GL.glViewport(0, 0, 800, 600)
        
        view = glm.lookAt(glm.vec3(50.0, 50.0, 50.0), glm.vec3(0.0, 0.0, 10.0), glm.vec3(0.0, 1.0, 0.0))
        projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)
        
        GL.glUseProgram(shader_program)
        viewLoc = GL.glGetUniformLocation(shader_program, "view")
        projectionLoc = GL.glGetUniformLocation(shader_program, "projection")
        GL.glUniformMatrix4fv(viewLoc, 1, GL.GL_FALSE, glm.value_ptr(view))
        GL.glUniformMatrix4fv(projectionLoc, 1, GL.GL_FALSE, glm.value_ptr(projection))
        GL.glUniform3f(GL.glGetUniformLocation(shader_program, "viewPos"), 50.0, 50.0, 50.0)
        
        # Draw light
        lightPos = glm.vec3(0.0, 50.0, 0.0)
        GL.glUniform3f(GL.glGetUniformLocation(shader_program, "lightPos"), *lightPos)
        GL.glUniform3f(GL.glGetUniformLocation(shader_program, "lightColor"), 1.0, 1.0, 1.0)
        
        # Draw original voxel
        model = glm.mat4(1.0)
        model = glm.scale(model, glm.vec3(0.6, 0.6, 0.6))   
        draw_model(shader_program, cube_vao, cube_vbo, original_voxel, model, (0.0, 1.0, 0.0), (0, 0, -40))
        
        # Draw predicted voxel
        draw_model(shader_program, cube_vao, cube_vbo, predicted_voxel, model, (0.0, 0.0, 1.0), (0, 0, 40))
        
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        check_gl_error()
    
    glfw.terminate()
