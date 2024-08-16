# %%
import OpenGL.GL as GL
import OpenGL.GLUT as GLUT
import OpenGL.GLU as GLU
import glfw
import numpy as np
import asyncio
import ctypes
import glm
import queue
import threading
from PIL import Image


original_voxel = None
predicted_voxel = None
texture_ids = [None, None]
gl_task_queue = queue.Queue()

# 定義頂點著色器
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;  // 修改为vec2

out vec3 FragPos;
out vec2 TexCoord;  // 输出纹理坐标
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    TexCoord = aTexCoord;  // 传递纹理坐标
    Normal = mat3(transpose(inverse(model))) * vec3(0.0, 0.0, 1.0);  
    gl_Position = projection * view * vec4(FragPos, 1.0);
}

"""

# 定義片段著色器
fragment_shader_source = """
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;
uniform sampler2D texture1;
uniform bool isEdge;
uniform bool isTextured;

void main()
{
    if (isTextured) {
        FragColor = texture(texture1, TexCoord);  // 使用纹理坐标进行纹理采样
    } else if (isEdge) {
        FragColor = vec4(1.0, 1.0, 1.0, 1.0);  // 白色边缘
    } else {
        // 原始的光照计算逻辑
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
        
        vec3 result = (ambient + diffuse + specular) * objectColor;
        FragColor = vec4(result, 1.0);
    }
}

"""

# %%
def update_voxel(predicted, original, texture, texture2):
    global original_voxel, predicted_voxel, gl_task_queue
    original_voxel = original
    predicted_voxel = predicted

    # 處理第一個紋理
    texture = texture.transpose(1, 2, 0)
    texture = np.array(np.uint8(texture * 255))
    image = Image.fromarray(texture, 'RGBA').transpose(Image.FLIP_TOP_BOTTOM)
    img_data = image.tobytes()
    
    # 處理第二個紋理
    texture2 = texture2.transpose(1, 2, 0)
    texture2 = np.array(np.uint8(texture2 * 255))
    texture2 = np.stack([texture2, texture2, texture2, np.ones_like(texture2) * 255], axis=-1)
    image2 = Image.fromarray(texture2, 'RGBA').transpose(Image.FLIP_TOP_BOTTOM)
    img_data2 = image2.tobytes()


    # 將兩個紋理的更新任務添加到隊列
    gl_task_queue.put(lambda: update_texture(img_data, image.width, image.height, 0))
    gl_task_queue.put(lambda: update_texture(img_data2, image2.width, image2.height, 1))
    
def update_texture(img_data, width, height, index):
    global texture_ids

    current_context = glfw.get_current_context()
    if not current_context:
        raise RuntimeError("No current OpenGL context")
    
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_ids[index])
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)


# %%
def load_texture(path):
    global texture_id
    image = Image.open(path)
    print("max: {}, min: {}".format(np.max(np.array(image)), np.min(np.array(image))))
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = image.convert("RGBA").tobytes()
    
    texture_id = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
    
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, image.width, image.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    

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
def setup_quad():
    vertices = np.array([
        # Positions        # Texture Coords
        -1.0, -1.0, 0.0,   0.0, 0.0,
         1.0, -1.0, 0.0,   1.0, 0.0,
         1.0,  1.0, 0.0,   1.0, 1.0,
        -1.0,  1.0, 0.0,   0.0, 1.0,
    ], dtype=np.float32)
    
    indices = np.array([
        0, 1, 2,
        2, 3, 0,
    ], dtype=np.uint32)

    vao = GL.glGenVertexArrays(1)
    vbo = GL.glGenBuffers(1)
    ebo = GL.glGenBuffers(1)
    
    GL.glBindVertexArray(vao)
    
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
    
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
    GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)
    
    # Position attribute
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 5 * 4, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(0)
    
    # Texture coord attribute
    GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
    GL.glEnableVertexAttribArray(1)
    
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)
    
    return vao

def draw_quad(shader_program, vao, texture, window_width, window_height, position):
    GL.glUseProgram(shader_program)
    
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glUniform1i(GL.glGetUniformLocation(shader_program, "texture1"), 0)

    GL.glUniform1i(GL.glGetUniformLocation(shader_program, "isTextured"), True)
    GL.glUniform1i(GL.glGetUniformLocation(shader_program, "isEdge"), False)
    
    quad_width = window_width // 3
    quad_height = window_height // 3
    
    if position == "top_left":
        GL.glViewport(0, window_height - quad_height, quad_width, quad_height)
    elif position == "center":
        GL.glViewport(quad_width, window_height - quad_height, quad_width, quad_height)
    
    ortho = glm.ortho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    projectionLoc = GL.glGetUniformLocation(shader_program, "projection")
    GL.glUniformMatrix4fv(projectionLoc, 1, GL.GL_FALSE, glm.value_ptr(ortho))
    
    model = glm.mat4(1.0)
    model = glm.translate(model, glm.vec3(0, 0, 0.0))
    modelLoc = GL.glGetUniformLocation(shader_program, "model")
    GL.glUniformMatrix4fv(modelLoc, 1, GL.GL_FALSE, glm.value_ptr(model))
    
    view = glm.mat4(1.0)
    viewLoc = GL.glGetUniformLocation(shader_program, "view")
    GL.glUniformMatrix4fv(viewLoc, 1, GL.GL_FALSE, glm.value_ptr(view))
    
    GL.glBindVertexArray(vao)   
    GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)
    GL.glBindVertexArray(0)
    
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glUniform1i(GL.glGetUniformLocation(shader_program, "isTextured"), False)
    
    GL.glViewport(0, 0, window_width, window_height)
        
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
    global gl_task_queue
    window = gl_init()
    if not window:
        return

    shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
    cube_vao, cube_vbo = setup_cube()
    quad_vao = setup_quad()
    
    texture_ids[0] = GL.glGenTextures(1)
    texture_ids[1] = GL.glGenTextures(1)
    
    while not glfw.window_should_close(window):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        # 处理队列中的 OpenGL 任务
        while not gl_task_queue.empty():
            task = gl_task_queue.get()
            task()
        GL.glViewport(0, 0, 800, 600)
        
        view = glm.lookAt(glm.vec3(50.0, 50.0, 50.0), glm.vec3(-5.0, 0.0, 10.0), glm.vec3(0.0, 1.0, 0.0))
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
        # Draw 2D Quad with texture
        window_width, window_height = glfw.get_framebuffer_size(window)
        draw_quad(shader_program, quad_vao, texture_ids[0], window_width, window_height, "top_left")
        draw_quad(shader_program, quad_vao, texture_ids[1], window_width, window_height, "center")
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        check_gl_error()
    
    glfw.terminate()

if(__name__ == "__main__"):
    gl_main()