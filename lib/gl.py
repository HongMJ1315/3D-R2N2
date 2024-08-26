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
import freetype
from lib.gl_matrix import move_camera_up_down, move_camera_left_right
from PIL import Image
from lib.config import GL_FONTS, GL_FONTS_SIZE, GL_VISUALIZE_THRESHOLD

threasold = int(GL_VISUALIZE_THRESHOLD * 10 - 1)

original_voxel = None
predicted_voxel = None
predicted_voxel_array = None
predicted_texture_array = None
texture_index = 0
texture_ids = [None, None]
gl_task_queue = queue.Queue()

font = None
text_texture = None
text_content = ""

color_code_texture = None

key_state = {}

view_pos = glm.vec3(50, 50, 50)
center_pos = glm.vec3(-5.0, 0.0, 10.0)

GRADIENT_COLORS = [
    ((0.0, 0.0, 1.0), 0.1), #0.1
    ((0.0, 0.5, 1.0), 0.1), #0.2
    ((0.0, 1.0, 1.0), 0.1), #0.3
    ((0.0, 1.0, 0.5), 0.1), #0.4
    ((0.0, 1.0, 0.0), 0.1), #0.5
    ((0.5, 1.0, 0.0), 0.1), #0.6
    ((1.0, 1.0, 0.0), 0.1), #0.7
    ((1.0, 0.5, 0.0), 0.1), #0.8
    ((1.0, 0.0, 0.0), 0.1), #0.9
    ((0.0, 0.0, 0.0), 0.0)  #error
]
    
# %%
def update_train_voxel(predicted, original, texture, texture2):
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
    
def update_test_data():
    global gl_task_queue, texture_index, predicted_texture_array, predicted_voxel_array, predicted_voxel
    texture = predicted_texture_array[texture_index].transpose(1, 2, 0)
    texture = texture[:,:,:4]
    predicted_voxel = predicted_voxel_array[texture_index][0]
    texture = np.array(np.uint8(texture * 255))
    image = Image.fromarray(texture, 'RGBA').transpose(Image.FLIP_TOP_BOTTOM)
    img_data = image.tobytes()
    
    gl_task_queue.put(lambda: update_texture(img_data, image.width, image.height, 0))

def update_test_voxel(predicted_array, texture_array):
    global predicted_voxel_array, predicted_texture_array, texture_index
    predicted_voxel_array = predicted_array
    predicted_texture_array = texture_array
    texture_index = 0
    update_text(f"Predicted Voxel: View {texture_index + 1}")
    update_test_data()
    

def update_texture(img_data, width, height, index):
    global texture_ids

    current_context = glfw.get_current_context()
    if not current_context:
        raise RuntimeError("No current OpenGL context")
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_ids[index])
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

def update_text(new_text):
    global text_content, gl_task_queue
    text_content = new_text
# %%
def load_texture(path):
    global texture_id
    img = Image.open(path)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = img.convert("RGBA").tobytes()
    
    texture_id = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
    
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width, img.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
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
    
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
    setup_font(GL_FONTS, GL_FONTS_SIZE)  # 使用適當的字體路徑和大小
    
    for i in range(255):
        # turn i to char
        key_state[chr(i)] = False
    return window

# %%
def check_gl_error():
    error = GL.glGetError()
    if error != GL.GL_NO_ERROR:
        print(f"GL Error: {error}")
        
# %%
def setup_font(font_path, font_size):
    global font
    font = freetype.Face(font_path)
    font.set_char_size(font_size * 45)

def create_text_texture(text):
    global text_texture, font
    
    width, height = 0, 0
    for char in text:
        font.load_char(char)
        width += font.glyph.advance.x >> 6
        height = max(height, font.glyph.bitmap.rows)
    
    img = np.zeros((height, width, 4), dtype=np.ubyte)
    
    x = 0
    for char in text:
        font.load_char(char)
        bitmap = font.glyph.bitmap
        y = height - font.glyph.bitmap_top
        glyph_width = bitmap.width
        glyph_height = bitmap.rows
        
        y = max(0, y)
        
        buffer_array = np.array(bitmap.buffer, dtype=np.ubyte).reshape(glyph_height, glyph_width)
        
        y_end = min(y + glyph_height, height)
        x_end = min(x + glyph_width, width)
        
        copy_height = y_end - y
        copy_width = x_end - x
        
        img[y:y_end, x:x_end, 0] = 255
        img[y:y_end, x:x_end, 1] = 255
        img[y:y_end, x:x_end, 2] = 255
        img[y:y_end, x:x_end, 3] = buffer_array[:copy_height, :copy_width]
        
        x += font.glyph.advance.x >> 6
    
    img = Image.fromarray(img, 'RGBA')
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = img.tobytes()
    
    if text_texture is None:
        text_texture = GL.glGenTextures(1)
    
    # print("image size: {}, {}, max: {}, min: {}".format(img.width, img.height, np.max(np.array(img)), np.min(np.array(img))))
        
    GL.glBindTexture(GL.GL_TEXTURE_2D, text_texture)
    
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width, img.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    
    return width, height

def draw_text(shader_program, quad_vao, window_width, window_height):
    global text_texture, text_content
    
    if text_texture is None:
        create_text_texture(text_content)
    
    text_width, text_height = create_text_texture(text_content)

    GL.glUseProgram(shader_program)
    
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, text_texture)
    GL.glUniform1i(GL.glGetUniformLocation(shader_program, "texture1"), 0)

    GL.glUniform1i(GL.glGetUniformLocation(shader_program, "isTextured"), True)
    GL.glUniform1i(GL.glGetUniformLocation(shader_program, "isEdge"), False)
    
    GL.glViewport(window_width - text_width - 10, 10, text_width, text_height)
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
    
    GL.glBindVertexArray(quad_vao)   
    GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)
    GL.glBindVertexArray(0)
    
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glUniform1i(GL.glGetUniformLocation(shader_program, "isTextured"), False)
    
    GL.glViewport(0, 0, window_width, window_height)
    
def create_color_code_texture():
    global color_code_texture, font, GRADIENT_COLORS, threasold
    
    width, height = 0, 0
    color_code_text_array = []
    color_code_text_color = []
    
    threadholds = threasold + 1
    for i in range(threadholds, 10):
        if i == threadholds: sep = ""
        else: sep = " | "
        
        text = sep + "0." + str(i)

        for char in text:
            font.load_char(char)
            width += font.glyph.advance.x >> 6
            height = max(height, font.glyph.bitmap.rows)
        
        if(sep != ""):
            color_code_text_array.append(sep)
        color_code_text_array.append("0." + str(i))
        color_code_text_color.append(GRADIENT_COLORS[i - 1][0])


    img = np.zeros((height, width, 4), dtype=np.ubyte)
    
    x = 0
    color_index = 0
    for string in color_code_text_array:
        for char in string:
            font.load_char(char)
            bitmap = font.glyph.bitmap
            y = height - font.glyph.bitmap_top
            glyph_width = bitmap.width
            glyph_height = bitmap.rows

            y = max(0, y)

            buffer_array = np.array(bitmap.buffer, dtype=np.ubyte).reshape(glyph_height, glyph_width)

            y_end = min(y + glyph_height, height)
            x_end = min(x + glyph_width, width)

            copy_height = y_end - y
            copy_width = x_end - x
            
            if(color_index >= len(color_code_text_color)):
                color = (0.0, 0.0, 0.0)
            else:
                color = color_code_text_color[color_index]
                if((char == "|") and color_index < len(color_code_text_color)):
                    color_index += 1
                    color = (1, 1, 1)

            img[y:y_end, x:x_end, 0] = color[0] * 255
            img[y:y_end, x:x_end, 1] = color[1] * 255
            img[y:y_end, x:x_end, 2] = color[2] * 255
            img[y:y_end, x:x_end, 3] = buffer_array[:copy_height, :copy_width]

            x += font.glyph.advance.x >> 6
            
            
    img = Image.fromarray(img, 'RGBA')
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = img.tobytes()
    
    if color_code_texture is None:
        color_code_texture = GL.glGenTextures(1)
        
    GL.glBindTexture(GL.GL_TEXTURE_2D, color_code_texture)
    
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width, img.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    
    return width, height

def draw_color_code(shader_program, quad_vao, window_width, window_height):
    global color_code_texture
    
    if color_code_texture is None:
        create_color_code_texture()
    
    text_width, text_height = create_color_code_texture()

    GL.glUseProgram(shader_program)
    
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, color_code_texture)
    GL.glUniform1i(GL.glGetUniformLocation(shader_program, "texture1"), 0)

    GL.glUniform1i(GL.glGetUniformLocation(shader_program, "isTextured"), True)
    GL.glUniform1i(GL.glGetUniformLocation(shader_program, "isEdge"), False)

    GL.glViewport(window_width - text_width - 10, 30, text_width, text_height)
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
    
    GL.glBindVertexArray(quad_vao)   
    GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)
    GL.glBindVertexArray(0)
    
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glUniform1i(GL.glGetUniformLocation(shader_program, "isTextured"), False)
    
    GL.glViewport(0, 0, window_width, window_height)
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
def get_gradient_color(value):
    if(value < threasold):
        return GRADIENT_COLORS[-1]
    index =  threasold
    if(index >= len(GRADIENT_COLORS) or index < 0):
        return GRADIENT_COLORS[-1]
    return GRADIENT_COLORS[index]

def draw_model(shader_program, vao, vbo, voxel, model, color, loc):
    if voxel is None:
        return
    
    GL.glUseProgram(shader_program)
    
    GL.glDepthMask(GL.GL_FALSE)
    
    isEdgeLoc = GL.glGetUniformLocation(shader_program, "isEdge")

    modelLoc = GL.glGetUniformLocation(shader_program, "model")
    for i in range(voxel.shape[0]):
        for j in range(voxel.shape[1]):
            for k in range(voxel.shape[2]):
                if voxel[i, j, k] == True or voxel[i, j, k] >= 0.1:
                    if(voxel[i, j, k] != True):
                        color, alpha = get_gradient_color(voxel[i, j, k] * 10)
                    else:
                        color = (0.0, 1.0, 0.0); alpha = 1.0
                    
                    GL.glUniform4f(GL.glGetUniformLocation(shader_program, "objectColor"), *color, alpha)
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
    
    GL.glDepthMask(GL.GL_TRUE)
# %%
def draw_train(window, shader_program, cube_vao, cube_vbo, quad_vao):
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    # 处理队列中的 OpenGL 任务
    while not gl_task_queue.empty():
        task = gl_task_queue.get()
        task()
    GL.glViewport(0, 0, 800, 600)
    
    view = glm.lookAt(view_pos, center_pos, glm.vec3(0.0, 1.0, 0.0))
    projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)
    
    GL.glUseProgram(shader_program)
    viewLoc = GL.glGetUniformLocation(shader_program, "view")
    projectionLoc = GL.glGetUniformLocation(shader_program, "projection")
    GL.glUniformMatrix4fv(viewLoc, 1, GL.GL_FALSE, glm.value_ptr(view))
    GL.glUniformMatrix4fv(projectionLoc, 1, GL.GL_FALSE, glm.value_ptr(projection))
    GL.glUniform3f(GL.glGetUniformLocation(shader_program, "viewPos"), *view_pos)
    
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
    draw_text(shader_program, quad_vao, window_width, window_height)
    draw_color_code(shader_program, quad_vao, window_width, window_height)
    glfw.swap_buffers(window)
    glfw.poll_events()
    
    check_gl_error()
        
def draw_test(window, shader_program, cube_vao, cube_vbo, quad_vao):
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
    # 处理队列中的 OpenGL 任务
    while not gl_task_queue.empty():
        task = gl_task_queue.get()
        task()
    GL.glViewport(0, 0, 800, 600)
    
    view = glm.lookAt(view_pos, center_pos, glm.vec3(0.0, 1.0, 0.0))
    projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)
    
    GL.glUseProgram(shader_program)
    viewLoc = GL.glGetUniformLocation(shader_program, "view")
    projectionLoc = GL.glGetUniformLocation(shader_program, "projection")
    GL.glUniformMatrix4fv(viewLoc, 1, GL.GL_FALSE, glm.value_ptr(view))
    GL.glUniformMatrix4fv(projectionLoc, 1, GL.GL_FALSE, glm.value_ptr(projection))
    GL.glUniform3f(GL.glGetUniformLocation(shader_program, "viewPos"), *view_pos)
    
    # Draw light
    lightPos = glm.vec3(0.0, 50.0, 0.0)
    GL.glUniform3f(GL.glGetUniformLocation(shader_program, "lightPos"), *lightPos)
    GL.glUniform3f(GL.glGetUniformLocation(shader_program, "lightColor"), 1.0, 1.0, 1.0)
    
    # Draw original voxel
    model = glm.mat4(1.0)
    model = glm.scale(model, glm.vec3(0.6, 0.6, 0.6))   
    if(predicted_texture_array is not None):
        draw_model(shader_program, cube_vao, cube_vbo, predicted_voxel, model, (0.0, 1.0, 0.0), (0, 0, 0))
    
    # Draw 2D Quad with texture
    window_width, window_height = glfw.get_framebuffer_size(window)
    draw_quad(shader_program, quad_vao, texture_ids[0], window_width, window_height, "top_left")
    draw_text(shader_program, quad_vao, window_width, window_height)
    draw_color_code(shader_program, quad_vao, window_width, window_height)
    glfw.swap_buffers(window)
    glfw.poll_events()
    
    check_gl_error()

def key_callback(window, key, scancode, action, mods):
    global texture_index, key_state, threasold
    if key == glfw.KEY_SPACE and action == glfw.PRESS:
        texture_index = (texture_index + 1) % len(predicted_texture_array)
        update_text(f"Predicted Voxel: View {texture_index + 1}")
        update_test_data()
        
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
        
    if(key == glfw.KEY_W and action == glfw.PRESS):
        key_state['w'] = True
    elif(key == glfw.KEY_W and action == glfw.RELEASE):
        key_state['w'] = False
        
    if(key == glfw.KEY_S and action == glfw.PRESS):
        key_state['s'] = True
    elif(key == glfw.KEY_S and action == glfw.RELEASE):
        key_state['s'] = False
        
    if(key == glfw.KEY_A and action == glfw.PRESS):
        key_state['a'] = True
    elif(key == glfw.KEY_A and action == glfw.RELEASE):
        key_state['a'] = False
        
    if(key == glfw.KEY_D and action == glfw.PRESS):
        key_state['d'] = True
    elif(key == glfw.KEY_D and action == glfw.RELEASE):
        key_state['d'] = False
    
    if(key == glfw.KEY_EQUAL and action == glfw.PRESS):
        if(threasold < 8):
            threasold += 1

    if(key == glfw.KEY_MINUS and action == glfw.PRESS):
        if(threasold >= 1):
            threasold -= 1
            
# %%
def gl_main(events):
    global gl_task_queue, text_texture, key_state, view_pos, center_pos, texture_ids, color_code_texture, text_content
    window = gl_init()
    
    glfw.set_key_callback(window, key_callback)
    if not window:
        return

    # read shader source from file and create shader program
    with open("shaders/phong.vert", "r") as f:
        vertex_shader_source = f.read()
    with open("shaders/phong.frag", "r") as f:
        fragment_shader_source = f.read()
    shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
    cube_vao, cube_vbo = setup_cube()
    quad_vao = setup_quad()
    
    texture_ids[0] = GL.glGenTextures(1)
    texture_ids[1] = GL.glGenTextures(1)
    text_texture = GL.glGenTextures(1)
    color_code_texture = GL.glGenTextures(1)
    
    while not glfw.window_should_close(window):
        if(key_state['w']):
            view_pos = move_camera_up_down(center_pos, view_pos, 1)
        if(key_state['s']):
            view_pos = move_camera_up_down(center_pos, view_pos, -1)
        if(key_state['a']):
            view_pos = move_camera_left_right(center_pos, view_pos, -1)
        if(key_state['d']):
            view_pos = move_camera_left_right(center_pos, view_pos, 1)
            
        if(events == 'train'):
            draw_train(window, shader_program, cube_vao, cube_vbo, quad_vao)
        elif(events == 'test'):
            draw_test(window, shader_program, cube_vao, cube_vbo, quad_vao)
            
    glfw.terminate()

if(__name__ == "__main__"):
    gl_main()