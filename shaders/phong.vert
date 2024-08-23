#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec3 FragPos;
out vec2 TexCoord;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    TexCoord = aTexCoord;
    Normal = mat3(transpose(inverse(model))) * vec3(0.0, 0.0, 1.0);  
    gl_Position = projection * view * vec4(FragPos, 1.0);
}