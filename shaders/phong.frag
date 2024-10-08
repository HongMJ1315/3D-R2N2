#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec4 objectColor; 
uniform sampler2D texture1;
uniform bool isEdge;
uniform bool isTextured;

void main()
{
    if (isTextured) {
        FragColor = texture(texture1, TexCoord);
    } else if (isEdge) {
        FragColor = vec4(1.0, 1.0, 1.0, objectColor.a); 
    } else {
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
        
        vec3 result = (ambient + diffuse + specular) * objectColor.rgb;
        FragColor = vec4(result, objectColor.a);  // 使用 objectColor 的 alpha 值
    }
}
