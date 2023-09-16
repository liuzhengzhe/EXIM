
pointclouds_balls_vertex_shader = '''
#version 410

layout(location = 0) in vec3 position;
layout(location = 1) in int selected;
layout(location = 2) in int segmented;
layout(location = 3) in vec3 vertexNormal;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 selectedColor;
uniform vec3 unselectedColor;
out vec3 theColor;
out vec3 normalWorld;
out vec3 vertexPositionWorld;

vec3 color_table[4] = vec3[] (vec3(1.0, 1.0, 1.0), vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));
void main(void)
{
    vec4 worldPosition = modelViewMatrix * vec4(position, 1.0); 
    gl_Position = projectionMatrix * worldPosition;
    if (segmented < 0) {
        if (selected == 0){
            theColor = unselectedColor / 255.0;
        }
        else {
            theColor = selectedColor / 255.0;
        }
    } else {
        theColor = color_table[segmented];
    }
    normalWorld = vertexNormal;
    vertexPositionWorld = worldPosition.xyz;

}
'''