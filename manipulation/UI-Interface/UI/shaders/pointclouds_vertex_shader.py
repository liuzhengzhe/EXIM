
pointclouds_vertex_shader = '''
#version 410

layout(location = 0) in vec3 position;
layout(location = 1) in int selected;
layout(location = 2) in int segmented;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
out vec3 theColor;

vec3 color_table[4] = vec3[] (vec3(1.0, 1.0, 1.0), vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));
void main(void)
{
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = 5;
    if (segmented < 0) {
        if (selected == 0){
            theColor = vec3(1.0, 0.0, 0.0);
        }
        else {
            theColor = vec3(0.0, 0.0, 1.0);
        }
    } else {
        theColor = color_table[segmented];
    }

}
'''