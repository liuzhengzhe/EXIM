
polygon_vertex_shader = '''
#version 410

layout(location = 0) in vec2 position;
out vec3 theColor;
void main(void)
{
    gl_Position = vec4(position, 0.0, 1.0);
    theColor = vec3(0.0, 0.196, 0.125);
}
'''