
polygon_fragment_shader = '''
#version 410
out vec4 daColor;
in vec3 theColor;

void main()
{
    daColor = vec4(theColor, 1.0);
}
'''