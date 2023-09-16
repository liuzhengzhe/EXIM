
pointclouds_fragment_shader = '''
#version 410
out vec4 daColor;
in vec3 theColor;

void main()
{
    daColor = vec4(theColor, 1.0);
    // the circle
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
    if (dot(circCoord, circCoord) > 1.0) {
        discard;
    }
}
'''