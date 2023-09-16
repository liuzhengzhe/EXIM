
pointclouds_balls_fragment_shader = '''
#version 410
out vec4 daColor;
in vec3 theColor;
in vec3 normalWorld;
in vec3 vertexPositionWorld;

uniform vec3 eyePosition;
uniform vec3 pointLightPosition;

vec3 lightDirection = vec3(1, -1, -1);
float diffuseLightIntensity = 0.5;
float ambientLightIntensity = 0.4;
float specularLightIntensity = 0.0;
float c = 1.0;

float suppLightRatio = 0.5;
float constant_a = 0.2;
float constant_b = 0.05;
float constant_c = 0.05;
float pointLightIntensity = 0.5;
vec3 pointLightColor = vec3(1.0, 1.0, 1.0);

vec3 computeDiffuse(vec3 lightInput, vec3 normal) {
    vec3 lightVectorWorld = normalize(-lightInput);
    float brightness = dot(lightVectorWorld, normalize(normal));
	vec3 diffuseLight = clamp(vec3(brightness, brightness, brightness), 0, 1);
	return diffuseLight;
}

vec3 computeSpecular(vec3 lightInput, vec3 normal, vec3 eye, vec3 vertexPosition) {
	vec3 reflectedLightVectorWorld = normalize(reflect(-lightInput, normal));
    vec3 eyeVectorWorld = normalize(eye - vertexPosition);
    
    float s = clamp(dot(reflectedLightVectorWorld, eyeVectorWorld), 0, 1);
	s = pow(s, c);
	vec3 specularLight = vec3(s, s, s);
	return specularLight;
}

vec3 computePointLight(vec3 pointLightPosition, vec3 vertexPositionWorld, vec3 normalWorld, vec3 eyeVectorWorld) {
    vec3 lightDir = normalize(pointLightPosition - vertexPositionWorld); 
    //diffuse 
    float diff = max(dot(normalWorld, lightDir), 0.0);
    // specular shading
	vec3 reflectDir = reflect(-lightDir, normalWorld);
    float spec = pow(clamp(max(dot(eyeVectorWorld, reflectDir), 0.0), 0, 1), c);
	// attenuation
	float distance = length(pointLightPosition - vertexPositionWorld);
	float attenuation = 1.0 / (constant_a + constant_b * distance +
		constant_c * (distance * distance));
	
	// lights
	vec3 diffusePointLight = pointLightColor  * diff * attenuation;
	vec3 specularPointLight = pointLightColor * spec * attenuation;
	
	return diffuseLightIntensity * diffusePointLight + specularLightIntensity * specularPointLight;

}

void main()
{
    // compute the diffuse lighting
	// vec3 diffuseLight = computeDiffuse(lightDirection, normalWorld);
	
	// compute the spectular
	// vec3 specularLight = computeSpecular(lightDirection, normalWorld, eyePosition, vertexPositionWorld);
	
	// compute sub light
	// vec3 diffuseLight_supp = computeDiffuse(-lightDirection, normalWorld);
    //vec3 specularLight_supp = computeSpecular(-lightDirection, normalWorld, eyePosition, vertexPositionWorld);
	
    daColor = vec4 (
        (
         // directionalLightIntensity * diffuseLight
         ambientLightIntensity * pointLightColor
         // + specularLightIntensity * specularLight 
         // + directionalLightIntensity * diffuseLight_supp * suppLightRatio
         // + specularLightIntensity * specularLight_supp * suppLightRatio
         + computePointLight(pointLightPosition, vertexPositionWorld, normalWorld, eyePosition) * pointLightIntensity
         ) * theColor
        , 1.0
    );
}
'''