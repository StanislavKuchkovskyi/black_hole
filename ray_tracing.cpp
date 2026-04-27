// ray_tracing.cpp
// ---------------
// Basic CPU ray-tracer demonstrating sphere intersection and diffuse/shadow shading.
// This is the prototype renderer used before gravitational lensing was introduced.
//
// Pipeline:
//   1. Engine    – creates a window, compiles shaders, creates a full-screen quad.
//   2. Scene     – holds a list of spherical Objects and a point light.
//   3. Scene::trace() – for each ray, finds the nearest sphere, computes diffuse
//                        lighting, and checks for shadows.
//   4. main loop – fires one ray per pixel, collects colours, uploads to GPU texture.

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#include <cmath>
using namespace glm;

// Window dimensions (pixels)
const int WIDTH = 800;
const int HEIGHT = 600;
// Engine – owns the OpenGL window, a minimal passthrough shader, and the
// full-screen quad used to display the CPU-rendered pixel buffer each frame.
class Engine{
public:
    // GPU resources
    GLFWwindow* window;
    GLuint quadVAO;        // vertex array for the two-triangle quad
    GLuint texture;        // 2-D texture updated each frame with CPU pixels
    GLuint shaderProgram;  // passthrough vertex + texture-sample fragment shader

    Engine(){
        this->window = StartGLFW();
        this->shaderProgram = CreateShaderProgram();
        
        auto result = QuadVAO();
        this->quadVAO = result[0];
        this->texture = result[1];
    }

    // Initialise GLFW and GLEW; return the created window handle.
    GLFWwindow* StartGLFW(){
        if(!glfwInit()){
            std::cerr<<"glfw failed init, PANIC PANIC!"<<std::endl;
            return nullptr;
        }

        GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "ray tracer", NULL, NULL);
        glfwMakeContextCurrent(window);
        
        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) {
            std::cerr << "Failed to initialize GLEW." << std::endl;
            glfwTerminate();
            return nullptr;
        }

        glViewport(0, 0, WIDTH, HEIGHT);
        return window;
    };

    // Build a minimal GLSL program:
    //   Vertex shader   – passes NDC position and UV straight through.
    //   Fragment shader – samples the screen texture at the interpolated UV.
    GLuint CreateShaderProgram(){
        const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec2 aPos;  // Changed to vec2
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);  // Explicit z=0
            TexCoord = aTexCoord;
        })";

        const char* fragmentShaderSource = R"(
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D screenTexture;
        void main() {
            FragColor = texture(screenTexture, TexCoord);
        })";

        // vertex shader
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
        glCompileShader(vertexShader);

        // fragment shader
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
        glCompileShader(fragmentShader);

        GLuint shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        return shaderProgram;
    };

    // QuadVAO – allocates a full-screen quad (two triangles) and a GPU texture.
    // Vertex layout: [x, y, u, v]  (position in NDC, UV for texture sampling).
    std::vector<GLuint> QuadVAO(){
        float quadVertices[] = {
            // positions   // texCoords
            -1.0f,  1.0f,  0.0f, 1.0f,  // top left
            -1.0f, -1.0f,  0.0f, 0.0f,  // bottom left
            1.0f, -1.0f,  1.0f, 0.0f,  // bottom right

            -1.0f,  1.0f,  0.0f, 1.0f,  // top left
            1.0f, -1.0f,  1.0f, 0.0f,  // bottom right
            1.0f,  1.0f,  1.0f, 1.0f   // top right
        };
        
        GLuint VAO, VBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

        // attribute 0: 2-D position (NDC)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        // attribute 1: texture UV coordinates
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        std::vector<GLuint> VAOtexture = {VAO, texture};
        return VAOtexture;
    }

    // renderScene – uploads the CPU pixel buffer to the GPU texture and
    // presents the textured quad as the full-screen image.
    void renderScene(std::vector<unsigned char> pixels) {
        // Upload the RGB pixel buffer (1 byte per channel) to the bound texture.
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, 
                    GL_UNSIGNED_BYTE, pixels.data());

        // Draw the full-screen quad textured with the uploaded image.
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);

        GLint textureLocation = glGetUniformLocation(shaderProgram, "screenTexture");
        glUniform1i(textureLocation, 0); // bind to texture unit 0

        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6); // 6 vertices = 2 triangles = 1 quad

        glfwSwapBuffers(window);
        glfwPollEvents();
    };
};

// Ray – a geometric ray with an origin and a normalised direction.
struct Ray{
    vec3 direction; // unit vector in world space
    vec3 origin;    // world-space starting point
    Ray(vec3 o, vec3 d) : origin(o), direction(normalize(d)){}
};

// Material – surface properties of a sphere.
struct Material{
    vec3 color;      // base albedo (RGB in [0, 1])
    float specular;  // specular coefficient (unused in the current trace)
    float emission;  // emissive intensity (unused in the current trace)
    Material(vec3 c, float s, float e) : color(c), specular(s), emission(e) {}
};

// Object – a sphere in the scene with an associated material.
struct Object{
    vec3 centre;      // world-space centre
    float radius;     // radius in world units
    Material material;

    Object(vec3 c, float r, Material m) : centre(c), radius(r), material(m) {}

    // Intersect – analytic ray-sphere intersection using the quadratic formula.
    // Solves |origin + t·direction - centre|² = radius²  for the smallest positive t.
    // Returns false if there is no intersection in front of the ray origin.
    bool Intersect(Ray &ray, float &t){
        vec3 oc = ray.origin - centre;  // vector from centre to ray origin
        float a = glm::dot(ray.direction, ray.direction); // should be 1 (normalised dir)
        float b = 2.0f * glm::dot(oc, ray.direction);
        float c = glm::dot(oc, oc) - radius * radius;
        double discriminant = b*b - 4*a*c;
        if(discriminant < 0){return false;} // no real solution → no intersection

        // Take the smaller root (nearer intersection point)
        float intercept = (-b - sqrt(discriminant)) / (2.0f*a);
        if(intercept < 0){
            intercept = (-b + sqrt(discriminant)) / (2.0f*a); // try the farther root
            if(intercept<0){return false;} // both intersections are behind the origin
        }
        t = intercept;
        return true;
    };

    // Compute the outward surface normal at a point on the sphere.
    vec3 getNormal(vec3 &point) const{
        return normalize(point - centre);
    }
};

// Scene – owns all objects and the point light, and implements the trace function.
class Scene {
public:
    std::vector<Object> objs;
    vec3 lightPos; // world-space position of the point light

    Scene() : lightPos(5.0f, 5.0f, 5.0f) {}

    // trace – cast a single ray and return the colour of the nearest hit surface.
    //
    // Algorithm:
    //   1. Find the nearest Object that the ray intersects (smallest positive t).
    //   2. Compute the hit point and surface normal.
    //   3. Cast a shadow ray from the hit point toward the light.  If blocked,
    //      return ambient light only; otherwise blend ambient + diffuse.
    //   4. Return a dark blue background colour if the ray misses all objects.
    vec3 trace(Ray &ray){
        float closest = INFINITY;
        const Object* hitObj = nullptr;

        // Find the nearest sphere hit by the ray.
        for(auto& obj : objs){
            float t;
            if(obj.Intersect(ray, t)){
                if(t < closest) {
                    closest = t;
                    hitObj = &obj;
                }
            }
        };

        if(hitObj){
            vec3 hitPoint = ray.origin + ray.direction * closest; // 3-D intersection point
            vec3 normal   = hitObj->getNormal(hitPoint);          // outward surface normal
            vec3 lightDir = normalize(lightPos - hitPoint);       // direction toward the light

            // Lambertian diffuse: intensity ∝ cos(angle between normal and light direction)
            float diff = std::max(glm::dot(normal, lightDir), 0.0f);

            // Offset the shadow ray origin slightly along the normal to avoid self-intersection.
            Ray shadowRay(hitPoint + normal * 0.001f, lightDir);
            bool inShadow = false;
            
            // Test every object: if any blocks the shadow ray, this point is in shadow.
            for(auto& obj : objs) {
                float t;
                if(obj.Intersect(shadowRay, t)) {
                    inShadow = true;
                    break;
                }
            }

            vec3 color  = hitObj->material.color;
            float ambient = 0.1f; // minimum (ambient) light level

            if (inShadow) {
                return color * ambient;                     // shadowed: ambient only
            }

            return color * (ambient + diff * 0.9f);         // lit: ambient + diffuse
        }

        return vec3(0.0f, 0.0f, 0.1f); // miss – return a dark blue background
    }
};


// --- main loop ---- //
// Sets up the scene, then each frame:
//   1. Iterates over every pixel and fires a view ray.
//   2. Converts pixel (x, y) → NDC direction with perspective via aspect ratio.
//   3. Calls scene.trace() and stores the colour in the RGB pixel buffer.
//   4. Uploads the pixel buffer to the GPU and presents it on screen.
int main(){
    Engine engine;
    Scene scene;

    // Two coloured spheres placed in front of the camera (camera is at origin, looking -z).
    scene.objs = {
        Object(vec3(0.0f, 0.0f, -5.0f), 2.0f, Material(vec3(1.0f, 0.2f, 0.2f), 0.5f, 0.0f)),
        Object(vec3(3.0f, 0.0f, -7.0f), 1.5f, Material(vec3(0.2f, 1.0f, 0.2f), 0.5f, 0.0f))
    };

    // Pre-allocate the pixel buffer (RGB, 3 bytes per pixel).
    std::vector<unsigned char> pixels(WIDTH * HEIGHT * 3);
    while(!glfwWindowShouldClose(engine.window)){
        glClear(GL_COLOR_BUFFER_BIT);

        // Trace one ray per pixel.
        for(int y = 0; y < HEIGHT; ++y){
            for(int x = 0; x < WIDTH; ++x){
                float aspectRatio = float(WIDTH) / float(HEIGHT);
                float u = float(x) / float(WIDTH);
                float v = float(y) / float(HEIGHT);

                // Convert pixel coordinates to a view direction in camera space.
                // The camera looks along -Z; X is right, Y is up (flipped for screen).
                vec3 direction(
                    (2.0f * u - 1.0f) * aspectRatio, // remap [0,W] → [-aspect, +aspect]
                    -(2.0f * v - 1.0f),               // flip Y: top of screen = +Y
                    -1.0f                              // camera points toward -Z
                );
                Ray ray(vec3(0.0f, 0.0f, 0.0f), normalize(direction));
                vec3 color = scene.trace(ray);

                // Pack the floating-point colour into the RGB byte buffer.
                int index = (y * WIDTH + x) * 3;
                pixels[index + 0] = static_cast<unsigned char>(color.r * 255);
                pixels[index + 1] = static_cast<unsigned char>(color.g * 255);
                pixels[index + 2] = static_cast<unsigned char>(color.b * 255);
            }
        }
        
        engine.renderScene(pixels);
    }

    glfwTerminate();
}


// func dec's





