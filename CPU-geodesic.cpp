// CPU-geodesic.cpp
// ----------------
// 3-D CPU ray-tracer with optional Schwarzschild null-geodesic integration.
//
// Two rendering modes are supported (toggle with the 'G' key):
//   - Flat mode   (useGeodesics = false): straight-line intersection test against
//                  the Schwarzschild sphere.  Fast, but physically incorrect.
//   - Geodesic mode (useGeodesics = true): full null-geodesic march through 3-D
//                  Schwarzschild spacetime using a 6-dimensional RK4 integrator
//                  (r, θ, φ, dr/dλ, dθ/dλ, dφ/dλ).
//
// Physical quantities are in SI units.

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <chrono>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace glm;
using namespace std;
using Clock = std::chrono::high_resolution_clock;

// -- Global timing / frame-counting state -- //
double lastPrintTime = 0.0;  // wall-clock time of the last FPS print
int    framesCount   = 0;    // frames rendered since the last FPS print
// Physical constants (SI)
double c = 299792458.0;  // speed of light (m/s)
double G = 6.67430e-11;  // gravitational constant (m³ kg⁻¹ s⁻²)
bool useGeodesics = false; // flag: full geodesic marching vs. straight-line test

// Camera – spherical-coordinate orbit camera centred on the black hole.
// The camera never leaves a sphere of radius 'radius' around 'target'.
// Controls:
//   - Left-drag          → orbit (change azimuth / elevation)
//   - Shift + Left-drag  → pan the target point
//   - Scroll             → dolly zoom (change radius)
struct Camera {
    vec3 pos;     // Cartesian world-space position (recomputed from spherical coords)
    vec3 target;  // Look-at point (usually the black hole centre)
    float fovY;   // Vertical field of view in degrees
    float azimuth, elevation, radius; // Spherical coordinates of the camera
    float minRadius = 1e12f, maxRadius = 1e20f; // Zoom clamp limits
    bool dragging = false; // True while a mouse button is held
    bool panning = false;  // True when Shift + drag for panning
    double lastX = 0, lastY = 0; // Last known cursor position

    // Adjustable speed constants
    float orbitSpeed = 0.008f; // radians per pixel for orbit drag
    float panSpeed = 0.001f;   // fraction of radius per pixel for pan drag
    float zoomSpeed = 1.08f;   // zoom factor per scroll tick (>1 = faster)

    Camera() : azimuth(0), elevation(M_PI / 2.0f), radius(6.34194e10), fovY(60.0f) {
        target = vec3(0, 0, 0);
        updateVectors();
    }

    // Recompute Cartesian 'pos' from the spherical (azimuth, elevation, radius).
    void updateVectors() {
        pos.x = target.x + radius * sin(elevation) * cos(azimuth);
        pos.y = target.y + radius * cos(elevation);
        pos.z = target.z + radius * sin(elevation) * sin(azimuth);
    }

    // Handle cursor movement: orbit or pan depending on the current mode.
    void processMouse(GLFWwindow* window, double xpos, double ypos) {
        float dx = float(xpos - lastX), dy = float(ypos - lastY);
        if (dragging && !panning) {
            // Orbit: adjust azimuth and elevation angles
            azimuth   -= dx * orbitSpeed;
            elevation -= dy * orbitSpeed;
            elevation = glm::clamp(elevation, 0.01f, float(M_PI)-0.01f);
        } else if (panning) {
            // Pan: slide the target in the camera's right-up plane
            vec3 forward = normalize(target - pos);
            vec3 right = normalize(cross(forward, vec3(0,1,0)));
            vec3 up = cross(right, forward);
            target += -right * dx * panSpeed * radius + up * dy * panSpeed * radius;
        }
        updateVectors();
        lastX = xpos; lastY = ypos;
    }

    // Handle scroll wheel: exponential zoom so the speed feels linear at all distances.
    void processScroll(double yoffset) {
        if (yoffset < 0)
            radius *= pow(zoomSpeed, -yoffset); // scroll down → zoom out
        else
            radius /= pow(zoomSpeed, yoffset);  // scroll up   → zoom in
        radius = glm::clamp(radius, minRadius, maxRadius);
        updateVectors();
    }

    // GLFW callbacks – these forward events to the Camera stored in the window user pointer.
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
        Camera* cam = (Camera*)glfwGetWindowUserPointer(window);
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS) {
                cam->dragging = true;
                cam->panning = (mods & GLFW_MOD_SHIFT); // Shift held → pan
                double x, y; glfwGetCursorPos(window, &x, &y);
                cam->lastX = x; cam->lastY = y;
            } else if (action == GLFW_RELEASE) {
                cam->dragging = false;
                cam->panning = false;
            }
        }
    }
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
        Camera* cam = (Camera*)glfwGetWindowUserPointer(window);
        cam->processMouse(window, xpos, ypos);
    }
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        Camera* cam = (Camera*)glfwGetWindowUserPointer(window);
        cam->processScroll(yoffset);
    }
};
Camera camera;

struct Ray;
void rk4Step(Ray& ray, double dλ, double rs);

// Engine manages the OpenGL window and the full-screen quad used to display
// the CPU ray-traced result.  Each frame the CPU fills a pixel buffer, uploads
// it as a texture, and draws a full-screen textured quad.
struct Engine {
    // -- Full-screen quad + texture used to display CPU ray-tracing results -- //
    GLFWwindow* window;
    GLuint quadVAO;      // VAO for the two-triangle quad
    GLuint texture;      // GPU texture updated every frame with the pixel buffer
    GLuint shaderProgram;// Minimal passthrough shader: vertex position + UV
    int WIDTH = 800;
    int HEIGHT = 600;
    float width = 100000000000.0f;  // World half-width in metres (unused in 3-D mode)
    float height = 75000000000.0f;  // World half-height in metres (unused in 3-D mode)
    
    Engine() {
        if (!glfwInit()) {
            cerr << "GLFW init failed\n";
            exit(EXIT_FAILURE);
        }
        // Request OpenGL 3.3 Core Profile (required for VAOs / shader programs)
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Black Hole", nullptr, nullptr);
        if (!window) {
            cerr << "Failed to create GLFW window\n";
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
        glfwMakeContextCurrent(window);
        glewExperimental = GL_TRUE;
        GLenum glewErr = glewInit();
        if (glewErr != GLEW_OK) {
            cerr << "Failed to initialize GLEW: "
                << (const char*)glewGetErrorString(glewErr)
                << "\n";
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
        cout << "OpenGL " << glGetString(GL_VERSION) << "\n";
        this->shaderProgram = CreateShaderProgram();

        auto result = QuadVAO();
        this->quadVAO = result[0];
        this->texture = result[1];
    }

    // CreateShaderProgram – builds a minimal passthrough GLSL program.
    // The vertex shader maps NDC positions (-1..1) straight through.
    // The fragment shader samples from the CPU pixel buffer uploaded as a texture.
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

    // QuadVAO – creates a full-screen quad (two triangles) covering NDC [-1,1]×[-1,1]
    // and allocates a GPU texture to hold the ray-traced pixel buffer.
    vector<GLuint> QuadVAO(){
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

        // attribute 0: position (2 floats)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        // attribute 1: UV (2 floats, offset by 2 floats)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        vector<GLuint> VAOtexture = {VAO, texture};
        return VAOtexture;
    }

    // renderScene – uploads the CPU pixel buffer to the GPU texture and
    // draws the full-screen quad to display the ray-traced image.
    void renderScene(const vector<unsigned char>& pixels, int texWidth, int texHeight) {
        // Upload the new pixel buffer (RGB, 1 byte per channel) as a 2-D texture
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texWidth, texHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

        // Clear the framebuffer and draw the textured quad
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);

        GLint textureLocation = glGetUniformLocation(shaderProgram, "screenTexture");
        glUniform1i(textureLocation, 0); // texture unit 0

        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6); // two triangles = one quad

        glfwSwapBuffers(window);
        glfwPollEvents();
    };

    // Toggle between straight-line and full geodesic ray-tracing with the 'G' key.
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (action == GLFW_PRESS) {
            if (key == GLFW_KEY_G) {
                useGeodesics = !useGeodesics;
                cout << "Geodesics: " << (useGeodesics ? "ON\n" : "OFF\n");
            }
        }
    }
};
Engine engine;

// BlackHole stores the event-horizon parameters.
// Intercept() returns true if a given 3-D point is inside the Schwarzschild radius.
struct BlackHole {
    vec3 position;  // world-space centre
    double mass;    // kg
    double radius;  // unused visual radius
    double r_s;     // Schwarzschild radius (event horizon) in metres

    BlackHole(vec3 pos, float m) : position(pos), mass(m) {r_s = 2.0 * G * mass / (c*c);}

    // Returns true when (px, py, pz) is within the event horizon.
    bool Intercept(float px, float py, float pz) const {
        float dx = px - position.x;
        float dy = py - position.y;
        float dz = pz - position.z;
        float dist2 = dx * dx + dy * dy + dz * dz;
        return dist2 < r_s * r_s;
    }
};
// Sagittarius A* – the supermassive black hole at the centre of the Milky Way.
BlackHole SagA(vec3(0.0f, 0.0f, 0.0f), 8.54e36);

// Ray represents a photon in 3-D Schwarzschild spacetime.
// The state vector is (r, θ, φ, dr/dλ, dθ/dλ, dφ/dλ) in spherical coords.
// Cartesian (x, y, z) are kept in sync after each RK4 step.
struct Ray{
    // -- Cartesian coordinates (for intersection tests and Cartesian output) -- //
    double x;  double y;  double z;
    // -- Spherical coordinates (used by the integrator) -- //
    double r;      // radial distance
    double phi;    // azimuthal angle [0, 2π)
    double theta;  // polar angle [0, π]
    double dr;     double dphi;    double dtheta;  // time derivatives of spherical coords
    // -- Conserved quantities -- //
    double E;  // energy per unit mass  (time-translation symmetry)
    double L;  // angular momentum per unit mass (rotational symmetry about z-axis)

    Ray(vec3 pos, vec3 dir) : x(pos.x), y(pos.y), z(pos.z) {
        // Step 1: convert Cartesian position to spherical (r, θ, φ)
        r = sqrt(x*x + y*y + z*z);
        theta = acos(z / r);
        phi = atan2(y, x);

        // Step 2: project Cartesian velocity onto the spherical basis
        double dx = dir.x, dy = dir.y, dz = dir.z;
        dr     = sin(theta)*cos(phi)*dx + sin(theta)*sin(phi)*dy + cos(theta)*dz;
        dtheta = cos(theta)*cos(phi)*dx + cos(theta)*sin(phi)*dy - sin(theta)*dz;
        dtheta /= r;  // normalise by r to get angular velocity
        dphi   = -sin(phi)*dx + cos(phi)*dy;
        dphi  /= (r * sin(theta));  // normalise by r·sin(θ)

        // Step 3: compute Schwarzschild conserved quantities
        L = r * r * sin(theta) * dphi;  // z-component of angular momentum
        double f = 1.0 - SagA.r_s / r;  // lapse function
        // dt/dλ from the null condition g_{μν} u^μ u^ν = 0
        double dt_dλ = sqrt((dr*dr)/f + r*r*dtheta*dtheta + r*r*sin(theta)*sin(theta)*dphi*dphi);
        E = f * dt_dλ;
    }

    // Advance the ray by one RK4 step.  Stops at the event horizon.
    void step(double dλ, double rs) {
        if (r <= rs) return; // photon is captured – stop marching
        rk4Step(*this, dλ, rs);
        // Reconstruct Cartesian coordinates from spherical
        this->x = r * sin(theta) * cos(phi);
        this->y = r * sin(theta) * sin(phi);
        this->z = r * cos(theta);
    }
};

// raytrace – CPU ray-tracer for the 3-D black hole simulation.
//
// For each screen pixel:
//   1. Build a view ray from the camera through the pixel (perspective projection).
//   2. Depending on useGeodesics:
//      a. Flat mode   – analytic sphere-ray intersection against the Schwarzschild sphere.
//      b. Geodesic    – march the null geodesic step-by-step with RK4 until the ray
//                        hits the event horizon (red) or escapes to infinity (black).
// The result is written into a flat RGB pixel buffer (3 bytes per pixel, row-major).
// The loop is parallelised with OpenMP for multi-core CPU utilisation.
void raytrace(vector<unsigned char>& pixels, int W, int H) {
    pixels.resize(W * H * 3);

    // Build an orthonormal camera basis from the camera's position and look-at point.
    vec3 forward = normalize(camera.target - camera.pos);
    vec3 right   = normalize(cross(forward, vec3(0,1,0)));
    vec3 up      = cross(right, forward);
    float aspect = float(W) / float(H);
    float tanHalfFov = tan(radians(camera.fovY) * 0.5f);

    // Parallelise over rows; each row is independent so no data races occur.
    #pragma omp parallel for schedule(dynamic, 4)
    for(int y = 0; y < H; ++y) {
        for(int x = 0; x < W; ++x) {
            // Convert pixel (x, y) to Normalised Device Coordinates in [-1, 1]:
            float u = (2.0f * (x + 0.5f) / float(W)  - 1.0f) * aspect * tanHalfFov;
            float v = (1.0f - 2.0f * (y + 0.5f) / float(H))        * tanHalfFov;
            vec3 dir = normalize(u*right + v*up + forward);

            Ray ray(camera.pos,  dir);

            const int MAX_STEPS = 10000;   // maximum number of RK4 steps per ray
            const double D_LAMBDA = 1e7;   // affine-parameter step size (metres)
            const double ESCAPE_R = 1e14;  // if ray reaches this radius, it escaped

            vec3 color(0.0f); // default colour is black (background)

            if (!useGeodesics) {
                // --- Flat (straight-line) mode ---
                // Solve |origin + t*dir|² = r_s² for t (quadratic equation).
                double b    = 2.0 * dot(camera.pos, dir);
                double c0   = dot(camera.pos, camera.pos) - SagA.r_s*SagA.r_s;
                double disc = b*b - 4.0*c0;
                if (disc > 0.0) {
                    double t1 = (-b - sqrt(disc)) * 0.5;
                    double t2 = (-b + sqrt(disc)) * 0.5;
                    if (t1 > 0.0 || t2 > 0.0)
                        color = vec3(1.0f, 0.0f, 0.0f); // ray intersects event horizon
                }
            }
            else {
                // --- Full null-geodesic march ---
                Ray ray(camera.pos, dir);
                for(int i = 0; i < MAX_STEPS; ++i) {
                    if (SagA.Intercept(ray.x, ray.y, ray.z)) {
                        color = vec3(1.0f, 0.0f, 0.0f); // photon captured
                        break;
                    }
                    ray.step(D_LAMBDA, SagA.r_s);
                    if (ray.r > ESCAPE_R) {
                        // Escaped to infinity – pixel stays black
                        break;
                    }
                }
            }

            // Pack colour into the RGB byte buffer (row-major, 3 bytes per pixel).
            int idx = (y * W + x) * 3;
            pixels[idx+0] = (unsigned char)(color.r * 255);
            pixels[idx+1] = (unsigned char)(color.g * 255);
            pixels[idx+2] = (unsigned char)(color.b * 255);
        }
    }
}

// geodesicRHS – 3-D Schwarzschild null geodesic right-hand side.
//
// The 6-component state is (r, θ, φ, dr/dλ, dθ/dλ, dφ/dλ).
// The equations come from the Christoffel symbols of the Schwarzschild metric:
//
//   d²r/dλ² = -(rs/2r²)·f·(dt/dλ)²  +  (rs/2r²f)·(dr/dλ)²
//             + r·((dθ/dλ)² + sin²θ·(dφ/dλ)²)
//
//   d²θ/dλ² = -(2/r)·(dr/dλ)·(dθ/dλ)  +  sinθ·cosθ·(dφ/dλ)²
//
//   d²φ/dλ² = -(2/r)·(dr/dλ)·(dφ/dλ)  -  (2cosθ/sinθ)·(dθ/dλ)·(dφ/dλ)
void geodesicRHS(const Ray& ray, double rhs[6], double rs) {
    double r      = ray.r;
    double theta  = ray.theta;
    double dr     = ray.dr;
    double dtheta = ray.dtheta;
    double dphi   = ray.dphi;
    double E      = ray.E;

    double f = 1.0 - rs / r;        // Schwarzschild lapse function
    double dt_dlambda = E / f;       // temporal velocity from conserved energy

    // First derivatives (by definition):
    rhs[0] = dr;      // dr/dλ
    rhs[1] = dtheta;  // dθ/dλ
    rhs[2] = dphi;    // dφ/dλ

    // Second derivatives (Christoffel-symbol equations):
    rhs[3] =                                                    // d²r/dλ²
        - (rs / (2 * r * r)) * f * dt_dlambda * dt_dlambda
        + (rs / (2 * r * r * f)) * dr * dr
        + r * (dtheta * dtheta + sin(theta) * sin(theta) * dphi * dphi);

    rhs[4] =                                                    // d²θ/dλ²
        - (2.0 / r) * dr * dtheta
        + sin(theta) * cos(theta) * dphi * dphi;

    rhs[5] =                                                    // d²φ/dλ²
        - (2.0 / r) * dr * dphi
        - 2.0 * cos(theta) / sin(theta) * dtheta * dphi;
}

// addState – helper: out[i] = a[i] + factor * b[i]  for a 6-element state vector.
void addState(const double a[6], const double b[6], double factor, double out[6]) {
    for (int i = 0; i < 6; i++)
        out[i] = a[i] + b[i] * factor;
}

// rk4Step – 4th-order Runge-Kutta integrator for a single affine-parameter step dλ.
// The 6-D state (r, θ, φ, dr, dθ, dφ) is updated in-place.
void rk4Step(Ray& ray, double dλ, double rs) {
    double y0[6] = { ray.r, ray.theta, ray.phi, ray.dr, ray.dtheta, ray.dphi };
    double k1[6], k2[6], k3[6], k4[6], temp[6];

    // k1: slope at the start of the step
    geodesicRHS(ray, k1, rs);
    addState(y0, k1, dλ/2.0, temp);
    Ray r2 = ray;
    r2.r = temp[0]; r2.theta = temp[1]; r2.phi = temp[2];
    r2.dr = temp[3]; r2.dtheta = temp[4]; r2.dphi = temp[5];

    // k2: slope at the midpoint estimated using k1
    geodesicRHS(r2, k2, rs);
    addState(y0, k2, dλ/2.0, temp);
    Ray r3 = ray;
    r3.r = temp[0]; r3.theta = temp[1]; r3.phi = temp[2];
    r3.dr = temp[3]; r3.dtheta = temp[4]; r3.dphi = temp[5];

    // k3: slope at the midpoint estimated using k2
    geodesicRHS(r3, k3, rs);
    addState(y0, k3, dλ, temp);
    Ray r4 = ray;
    r4.r = temp[0]; r4.theta = temp[1]; r4.phi = temp[2];
    r4.dr = temp[3]; r4.dtheta = temp[4]; r4.dphi = temp[5];

    // k4: slope at the end of the step estimated using k3
    geodesicRHS(r4, k4, rs);

    // Weighted average: y₁ = y₀ + (dλ/6)·(k1 + 2k2 + 2k3 + k4)
    ray.r      += (dλ/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
    ray.theta  += (dλ/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
    ray.phi    += (dλ/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]);
    ray.dr     += (dλ/6.0)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3]);
    ray.dtheta += (dλ/6.0)*(k1[4] + 2*k2[4] + 2*k3[4] + k4[4]);
    ray.dphi   += (dλ/6.0)*(k1[5] + 2*k2[5] + 2*k3[5] + k4[5]);
}

// setupCameraCallbacks – registers GLFW input callbacks, forwarding all events
// to the Camera struct via the window user pointer.
void setupCameraCallbacks(GLFWwindow* window) {
    glfwSetWindowUserPointer(window, &camera);
    glfwSetMouseButtonCallback(window, Camera::mouseButtonCallback);
    glfwSetCursorPosCallback(window, Camera::cursorPosCallback);
    glfwSetScrollCallback(window, Camera::scrollCallback);
    glfwSetKeyCallback(window, Engine::keyCallback);
}

// -- MAIN -- //
// Each frame:
//   1. CPU ray-traces the scene into a pixel buffer.
//   2. Uploads the buffer to a GPU texture and displays it with a full-screen quad.
//   3. Counts and prints frames per second every second.
int main() {
    setupCameraCallbacks(engine.window);
    vector<unsigned char> pixels(engine.WIDTH * engine.HEIGHT * 3);

    auto t0 = Clock::now();
    lastPrintTime = std::chrono::duration<double>(t0.time_since_epoch()).count();

    while (!glfwWindowShouldClose(engine.window)) {
        raytrace(pixels, engine.WIDTH, engine.HEIGHT);
        engine.renderScene(pixels, engine.WIDTH, engine.HEIGHT);

        // FPS counting: print the number of frames rendered in the last second.
        framesCount++;
        auto t1 = Clock::now();
        double now = std::chrono::duration<double>(t1.time_since_epoch()).count();
        if (now - lastPrintTime >= 1.0) {
            cout << "FPS: " << framesCount / (now - lastPrintTime) << "\n";
            framesCount   = 0;
            lastPrintTime = now;
        }

    }

    glfwDestroyWindow(engine.window);
    glfwTerminate();
    return 0;
}


















        // 2) FPS counting
        // framesCount++;
        // auto t1 = Clock::now();
        // double now = std::chrono::duration<double>(t1.time_since_epoch()).count();
        // if (now - lastPrintTime >= 1.0) {
        //     cout << "FPS: " << framesCount / (now - lastPrintTime) << "\n";
        //     framesCount   = 0;
        //     lastPrintTime = now;
        // }
        //raytrace(pixels, engine.WIDTH, engine.HEIGHT);
