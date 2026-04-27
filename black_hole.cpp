// black_hole.cpp
// --------------
// Main 3-D black-hole simulation: GPU compute shader (geodesic.comp) ray-traces
// the Schwarzschild spacetime while the CPU renders an embedded spacetime grid and
// simulates Newtonian gravity between scene objects.
//
// Architecture:
//   - Camera        – spherical-orbit camera that always points at the black hole.
//   - BlackHole     – Schwarzschild parameters (mass, r_s, event-horizon test).
//   - ObjectData    – N-body objects (yellow/red spheres + the black hole itself).
//   - Engine        – OpenGL 4.3 window, compute program, UBOs, grid mesh, display quad.
//   - main()        – per-frame: N-body gravity update → grid rebuild → GPU ray-trace
//                     → composite compute result + grid overlay → present.
//
// Key controls:
//   Left-drag       → orbit camera
//   Scroll          → zoom camera
//   Right-click     → toggle Newtonian gravity simulation
//   G key           → toggle gravity (same as right-click)

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
#include <fstream>
#include <sstream>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace glm;
using namespace std;
using Clock = std::chrono::high_resolution_clock;

// -- Global state -- //
double lastPrintTime = 0.0; // wall-clock time of the last FPS print
int    framesCount   = 0;   // frames since the last FPS print
double c = 299792458.0;     // speed of light (m/s)
double G = 6.67430e-11;     // gravitational constant (m³ kg⁻¹ s⁻²)
struct Ray;
bool Gravity = false; // true = Newtonian gravity between objects is active

// Camera – spherical-orbit camera fixed on the black hole at the origin.
// Panning is intentionally disabled to keep the black hole centred at all times.
// The 'moving' flag is forwarded to the compute shader (reserved for future LOD).
struct Camera {
    vec3 target = vec3(0.0f, 0.0f, 0.0f); // always look at the black hole centre
    float radius = 6.34194e10f;            // distance from the black hole (metres)
    float minRadius = 1e10f, maxRadius = 1e12f; // zoom clamp limits

    float azimuth   = 0.0f;               // horizontal orbit angle (radians)
    float elevation = M_PI / 2.0f;        // vertical orbit angle (radians, [0, π])

    float orbitSpeed = 0.01f;   // radians per pixel for orbit drag
    float panSpeed   = 0.01f;   // (unused – panning is disabled)
    double zoomSpeed = 25e9f;   // metres per scroll tick

    bool dragging = false; // true while a mouse button is held
    bool panning  = false; // always false (panning disabled)
    bool moving   = false; // true while the user is interacting
    double lastX = 0.0, lastY = 0.0;

    // position() – compute the Cartesian camera position from spherical coordinates.
    vec3 position() const {
        float clampedElevation = glm::clamp(elevation, 0.01f, float(M_PI) - 0.01f);
        return vec3(
            radius * sin(clampedElevation) * cos(azimuth),
            radius * cos(clampedElevation),
            radius * sin(clampedElevation) * sin(azimuth)
        );
    }

    // update() – keep the target pinned to origin and refresh the 'moving' flag.
    void update() {
        target = vec3(0.0f, 0.0f, 0.0f);
        moving = dragging | panning;
    }

    // processMouseMove – orbit the camera when the user drags with the left button.
    void processMouseMove(double x, double y) {
        float dx = float(x - lastX);
        float dy = float(y - lastY);

        if (dragging && panning) {
            // Panning is disabled to keep camera locked on the black hole.
        }
        else if (dragging && !panning) {
            azimuth   += dx * orbitSpeed;
            elevation -= dy * orbitSpeed;
            elevation = glm::clamp(elevation, 0.01f, float(M_PI) - 0.01f);
        }

        lastX = x;
        lastY = y;
        update();
    }

    // processMouseButton – left/middle drag → orbit; right-click → toggle gravity.
    void processMouseButton(int button, int action, int mods, GLFWwindow* win) {
        if (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_MIDDLE) {
            if (action == GLFW_PRESS) {
                dragging = true;
                panning  = false; // panning intentionally disabled
                glfwGetCursorPos(win, &lastX, &lastY);
            } else if (action == GLFW_RELEASE) {
                dragging = false;
                panning  = false;
            }
        }
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            // Hold right mouse button to enable Newtonian gravity simulation.
            Gravity = (action == GLFW_PRESS);
        }
    }

    // processScroll – linear zoom (subtract metres per tick).
    void processScroll(double xoffset, double yoffset) {
        radius -= yoffset * zoomSpeed;
        radius = glm::clamp(radius, minRadius, maxRadius);
        update();
    }

    // processKey – 'G' toggles the Newtonian gravity simulation.
    void processKey(int key, int scancode, int action, int mods) {
        if (action == GLFW_PRESS && key == GLFW_KEY_G) {
            Gravity = !Gravity;
            cout << "[INFO] Gravity turned " << (Gravity ? "ON" : "OFF") << endl;
        }
    }
};
Camera camera;

// BlackHole – stores the event-horizon parameters of Sagittarius A*.
// Intercept() returns true when a point is inside the Schwarzschild radius.
struct BlackHole {
    vec3 position;  // world-space centre (always origin)
    double mass;    // mass in kg
    double radius;  // unused visual radius
    double r_s;     // Schwarzschild radius (event horizon) in metres

    BlackHole(vec3 pos, float m) : position(pos), mass(m) {r_s = 2.0 * G * mass / (c*c);}

    bool Intercept(float px, float py, float pz) const {
        double dx = double(px) - double(position.x);
        double dy = double(py) - double(position.y);
        double dz = double(pz) - double(position.z);
        double dist2 = dx * dx + dy * dy + dz * dz;
        return dist2 < r_s * r_s;
    }
};
// Sagittarius A* – mass ≈ 4.15 million solar masses (8.54 × 10^36 kg)
BlackHole SagA(vec3(0.0f, 0.0f, 0.0f), 8.54e36);
struct ObjectData {
// ObjectData – represents a scene sphere that can interact gravitationally.
// posRadius.xyz is the world-space centre; posRadius.w is the radius.
// The velocity field is used by the Newtonian gravity integrator in main().
struct ObjectData {
    vec4 posRadius; // xyz = centre (metres), w = radius (metres)
    vec4 color;     // rgba colour (sent to the compute shader via objectsUBO)
    float  mass;    // kg (used for gravity calculation)
    vec3 velocity = vec3(0.0f, 0.0f, 0.0f); // current velocity vector (m per frame)
};

// Initial set of scene objects:
//   • Two solar-mass spheres placed 400 Gm from the black hole along X and Z.
//   • The black hole itself (represented as a sphere of radius r_s, black colour).
vector<ObjectData> objects = {
    { vec4(4e11f, 0.0f, 0.0f, 4e10f)   , vec4(1,1,0,1), 1.98892e30 }, // yellow sphere along +X
    { vec4(0.0f, 0.0f, 4e11f, 4e10f)   , vec4(1,0,0,1), 1.98892e30 }, // red sphere along +Z
    { vec4(0.0f, 0.0f, 0.0f, SagA.r_s) , vec4(0,0,0,1), static_cast<float>(SagA.mass)  }, // black hole
};

// Engine – manages all GPU resources:
//   • GLFW window + GLEW initialisation
//   • Compute shader program (geodesic.comp) + three UBOs (camera, disk, objects)
//   • Passthrough shader + full-screen quad for displaying the compute result
//   • Grid shader + mesh for the spacetime-curvature overlay
struct Engine {
    GLuint gridShaderProgram; // grid.vert + grid.frag
    // -- Full-screen quad + texture for displaying the compute shader output -- //
    GLFWwindow* window;
    GLuint quadVAO;          // two-triangle quad covering NDC [-1,1]×[-1,1]
    GLuint texture;          // RGBA8 texture written by the compute shader
    GLuint shaderProgram;    // passthrough vertex + texture-sample fragment shader
    GLuint computeProgram = 0; // geodesic.comp compute program
    // -- Uniform Buffer Objects (std140 layout, matching geodesic.comp bindings) -- //
    GLuint cameraUBO  = 0;   // binding 1 – camera position and basis vectors
    GLuint diskUBO    = 0;   // binding 2 – accretion disk radii
    GLuint objectsUBO = 0;   // binding 3 – scene sphere list
    // -- Spacetime-curvature grid mesh -- //
    GLuint gridVAO = 0;
    GLuint gridVBO = 0;      // vertex positions (warped on CPU each frame)
    GLuint gridEBO = 0;      // index buffer for GL_LINES rendering
    int gridIndexCount = 0;

    int WIDTH  = 800;         // display window width  (pixels)
    int HEIGHT = 600;         // display window height (pixels)
    int COMPUTE_WIDTH  = 200; // compute shader output width  (lower res for performance)
    int COMPUTE_HEIGHT = 150; // compute shader output height
    float width  = 100000000000.0f; // world half-width  in metres (unused in 3-D mode)
    float height =  75000000000.0f; // world half-height in metres (unused in 3-D mode)
    
    Engine() {
        if (!glfwInit()) {
            cerr << "GLFW init failed\n";
            exit(EXIT_FAILURE);
        }
        // Request OpenGL 4.3 Core Profile (required for compute shaders and image units)
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
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

        // Build the three shader programs
        this->shaderProgram   = CreateShaderProgram();                      // display quad
        gridShaderProgram     = CreateShaderProgram("grid.vert", "grid.frag"); // grid overlay
        computeProgram        = CreateComputeProgram("geodesic.comp");       // ray-tracer

        // Allocate the Camera UBO (binding = 1, ~128 bytes for the Camera struct)
        glGenBuffers(1, &cameraUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
        glBufferData(GL_UNIFORM_BUFFER, 128, nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, cameraUBO);

        // Allocate the Disk UBO (binding = 2, 4 floats: r1, r2, num, thickness)
        glGenBuffers(1, &diskUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, diskUBO);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, diskUBO);

        // Allocate the Objects UBO (binding = 3, space for 16 spheres + count)
        glGenBuffers(1, &objectsUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, objectsUBO);
        GLsizeiptr objUBOSize = sizeof(int) + 3 * sizeof(float)   // numObjects + padding
            + 16 * (sizeof(vec4) + sizeof(vec4))                   // posRadius + color
            + 16 * sizeof(float);                                   // mass[16]
        glBufferData(GL_UNIFORM_BUFFER, objUBOSize, nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 3, objectsUBO);

        auto result = QuadVAO();
        this->quadVAO = result[0];
        this->texture = result[1];
    }
    // generateGrid – builds the spacetime-curvature grid mesh on the CPU each frame.
    //
    // The grid is a 25×25 flat lattice in the x-z plane.  Each vertex's y-coordinate
    // is warped using the Schwarzschild embedding function:
    //   Δy = 2·√(r_s·(dist - r_s))
    // This reproduces the classic "rubber sheet" funnel that visualises how mass
    // curves spacetime.  Contributions from all objects are summed at each vertex.
    void generateGrid(const vector<ObjectData>& objects) {
        const int gridSize = 25;
        const float spacing = 1e10f;  // distance between grid lines (10 Gm)

        vector<vec3> vertices;
        vector<GLuint> indices;

        for (int z = 0; z <= gridSize; ++z) {
            for (int x = 0; x <= gridSize; ++x) {
                float worldX = (x - gridSize / 2) * spacing;
                float worldZ = (z - gridSize / 2) * spacing;

                float y = 0.0f; // will be warped downward by gravity

                // Accumulate Schwarzschild embedding displacement from every object.
                for (const auto& obj : objects) {
                    vec3 objPos = vec3(obj.posRadius);
                    double mass = obj.mass;

                    double r_s = 2.0 * G * mass / (c * c); // Schwarzschild radius
                    double dx = worldX - objPos.x;
                    double dz = worldZ - objPos.z;
                    double dist = sqrt(dx * dx + dz * dz); // 2-D radial distance from object

                    if (dist > r_s) {
                        // Embedding formula: Δy = 2√(r_s · (dist - r_s))
                        double deltaY = 2.0 * sqrt(r_s * (dist - r_s));
                        y += static_cast<float>(deltaY) - 3e10f; // offset so far field is flat
                    } else {
                        // Inside the Schwarzschild sphere: pin to a deep constant pit.
                        y += 2.0f * static_cast<float>(sqrt(r_s * r_s)) - 3e10f;
                    }
                }

                vertices.emplace_back(worldX, y, worldZ);
            }
        }

        // Build line indices: each interior vertex connects right (+x) and down (+z).
        for (int z = 0; z < gridSize; ++z) {
            for (int x = 0; x < gridSize; ++x) {
                int i = z * (gridSize + 1) + x;
                indices.push_back(i);
                indices.push_back(i + 1);          // horizontal line

                indices.push_back(i);
                indices.push_back(i + gridSize + 1); // vertical line
            }
        }

        // Upload vertex and index data to GPU (re-use the same VAO/VBO/EBO each frame).
        if (gridVAO == 0) glGenVertexArrays(1, &gridVAO);
        if (gridVBO == 0) glGenBuffers(1, &gridVBO);
        if (gridEBO == 0) glGenBuffers(1, &gridEBO);

        glBindVertexArray(gridVAO);

        glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec3), vertices.data(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gridEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0); // location = 0 matches grid.vert
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), (void*)0);

        gridIndexCount = indices.size();
        glBindVertexArray(0);
    }

    // drawGrid – renders the warped spacetime grid as semi-transparent grey lines.
    // Depth testing is disabled so the grid always appears on top of the ray-traced image.
    void drawGrid(const mat4& viewProj) {
        glUseProgram(gridShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(gridShaderProgram, "viewProj"),
                        1, GL_FALSE, glm::value_ptr(viewProj));
        glBindVertexArray(gridVAO);

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glDrawElements(GL_LINES, gridIndexCount, GL_UNSIGNED_INT, 0);

        glBindVertexArray(0);
        glEnable(GL_DEPTH_TEST);
    }

    // drawFullScreenQuad – blits the compute shader texture to the screen.
    // Depth testing is disabled so the ray-traced image acts as a full-screen background.
    void drawFullScreenQuad() {
        glUseProgram(shaderProgram);
        glBindVertexArray(quadVAO);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(glGetUniformLocation(shaderProgram, "screenTexture"), 0);

        glDisable(GL_DEPTH_TEST);      // draw behind everything else
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 6); // 2 triangles = 1 quad
        glEnable(GL_DEPTH_TEST);
    }
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
    GLuint CreateShaderProgram(const char* vertPath, const char* fragPath) {
        auto loadShader = [](const char* path, GLenum type) -> GLuint {
            std::ifstream in(path);
            if (!in.is_open()) {
                std::cerr << "Failed to open shader: " << path << "\n";
                exit(EXIT_FAILURE);
            }
            std::stringstream ss;
            ss << in.rdbuf();
            std::string srcStr = ss.str();
            const char* src = srcStr.c_str();

            GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &src, nullptr);
            glCompileShader(shader);

            GLint success;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                GLint logLen;
                glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
                std::vector<char> log(logLen);
                glGetShaderInfoLog(shader, logLen, nullptr, log.data());
                std::cerr << "Shader compile error (" << path << "):\n" << log.data() << "\n";
                exit(EXIT_FAILURE);
            }
            return shader;
        };

        GLuint vertShader = loadShader(vertPath, GL_VERTEX_SHADER);
        GLuint fragShader = loadShader(fragPath, GL_FRAGMENT_SHADER);

        GLuint program = glCreateProgram();
        glAttachShader(program, vertShader);
        glAttachShader(program, fragShader);
        glLinkProgram(program);

        GLint linkSuccess;
        glGetProgramiv(program, GL_LINK_STATUS, &linkSuccess);
        if (!linkSuccess) {
            GLint logLen;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLen);
            std::vector<char> log(logLen);
            glGetProgramInfoLog(program, logLen, nullptr, log.data());
            std::cerr << "Shader link error:\n" << log.data() << "\n";
            exit(EXIT_FAILURE);
        }

        glDeleteShader(vertShader);
        glDeleteShader(fragShader);

        return program;
    }
    GLuint CreateComputeProgram(const char* path) {
        // 1) read GLSL source
        std::ifstream in(path);
        if(!in.is_open()) {
            std::cerr << "Failed to open compute shader: " << path << "\n";
            exit(EXIT_FAILURE);
        }
        std::stringstream ss;
        ss << in.rdbuf();
        std::string srcStr = ss.str();
        const char* src = srcStr.c_str();

        // 2) compile
        GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(cs, 1, &src, nullptr);
        glCompileShader(cs);
        GLint ok; 
        glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
        if(!ok) {
            GLint logLen;
            glGetShaderiv(cs, GL_INFO_LOG_LENGTH, &logLen);
            std::vector<char> log(logLen);
            glGetShaderInfoLog(cs, logLen, nullptr, log.data());
            std::cerr << "Compute shader compile error:\n" << log.data() << "\n";
            exit(EXIT_FAILURE);
        }

        // 3) link
        GLuint prog = glCreateProgram();
        glAttachShader(prog, cs);
        glLinkProgram(prog);
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if(!ok) {
            GLint logLen;
            glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &logLen);
            std::vector<char> log(logLen);
            glGetProgramInfoLog(prog, logLen, nullptr, log.data());
            std::cerr << "Compute shader link error:\n" << log.data() << "\n";
            exit(EXIT_FAILURE);
        }

        glDeleteShader(cs);
        return prog;
    }
    // dispatchCompute – runs the geodesic.comp compute shader to ray-trace the scene.
    //
    // Steps:
    //   1. Re-allocate the output texture at the correct resolution.
    //   2. Bind the compute program and upload current-frame UBO data.
    //   3. Bind the texture as a write-only image unit (binding = 0).
    //   4. Dispatch a 2-D grid of 16×16 thread groups covering the render resolution.
    //   5. Insert a memory barrier so the texture can be safely read as a sampler later.
    void dispatchCompute(const Camera& cam) {
        int cw = cam.moving ? COMPUTE_WIDTH  : 200; // render width
        int ch = cam.moving ? COMPUTE_HEIGHT : 150; // render height

        // 1) Ensure the output texture matches the desired resolution.
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cw, ch, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        // 2) Activate the compute program and upload all UBOs.
        glUseProgram(computeProgram);
        uploadCameraUBO(cam);
        uploadDiskUBO();
        uploadObjectsUBO(objects);

        // 3) Bind the texture as a write-only image (image unit 0 → layout binding = 0).
        glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

        // 4) Dispatch enough work groups to cover every pixel (each group is 16×16 threads).
        GLuint groupsX = (GLuint)std::ceil(cw / 16.0f);
        GLuint groupsY = (GLuint)std::ceil(ch / 16.0f);
        glDispatchCompute(groupsX, groupsY, 1);

        // 5) Wait until all image writes are complete before sampling the texture.
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    // uploadCameraUBO – packs the camera state into the std140 Camera UBO (binding = 1).
    // Computes the orthonormal camera basis (right, up, forward) from position and target.
    void uploadCameraUBO(const Camera& cam) {
        struct UBOData {
            vec3 pos;     float _pad0; // std140 requires vec3 padded to 16 bytes
            vec3 right;   float _pad1;
            vec3 up;      float _pad2;
            vec3 forward; float _pad3;
            float tanHalfFov; // tan(fovY/2) for ray direction scaling
            float aspect;     // width/height for horizontal FOV correction
            bool moving;      // reserved for LOD (currently unused in shader)
            int _pad4;
        } data;

        vec3 fwd   = normalize(cam.target - cam.position()); // look direction
        vec3 up    = vec3(0, 1, 0);                          // world up (disk lies in x-z)
        vec3 right = normalize(cross(fwd, up));              // camera right
        up         = cross(right, fwd);                      // re-orthogonalise up

        data.pos        = cam.position();
        data.right      = right;
        data.up         = up;
        data.forward    = fwd;
        data.tanHalfFov = tan(radians(60.0f * 0.5f));
        data.aspect     = float(WIDTH) / float(HEIGHT);
        data.moving     = cam.dragging || cam.panning;

        glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(UBOData), &data);
    }

    // uploadObjectsUBO – packs scene objects into the Objects UBO (binding = 3).
    // Capped at 16 objects (matching the shader array size).
    void uploadObjectsUBO(const vector<ObjectData>& objs) {
        struct UBOData {
            int   numObjects;              // actual count
            float _pad0, _pad1, _pad2;    // std140 padding to 16-byte boundary
            vec4  posRadius[16];           // xyz = centre, w = radius
            vec4  color[16];              // rgba colour
            float mass[16];               // kg (for future N-body use in shader)
        } data;

        size_t count = std::min(objs.size(), size_t(16));
        data.numObjects = static_cast<int>(count);

        for (size_t i = 0; i < count; ++i) {
            data.posRadius[i] = objs[i].posRadius;
            data.color[i]     = objs[i].color;
            data.mass[i]      = objs[i].mass;
        }

        glBindBuffer(GL_UNIFORM_BUFFER, objectsUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(data), &data);
    }

    // uploadDiskUBO – writes accretion disk parameters into the Disk UBO (binding = 2).
    // The inner radius is just outside the Innermost Stable Circular Orbit (ISCO ≈ 3r_s).
    void uploadDiskUBO() {
        float r1        = SagA.r_s * 2.2f; // inner edge (just outside event horizon)
        float r2        = SagA.r_s * 5.2f; // outer edge
        float num       = 2.0;             // ring count (cosmetic, not used by current shader)
        float thickness = 1e9f;            // std140 padding placeholder
        float diskData[4] = { r1, r2, num, thickness };

        glBindBuffer(GL_UNIFORM_BUFFER, diskUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(diskData), diskData);
    }

    // QuadVAO – allocates the full-screen quad (two triangles, NDC [-1,1]×[-1,1])
    // and creates the RGBA8 texture that the compute shader writes into.
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

        // attribute 0: 2-D position (NDC)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        // attribute 1: UV coordinates for texture sampling
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // Allocate the output texture at the compute resolution (bilinear filtering for upscale).
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, COMPUTE_WIDTH, COMPUTE_HEIGHT,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        vector<GLuint> VAOtexture = {VAO, texture};
        return VAOtexture;
    }

    // renderScene – (legacy) clear + blit the compute texture with the passthrough shader.
    void renderScene() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(quadVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glfwSwapBuffers(window);
        glfwPollEvents();
    };
};
Engine engine;

// setupCameraCallbacks – registers GLFW input callbacks and forwards them to
// the Camera struct via the window user pointer.
void setupCameraCallbacks(GLFWwindow* window) {
    glfwSetWindowUserPointer(window, &camera);

    glfwSetMouseButtonCallback(window, [](GLFWwindow* win, int button, int action, int mods) {
        Camera* cam = (Camera*)glfwGetWindowUserPointer(win);
        cam->processMouseButton(button, action, mods, win);
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow* win, double x, double y) {
        Camera* cam = (Camera*)glfwGetWindowUserPointer(win);
        cam->processMouseMove(x, y);
    });

    glfwSetScrollCallback(window, [](GLFWwindow* win, double xoffset, double yoffset) {
        Camera* cam = (Camera*)glfwGetWindowUserPointer(win);
        cam->processScroll(xoffset, yoffset);
    });

    glfwSetKeyCallback(window, [](GLFWwindow* win, int key, int scancode, int action, int mods) {
        Camera* cam = (Camera*)glfwGetWindowUserPointer(win);
        cam->processKey(key, scancode, action, mods);
    });
}


// -- MAIN – per-frame render loop -- //
//
// Each frame:
//   1. Newtonian gravity (optional, toggle with G / right-click):
//      For every ordered pair of objects compute F = GMm/r² and integrate
//      velocity → position using explicit Euler integration.
//
//   2. Spacetime grid (CPU):
//      Rebuild the warped grid mesh using the Schwarzschild embedding function,
//      upload it to the GPU, and draw it with the grid shader.
//
//   3. GPU ray-tracer (compute shader):
//      Dispatch geodesic.comp to march null geodesics and write pixel colours
//      to the compute texture, then blit it to the screen with the quad shader.
//
//   4. Present:
//      Swap front/back buffers and poll for input events.
int main() {
    setupCameraCallbacks(engine.window);
    vector<unsigned char> pixels(engine.WIDTH * engine.HEIGHT * 3);


    auto t0 = Clock::now();
    lastPrintTime = chrono::duration<double>(t0.time_since_epoch()).count();

    double lastTime = glfwGetTime();
    int   renderW  = 800, renderH = 600, numSteps = 80000;
    while (!glfwWindowShouldClose(engine.window)) {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // optional, but good practice
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        double now   = glfwGetTime();
        double dt    = now - lastTime;   // seconds since last frame
        lastTime     = now;

        // -- 1. Newtonian gravity integration (O(N²) – fine for ≤ 16 objects) -- //
        // For each pair (obj, obj2) compute the gravitational force F = GMm/r²
        // and add the resulting acceleration to obj's velocity, then integrate position.
        for (auto& obj : objects) {
            for (auto& obj2 : objects) {
                if (&obj == &obj2) continue; // skip self-interaction
                float dx  = obj2.posRadius.x - obj.posRadius.x;
                float dy  = obj2.posRadius.y - obj.posRadius.y;
                float dz  = obj2.posRadius.z - obj.posRadius.z;
                float distance = sqrt(dx * dx + dy * dy + dz * dz);
                if (distance > 0) {
                    // Unit vector from obj toward obj2
                    vector<double> direction = {dx / distance, dy / distance, dz / distance};
                    // Newton's law of gravity: F = G·m1·m2 / r²
                    double Gforce = (G * obj.mass * obj2.mass) / (distance * distance);
                    // Acceleration on obj due to obj2:  a = F / m1
                    double acc1 = Gforce / obj.mass;
                    std::vector<double> acc = {direction[0] * acc1, direction[1] * acc1, direction[2] * acc1};
                    if (Gravity) {
                        // Explicit Euler: v += a,  x += v  (dt = 1 frame, not physically scaled)
                        obj.velocity.x += acc[0];
                        obj.velocity.y += acc[1];
                        obj.velocity.z += acc[2];

                        obj.posRadius.x += obj.velocity.x;
                        obj.posRadius.y += obj.velocity.y;
                        obj.posRadius.z += obj.velocity.z;
                        cout << "velocity: " <<obj.velocity.x<<", " <<obj.velocity.y<<", " <<obj.velocity.z<<endl;
                    }
                }
            }
        }

        // -- 2. Spacetime curvature grid (CPU → GPU) -- //
        // Rebuild the warped grid mesh each frame to reflect updated object positions.
        engine.generateGrid(objects);
        // Build the view-projection matrix for the grid overlay (same FOV as the compute shader).
        mat4 view     = lookAt(camera.position(), camera.target, vec3(0,1,0));
        mat4 proj     = perspective(radians(60.0f), float(engine.COMPUTE_WIDTH)/engine.COMPUTE_HEIGHT, 1e9f, 1e14f);
        mat4 viewProj = proj * view;
        engine.drawGrid(viewProj); // draw the warped grid with the grid shader

        // -- 3. GPU ray-tracer (compute shader) -- //
        glViewport(0, 0, engine.WIDTH, engine.HEIGHT); // full window viewport for the blit
        engine.dispatchCompute(camera);   // run geodesic.comp for all pixels
        engine.drawFullScreenQuad();      // blit the compute texture to the screen

        // -- 4. Present -- //
        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }

    glfwDestroyWindow(engine.window);
    glfwTerminate();
    return 0;
}
