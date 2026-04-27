// 2D_lensing.cpp
// ---------------
// Top-down 2-D simulation of gravitational lensing by a Schwarzschild black hole.
//
// Each "Ray" is a massless photon whose path is governed by the null geodesic
// equations of the Schwarzschild metric restricted to the equatorial plane.
// The path is integrated numerically with a classical 4th-order Runge-Kutta (RK4)
// scheme, stepping in the affine parameter λ.
//
// Physical quantities are stored in SI units (metres, kilograms, seconds).

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace glm;
using namespace std;

// Physical constants (SI)
double c = 299792458.0;  // speed of light (m/s)
double G = 6.67430e-11;  // gravitational constant (m³ kg⁻¹ s⁻²)

struct Ray;
void rk4Step(Ray& ray, double dλ, double rs);

// --- Structs --- //

// Engine manages the OpenGL window and the 2-D orthographic camera used for
// the top-down lensing view.  It wraps GLFW (window/events) and GLEW
// (modern OpenGL function loading).
struct Engine {
    GLFWwindow* window;
    int WIDTH = 800;
    int HEIGHT = 600;
    float width = 100000000000.0f;  // Half-width of the viewport in metres (~100 Gm)
    float height = 75000000000.0f; // Half-height of the viewport in metres (~75 Gm)

    // Navigation state – panning with the middle mouse button
    float offsetX = 0.0f, offsetY = 0.0f;
    float zoom = 1.0f;
    bool middleMousePressed = false;
    double lastMouseX = 0.0, lastMouseY = 0;

    Engine() {
        // Initialise GLFW (window system abstraction)
        if (!glfwInit()) {
            cerr << "Failed to initialize GLFW" << endl;
            exit(EXIT_FAILURE);
        }
        window = glfwCreateWindow(WIDTH, HEIGHT, "Black Hole Simulation", NULL, NULL);
        if (!window) {
            cerr << "Failed to create GLFW window" << endl;
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
        glfwMakeContextCurrent(window);

        // Initialise GLEW so modern OpenGL entry-points are available
        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) {
            cerr << "Failed to initialize GLEW" << endl;
            glfwDestroyWindow(window);
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
        glViewport(0, 0, WIDTH, HEIGHT);;
    }

    // run() is called once per frame.  It clears the screen and sets up an
    // orthographic projection that maps world-space metres to screen pixels.
    // The offsets allow the user to pan the view interactively.
    void run() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        double left   = -width + offsetX;
        double right  =  width + offsetX;
        double bottom = -height + offsetY;
        double top    =  height + offsetY;
        // glOrtho sets up a 2-D orthographic clip volume (no perspective distortion)
        glOrtho(left, right, bottom, top, -1.0, 1.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }
};
Engine engine;

// BlackHole holds the physical parameters of the black hole.
// r_s (Schwarzschild radius) = 2GM/c² defines the event horizon.
// The draw() method renders a filled red circle of radius r_s using a GL_TRIANGLE_FAN.
struct BlackHole {
    vec3 position;   // world-space centre (here always the origin)
    double mass;     // mass in kg
    double radius;   // visual radius (unused; r_s is used instead)
    double r_s;      // Schwarzschild radius in metres

    // Constructor: computes r_s from mass using r_s = 2GM/c²
    BlackHole(vec3 pos, float m) : position(pos), mass(m) {r_s = 2.0 * G * mass / (c*c);}

    // Draws the event horizon as a solid red circle centred at the origin.
    void draw() {
        glBegin(GL_TRIANGLE_FAN);
        glColor3f(1.0f, 0.0f, 0.0f);    // red colour for the event horizon
        glVertex2f(0.0f, 0.0f);         // centre vertex of the fan
        for(int i = 0; i <= 100; i++) {
            float angle = 2.0f * M_PI * i / 100;
            float x = r_s * cos(angle);
            float y = r_s * sin(angle);
            glVertex2f(x, y);
        }
        glEnd();
    }
};
// Sagittarius A* – the supermassive black hole at the centre of the Milky Way.
// Mass ≈ 4.15 million solar masses (8.54 × 10^36 kg).
// The Schwarzschild radius r_s = 2GM/c² is computed automatically in the constructor.
BlackHole SagA(vec3(0.0f, 0.0f, 0.0f), 8.54e36);

// Ray represents a single photon travelling through the 2-D Schwarzschild spacetime.
// Because the metric is spherically symmetric and the photon stays in the equatorial
// plane (θ = π/2), we need only two spatial coordinates (r, φ) plus their
// affine-parameter derivatives (dr/dλ, dφ/dλ).
struct Ray{
    // -- Cartesian position (used only for drawing) -- //
    double x;   double y;
    // -- Polar coordinates (used by the integrator) -- //
    double r;   double phi;   // radial distance and azimuthal angle
    double dr;  double dphi;  // velocities in polar coords
    vector<vec2> trail;       // historical positions for the fading-trail effect
    // -- Conserved quantities along the null geodesic -- //
    // E = energy per unit mass  (related to time-translation symmetry)
    // L = angular momentum per unit mass  (related to rotational symmetry)
    double E, L;

    // Constructor: given Cartesian position 'pos' and Cartesian velocity direction 'dir',
    // converts to polar coords and seeds the Schwarzschild conserved quantities.
    Ray(vec2 pos, vec2 dir) : x(pos.x), y(pos.y), r(sqrt(pos.x * pos.x + pos.y * pos.y)), phi(atan2(pos.y, pos.x)), dr(dir.x), dphi(dir.y) {
        // step 1) get polar coords (r, phi) :
        this->r = sqrt(x*x + y*y);
        this->phi = atan2(y, x);
        // step 2) seed velocities – project Cartesian dir onto polar basis:
        dr   =  dir.x * cos(phi) + dir.y * sin(phi); // radial component
        dphi = (-dir.x * sin(phi) + dir.y * cos(phi)) / r; // tangential component / r
        // step 3) store Schwarzschild conserved quantities.
        // f(r) = 1 - r_s/r  is the Schwarzschild lapse function.
        L = r*r * dphi;
        double f = 1.0 - SagA.r_s/r;
        double dt_dλ = sqrt( (dr*dr)/(f*f) + (r*r*dphi*dphi)/f );
        E = f * dt_dλ;
        // step 4) start trail :
        trail.push_back({x, y});
    }

    // Draw all rays in the scene.  Each ray is shown as a small red dot at its
    // current position, plus a fading white line showing its past trajectory.
    void draw(const std::vector<Ray>& rays) {
        // draw current ray positions as points
        glPointSize(2.0f);
        glColor3f(1.0f, 0.0f, 0.0f);
        glBegin(GL_POINTS);
          for (const auto& ray : rays) {
              glVertex2f(ray.x, ray.y);
          }
        glEnd();
    
        // turn on blending for the trails
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glLineWidth(1.0f);
    
        // draw each trail with fading alpha so older segments are more transparent
        for (const auto& ray : rays) {
            size_t N = ray.trail.size();
            if (N < 2) continue;
    
            glBegin(GL_LINE_STRIP);
            for (size_t i = 0; i < N; ++i) {
                // older points (i=0) get alpha≈0, newer get alpha≈1
                float alpha = float(i) / float(N - 1);
                glColor4f(1.0f, 1.0f, 1.0f, std::max(alpha, 0.05f));
                glVertex2f(ray.trail[i].x, ray.trail[i].y);
            }
            glEnd();
        }
    
        glDisable(GL_BLEND);
    }

    // Advance the ray by one affine-parameter step dλ using the RK4 integrator.
    // Stops integration once the photon crosses the event horizon (r ≤ r_s).
    void step(double dλ, double rs) {
        // 1) integrate (r,φ,dr,dφ) forward in affine parameter λ
        if(r <= rs) return; // stop if inside the event horizon
        rk4Step(*this, dλ, rs);

        // 2) convert back to cartesian x,y for drawing
        x = r * cos(phi);
        y = r * sin(phi);

        // 3) record the current position in the trail
        trail.push_back({ float(x), float(y) });
    }
};
vector<Ray> rays;

// geodesicRHS – Right-Hand Side of the 2-D Schwarzschild null geodesic equations.
//
// The photon's state is (r, φ, dr/dλ, dφ/dλ).  These equations come from
// the Euler-Lagrange equations for a null (massless) particle in the
// Schwarzschild metric restricted to the equatorial plane (θ = π/2):
//
//   d²r/dλ²  = -(rs/2r²)·f·(dt/dλ)²  +  (rs/2r²f)·(dr/dλ)²  +  r·(dφ/dλ)²
//   d²φ/dλ²  = -2·(dr/dλ)·(dφ/dλ) / r
//
// where f(r) = 1 - rs/r  is the Schwarzschild lapse function and
// dt/dλ = E/f  uses the conserved energy E.
void geodesicRHS(const Ray& ray, double rhs[4], double rs) {
    double r    = ray.r;
    double dr   = ray.dr;
    double dphi = ray.dphi;
    double E    = ray.E;

    double f = 1.0 - rs/r; // lapse function

    // First derivatives (by definition):
    rhs[0] = dr;    // dr/dλ
    rhs[1] = dphi;  // dφ/dλ

    // Second derivative of r from the null geodesic equation:
    double dt_dλ = E / f;
    rhs[2] = 
        - (rs/(2*r*r)) * f * (dt_dλ*dt_dλ)   // gravitational attraction term
        + (rs/(2*r*r*f)) * (dr*dr)             // correction from Schwarzschild metric
        + (r - rs) * (dphi*dphi);              // centrifugal term

    // Second derivative of φ (Christoffel symbol Γ^φ_{rφ}):
    rhs[3] = -2.0 * dr * dphi / r;
}

// addState – helper used inside rk4Step.
// Computes out[i] = a[i] + factor * b[i]  for i in [0,4).
// This avoids repeating the same loop in the RK4 body.
void addState(const double a[4], const double b[4], double factor, double out[4]) {
    for (int i = 0; i < 4; i++)
        out[i] = a[i] + b[i] * factor;
}

// rk4Step – Classical 4th-order Runge-Kutta integrator for one affine-parameter step dλ.
//
// RK4 approximates the ODE solution as a weighted average of four slope estimates:
//   k1 = f(y₀)
//   k2 = f(y₀ + dλ/2 · k1)
//   k3 = f(y₀ + dλ/2 · k2)
//   k4 = f(y₀ + dλ   · k3)
//   y₁ = y₀ + (dλ/6)·(k1 + 2k2 + 2k3 + k4)
//
// Each k evaluates the geodesic right-hand side at a temporary state,
// so we copy the ray into r2/r3/r4 and patch the relevant fields.
void rk4Step(Ray& ray, double dλ, double rs) {
    double y0[4] = { ray.r, ray.phi, ray.dr, ray.dphi };
    double k1[4], k2[4], k3[4], k4[4], temp[4];

    // k1: slope at the beginning of the step
    geodesicRHS(ray, k1, rs);
    addState(y0, k1, dλ/2.0, temp);
    Ray r2 = ray; r2.r=temp[0]; r2.phi=temp[1]; r2.dr=temp[2]; r2.dphi=temp[3];

    // k2: slope at the midpoint (estimated using k1)
    geodesicRHS(r2, k2, rs);
    addState(y0, k2, dλ/2.0, temp);
    Ray r3 = ray; r3.r=temp[0]; r3.phi=temp[1]; r3.dr=temp[2]; r3.dphi=temp[3];

    // k3: slope at the midpoint (estimated using k2)
    geodesicRHS(r3, k3, rs);
    addState(y0, k3, dλ, temp);
    Ray r4 = ray; r4.r=temp[0]; r4.phi=temp[1]; r4.dr=temp[2]; r4.dphi=temp[3];

    // k4: slope at the end of the step (estimated using k3)
    geodesicRHS(r4, k4, rs);

    // Combine slopes with RK4 weights (1, 2, 2, 1) / 6:
    ray.r    += (dλ/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
    ray.phi  += (dλ/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
    ray.dr   += (dλ/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]);
    ray.dphi += (dλ/6.0)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3]);
}


// Main render loop for the 2-D lensing simulation.
// Each frame:
//   1. engine.run()     – clear screen, set orthographic projection
//   2. SagA.draw()      – draw the event horizon as a red circle
//   3. ray.step(...)    – advance each photon by one RK4 step (dλ = 1 metre of affine parameter)
//   4. ray.draw(rays)   – paint the current positions and fading trails
//   5. glfwSwapBuffers  – display the rendered frame
//   6. glfwPollEvents   – handle keyboard / mouse input
int main () {
    //rays.push_back(Ray(vec2(-1e11, 3.27606302719999999e10), vec2(c, 0.0f)));
    while(!glfwWindowShouldClose(engine.window)) {
        engine.run();
        SagA.draw();

        for (auto& ray : rays) {
            ray.step(1.0f, SagA.r_s);
            ray.draw(rays);
        }

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }

    return 0;
}
