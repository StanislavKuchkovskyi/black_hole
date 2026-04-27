// grid.vert
// ---------
// Minimal vertex shader for the spacetime-curvature grid overlay.
// Receives pre-warped 3-D vertex positions from the CPU (the grid mesh is
// bent on the CPU using the Schwarzschild embedding function and uploaded
// to the GPU via a VBO).  A combined view-projection matrix is applied to
// project the vertices to clip space.

#version 330 core
layout(location = 0) in vec3 aPos; // world-space position of the grid vertex
uniform mat4 viewProj;             // combined view * projection matrix (from CPU)
void main() {
    gl_Position = viewProj * vec4(aPos, 1.0);
}
