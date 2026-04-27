// grid.frag
// ---------
// Fragment shader for the spacetime-curvature grid overlay.
// All grid lines are rendered in a uniform semi-transparent grey so that
// the underlying ray-traced image remains visible beneath the grid.

#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(0.5, 0.5, 0.5, 0.7); // translucent grey grid lines
}
