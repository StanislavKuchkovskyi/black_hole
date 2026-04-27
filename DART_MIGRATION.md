# Dart Migration Guide

> **Goal:** port this C++ / OpenGL black-hole simulation to Dart so it can run
> on any platform supported by the Flutter or Dart Native toolchains.

---

## 1. Why Dart?

| Concern | C++ / OpenGL (current) | Dart (target) |
|---|---|---|
| Platform | Windows / Linux / macOS | Web, Android, iOS, Desktop (Flutter) |
| GPU access | OpenGL 4.3 (compute shaders) | `dart:ui` / Flutter GPU, WebGL (via `package:web`) |
| Math library | GLM (C++ header-only) | `package:vector_math` |
| Build system | CMake + vcpkg | `dart pub get` |
| Parallelism | OpenMP + GPU compute | Dart Isolates + WebGL/WASM |

---

## 2. Recommended packages

| C++ dependency | Dart equivalent |
|---|---|
| GLM (glm::vec3, mat4 …) | [`vector_math`](https://pub.dev/packages/vector_math) |
| GLFW + GLEW (window / OpenGL) | [`flutter_gl`](https://pub.dev/packages/flutter_gl) or direct WebGL via `package:web` |
| OpenMP (CPU parallelism) | `dart:isolate` + `Isolate.run()` |
| OpenGL compute shaders | Flutter GPU shaders (`flutter_gpu`) or WebGL 2 compute (experimental) |
| `<chrono>` timing | `Stopwatch` / `DateTime.now()` |

---

## 3. Step-by-step migration plan

### Step 1 – Set up a Dart/Flutter project

```bash
flutter create black_hole_dart
cd black_hole_dart
```

Add dependencies to `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  vector_math: ^2.1.4
  flutter_gl: ^0.0.7        # or package:web for WebGL
```

---

### Step 2 – Port the physics (pure Dart, no GPU)

The physics code is pure math with no platform-specific dependencies.
Port these first; they are straightforward 1:1 translations.

#### Physical constants

```dart
// constants.dart
const double c = 299792458.0; // speed of light (m/s)
const double G = 6.67430e-11; // gravitational constant (m³ kg⁻¹ s⁻²)
```

#### BlackHole struct

```dart
// black_hole.dart
import 'package:vector_math/vector_math.dart';
import 'constants.dart';

class BlackHole {
  final Vector3 position;
  final double mass;
  late final double rs; // Schwarzschild radius

  BlackHole(this.position, this.mass) {
    rs = 2.0 * G * mass / (c * c);
  }

  bool intercept(double px, double py, double pz) {
    final dx = px - position.x;
    final dy = py - position.y;
    final dz = pz - position.z;
    return dx * dx + dy * dy + dz * dz < rs * rs;
  }
}
```

#### Ray struct (3-D Schwarzschild)

```dart
// ray.dart
import 'dart:math';
import 'package:vector_math/vector_math.dart';
import 'black_hole.dart';

class Ray {
  double x, y, z;
  double r, theta, phi;
  double dr, dtheta, dphi;
  double E, L;

  Ray(Vector3 pos, Vector3 dir, BlackHole bh)
      : x = pos.x, y = pos.y, z = pos.z,
        r = 0, theta = 0, phi = 0,
        dr = 0, dtheta = 0, dphi = 0,
        E = 0, L = 0 {
    r     = sqrt(x * x + y * y + z * z);
    theta = acos(z / r);
    phi   = atan2(y, x);

    final dx = dir.x, dy = dir.y, dz = dir.z;
    dr     =  sin(theta)*cos(phi)*dx + sin(theta)*sin(phi)*dy + cos(theta)*dz;
    dtheta = (cos(theta)*cos(phi)*dx + cos(theta)*sin(phi)*dy - sin(theta)*dz) / r;
    dphi   = (-sin(phi)*dx + cos(phi)*dy) / (r * sin(theta));

    L = r * r * sin(theta) * dphi;
    final f = 1.0 - bh.rs / r;
    final dtdL = sqrt((dr * dr) / f +
        r * r * dtheta * dtheta +
        r * r * sin(theta) * sin(theta) * dphi * dphi);
    E = f * dtdL;
  }

  void step(double dLambda, double rs) {
    if (r <= rs) return;
    _rk4Step(dLambda, rs);
    x = r * sin(theta) * cos(phi);
    y = r * sin(theta) * sin(phi);
    z = r * cos(theta);
  }

  // ... _geodesicRHS and _rk4Step identical to C++ logic
}
```

#### RK4 integrator

The RK4 logic maps directly.  Replace C-style arrays with Dart `List<double>`:

```dart
List<double> _geodesicRHS(double r, double theta, double dr,
    double dtheta, double dphi, double E, double rs) {
  final f = 1.0 - rs / r;
  final dtdL = E / f;
  return [
    dr, dtheta, dphi,
    -(rs / (2 * r * r)) * f * dtdL * dtdL
        + (rs / (2 * r * r * f)) * dr * dr
        + r * (dtheta * dtheta + sin(theta) * sin(theta) * dphi * dphi),
    -2.0 * dr * dtheta / r + sin(theta) * cos(theta) * dphi * dphi,
    -2.0 * dr * dphi / r - 2.0 * cos(theta) / sin(theta) * dtheta * dphi,
  ];
}
```

---

### Step 3 – Port the renderer

#### Option A – Flutter canvas (CPU, no GPU shaders)

Render into a `Uint8List` pixel buffer and display it with `decodeImageFromPixels`:

```dart
// renderer.dart
import 'dart:typed_data';
import 'dart:ui' as ui;

Future<ui.Image> renderFrame(int width, int height, Camera cam, BlackHole bh) async {
  final pixels = Uint8List(width * height * 4); // RGBA

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // Build view ray, march geodesic, determine colour …
      final color = _tracePixel(x, y, width, height, cam, bh);
      final idx = (y * width + x) * 4;
      pixels[idx + 0] = (color.x * 255).clamp(0, 255).toInt();
      pixels[idx + 1] = (color.y * 255).clamp(0, 255).toInt();
      pixels[idx + 2] = (color.z * 255).clamp(0, 255).toInt();
      pixels[idx + 3] = 255;
    }
  }

  final completer = Completer<ui.Image>();
  ui.decodeImageFromPixels(pixels, width, height, ui.PixelFormat.rgba8888,
      completer.complete);
  return completer.future;
}
```

Display the image inside a `CustomPainter`:

```dart
class BlackHolePainter extends CustomPainter {
  final ui.Image frame;
  BlackHolePainter(this.frame);

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawImage(frame, Offset.zero, Paint());
  }

  @override
  bool shouldRepaint(covariant BlackHolePainter old) => true;
}
```

#### Option B – WebGL / flutter_gl (GPU path)

Use `flutter_gl` to create an OpenGL ES context, compile GLSL shaders (adapt the
existing `.vert` / `.frag` / `.comp`), and draw with the same pipeline as the C++
code.  The shader GLSL source can be reused almost verbatim (change `#version 430`
to `#version 310 es` for OpenGL ES 3.1 compute support).

---

### Step 4 – Parallelise with Dart Isolates

Replace the OpenMP `#pragma omp parallel for` with Dart Isolates:

```dart
import 'dart:isolate';

Future<void> renderParallel(Uint8List pixels, int width, int height) async {
  final rowsPerIsolate = height ~/ Platform.numberOfProcessors;
  final futures = <Future>[];

  for (int i = 0; i < Platform.numberOfProcessors; i++) {
    final startY = i * rowsPerIsolate;
    final endY   = (i == Platform.numberOfProcessors - 1)
        ? height : startY + rowsPerIsolate;

    futures.add(Isolate.run(() {
      for (int y = startY; y < endY; y++) {
        // trace row y …
      }
    }));
  }

  await Future.wait(futures);
}
```

---

### Step 5 – Camera and input

Replace GLFW callbacks with Flutter gesture detectors:

```dart
GestureDetector(
  onScaleUpdate: (details) => camera.zoom(details.scale),
  onPanUpdate: (details) => camera.orbit(details.delta),
  child: CustomPaint(painter: BlackHolePainter(currentFrame)),
)
```

---

## 4. File mapping

| C++ file | Dart equivalent |
|---|---|
| `2D_lensing.cpp` | `lib/simulation/lensing_2d.dart` |
| `CPU-geodesic.cpp` | `lib/simulation/geodesic_3d.dart` |
| `black_hole.cpp` | `lib/simulation/black_hole_gpu.dart` |
| `ray_tracing.cpp` | `lib/simulation/ray_tracer.dart` |
| `geodesic.comp` | `shaders/geodesic.glsl` (WebGL ES 3.1) |
| `grid.vert` / `grid.frag` | `shaders/grid.vert` / `shaders/grid.frag` |

---

## 5. Known challenges

| Challenge | Notes |
|---|---|
| **OpenGL 4.3 compute shaders** | Not available on iOS or older Android.  Use WebGL 2 compute (Chrome flag) or a CPU fallback. |
| **`double` precision** | Dart's `double` is 64-bit (same as C++ `double`), so numerical results should match. |
| **Performance** | Dart JIT is slower than C++.  Use AOT compilation (`flutter build`) and Isolates. |
| **GLM → vector_math** | `vector_math` covers vec2/3/4 and mat4; all GLM functions used here have equivalents. |
| **OMP parallelism** | Dart Isolates share no memory; pass data via `SendPort`/`ReceivePort` or use `ffi` for shared buffers. |

---

## 6. Useful resources

- [Flutter GPU (experimental)](https://docs.flutter.dev/perf/rendering-performance#flutter-gpu)
- [`vector_math` package](https://pub.dev/packages/vector_math)
- [`flutter_gl` package](https://pub.dev/packages/flutter_gl)
- [Dart Isolates documentation](https://dart.dev/language/isolates)
- [WebGL 2 Compute (draft spec)](https://www.khronos.org/registry/webgl/specs/latest/2.0-compute/)
