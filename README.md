# picoCFD

A lightweight Computational Fluid Dynamics simulation framework written in Rust.

## Description

picoCFD is a minimal (I mean, very minimal) implementation of the Navier-Stokes equations for incompressible fluid flow. The simulation uses a grid-based approach with finite difference methods for solving fluid dynamics in real-time.

The project is intended as a learning exercise for understanding the basics of fluid dynamics and numerical methods. It is not intended for production use or serious research applications. As such, the entire project is implemented in a single ~360 line file with minimal dependencies.

## Features

- Real-time fluid simulation using Navier-Stokes equations
- Hardware-accelerated rendering with the `pixels` crate
- Efficient grid-based computation
- Lid-driven cavity flow demonstration
- Customizable simulation parameters:
  - Viscosity
  - Grid resolution
  - Time step
  - Flow velocity

## Dependencies

- Rust
- `pixels` crate
- `winit` crate
- `rand` crate

## Example Usage

Run the simulation:
```sh
cargo run --release
```

The simulation demonstrates a classic lid-driven cavity flow problem:
- Top wall moves at constant velocity
- Other walls are stationary
- Fluid motion is driven by viscous drag
- Visualization uses color mapping:
  - Red → High velocity
  - Blue → Low velocity

## Implementation Details

Core components:
- Grid-based velocity field (u, v components)
- Pressure field computation
- Semi-Lagrangian advection

## License

This project is licensed under the MIT License.
