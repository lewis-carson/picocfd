use pixels::{Error, Pixels, SurfaceTexture};
use winit::{
    dpi::PhysicalSize,
    event::{Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

// Core simulation parameters
const DT: f32 = 0.016;    // Time step targeting 60 FPS
const VISC: f32 = 0.001;  // Fluid viscosity - controls flow turbulence
const NITER: usize = 8;   // Solver iteration count - balance of accuracy vs performance

// Display configuration
const VIRTUAL_WIDTH: u32 = 150;   // Internal simulation grid width
const VIRTUAL_HEIGHT: u32 = 150;  // Internal simulation grid height
const UPSCALE: u32 = 4;          // Window scaling multiplier
const ACTUAL_WIDTH: u32 = VIRTUAL_WIDTH * UPSCALE;    // Physical window width
const ACTUAL_HEIGHT: u32 = VIRTUAL_HEIGHT * UPSCALE;  // Physical window height

// Simulation parameters
const LID_VELOCITY: f32 = 10.0;  // Speed of the top wall driving the flow

fn main() -> Result<(), Error> {
    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Simple CFD Simulation")
        .with_inner_size(PhysicalSize::new(ACTUAL_WIDTH, ACTUAL_HEIGHT)) // Larger window size
        .build(&event_loop)
        .unwrap();

    let size = window.inner_size();
    let mut pixels = {
        let surface_texture = SurfaceTexture::new(size.width, size.height, &window);
        // Use virtual resolution for the pixel buffer
        Pixels::new(VIRTUAL_WIDTH, VIRTUAL_HEIGHT, surface_texture)?
    };

    let mut simulation = Simulation::new(VIRTUAL_WIDTH as usize, VIRTUAL_HEIGHT as usize);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::RedrawRequested(_) => {
                simulation.update();
                simulation.draw(pixels.get_frame_mut());

                if pixels.render().is_err() {
                    *control_flow = ControlFlow::Exit;
                    return;
                }
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(new_size) => {
                    pixels.resize_surface(new_size.width, new_size.height).unwrap();
                    // Keep the buffer at virtual resolution
                    pixels.resize_buffer(VIRTUAL_WIDTH, VIRTUAL_HEIGHT).unwrap();

                    simulation = Simulation::new(VIRTUAL_WIDTH as usize, VIRTUAL_HEIGHT as usize);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    pixels.resize_surface(new_inner_size.width, new_inner_size.height).unwrap();
                    // Keep the buffer at virtual resolution
                    pixels.resize_buffer(VIRTUAL_WIDTH, VIRTUAL_HEIGHT).unwrap();

                    simulation = Simulation::new(
                        VIRTUAL_WIDTH as usize,
                        VIRTUAL_HEIGHT as usize,
                    );
                }
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                _ => {}
            },
            _ => {}
        }
    });
}

struct Simulation {
    width: usize,     // Grid width
    height: usize,    // Grid height
    u: Vec<f32>,      // Horizontal velocity field
    v: Vec<f32>,      // Vertical velocity field
    u_prev: Vec<f32>, // Previous horizontal velocity state
    v_prev: Vec<f32>, // Previous vertical velocity state
}

impl Simulation {
    fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        let u = vec![0.0; size];
        let v = vec![0.0; size];
        let u_prev = vec![0.0; size];
        let v_prev = vec![0.0; size];

        Self { width, height, u, v, u_prev, v_prev }
    }

    fn update(&mut self) {
        self.vel_step();
        self.apply_boundary_conditions();
    }

    fn vel_step(&mut self) {
        let dt = DT;
        let visc = VISC;
        let size = self.width * self.height;
        
        // Allocate scratch space for intermediate calculations
        let mut u_temp = vec![0.0; size];
        let mut v_temp = vec![0.0; size];
        
        // Phase 1: Apply external forces
        for i in 0..size {
            u_temp[i] = self.u[i] + dt * self.u_prev[i];
            v_temp[i] = self.v[i] + dt * self.v_prev[i];
        }
        
        // Phase 2: Apply viscous diffusion
        self.diffuse_step(1, &mut u_temp, &self.u, visc, dt);
        self.diffuse_step(2, &mut v_temp, &self.v, visc, dt);
        
        // Phase 3: Enforce mass conservation
        self.project_step(&mut u_temp, &mut v_temp);
        
        // Phase 4: Transport velocities
        let (mut u_next, mut v_next) = (vec![0.0; size], vec![0.0; size]);
        self.advect_step(1, &mut u_next, &u_temp, &u_temp, &v_temp, dt);
        self.advect_step(2, &mut v_next, &v_temp, &u_temp, &v_temp, dt);
        
        // Phase 5: Final mass conservation step
        self.project_step(&mut u_next, &mut v_next);
        
        // Update velocity fields with new values
        self.u.copy_from_slice(&u_next);
        self.v.copy_from_slice(&v_next);
    }

    fn draw(&self, frame: &mut [u8]) {
        // Scale factor for mapping simulation grid to display resolution
        let scale_factor = (self.width * self.height) as f32 / (VIRTUAL_WIDTH * VIRTUAL_HEIGHT) as f32;
        
        // Convert velocity field to color pixels
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            // Map display coordinates to simulation grid
            let vx = (i % VIRTUAL_WIDTH as usize) as f32;
            let vy = (i / VIRTUAL_WIDTH as usize) as f32;
            let sx = (vx * scale_factor) as usize;
            let sy = (vy * scale_factor) as usize;
            let si = sy * self.width + sx;
            
            // Calculate fluid speed at this point
            let speed = (self.u[si].powi(2) + self.v[si].powi(2)).sqrt();
            let value = (speed * 10.0).clamp(0.0, 1.0);
            
            // Convert to RGBA color
            let (r, g, b) = spectrum_color(value);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
            pixel[3] = 0xff;  // Full opacity
        }
    }

    fn apply_boundary_conditions(&mut self) {
        let w = self.width;
        let h = self.height;

        for x in 0..w {
            let top = x;
            let bottom = (h - 1) * w + x;

            // Top boundary (lid) moving with increased velocity u = 2.0
            self.u[top] = LID_VELOCITY;
            self.v[top] = 0.0;

            // Bottom boundary
            self.u[bottom] = 0.0;
            self.v[bottom] = 0.0;
        }

        for y in 0..h {
            let left = y * w;
            let right = y * w + w - 1;

            // Left boundary
            self.u[left] = 0.0;
            self.v[left] = 0.0;

            // Right boundary
            self.u[right] = 0.0;
            self.v[right] = 0.0;
        }
    }

    fn project(
        &self,
        u: &mut [f32],
        v: &mut [f32],
        p: &mut [f32],
        div: &mut [f32],
    ) {
        let h = 1.0 / (self.width as f32);

        for j in 1..self.height - 1 {
            for i in 1..self.width - 1 {
                let idx = j * self.width + i;
                div[idx] = -0.5 * h * (u[idx + 1] - u[idx - 1] + v[idx + self.width] - v[idx - self.width]);
                p[idx] = 0.0;
            }
        }
        self.set_bnd(0, div);
        self.set_bnd(0, p);

        for _ in 0..NITER {
            for j in 1..self.height - 1 {
                for i in 1..self.width - 1 {
                    let idx = j * self.width + i;
                    p[idx] = (div[idx] + p[idx - 1] + p[idx + 1] + p[idx - self.width] + p[idx + self.width]) / 4.0;
                }
            }
            self.set_bnd(0, p);
        }

        for j in 1..self.height - 1 {
            for i in 1..self.width - 1 {
                let idx = j * self.width + i;
                u[idx] -= 0.5 * (p[idx + 1] - p[idx - 1]) / h;
                v[idx] -= 0.5 * (p[idx + self.width] - p[idx - self.width]) / h;
            }
        }
        self.set_bnd(1, u);
        self.set_bnd(2, v);
    }

    fn linear_solve(
        &self,
        b: usize,
        x: &mut [f32],
        x0: &[f32],
        a: f32,
        c: f32,
    ) {
        for _ in 0..NITER {
            for j in 1..self.height - 1 {
                for i in 1..self.width - 1 {
                    let idx = j * self.width + i;
                    x[idx] = (
                        x0[idx] + a * (
                            x[idx - 1] + x[idx + 1] + x[idx - self.width] + x[idx + self.width]
                        )
                    ) / c;
                }
            }
            self.set_bnd(b, x);
        }
    }

    fn set_bnd(&self, b: usize, x: &mut [f32]) {
        // Handle boundary conditions for velocity components
        // b=1: horizontal velocity, b=2: vertical velocity
        let w = self.width;
        let h = self.height;

        // Apply boundary conditions to edges
        for i in 1..w - 1 {
            x[i] = if b == 2 { -x[w + i] } else { x[w + i] };                    // Top wall
            x[(h - 1) * w + i] = if b == 2 { -x[(h - 2) * w + i] } 
                                 else { x[(h - 2) * w + i] };                     // Bottom wall
        }

        for j in 1..h - 1 {
            // Left and right boundaries
            x[j * w] = if b == 1 { -x[j * w + 1] } else { x[j * w + 1] }; // Left
            x[j * w + w - 1] = if b == 1 { -x[j * w + w - 2] } else { x[j * w + w - 2] }; // Right
        }

        // Corners
        x[0] = 0.5 * (x[1] + x[w]);
        x[w - 1] = 0.5 * (x[w - 2] + x[2 * w - 1]);
        x[(h - 1) * w] = 0.5 * (x[(h - 2) * w] + x[(h - 1) * w + 1]);
        x[h * w - 1] = 0.5 * (x[h * w - 2] + x[(h - 1) * w - 1]);
    }

    fn diffuse_step(&self, b: usize, x: &mut [f32], x0: &[f32], diff: f32, dt: f32) {
        let a = dt * diff * (self.width - 2) as f32 * (self.height - 2) as f32;
        self.linear_solve(b, x, x0, a, 1.0 + 4.0 * a);
    }

    fn project_step(&self, u: &mut [f32], v: &mut [f32]) {
        let size = self.width * self.height;
        let mut div = vec![0.0; size];
        let mut p = vec![0.0; size];
        
        self.project(u, v, &mut p, &mut div);
    }

    fn advect_step(
        &self,
        b: usize,
        d: &mut [f32],
        d0: &[f32],
        u: &[f32],
        v: &[f32],
        dt: f32,
    ) {
        let nx = self.width as f32;
        let ny = self.height as f32;
        let dt0x = dt * (nx - 2.0);
        let dt0y = dt * (ny - 2.0);

        for j in 1..self.height - 1 {
            for i in 1..self.width - 1 {
                let idx = j * self.width + i;
                let mut x = i as f32 - dt0x * u[idx];
                let mut y = j as f32 - dt0y * v[idx];

                x = x.clamp(0.5, nx - 1.5);
                y = y.clamp(0.5, ny - 1.5);

                let i0 = x.floor() as usize;
                let i1 = i0 + 1;
                let j0 = y.floor() as usize;
                let j1 = j0 + 1;

                let s1 = x - i0 as f32;
                let s0 = 1.0 - s1;
                let t1 = y - j0 as f32;
                let t0 = 1.0 - t1;

                d[idx] = s0 * (t0 * d0[j0 * self.width + i0] + t1 * d0[j1 * self.width + i0]) +
                         s1 * (t0 * d0[j0 * self.width + i1] + t1 * d0[j1 * self.width + i1]);
            }
        }
        self.set_bnd(b, d);
    }
}

// Color mapping function for visualization
// Maps scalar values to a color gradient: Red → Yellow → Green → Cyan → Blue
fn spectrum_color(t: f32) -> (u8, u8, u8) {
    let t = t * 4.0; // Expand input to cover four color transitions
    
    let (r, g, b) = match t.floor() as i32 {
        0 => (255, (t * 255.0) as u8, 0),                    // Red → Yellow
        1 => ((255.0 * (2.0 - t)) as u8, 255, 0),           // Yellow → Green
        2 => (0, 255, ((t - 2.0) * 255.0) as u8),           // Green → Cyan
        _ => (0, (255.0 * (4.0 - t)) as u8, 255)            // Cyan → Blue
    };
    
    (r, g, b)
}
