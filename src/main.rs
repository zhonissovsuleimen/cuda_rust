use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use minifb::{Key, Window, WindowOptions};

use crate::{
  consts::{HEIGHT, SIZE, WIDTH},
  cuda::kernels,
  particle::Particle,
};

mod consts;
mod cuda;
mod particle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut window = Window::new(
    "Test - ESC to exit",
    WIDTH,
    HEIGHT,
    WindowOptions {
      resize: true,
      scale_mode: minifb::ScaleMode::Stretch,
      ..Default::default()
    },
  )?;

  window.set_target_fps(60);

  let ctx = CudaContext::new(0)?;
  let stream = ctx.default_stream();
  let launch_config = LaunchConfig1D::new((SIZE as u32).div_ceil(256), 256, 0);

  let mut host_screen = vec![0u32; SIZE];
  let mut device_screen = DeviceBuffer::from_host(&stream, &host_screen)?;

  let mut host_particles: Vec<Particle> = (0..100).map(|_| Particle::random()).collect();
  let mut device_particles_a = DeviceBuffer::from_host(&stream, &host_particles)?;
  let mut device_particles_b = DeviceBuffer::from_host(&stream, &host_particles)?;
  let mut flip = true;

  let module = unsafe { kernels::load(&ctx)? };

  let prep_clear = module.prepare_clear(launch_config)?;
  let prep_draw = module.prepare_draw(launch_config)?;
  let prep_solve = module.prepare_solve(launch_config)?;

  while window.is_open() && !window.is_key_down(Key::Escape) {
    module.clear(&stream, &prep_clear, &mut device_screen)?;

    if flip {
      #[rustfmt::skip]
      module.solve(&stream, &prep_solve, &device_particles_a, &mut device_particles_b, 1.0)?;

      unsafe {
        let screen_ptr = device_screen.cu_deviceptr() as *mut u32;
        module.draw(&stream, &prep_draw, &device_particles_b, screen_ptr)?;
      }
    } else {
      #[rustfmt::skip]
      module.solve(&stream, &prep_solve, &device_particles_b, &mut device_particles_a, 1.0)?;

      unsafe {
        let screen_ptr = device_screen.cu_deviceptr() as *mut u32;
        module.draw(&stream, &prep_draw, &device_particles_a, screen_ptr)?;
      }
    }
    flip = !flip;

    host_screen = device_screen.to_host_vec(&stream)?;

    window
      .update_with_buffer(&host_screen, WIDTH, HEIGHT)
      .unwrap();
  }
  Ok(())
}
