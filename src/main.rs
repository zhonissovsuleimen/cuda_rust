use std::{mem, time::Instant};

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use minifb::{Key, Window, WindowOptions};

use crate::{
  consts::{HEIGHT, SIZE, WIDTH},
  cuda::kernels,
  pic::PIC,
};

mod cell;
mod consts;
mod cuda;
mod particle;
mod pic;

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

  let mut pic = PIC::new(10_000);

  let mut device_cells_input = DeviceBuffer::from_host(&stream, &pic.cells)?;
  let mut device_cells_output = DeviceBuffer::from_host(&stream, &pic.cells)?;
  let mut device_particles_input = DeviceBuffer::from_host(&stream, &pic.particles)?;
  let mut device_particles_output = DeviceBuffer::from_host(&stream, &pic.particles)?;
  let mut device_offsets = DeviceBuffer::from_host(&stream, pic.offsets())?;
  let mut device_counts = DeviceBuffer::from_host(&stream, pic.counts())?;

  let module = unsafe { kernels::load(&ctx)? };

  let prep_clear_screen = module.prepare_clear_screen(launch_config)?;
  let prep_ptc = module.prepare_particle_to_cell(launch_config)?;
  let prep_ctp = module.prepare_cell_to_particle(launch_config)?;
  let prep_draw_screen = module.prepare_draw(launch_config)?;
  let prep_upd_particles = module.prepare_upd_particles(launch_config)?;
  let prep_clear_cells = module.prepare_clear_cells(launch_config)?;
  let prep_upd_cells = module.prepare_upd_cells(launch_config)?;

  let mut timer = Instant::now();
  while window.is_open() && !window.is_key_down(Key::Escape) {
    let dt = timer.elapsed().as_secs_f32();
    timer = Instant::now();

    pic.particles = device_particles_input.to_host_vec(&stream)?;
    pic.update();
    device_particles_input.copy_from_host(&stream, &pic.particles)?;
    device_offsets.copy_from_host(&stream, pic.offsets())?;
    device_counts.copy_from_host(&stream, pic.counts())?;

    module.clear_cells(&stream, &prep_clear_cells, &mut device_cells_input)?;
    #[rustfmt::skip]
    module.particle_to_cell(&stream, &prep_ptc, &device_particles_input, &device_offsets, &device_counts, &mut device_cells_input)?;
    #[rustfmt::skip]
    module.upd_cells(&stream, &prep_upd_cells, &device_cells_input, &mut device_cells_output)?;

    #[rustfmt::skip]
    module.cell_to_particle(&stream, &prep_ctp, &device_cells_output, &mut device_particles_input, dt)?;

    #[rustfmt::skip]
    module.upd_particles(&stream, &prep_upd_particles, &device_particles_input, &mut device_particles_output, dt)?;

    mem::swap(&mut device_particles_input, &mut device_particles_output);

    module.clear_screen(&stream, &prep_clear_screen, &mut device_screen)?;
    #[rustfmt::skip]
    module.draw(&stream, &prep_draw_screen, &device_cells_output, &mut device_screen)?;

    host_screen = device_screen.to_host_vec(&stream)?;

    window
      .update_with_buffer(&host_screen, WIDTH, HEIGHT)
      .unwrap();
  }
  Ok(())
}
