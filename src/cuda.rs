use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
  use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicU32};

  use crate::{
    cell::Cell,
    consts::{HEIGHT, WIDTH},
    particle::Particle,
  };

  use super::*;

  const GRAVITY: f32 = 0.1;
  const ENERGY_LOSS: f32 = 0.9;

  #[kernel]
  #[launch_bounds(256)]
  #[launch_contract(domain = 1, block = (256, 1, 1))]
  pub fn upd_particles(input: &[Particle], mut output: DisjointSlice<Particle>, dt: f32) {
    let idx = thread::index_1d();
    let i = idx.get();

    if let Some(out) = output.get_mut(idx) {
      let input = &input[i];

      *out = *input;

      out.pos[0] += out.vel[0] * dt;
      out.pos[1] += out.vel[1] * dt;

      if out.pos[0] < 0.0 || out.pos[0] >= WIDTH as f32 {
        out.pos[0] = out.pos[0].clamp(0.0, WIDTH as f32 - 1.0);
        out.vel[0] *= -ENERGY_LOSS;
      }

      if out.pos[1] < 0.0 || out.pos[1] >= HEIGHT as f32 {
        out.pos[1] = out.pos[1].clamp(0.0, HEIGHT as f32 - 1.0);
        out.vel[1] *= -ENERGY_LOSS;
      }

      out.vel[1] += GRAVITY * dt;
    }
  }

  #[kernel]
  #[launch_bounds(256)]
  #[launch_contract(domain = 1, block = (256, 1, 1))]
  pub fn clear_cells(mut cells: DisjointSlice<Cell>) {
    let idx = thread::index_1d();

    if let Some(cell) = cells.get_mut(idx) {
      cell.clear();
    }
  }

  #[kernel]
  #[launch_bounds(256)]
  #[launch_contract(domain = 1, block = (256, 1, 1))]
  pub unsafe fn upd_cells(particles: &[Particle], cells: *mut Cell) {
    let i = thread::index_1d().get();

    if let Some(p) = particles.get(i) {
      let cell_id = p.pos[0] as usize + p.pos[1] as usize * WIDTH;

      unsafe {
        let cell = cells.add(cell_id);

        DeviceAtomicU32::from_ptr(&raw mut (*cell).count).fetch_add(1, AtomicOrdering::Relaxed);
        DeviceAtomicF32::from_ptr(&raw mut (*cell).vel[0])
          .fetch_add(p.vel[0], AtomicOrdering::Relaxed);
        DeviceAtomicF32::from_ptr(&raw mut (*cell).vel[1])
          .fetch_add(p.vel[1], AtomicOrdering::Relaxed);
      }
    }
  }

  #[kernel]
  #[launch_bounds(256)]
  #[launch_contract(domain = 1, block = (256, 1, 1))]
  pub fn clear_screen(mut screen: DisjointSlice<u32>) {
    let idx = thread::index_1d();

    if let Some(pixel) = screen.get_mut(idx) {
      *pixel = 0;
    }
  }
  #[kernel]
  #[launch_bounds(256)]
  #[launch_contract(domain = 1, block = (256, 1, 1))]
  pub fn draw(cells: &[Cell], mut screen: DisjointSlice<u32>) {
    let idx = thread::index_1d();
    let i = idx.get();

    if let Some(pixel) = screen.get_mut(idx) {
      if cells[i].count == 0 {
        *pixel = 0;
        return;
      }

      let max = 5;
      let intensity = cells[i].count.clamp(0, max) as f32 / max as f32;
      let c = (0xFFu8 as f32 * (1.0 - intensity)) as u32;

      *pixel = (c << 16) | (c << 8) | 0xFF;
    }
  }
}
