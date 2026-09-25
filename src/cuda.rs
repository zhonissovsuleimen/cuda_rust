use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
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
  pub fn upd_cells(
    particles: &[Particle],
    offsets: &[usize],
    counts: &[usize],
    mut cells: DisjointSlice<Cell>,
  ) {
    let idx = thread::index_1d();
    let i = idx.get();

    if let Some(c) = cells.get_mut(idx) {
      let start = offsets[i];
      let count = counts[i];

      for pid in start..(start + count) {
        let particle = particles[pid];

        c.count += 1;
        c.vel[0] += particle.vel[0];
        c.vel[1] += particle.vel[1];
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
