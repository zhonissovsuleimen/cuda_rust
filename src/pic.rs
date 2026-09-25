use crate::{
  cell::Cell,
  consts::{SIZE, WIDTH},
  particle::Particle,
};

pub struct PIC {
  pub particles: Vec<Particle>,
  pub cells: Vec<Cell>,

  cell_offsets: Vec<usize>,
  cell_counts: Vec<usize>,
}

impl PIC {
  pub fn new(count: usize) -> Self {
    Self {
      particles: (0..count).map(|_| Particle::random()).collect(),
      cells: vec![Cell::empty(); SIZE],
      cell_offsets: vec![0; SIZE],
      cell_counts: vec![0; SIZE],
    }
  }

  pub fn offsets(&self) -> &Vec<usize> {
    &self.cell_offsets
  }

  pub fn counts(&self) -> &Vec<usize> {
    &self.cell_counts
  }

  pub fn update(&mut self) {
    self.cell_counts.fill(0);

    for particle in &self.particles {
      let x = particle.pos[0].floor() as usize;
      let y = particle.pos[1].floor() as usize;

      let cell = x + y * WIDTH;

      self.cell_counts[cell] += 1;
    }

    let mut offset = 0;
    for cell in 0..SIZE {
      self.cell_offsets[cell] = offset;
      offset += self.cell_counts[cell];
    }

    let mut new_particles = vec![Particle::zeros(); self.particles.len()];

    let mut lookup = self.cell_offsets.clone();
    for &particle in &self.particles {
      let x = particle.pos[0].floor() as usize;
      let y = particle.pos[1].floor() as usize;

      let cell = x + y * WIDTH;

      let destination = lookup[cell];
      new_particles[destination] = particle;
      lookup[cell] += 1;
    }

    self.particles = new_particles;
  }
}
