use cuda_core::DeviceCopy;

use crate::consts::{HEIGHT, WIDTH};

#[derive(Clone, Copy, DeviceCopy)]
pub struct Particle {
  pub pos: [f32; 2],
  pub vel: [f32; 2],
}

impl Particle {
  pub fn random() -> Self {
    Self {
      pos: [
        rand::random_range(0.0..WIDTH as f32),
        rand::random_range(0.0..HEIGHT as f32),
      ],
      vel: [
        rand::random_range(-1.0..=1.0 as f32),
        rand::random_range(-1.0..=1.0 as f32),
      ],
    }
  }
}
