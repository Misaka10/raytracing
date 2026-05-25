use std::cell::RefCell;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

thread_local! {
    static THREAD_RNG: RefCell<SmallRng> = RefCell::new(
        SmallRng::from_entropy()
    );
}

pub fn with_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut SmallRng) -> R,
{
    THREAD_RNG.with(|rng| f(&mut rng.borrow_mut()))
}

pub fn random_f64() -> f64 {
    with_rng(|rng| rng.gen())
}

pub fn random_f64_range(min: f64, max: f64) -> f64 {
    with_rng(|rng| rng.gen_range(min..max))
}
