use criterion::{criterion_group, criterion_main, Criterion};

fn sin(freq: f32, secs: f32) -> Vec<f32> {
    (0..((secs * 48_000.0) as usize))
        .map(|x| (x as f32 * freq * 2.0 * std::f32::consts::PI / 48_000.0).sin() * i16::MAX as f32)
        .collect()
}

pub fn bench_sin(c: &mut Criterion) {
    let input = sin(440.0, 1.0);
    let mut output = [0.0; nnnoiseless::DenoiseState::FRAME_SIZE];
    c.bench_function("nnnoiseless sin/440/1", |b| {
        b.iter(|| {
            let mut state = nnnoiseless::DenoiseState::new();
            for chunk in input.chunks_exact(nnnoiseless::DenoiseState::FRAME_SIZE) {
                state.process_frame(&mut output[..], chunk);
            }
        })
    });
}

criterion_group!(benches, bench_sin,);
criterion_main!(benches);
