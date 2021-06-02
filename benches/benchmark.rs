use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn sin_benchmark(c: &mut Criterion) {
    c.bench_function("sin(0.1)", |b| b.iter(|| black_box(0.1f32).sin()));
}

pub fn tan_benchmark(c: &mut Criterion) {
    c.bench_function("tan(0.1)", |b| b.iter(|| black_box(0.1f32).tan()));
}

pub fn div_benchmark(c: &mut Criterion) {
    c.bench_function("2.0 / 0.1", |b| b.iter(|| black_box(2.0f32) / 0.1));
}

criterion_group!(benches, sin_benchmark, tan_benchmark, div_benchmark);
criterion_main!(benches);
