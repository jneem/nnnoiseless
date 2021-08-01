pub(crate) const BIQUAD_HP: Biquad = Biquad {
    a: [-1.99599, 0.99600],
    b: [-2.0, 1.0],
};

pub(crate) struct Biquad {
    pub a: [f32; 2],
    pub b: [f32; 2],
}

impl Biquad {
    pub fn filter(&self, output: &mut [f32], mem: &mut [f32; 2], input: &[f32]) {
        let a0 = self.a[0] as f64;
        let a1 = self.a[1] as f64;
        let b0 = self.b[0] as f64;
        let b1 = self.b[1] as f64;
        for (&x, y) in input.iter().zip(output) {
            let x64 = x as f64;
            let y64 = x64 + mem[0] as f64;
            mem[0] = (mem[1] as f64 + (b0 * x64 - a0 * y64)) as f32;
            mem[1] = (b1 * x64 - a1 * y64) as f32;
            *y = y64 as f32;
        }
    }
}
