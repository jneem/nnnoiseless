// Copyright 2020 Henrik Enquist
// Copyright 2020 Joe Neeman
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// This file was derived from the `realfft` crate at https://github.com/HEnquist/realfft
// Long-term, this will hopefully make it into `rustfft`.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::util::zip4;

#[derive(Clone)]
pub struct RealFft {
    sin_cos: &'static [(f32, f32)],
    length: usize,
    scratch: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    forward: std::sync::Arc<dyn rustfft::Fft<f32>>,
    inverse: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

pub fn precompute_sin_cos_table(sin_cos: &mut [(f32, f32)]) {
    let pi = std::f64::consts::PI;
    let len = sin_cos.len() as f64;
    for (k, (ref mut sin, ref mut cos)) in sin_cos.iter_mut().enumerate() {
        *sin = (k as f64 * pi / len).sin() as f32;
        *cos = (k as f64 * pi / len).cos() as f32;
    }
}

impl RealFft {
    pub fn new(sin_cos_table: &'static [(f32, f32)]) -> RealFft {
        let length = sin_cos_table.len() * 2;
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(length / 2);
        let inverse = planner.plan_fft_inverse(length / 2);
        let scratch = vec![0.0.into(); length / 2 + 1];
        let fft_scratch_len = forward
            .get_outofplace_scratch_len()
            .max(inverse.get_outofplace_scratch_len());
        let fft_scratch = vec![0.0.into(); fft_scratch_len];
        RealFft {
            sin_cos: sin_cos_table,
            length,
            scratch,
            fft_scratch,
            forward,
            inverse,
        }
    }

    /// Transform a vector of 2*N real-valued samples, storing the result in the N+1 element long complex output vector.
    /// The input buffer is used as scratch space, so the contents of input should be considered garbage after calling.
    pub fn forward(&mut self, input: &mut [f32], output: &mut [Complex<f32>]) {
        assert_eq!(input.len(), self.length);
        assert_eq!(output.len(), self.length / 2 + 1);

        let scratch = &mut self.scratch;
        let fft_scratch = &mut self.fft_scratch;

        let fftlen = self.length / 2;

        let mut buf_in = unsafe {
            let ptr = input.as_mut_ptr() as *mut Complex<f32>;
            let len = input.len();
            std::slice::from_raw_parts_mut(ptr, len / 2)
        };

        self.forward.process_outofplace_with_scratch(
            &mut buf_in,
            &mut scratch[0..fftlen],
            &mut fft_scratch[..],
        );

        scratch[fftlen] = scratch[0];

        for (&buf, &buf_rev, &(sin, cos), out) in zip4(
            &scratch[..],
            scratch.iter().rev(),
            self.sin_cos,
            &mut output[..],
        ) {
            let xr = 0.5
                * ((buf.re + buf_rev.re) + cos * (buf.im + buf_rev.im)
                    - sin * (buf.re - buf_rev.re));
            let xi = 0.5
                * ((buf.im - buf_rev.im)
                    - sin * (buf.im + buf_rev.im)
                    - cos * (buf.re - buf_rev.re));
            *out = Complex::new(xr, xi);
        }
        output[fftlen] = Complex::new(scratch[0].re - scratch[0].im, 0.0);
    }

    /// Transform a complex spectrum of N+1 values and store the real result in the 2*N long output.
    pub fn inverse(&mut self, input: &[Complex<f32>], output: &mut [f32]) {
        assert_eq!(input.len(), self.length / 2 + 1);
        assert_eq!(output.len(), self.length);
        let fftlen = self.length / 2;
        let scratch = &mut self.scratch;
        let fft_scratch = &mut self.fft_scratch;

        for (&buf, &buf_rev, &(sin, cos), fft_input) in
            zip4(input, input.iter().rev(), self.sin_cos, &mut scratch[..])
        {
            let xr = 0.5
                * ((buf.re + buf_rev.re)
                    - cos * (buf.im + buf_rev.im)
                    - sin * (buf.re - buf_rev.re));
            let xi = 0.5
                * ((buf.im - buf_rev.im) + cos * (buf.re - buf_rev.re)
                    - sin * (buf.im + buf_rev.im));
            *fft_input = Complex::new(xr, xi);
        }

        // FFT and store result in buffer_out
        let mut buf_out = unsafe {
            let ptr = output.as_mut_ptr() as *mut Complex<f32>;
            let len = output.len();
            std::slice::from_raw_parts_mut(ptr, len / 2)
        };
        self.inverse.process_outofplace_with_scratch(
            &mut scratch[..fftlen],
            &mut buf_out,
            &mut fft_scratch[..],
        );
    }
}
