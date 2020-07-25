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
use rustfft::FFTplanner;

pub struct RealFft {
    sin_cos: Vec<(f32, f32)>,
    length: usize,
    forward: std::sync::Arc<dyn rustfft::FFT<f32>>,
    inverse: std::sync::Arc<dyn rustfft::FFT<f32>>,
}

fn zip4<A, B, C, D>(
    a: A,
    b: B,
    c: C,
    d: D,
) -> impl Iterator<Item = (A::Item, B::Item, C::Item, D::Item)>
where
    A: IntoIterator,
    B: IntoIterator,
    C: IntoIterator,
    D: IntoIterator,
{
    a.into_iter()
        .zip(b.into_iter().zip(c.into_iter().zip(d)))
        .map(|(w, (x, (y, z)))| (w, x, y, z))
}

impl RealFft {
    pub fn new(length: usize) -> RealFft {
        assert!(length % 2 == 0);
        let mut sin_cos = Vec::with_capacity(length / 2);
        let pi = std::f64::consts::PI;
        for k in 0..length / 2 {
            let sin = (k as f64 * pi / (length / 2) as f64).sin() as f32;
            let cos = (k as f64 * pi / (length / 2) as f64).cos() as f32;
            sin_cos.push((sin, cos));
        }
        let mut forward_planner = FFTplanner::<f32>::new(false);
        let mut inverse_planner = FFTplanner::<f32>::new(true);
        let forward = forward_planner.plan_fft(length / 2);
        let inverse = inverse_planner.plan_fft(length / 2);
        RealFft {
            sin_cos,
            length,
            forward,
            inverse,
        }
    }

    /// Transform a vector of 2*N real-valued samples, storing the result in the N+1 element long complex output vector.
    /// The input buffer is used as scratch space, so the contents of input should be considered garbage after calling.
    ///
    /// `scratch` is a buffer of size length / 2 + 1
    pub fn forward(
        &self,
        input: &mut [f32],
        output: &mut [Complex<f32>],
        scratch: &mut [Complex<f32>],
    ) {
        assert_eq!(input.len(), self.length);
        assert_eq!(output.len(), self.length / 2 + 1);
        assert_eq!(scratch.len(), self.length / 2 + 1);

        let fftlen = self.length / 2;

        let mut buf_in = unsafe {
            let ptr = input.as_mut_ptr() as *mut Complex<f32>;
            let len = input.len();
            std::slice::from_raw_parts_mut(ptr, len / 2)
        };

        // FFT and store result in buffer_out
        self.forward.process(&mut buf_in, &mut scratch[0..fftlen]);

        scratch[fftlen] = scratch[0];

        for (&buf, &buf_rev, &(sin, cos), out) in zip4(
            &scratch[..],
            scratch.iter().rev(),
            &self.sin_cos,
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
    pub fn inverse(
        &self,
        input: &[Complex<f32>],
        output: &mut [f32],
        scratch: &mut [Complex<f32>],
    ) {
        assert_eq!(input.len(), self.length / 2 + 1);
        assert_eq!(output.len(), self.length);
        assert_eq!(scratch.len(), self.length / 2);

        for (&buf, &buf_rev, &(sin, cos), fft_input) in
            zip4(input, input.iter().rev(), &self.sin_cos, &mut scratch[..])
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
        self.inverse.process(scratch, &mut buf_out);
    }
}
