use crate::{PITCH_BUF_SIZE, PITCH_FRAME_SIZE, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD};

#[derive(Clone)]
pub(crate) struct PitchFinder {
    last_period: usize,
    last_gain: f32,
    // A buffer of size PITCH_BUF_SIZE / 2.
    pitch_buf: Vec<f32>,
    // Scratch buffer of size PITCH_MAX_PERIOD + 1. We'll also use it for a scratch buffer of size
    // PITCH_FRAME_SIZE  / 4 + (PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD) / 4,
    // which is smaller.
    scratch: Vec<f32>,
    // Scratch buffer of size PITCH_FRAME_SIZE / 4.
    scratch2: Vec<f32>,
    // Scratch buffer of length (PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD) / 2.
    scratch3: Vec<f32>,
}

impl PitchFinder {
    pub(crate) fn new() -> PitchFinder {
        assert!(
            PITCH_MAX_PERIOD + 1
                >= PITCH_FRAME_SIZE / 4 + (PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD) / 4
        );

        let pitch_buf = vec![0.0; PITCH_BUF_SIZE / 2];
        let scratch = vec![0.0; PITCH_MAX_PERIOD + 1];
        let scratch2 = vec![0.0; PITCH_FRAME_SIZE / 4];
        let scratch3 = vec![0.0; (PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD) / 2];
        PitchFinder {
            last_period: 0,
            last_gain: 0.0,
            pitch_buf,
            scratch,
            scratch2,
            scratch3,
        }
    }

    /// Finds the main pitch of an audio signal, and also something gain something something.
    ///
    /// `input` is a buffer of size `PITCH_BUF_SIZE`
    ///
    /// Returns the period of the detected pitch, and the detected gain.
    pub(crate) fn process(&mut self, input: &[f32]) -> (usize, f32) {
        pitch_downsample(input, &mut self.pitch_buf);

        let pitch_idx = self.pitch_search();
        let pitch_idx = PITCH_MAX_PERIOD - pitch_idx;
        let (period, gain) = self.remove_doubling(pitch_idx);
        self.last_period = period;
        self.last_gain = gain;
        (period, gain)
    }

    // Finds the pitch of the signal in self.pitch_buf. Returns the period of the pitch.
    //
    // Note that the data in self.pitch_buf is already downsampled by a factor of 2. The return
    // value is the period of the *original* signal.
    //
    // Roughly speaking, this works by taking two adjacent frames of the input and finding the
    // offset with maximal cross-correlation.
    fn pitch_search(&mut self) -> usize {
        let x_lp = &self.pitch_buf[(PITCH_MAX_PERIOD / 2)..];
        let y = &self.pitch_buf[..];
        let len = PITCH_FRAME_SIZE;
        let max_pitch = PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD;
        let x_lp4 = &mut self.scratch2[..(len / 4)];
        let y_lp4 = &mut self.scratch[..(len / 4 + max_pitch / 4)];
        let xcorr = &mut self.scratch3[..];

        // The signal in `self.pitch_buf` was already downsampled by a factor of 2. Downsample it
        // again.
        for j in 0..x_lp4.len() {
            x_lp4[j] = x_lp[2 * j];
        }
        for j in 0..y_lp4.len() {
            y_lp4[j] = y[2 * j];
        }

        // Use brute-force for the 4x downsampled data.
        pitch_xcorr(&x_lp4, &y_lp4, &mut xcorr[0..(max_pitch / 4)]);
        let (best_pitch, second_best_pitch) =
            find_best_pitch(&xcorr[0..(max_pitch / 4)], &y_lp4, len / 4);

        // Do a finer search on the 2x downsampled data. We still do pitch_search by brute force,
        // but this time we only compute a few candidate values of the cross-correlation.
        for i in 0..(max_pitch / 2) {
            xcorr[i] = 0.0;
            if (i as isize - 2 * best_pitch as isize).abs() > 2
                && (i as isize - 2 * second_best_pitch as isize).abs() > 2
            {
                continue;
            }
            xcorr[i] = inner_prod(&x_lp[..], &y[i..], len / 2).max(-1.0);
        }
        let (best_pitch, _) = find_best_pitch(&xcorr, &y, len / 2);

        // Use pseudo-interpolation to get the final pitch for the original signal.
        let offset: isize = if best_pitch > 0 && best_pitch < (max_pitch / 2) - 1 {
            let a = xcorr[best_pitch - 1];
            let b = xcorr[best_pitch];
            let c = xcorr[best_pitch + 1];
            if c - a > 0.7 * (b - a) {
                1
            } else if a - c > 0.7 * (b - c) {
                -1
            } else {
                0
            }
        } else {
            0
        };
        (2 * best_pitch as isize - offset) as usize
    }

    // TODO: document this.
    fn remove_doubling(&mut self, pitch_idx: usize) -> (usize, f32) {
        let x = &self.pitch_buf[..];

        // All these quantities get divided by 2 because we're working with downsampled data.
        let min_period = PITCH_MIN_PERIOD / 2;
        let max_period = PITCH_MAX_PERIOD / 2;
        let n = PITCH_FRAME_SIZE / 2;
        let t0 = (pitch_idx / 2).min(max_period - 1);
        let prev_period = self.last_period / 2;

        let yy_lookup = &mut self.scratch[..];
        let mut t = t0;

        // Note that because we can't index with negative numbers, the x in the C code is our
        // x[max_period..].
        let xx = inner_prod(&x[max_period..], &x[max_period..], n);
        let mut xy = inner_prod(&x[max_period..], &x[(max_period - t0)..], n);
        yy_lookup[0] = xx;

        let mut yy = xx;
        for i in 1..=max_period {
            yy += x[max_period - i] * x[max_period - i]
                - x[max_period + n - i] * x[max_period + n - i];
            yy_lookup[i] = yy.max(0.0);
        }

        yy = yy_lookup[t0];
        let mut best_xy = xy;
        let mut best_yy = yy;

        let g0 = pitch_gain(xy, xx, yy);
        let mut g = g0;

        // Look for any pitch at T/k */
        for k in 2..=15 {
            let t1 = (2 * t0 + k) / (2 * k);
            if t1 < min_period {
                break;
            }
            // Look for another strong correlation at t1b
            let t1b = if k == 2 {
                if t1 + t0 > max_period {
                    t0
                } else {
                    t0 + t1
                }
            } else {
                (2 * SECOND_CHECK[k] * t0 + k) / (2 * k)
            };
            xy = inner_prod(&x[max_period..], &x[(max_period - t1)..], n);
            let xy2 = inner_prod(&x[max_period..], &x[(max_period - t1b)..], n);
            xy = (xy + xy2) / 2.0;
            yy = (yy_lookup[t1] + yy_lookup[t1b]) / 2.0;

            let g1 = pitch_gain(xy, xx, yy);
            let cont = if (t1 as isize - prev_period as isize).abs() <= 1 {
                self.last_gain
            } else if (t1 as isize - prev_period as isize).abs() <= 2 && 5 * k * k < t0 {
                self.last_gain / 2.0
            } else {
                0.0
            };

            // Bias against very high pitch (very short period) to avoid false-positives due to
            // short-term correlation.
            let thresh = if t1 < 3 * min_period {
                (0.85 * g0 - cont).max(0.4)
            } else if t1 < 2 * min_period {
                (0.9 * g0 - cont).max(0.5)
            } else {
                (0.7 * g0 - cont).max(0.3)
            };
            if g1 > thresh {
                best_xy = xy;
                best_yy = yy;
                t = t1;
                g = g1;
            }
        }

        let best_xy = best_xy.max(0.0);
        let pg = if best_yy <= best_xy {
            1.0
        } else {
            best_xy / (best_yy + 1.0)
        };

        let mut xcorr = [0.0; 3];
        for k in 0..3 {
            xcorr[k] = inner_prod(&x[max_period..], &x[(max_period - (t + k - 1))..], n);
        }
        let offset: isize = if xcorr[2] - xcorr[0] > 0.7 * (xcorr[1] - xcorr[0]) {
            1
        } else if xcorr[0] - xcorr[2] > 0.7 * (xcorr[1] - xcorr[2]) {
            -1
        } else {
            0
        };

        let pg = pg.min(g);
        let t0 = (2 * t).wrapping_add(offset as usize).max(PITCH_MIN_PERIOD);

        (t0, pg)
    }
}

/// Computes the inner product of `xs[..n]` and `ys[..n]`.
fn inner_prod(xs: &[f32], ys: &[f32], n: usize) -> f32 {
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;

    let n_4 = n - n % 4;
    for (x, y) in xs[..n_4].chunks_exact(4).zip(ys[..n_4].chunks_exact(4)) {
        sum0 += x[0] * y[0];
        sum1 += x[1] * y[1];
        sum2 += x[2] * y[2];
        sum3 += x[3] * y[3];
    }

    let mut sum = sum0 + sum1 + sum2 + sum3;
    for (&x, &y) in xs[n_4..n].iter().zip(&ys[n_4..n]) {
        sum += x * y;
    }
    sum
}

/// Does linear predictive coding (LPC) for a signal. The LPC coefficients are put into `lpc`,
/// which should have the same length as `ac`.
///
/// Very quick summary, mostly for my own understanding: the idea of LPC is to approximate a signal
/// x by shifted versions of it, so x[t] is approximately $\sum_i a_i x_{t-i}$, where the $a_i$ are
/// the LPC coefficients. This function determines the LPC coefficients using linear regression,
/// where the main observation is that this only requires a few auto-correlations. Therefore, the
/// function takes the autocorrelations as a parameter instead of the original signal.
///
/// This function solves the linear regression iteratively by first solving the smaller versions
/// (i.e., first solve the linear regression for one lag, then for two lags, and so on).
fn lpc(lpc: &mut [f32], ac: &[f32]) {
    let p = lpc.len();
    let mut error = ac[0];

    for b in lpc.iter_mut() {
        *b = 0.0;
    }

    if ac[0] == 0.0 {
        return;
    }

    for i in 0..p {
        // Sum up this iteration's reflection coefficient
        let mut rr = 0.0;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];
        let r = -rr / error;
        // Update LPC coefficients and total error
        lpc[i] = r;
        for j in 0..((i + 1) / 2) {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            lpc[j] = tmp1 + r * tmp2;
            lpc[i - 1 - j] = tmp2 + r * tmp1;
        }

        error = error - r * r * error;
        // Bail out once we get 30 dB gain
        if error < 0.001 * ac[0] {
            return;
        }
    }
}

/// Computes various terms of the cross-correlation between x and y (the number of terms to compute
/// is determined by the size of `xcorr`).
fn pitch_xcorr(xs: &[f32], ys: &[f32], xcorr: &mut [f32]) {
    // The un-optimized version of this function is:
    //
    // for i in 0..xcorr.len() {
    //    xcorr[i] = xs.iter().zip(&ys[i..]).map(|(&x, &y)| x * y).sum();
    // }
    //
    // To optimize it, we unroll both the outer and inner loops four times each. This is a huge win
    // because it improves the pattern of access to ys. The compiler does a good job of vectorizing
    // the inner loop. (Maybe if we unrolled 8 times, it would be better on AVX?)

    let xcorr_len_4 = xcorr.len() - xcorr.len() % 4;
    let xs_len_4 = xs.len() - xs.len() % 4;

    for i in (0..xcorr_len_4).step_by(4) {
        let mut c0 = 0.0;
        let mut c1 = 0.0;
        let mut c2 = 0.0;
        let mut c3 = 0.0;

        let mut y0 = ys[i + 0];
        let mut y1 = ys[i + 1];
        let mut y2 = ys[i + 2];
        let mut y3 = ys[i + 3];

        for (x, y) in xs.chunks_exact(4).zip(ys[(i + 4)..].chunks_exact(4)) {
            c0 += x[0] * y0;
            c1 += x[0] * y1;
            c2 += x[0] * y2;
            c3 += x[0] * y3;

            y0 = y[0];
            c0 += x[1] * y1;
            c1 += x[1] * y2;
            c2 += x[1] * y3;
            c3 += x[1] * y0;

            y1 = y[1];
            c0 += x[2] * y2;
            c1 += x[2] * y3;
            c2 += x[2] * y0;
            c3 += x[2] * y1;

            y2 = y[2];
            c0 += x[3] * y3;
            c1 += x[3] * y0;
            c2 += x[3] * y1;
            c3 += x[3] * y2;

            y3 = y[3];
        }

        for j in xs_len_4..xs.len() {
            c0 += xs[j] * ys[i + 0 + j];
            c1 += xs[j] * ys[i + 1 + j];
            c2 += xs[j] * ys[i + 2 + j];
            c3 += xs[j] * ys[i + 3 + j];
        }
        xcorr[i + 0] = c0;
        xcorr[i + 1] = c1;
        xcorr[i + 2] = c2;
        xcorr[i + 3] = c3;
    }

    for i in xcorr_len_4..xcorr.len() {
        xcorr[i] = xs.iter().zip(&ys[i..]).map(|(&x, &y)| x * y).sum();
    }
}

/// Returns the indices with the largest and second-largest normalized auto-correlation.
///
/// `xcorr` is the autocorrelation of `ys`, taken with windows of length `len`.
///
/// To be a little more precise, the function that we're maximizing is xcorr[i] * xcorr[i],
/// divided by the squared norm of ys[i..(i+len)] (but with a bit of fudging to avoid dividing
/// by small things).
fn find_best_pitch(xcorr: &[f32], ys: &[f32], len: usize) -> (usize, usize) {
    let mut best_num = -1.0;
    let mut second_best_num = -1.0;
    let mut best_den = 0.0;
    let mut second_best_den = 0.0;
    let mut best_pitch = 0;
    let mut second_best_pitch = 1;
    let mut y_sq_norm = 1.0;
    for y in &ys[0..len] {
        y_sq_norm += y * y;
    }
    for (i, &corr) in xcorr.iter().enumerate() {
        if corr > 0.0 {
            let num = corr * corr;
            if num * second_best_den > second_best_num * y_sq_norm {
                if num * best_den > best_num * y_sq_norm {
                    second_best_num = best_num;
                    second_best_den = best_den;
                    second_best_pitch = best_pitch;
                    best_num = num;
                    best_den = y_sq_norm;
                    best_pitch = i;
                } else {
                    second_best_num = num;
                    second_best_den = y_sq_norm;
                    second_best_pitch = i;
                }
            }
        }
        y_sq_norm += ys[i + len] * ys[i + len] - ys[i] * ys[i];
        y_sq_norm = y_sq_norm.max(1.0);
    }
    (best_pitch, second_best_pitch)
}

fn fir5_in_place(xs: &mut [f32], num: &[f32]) {
    let num0 = num[0];
    let num1 = num[1];
    let num2 = num[2];
    let num3 = num[3];
    let num4 = num[4];

    let mut mem0 = 0.0;
    let mut mem1 = 0.0;
    let mut mem2 = 0.0;
    let mut mem3 = 0.0;
    let mut mem4 = 0.0;

    for x in xs {
        let out = *x + num0 * mem0 + num1 * mem1 + num2 * mem2 + num3 * mem3 + num4 * mem4;
        mem4 = mem3;
        mem3 = mem2;
        mem2 = mem1;
        mem1 = mem0;
        mem0 = *x;
        *x = out;
    }
}

/// Computes the autocorrelation of the sequence `x` (the number of terms to compute is determined
/// by the length of `ac`).
fn celt_autocorr(x: &[f32], ac: &mut [f32]) {
    let n = x.len();
    let lag = ac.len() - 1;
    let fast_n = n - lag;
    pitch_xcorr(&x[0..fast_n], x, ac);

    for k in 0..ac.len() {
        let mut d = 0.0;
        for i in (k + fast_n)..n {
            d += x[i] * x[i - k];
        }
        ac[k] += d;
    }
}

fn pitch_downsample(x: &[f32], x_lp: &mut [f32]) {
    let mut ac = [0.0; 5];
    let mut lpc_coeffs = [0.0; 4];
    let mut lpc_coeffs2 = [0.0; 5];

    // It would be nice to write this using `windows()`, but unfortunately `windows(3).step_by(2)`
    // produces nasty assembly. Just as well this isn't a hot loop...
    for i in 1..(x.len() / 2) {
        x_lp[i] = ((x[2 * i - 1] + x[2 * i + 1]) / 2.0 + x[2 * i]) / 2.0;
    }
    x_lp[0] = (x[1] / 2.0 + x[0]) / 2.0;

    celt_autocorr(x_lp, &mut ac);

    // Noise floor -40 dB
    ac[0] *= 1.0001;
    // Lag windowing
    for i in 1..5 {
        ac[i] -= ac[i] * (0.008 * i as f32) * (0.008 * i as f32);
    }

    lpc(&mut lpc_coeffs, &ac);
    let mut tmp = 1.0;
    for i in 0..4 {
        tmp *= 0.9;
        lpc_coeffs[i] *= tmp;
    }
    // Add a zero
    lpc_coeffs2[0] = lpc_coeffs[0] + 0.8;
    lpc_coeffs2[1] = lpc_coeffs[1] + 0.8 * lpc_coeffs[0];
    lpc_coeffs2[2] = lpc_coeffs[2] + 0.8 * lpc_coeffs[1];
    lpc_coeffs2[3] = lpc_coeffs[3] + 0.8 * lpc_coeffs[2];
    lpc_coeffs2[4] = 0.8 * lpc_coeffs[3];

    fir5_in_place(x_lp, &lpc_coeffs2);
}

fn pitch_gain(xy: f32, xx: f32, yy: f32) -> f32 {
    xy / (1.0 + xx * yy).sqrt()
}

const SECOND_CHECK: [usize; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];
