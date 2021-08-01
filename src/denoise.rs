use std::borrow::Cow;

use crate::{
    common, Complex, RnnModel, CEPS_MEM, FRAME_SIZE, FREQ_SIZE, NB_BANDS, NB_DELTA_CEPS,
    NB_FEATURES, PITCH_BUF_SIZE, WINDOW_SIZE,
};

/// This is the main entry-point into `nnnoiseless`. It mainly contains the various memory buffers
/// that are used while denoising. As such, this is quite a large struct, and should probably be
/// kept behind some kind of pointer.
///
/// # Example
///
/// ```rust
/// # use nnnoiseless::DenoiseState;
/// // One second of 440Hz sine wave at 48kHz sample rate. Note that the input data consists of
/// // `f32`s, but the values should be in the range of an `i16`.
/// let sine: Vec<_> = (0..48_000)
///     .map(|x| (x as f32 * 440.0 * 2.0 * std::f32::consts::PI / 48_000.0).sin() * i16::MAX as f32)
///     .collect();
/// let mut output = Vec::new();
/// let mut out_buf = [0.0; DenoiseState::FRAME_SIZE];
/// let mut denoise = DenoiseState::new();
/// let mut first = true;
/// for chunk in sine.chunks_exact(DenoiseState::FRAME_SIZE) {
///     denoise.process_frame(&mut out_buf[..], chunk);
///
///     // We throw away the first output, as discussed in the documentation for
///     //`DenoiseState::process_frame`.
///     if !first {
///         output.extend_from_slice(&out_buf[..]);
///     }
///     first = false;
/// }
/// ```
#[derive(Clone)]
pub struct DenoiseState<'model> {
    /// This stores some of the previous input. Currently, whenever we get new input we shift this
    /// backwards and copy the new input at the end. It might be worth investigating a ring buffer.
    input_mem: Vec<f32>,
    /// This is some sort of ring buffer, storing the last bunch of cepstra.
    cepstral_mem: [[f32; crate::NB_BANDS]; crate::CEPS_MEM],
    /// The index pointing to the most recent cepstrum in `cepstral_mem`. The previous cepstra are
    /// at indices mem_id - 1, mem_id - 1, etc (wrapped appropriately).
    mem_id: usize,
    synthesis_mem: [f32; FRAME_SIZE],
    mem_hp_x: [f32; 2],
    lastg: [f32; crate::NB_BANDS],
    rnn: crate::rnn::RnnState<'model>,
    fft: crate::fft::RealFft,

    pitch_finder: crate::pitch::PitchFinder,
}

impl DenoiseState<'static> {
    /// A `DenoiseState` processes this many samples at a time.
    pub const FRAME_SIZE: usize = FRAME_SIZE;

    pub(crate) fn default() -> Self {
        DenoiseState::from_model_owned(Cow::Owned(RnnModel::default()))
    }

    /// Creates a new `DenoiseState`.
    pub fn new() -> Box<DenoiseState<'static>> {
        Box::new(Self::default())
    }

    /// Creates a new `DenoiseState` owning a custom model.
    ///
    /// The main difference between this method and `DenoiseState::with_model` is that here
    /// `DenoiseState` will own the model; this might be more convenient.
    pub fn from_model(model: RnnModel) -> Box<DenoiseState<'static>> {
        Box::new(DenoiseState::from_model_owned(Cow::Owned(model)))
    }
}

impl<'model> DenoiseState<'model> {
    /// Creates a new `DenoiseState` using a custom model.
    ///
    /// The main difference between this method and `DenoiseState::from_model` is that here
    /// `DenoiseState` will borrow the model; this might create some lifetime-related pain, but
    /// it means that the same model can be shared between multiple `DenoiseState`s.
    pub fn with_model(model: &'model RnnModel) -> Box<DenoiseState<'model>> {
        Box::new(DenoiseState::from_model_owned(Cow::Borrowed(model)))
    }

    pub(crate) fn from_model_owned(model: Cow<'model, RnnModel>) -> DenoiseState<'model> {
        DenoiseState {
            input_mem: vec![0.0; FRAME_SIZE.max(PITCH_BUF_SIZE)],
            cepstral_mem: [[0.0; NB_BANDS]; CEPS_MEM],
            mem_id: 0,
            synthesis_mem: [0.0; FRAME_SIZE],
            mem_hp_x: [0.0; 2],
            lastg: [0.0; NB_BANDS],
            fft: crate::fft::RealFft::new(crate::sin_cos_table()),
            rnn: crate::rnn::RnnState::new(model),
            pitch_finder: crate::pitch::PitchFinder::new(),
        }
    }

    // Returns the most recent chunk of input from our internal buffer.
    fn input(&self, len: usize) -> &[f32] {
        &self.input_mem[self.input_mem.len().checked_sub(len).unwrap()..]
    }

    fn find_pitch(&mut self) -> usize {
        let input = &self.input_mem[self.input_mem.len().checked_sub(PITCH_BUF_SIZE).unwrap()..];
        let (pitch, _gain) = self.pitch_finder.process(input);
        pitch
    }

    /// Performs an FFT on `input`, putting the result in `output`.
    fn forward_transform(&mut self, output: &mut [Complex], input: &mut [f32]) {
        self.fft.forward(input, output);

        // In the original RNNoise code, the forward transform is normalized and the inverse
        // tranform isn't. `rustfft` doesn't normalize either one, so we do it ourselves.
        let norm = common().wnorm;
        for x in &mut output[..] {
            *x *= norm;
        }
    }

    /// Fourier transforms the input, after looking back `lag` samples.
    ///
    /// The Fourier transform goes in `x` and the band energies go in `ex`.
    fn transform_input(&mut self, lag: usize, x: &mut [Complex], ex: &mut [f32]) {
        let mut buf = [0.0; WINDOW_SIZE];
        crate::apply_window(&mut buf[..], self.input(WINDOW_SIZE + lag));
        self.forward_transform(x, &mut buf[..]);
        crate::compute_band_corr(ex, x, x);
    }

    /// Performs an inverse FFT on `input`, putting the result in `output`.
    fn inverse_transform(&mut self, output: &mut [f32], input: &mut [Complex]) {
        self.fft.inverse(input, output);
    }

    /// Processes a chunk of samples.
    ///
    /// Both `output` and `input` should be slices of length `DenoiseState::FRAME_SIZE`, and they
    /// are assumed to be in 16-bit, 48kHz signed PCM format. Note that although the input and
    /// output are `f32`s, they are supposed to come from 16-bit integers. In particular, they
    /// should be in the range `[-32768.0, 32767.0]` instead of the range `[-1.0, 1.0]` which
    /// is more common for floating-point PCM.
    ///
    /// The current output of `process_frame` depends on the current input, but also on the
    /// preceding inputs. Because of this, you might prefer to discard the very first output; it
    /// will contain some fade-in artifacts.
    pub fn process_frame(&mut self, output: &mut [f32], input: &[f32]) -> f32 {
        process_frame(self, output, input)
    }
}

/// Computes the features of the current frame.
///
/// - `x` is the Fourier transform of the input, and `ex` are its band energies
/// - `p` is the Fourier transform of older input, with a lag of the pitch period; `ep` are its band
///     energies
/// - `exp` is the band correlation between `x` and `p`
/// - `features` are all the features of that get input to the neural network.
///
/// The return value is `true` if the input was pretty much silent.
fn compute_frame_features(
    state: &mut DenoiseState,
    x: &mut [Complex],
    p: &mut [Complex],
    ex: &mut [f32],
    ep: &mut [f32],
    exp: &mut [f32],
    features: &mut [f32],
) -> bool {
    let mut ly = [0.0; NB_BANDS];
    let mut tmp = [0.0; NB_BANDS];

    state.transform_input(0, x, ex);

    let pitch_idx = state.find_pitch();

    state.transform_input(pitch_idx, p, ep);
    crate::compute_band_corr(exp, x, p);
    for i in 0..NB_BANDS {
        exp[i] /= (0.001 + ex[i] * ep[i]).sqrt();
    }
    crate::dct(&mut tmp[..], exp);
    for i in 0..NB_DELTA_CEPS {
        features[NB_BANDS + 2 * NB_DELTA_CEPS + i] = tmp[i];
    }

    features[NB_BANDS + 2 * NB_DELTA_CEPS] -= 1.3;
    features[NB_BANDS + 2 * NB_DELTA_CEPS + 1] -= 0.9;
    features[NB_BANDS + 3 * NB_DELTA_CEPS] = 0.01 * (pitch_idx as f32 - 300.0);
    let mut log_max = -2.0;
    let mut follow = -2.0;
    let mut e = 0.0;
    for i in 0..NB_BANDS {
        ly[i] = (1e-2 + ex[i]).log10().max(log_max - 7.0).max(follow - 1.5);
        log_max = log_max.max(ly[i]);
        follow = (follow - 1.5).max(ly[i]);
        e += ex[i];
    }

    if e < 0.04 {
        /* If there's no audio, avoid messing up the state. */
        for i in 0..NB_FEATURES {
            features[i] = 0.0;
        }
        return true;
    }
    crate::dct(features, &ly[..]);
    features[0] -= 12.0;
    features[1] -= 4.0;
    let ceps_0_idx = state.mem_id;
    let ceps_1_idx = if state.mem_id < 1 {
        CEPS_MEM + state.mem_id - 1
    } else {
        state.mem_id - 1
    };
    let ceps_2_idx = if state.mem_id < 2 {
        CEPS_MEM + state.mem_id - 2
    } else {
        state.mem_id - 2
    };

    for i in 0..NB_BANDS {
        state.cepstral_mem[ceps_0_idx][i] = features[i];
    }
    state.mem_id += 1;

    let ceps_0 = &state.cepstral_mem[ceps_0_idx];
    let ceps_1 = &state.cepstral_mem[ceps_1_idx];
    let ceps_2 = &state.cepstral_mem[ceps_2_idx];
    for i in 0..NB_DELTA_CEPS {
        features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];
        features[NB_BANDS + i] = ceps_0[i] - ceps_2[i];
        features[NB_BANDS + NB_DELTA_CEPS + i] = ceps_0[i] - 2.0 * ceps_1[i] + ceps_2[i];
    }

    /* Spectral variability features. */
    let mut spec_variability = 0.0;
    if state.mem_id == CEPS_MEM {
        state.mem_id = 0;
    }
    for i in 0..CEPS_MEM {
        let mut min_dist = 1e15f32;
        for j in 0..CEPS_MEM {
            let mut dist = 0.0;
            for k in 0..NB_BANDS {
                let tmp = state.cepstral_mem[i][k] - state.cepstral_mem[j][k];
                dist += tmp * tmp;
            }
            if j != i {
                min_dist = min_dist.min(dist);
            }
        }
        spec_variability += min_dist;
    }

    features[NB_BANDS + 3 * NB_DELTA_CEPS + 1] = spec_variability / CEPS_MEM as f32 - 2.1;

    false
}

fn frame_synthesis(state: &mut DenoiseState, out: &mut [f32], y: &mut [Complex]) {
    let mut x = [0.0; WINDOW_SIZE];
    state.inverse_transform(&mut x[..], y);
    crate::apply_window_in_place(&mut x[..]);
    for i in 0..FRAME_SIZE {
        out[i] = x[i] + state.synthesis_mem[i];
        state.synthesis_mem[i] = x[FRAME_SIZE + i];
    }
}

fn pitch_filter(x: &mut [Complex], p: &[Complex], ex: &[f32], ep: &[f32], exp: &[f32], g: &[f32]) {
    let mut r = [0.0; NB_BANDS];
    let mut rf = [0.0; FREQ_SIZE];
    for i in 0..NB_BANDS {
        r[i] = if exp[i] > g[i] {
            1.0
        } else {
            let exp_sq = exp[i] * exp[i];
            let g_sq = g[i] * g[i];
            exp_sq * (1.0 - g_sq) / (0.001 + g_sq * (1.0 - exp_sq))
        };
        r[i] = 1.0_f32.min(0.0_f32.max(r[i])).sqrt();
        r[i] *= (ex[i] / (1e-8 + ep[i])).sqrt();
    }
    crate::interp_band_gain(&mut rf[..], &r[..]);
    for i in 0..FREQ_SIZE {
        x[i] += rf[i] * p[i];
    }

    let mut new_e = [0.0; NB_BANDS];
    crate::compute_band_corr(&mut new_e[..], x, x);
    let mut norm = [0.0; NB_BANDS];
    let mut normf = [0.0; FREQ_SIZE];
    for i in 0..NB_BANDS {
        norm[i] = (ex[i] / (1e-8 + new_e[i])).sqrt();
    }
    crate::interp_band_gain(&mut normf[..], &norm[..]);
    for i in 0..FREQ_SIZE {
        x[i] *= normf[i];
    }
}

fn process_frame(state: &mut DenoiseState, output: &mut [f32], input: &[f32]) -> f32 {
    let mut x_freq = [Complex::from(0.0); FREQ_SIZE];
    let mut p = [Complex::from(0.0); FREQ_SIZE];
    let mut ex = [0.0; NB_BANDS];
    let mut ep = [0.0; NB_BANDS];
    let mut exp = [0.0; NB_BANDS];
    let mut features = [0.0; NB_FEATURES];
    let mut g = [0.0; NB_BANDS];
    let mut gf = [1.0; FREQ_SIZE];
    let mut vad_prob = [0.0];

    // Shift our internal input buffer and copy the (filtered) input into it.
    let new_idx = state.input_mem.len() - FRAME_SIZE;
    for i in 0..new_idx {
        state.input_mem[i] = state.input_mem[i + FRAME_SIZE];
    }
    crate::biquad::BIQUAD_HP.filter(&mut state.input_mem[new_idx..], &mut state.mem_hp_x, input);
    let silence = compute_frame_features(
        state,
        &mut x_freq[..],
        &mut p[..],
        &mut ex[..],
        &mut ep[..],
        &mut exp[..],
        &mut features[..],
    );
    if !silence {
        state
            .rnn
            .compute(&mut g[..], &mut vad_prob[..], &features[..]);
        pitch_filter(&mut x_freq[..], &p[..], &ex[..], &ep[..], &exp[..], &g[..]);
        for i in 0..NB_BANDS {
            g[i] = g[i].max(0.6 * state.lastg[i]);
            state.lastg[i] = g[i];
        }
        crate::interp_band_gain(&mut gf[..], &g[..]);
        for i in 0..FREQ_SIZE {
            x_freq[i] *= gf[i];
        }
    }

    frame_synthesis(state, output, &mut x_freq[..]);
    vad_prob[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate static_assertions as sa;

    sa::assert_impl_all!(DenoiseState: Send, Sync);
}
