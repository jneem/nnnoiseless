use std::borrow::Cow;

use crate::{RnnModel, FRAME_SIZE, FREQ_SIZE, NB_BANDS};

/// This is the low-level entry-point into `nnnoiseless`: by using the `DenoiseState` directly,
/// you can denoise your audio while keeping copying to a minimum. For a higher-level
/// denoising experience, try [`DenoiseSignal`](crate::DenoiseSignal).
///
/// This struct directly contains various memory buffers that are used while denoising. As such,
/// this is quite a large struct, and should probably be kept behind some kind of pointer.
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
    /// Most recent gains that we applied.
    lastg: [f32; crate::NB_BANDS],
    rnn: crate::rnn::RnnState<'model>,
    feat: crate::features::DenoiseFeatures,
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
            lastg: [0.0; NB_BANDS],
            rnn: crate::rnn::RnnState::new(model),
            feat: crate::features::DenoiseFeatures::new(),
        }
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
        let mut g = [0.0; NB_BANDS];
        let mut gf = [1.0; FREQ_SIZE];
        let mut vad_prob = [0.0];

        self.feat.shift_and_filter_input(input);
        let silence = self.feat.compute_frame_features();
        if !silence {
            self.rnn
                .compute(&mut g[..], &mut vad_prob[..], self.feat.features());
            self.feat.pitch_filter(&g);
            for i in 0..NB_BANDS {
                g[i] = g[i].max(0.6 * self.lastg[i]);
                self.lastg[i] = g[i];
            }
            crate::interp_band_gain(&mut gf[..], &g[..]);
            self.feat.apply_gain(&gf);
        }

        self.feat.frame_synthesis(output);
        vad_prob[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate static_assertions as sa;

    sa::assert_impl_all!(DenoiseState: Send, Sync);
}
