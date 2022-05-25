//! An adaptation of our denoising to dasp Signals.

use dasp::frame::Frame;
use dasp::sample::Sample;
use dasp::signal::Signal;
use std::borrow::Cow;

use crate::{DenoiseState, RnnModel, FRAME_SIZE};

/// Applies denoising to a `Signal` (from the `dasp` crate).
///
/// Instantiate one of these with a `Signal` as input, and you'll get a `Signal` that yields
/// denoised audio. Note that the denoised `Signal` will be in floating-point, even if the
/// original signal wasn't.
///
/// # Example
/// ```rust
/// use nnnoiseless::dasp::signal::{self, Signal};
/// use nnnoiseless::dasp::sample::Sample;
/// use nnnoiseless::DenoiseSignal;
///
/// let noise = signal::noise(0);
/// let mut denoised = DenoiseSignal::new(noise);
/// for n in denoised.take(10_000) {
/// // ... do something with your denoised noise.
/// }
/// ```
#[derive(Clone)]
pub struct DenoiseSignal<'model, S: Signal> {
    input: S,
    states: Vec<DenoiseState<'model>>,
    in_bufs: Vec<[f32; FRAME_SIZE]>,
    out_bufs: Vec<[f32; FRAME_SIZE]>,
    out_idx: usize,
}

impl<'model, S: Signal> DenoiseSignal<'model, S> {
    /// Creates a new `DenoiseSignal` using the built-in default noise model.
    pub fn new(input: S) -> DenoiseSignal<'static, S> {
        DenoiseSignal {
            input,
            states: vec![DenoiseState::default(); S::Frame::CHANNELS],
            in_bufs: vec![[0.0; FRAME_SIZE]; S::Frame::CHANNELS],
            out_bufs: vec![[0.0; FRAME_SIZE]; S::Frame::CHANNELS],
            out_idx: 0,
        }
        .discard_first_frame()
    }

    /// Creates a new `DenoiseSignal` using a custom noise model.
    ///
    /// The main difference between this method and `DenoiseSignal::from_model` is that here
    /// `DenoiseSignal` will borrow the model and reuse it for the different channels in the
    /// signal.
    pub fn with_model(input: S, model: &'model RnnModel) -> DenoiseSignal<'model, S> {
        DenoiseSignal {
            input,
            states: vec![DenoiseState::from_model_owned(Cow::Borrowed(model)); S::Frame::CHANNELS],
            in_bufs: vec![[0.0; FRAME_SIZE]; S::Frame::CHANNELS],
            out_bufs: vec![[0.0; FRAME_SIZE]; S::Frame::CHANNELS],
            out_idx: 0,
        }
        .discard_first_frame()
    }

    /// Creates a new `DenoiseSignal` owning a custom noise model.
    ///
    /// The main difference between this method and `DenoiseSignal::with_model` is that here
    /// `DenoiseSignal` will take ownership of the model and make a clone for each channel in the
    /// signal. If the model is cheap to clone (for example, because it was created with
    /// [`RnnModel::from_static_bytes`](crate::RnnModel::from_static_bytes) then this is fine.
    pub fn from_model(input: S, model: RnnModel) -> DenoiseSignal<'static, S> {
        DenoiseSignal {
            input,
            states: vec![DenoiseState::from_model_owned(Cow::Owned(model)); S::Frame::CHANNELS],
            in_bufs: vec![[0.0; FRAME_SIZE]; S::Frame::CHANNELS],
            out_bufs: vec![[0.0; FRAME_SIZE]; S::Frame::CHANNELS],
            out_idx: 0,
        }
        .discard_first_frame()
    }

    fn discard_first_frame(mut self) -> Self {
        self.refill_out_bufs();
        self.refill_out_bufs();
        self
    }

    /// Returns true if the input was not exhausted.
    fn refill_out_bufs(&mut self) -> bool {
        if self.input.is_exhausted() {
            return false;
        }

        for i in 0..FRAME_SIZE {
            for (ch, samp) in self.input.next().to_float_frame().channels().enumerate() {
                // Our denoiser expects f32s, but they should be in the range of an i16.
                self.in_bufs[ch][i] = samp.to_sample::<f32>() * 32768.0;
            }
        }

        for ch in 0..S::Frame::CHANNELS {
            self.states[ch].process_frame(&mut self.out_bufs[ch][..], &self.in_bufs[ch][..]);
        }
        !self.input.is_exhausted()
    }
}

impl<'model, S: Signal> Signal for DenoiseSignal<'model, S> {
    type Frame = <<S as Signal>::Frame as Frame>::Float;

    fn is_exhausted(&self) -> bool {
        self.out_idx >= FRAME_SIZE
    }

    fn next(&mut self) -> Self::Frame {
        if self.out_idx >= FRAME_SIZE {
            return Self::Frame::EQUILIBRIUM;
        }

        let idx = self.out_idx;
        self.out_idx += 1;
        let ret = Frame::from_fn(|ch| {
            // Undo the denoiser's peculiar choice of scaling.
            let samp = (self.out_bufs[ch][idx] / 32768.0).clamp(-1.0, 1.0);
            samp.to_sample()
        });

        // Process the next frame early, because it makes `is_exhausted` more accurate.
        if self.out_idx >= FRAME_SIZE {
            if self.refill_out_bufs() {
                self.out_idx = 0;
            }
        }

        ret
    }
}
