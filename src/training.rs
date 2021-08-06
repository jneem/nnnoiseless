use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use anyhow::{Context, Result};
use clap::{crate_version, App, Arg};
use rand::Rng;
use rustfft::num_complex::Complex;

use nnnoiseless::biquad::{Biquad, BIQUAD_HP};
use nnnoiseless::util::zip3;
use nnnoiseless::{DenoiseState, FREQ_SIZE, NB_BANDS, NB_FEATURES};

// After this many frames, we re-randomize the gains and the filters.
const GAIN_CHANGE_COUNT: u32 = 2821;

fn main() -> Result<()> {
    let matches = App::new("nnnoiseless-gen-training-data")
        .version(crate_version!())
        .about("Generate data for training nnnoiseless models")
        .arg(
            Arg::with_name("SIGNAL")
                .help("clean audio signal data")
                .required(true),
        )
        .arg(
            Arg::with_name("NOISE")
                .help("audio noise data")
                .required(true),
        )
        .arg(
            Arg::with_name("COUNT")
                .help("number of frames to generate")
                .required(true),
        )
        .arg(
            Arg::with_name("output")
                .short("o")
                .takes_value(true)
                .help("output file (defaults to stdout)")
                .required(true),
        )
        .get_matches();

    let signal_name = matches.value_of("SIGNAL").unwrap();
    let noise_name = matches.value_of("NOISE").unwrap();
    let count: usize = matches
        .value_of("COUNT")
        .unwrap()
        .parse()
        .context("COUNT must be a non-negative integer")?;

    let signal_file = File::open(signal_name).context("failed to open signal file")?;
    let noise_file = File::open(noise_name).context("failed to open noise file")?;
    let out_name = matches.value_of("output").unwrap();
    // TODO: stream the output instead of storing it all.
    let mut output = Vec::<f32>::new();
    let mut sim = NoiseSimulator::new(BufReader::new(signal_file), BufReader::new(noise_file));

    // Fourier transforms and band energies of the signal, noise, and combination of the two.
    let mut sig_freq = [Complex::from(0.0); FREQ_SIZE];
    let mut noise_freq = [Complex::from(0.0); FREQ_SIZE];
    let mut comb_freq = [Complex::from(0.0); FREQ_SIZE];
    let mut comb_p_freq = [Complex::from(0.0); FREQ_SIZE];
    let mut sig_e = [0.0; NB_BANDS];
    let mut noise_e = [0.0; NB_BANDS];
    let mut noise_level = [0.0; NB_BANDS];
    let mut comb_ex = [0.0; NB_BANDS];
    let mut comb_ep = [0.0; NB_BANDS];
    let mut comb_exp = [0.0; NB_BANDS];

    let mut features = [0.0; NB_FEATURES];
    let mut gains = [0.0; NB_BANDS];

    // Note that we shouldn't really need the whole DenoiseStates here. For sig and noise, we only
    // need the fft parts. For combined, we also need to compute frame features.
    let mut sig_state = DenoiseState::new();
    let mut noise_state = DenoiseState::new();
    let mut comb_state = DenoiseState::new();

    for i in 0..count {
        if i % 1000 == 0 {
            eprint!("{}\r", i);
        }
        let frame = sim.next_frame()?;
        sig_state.shift_input(frame.signal);
        noise_state.shift_input(frame.noise);
        comb_state.shift_input(frame.combined);

        sig_state.transform_input(0, &mut sig_freq[..], &mut sig_e[..]);
        noise_state.transform_input(0, &mut noise_freq[..], &mut noise_e[..]);
        let silence = comb_state.compute_frame_features(
            &mut comb_freq[..],
            &mut comb_p_freq,
            &mut comb_ex,
            &mut comb_ep[..],
            &mut comb_exp,
            &mut features[..],
        );

        nnnoiseless::denoise::pitch_filter(
            &mut comb_freq[..],
            &comb_p_freq[..],
            &comb_ex[..],
            &comb_ep[..],
            &comb_exp[..],
            &gains[..],
        );

        let band_gain_cutoff = if silence { 0 } else { frame.band_gain_cutoff };
        for i in 0..band_gain_cutoff {
            gains[i] = if sig_e[i] < 5e-2 && comb_ex[i] < 5e-2 {
                -1.0
            } else {
                ((sig_e[i] + 1e-3) / (comb_ex[i] + 1e-3)).sqrt().min(1.0)
            };
        }
        for i in band_gain_cutoff..NB_BANDS {
            gains[i] = -1.0;
        }

        for (nl, ne) in noise_level.iter_mut().zip(&noise_e) {
            *nl = (*ne + 1e-2).log10();
        }

        output.extend_from_slice(&features[..]);
        output.extend_from_slice(&gains[..]);
        output.extend_from_slice(&noise_level[..]);
        output.extend_from_slice(&[frame.vad][..]);
    }
    write_hdf5(&output[..], count, NB_FEATURES + 2 * NB_BANDS + 1, out_name)?;

    Ok(())
}

struct NoiseSimulator {
    signal: BufReader<File>,
    noise: BufReader<File>,
    sig_filter: Biquad,
    noise_filter: Biquad,
    vad_count: i32,
    gain_change_count: u32,
    signal_gain: f32,
    noise_gain: f32,
    lowpass: usize,
    band_lp: usize,

    /// A buffer of length FRAME_SIZE * 2 (for reading in enough bytes to have FRAME_SIZE i16s).
    read_buf: Vec<u8>,
    sig_buf: Vec<f32>,
    noise_buf: Vec<f32>,
    out_buf: Vec<f32>,

    signal_hp_mem: [f32; 2],
    noise_hp_mem: [f32; 2],
    signal_resp_mem: [f32; 2],
    noise_resp_mem: [f32; 2],
}

struct NoisyFrame<'a> {
    signal: &'a [f32],
    noise: &'a [f32],
    combined: &'a [f32],
    band_gain_cutoff: usize,
    vad: f32,
}

// Reads from a file, looping back to the beginning once we get to the end.
fn read_loop(read: &mut BufReader<File>, output: &mut [u8]) -> Result<()> {
    match read.read_exact(output) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            read.seek(SeekFrom::Start(0))?;
            Ok(read.read_exact(output)?)
        }
        Err(e) => Err(e.into()),
    }
}

fn read_input(
    read: &mut BufReader<File>,
    buf: &mut [u8],
    out: &mut [f32],
    gain: f32,
) -> Result<f32> {
    let mut energy = 0.0;
    if gain != 0.0 {
        read_loop(read, buf)?;
        for (input, out) in buf.chunks_exact(2).zip(out) {
            let x = i16::from_le_bytes([input[0], input[1]]) as f32;
            *out = gain * x;
            energy += x * x;
        }
    } else {
        for out in out {
            *out = 0.0;
        }
    }
    Ok(energy)
}

fn random_filter() -> Biquad {
    let r = || 0.75 * (rand::random::<f32>() - 0.5);
    Biquad {
        a: [r(), r()],
        b: [r(), r()],
    }
}

fn write_hdf5(data: &[f32], height: usize, width: usize, filename: &str) -> Result<()> {
    let f = hdf5::File::create(filename).context("failed to open output file")?;
    let dataset = hdf5::DatasetBuilder::<f32>::new(&f).create("data", (height, width))?;
    dataset.write_raw(data)?;
    f.flush()?;
    f.close();
    Ok(())
}

impl NoiseSimulator {
    fn new(signal: BufReader<File>, noise: BufReader<File>) -> NoiseSimulator {
        NoiseSimulator {
            signal,
            noise,
            sig_filter: Biquad::default(),
            noise_filter: Biquad::default(),
            vad_count: 0,
            gain_change_count: 0,
            signal_gain: 1.0,
            noise_gain: 1.0,
            lowpass: nnnoiseless::FREQ_SIZE,
            band_lp: nnnoiseless::NB_BANDS - 1,

            read_buf: vec![0; nnnoiseless::FRAME_SIZE * 2],
            sig_buf: vec![0.0; nnnoiseless::FRAME_SIZE],
            noise_buf: vec![0.0; nnnoiseless::FRAME_SIZE],
            out_buf: vec![0.0; nnnoiseless::FRAME_SIZE],

            signal_hp_mem: [0.0, 0.0],
            noise_hp_mem: [0.0, 0.0],
            signal_resp_mem: [0.0, 0.0],
            noise_resp_mem: [0.0, 0.0],
        }
    }

    fn read_noise(&mut self) -> Result<()> {
        read_input(
            &mut self.noise,
            &mut self.read_buf,
            &mut self.noise_buf,
            self.noise_gain,
        )?;
        Ok(())
    }

    /// Returns the strength of the signal (before applying gain).
    fn read_signal(&mut self) -> Result<f32> {
        read_input(
            &mut self.signal,
            &mut self.read_buf,
            &mut self.sig_buf,
            self.signal_gain,
        )
        .map_err(Into::into)
    }

    fn randomize(&mut self) {
        let mut rng = rand::thread_rng();
        self.signal_gain = 10.0_f32.powf(rng.gen_range(-40..20) as f32 / 20.0);
        self.noise_gain = 10.0_f32.powf(rng.gen_range(-30..20) as f32 / 20.0);
        self.noise_gain *= self.signal_gain;
        if rng.gen_bool(0.1) {
            self.noise_gain = 0.0;
        }
        if rng.gen_bool(0.1) {
            self.signal_gain = 0.0;
        }

        self.sig_filter = random_filter();
        self.noise_filter = random_filter();

        self.lowpass =
            (nnnoiseless::FREQ_SIZE as f32 * 3000.0 / 24000.0 * 50.0_f32.powf(rng.gen())) as usize;

        self.band_lp = nnnoiseless::EBAND_5MS
            .iter()
            .position(|x| x << nnnoiseless::FRAME_SIZE_SHIFT > self.lowpass)
            .unwrap_or(nnnoiseless::NB_BANDS - 1);
    }

    /// Update our voice activity probability based on the energy of the signal.
    fn vad(&mut self, sig_e: f32) -> f32 {
        if sig_e > 1e9 {
            self.vad_count = 0;
        } else if sig_e > 1e8 {
            self.vad_count -= 5;
        } else if sig_e > 1e7 {
            self.vad_count += 1;
        } else {
            self.vad_count += 2;
        }
        self.vad_count = self.vad_count.clamp(0, 15);
        if self.vad_count >= 10 {
            0.0
        } else if self.vad_count > 0 {
            0.5
        } else {
            1.0
        }
    }

    fn next_frame(&mut self) -> Result<NoisyFrame<'_>> {
        self.gain_change_count += 1;
        if self.gain_change_count > GAIN_CHANGE_COUNT {
            self.gain_change_count = 0;
            self.randomize();
        }
        self.read_noise()?;
        let sig_e = self.read_signal()?;

        BIQUAD_HP.filter_in_place(&mut self.sig_buf[..], &mut self.signal_hp_mem);
        self.sig_filter
            .filter_in_place(&mut self.sig_buf[..], &mut self.signal_resp_mem);
        BIQUAD_HP.filter_in_place(&mut self.noise_buf[..], &mut self.noise_hp_mem);
        self.noise_filter
            .filter_in_place(&mut self.noise_buf[..], &mut self.noise_resp_mem);

        for (x, y, z) in zip3(&self.sig_buf, &self.noise_buf, &mut self.out_buf) {
            *z = *x + *y;
        }

        let vad = self.vad(sig_e);

        // Set the gain to -1 for this and all higher bands.
        let band_gain_cutoff = if vad == 0.0 && self.noise_gain == 0.0 {
            0
        } else {
            self.band_lp + 1
        };
        Ok(NoisyFrame {
            signal: &self.sig_buf[..],
            noise: &self.noise_buf[..],
            combined: &self.out_buf[..],
            band_gain_cutoff,
            vad,
        })
    }
}
