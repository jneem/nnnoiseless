use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{crate_version, Arg, Command};
use hound::WavReader;
use rand::seq::SliceRandom;
use rand::Rng;

use nnnoiseless::util::zip3;
use nnnoiseless::util::Biquad;
use nnnoiseless::DenoiseFeatures;
use nnnoiseless::{NB_BANDS, NB_FEATURES};

// After this many frames, we re-randomize the gains and the filters.
const GAIN_CHANGE_COUNT: u32 = 2821;

fn glob_paths<'a>(globs: impl Iterator<Item = &'a str>) -> Result<Vec<PathBuf>> {
    let mut ret = Vec::new();
    for glob in globs {
        let paths = glob::glob(glob).context(format!("while expanding glob {}", glob))?;
        for p in paths {
            ret.push(p.context(format!("while expanding glob {}", glob))?);
        }
    }
    Ok(ret)
}

fn main() -> Result<()> {
    let matches = Command::new("nnnoiseless-gen-training-data")
        .version(crate_version!())
        .about("Generate data for training nnnoiseless models")
        .arg(
            Arg::new("signal-glob")
                .help("wildcard for audio signal data")
                .long("signal-glob")
                .takes_value(true)
                .multiple_occurrences(true)
                .required(true),
        )
        .arg(
            Arg::new("noise-glob")
                .help("wildcard for audio noise data")
                .long("noise-glob")
                .takes_value(true)
                .multiple_occurrences(true)
                .required(true),
        )
        .arg(
            Arg::new("shuffle")
                .help("if set, shuffle the signal and noise files")
                .long("shuffle"),
        )
        .arg(
            Arg::new("count")
                .help("number of frames to generate")
                .long("count")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .takes_value(true)
                .help("output file (defaults to stdout)")
                .required(true),
        )
        .get_matches();

    let signal_globs = matches.values_of("signal-glob").unwrap();
    let noise_globs = matches.values_of("noise-glob").unwrap();
    let count: usize = matches
        .value_of("count")
        .unwrap()
        .parse()
        .context("count must be a non-negative integer")?;
    let shuffle = matches.is_present("shuffle");

    let mut signal_paths = glob_paths(signal_globs)?;
    let mut noise_paths = glob_paths(noise_globs)?;

    if shuffle {
        signal_paths.shuffle(&mut rand::thread_rng());
        noise_paths.shuffle(&mut rand::thread_rng());
    }

    let out_name = matches.value_of("output").unwrap();
    let out_file = hdf5::File::create(out_name).context("failed to open output file")?;
    let (width, height) = (NB_FEATURES + 2 * NB_BANDS + 1, count);
    let dataset = hdf5::DatasetBuilder::new(&out_file)
        .empty::<f32>()
        .shape((height, width))
        .create("data")?;
    // A buffer containing just enough output for a single frame.
    let mut output = Vec::<f32>::new();
    let signal_reader = SignalReader::new(signal_paths, count);
    let noise_reader = SignalReader::new(noise_paths, count);

    eprintln!(
        "Found {} clean files, reading about {} frames from each",
        signal_reader.paths.len(),
        signal_reader.frames_per_file
    );
    eprintln!(
        "Found {} noise files, reading about {} frames from each",
        noise_reader.paths.len(),
        noise_reader.frames_per_file
    );

    let mut sim = NoiseSimulator::new(signal_reader, noise_reader);

    let mut clean_features = DenoiseFeatures::new();
    let mut noise_features = DenoiseFeatures::new();
    let mut comb_features = DenoiseFeatures::new();

    let mut noise_level = [0.0; NB_BANDS];
    let mut gains = [0.0; NB_BANDS];

    for i in 0..count {
        if i % 1000 == 0 {
            eprint!("{}\r", i);
        }
        let frame = sim.next_frame()?;
        clean_features.shift_and_filter_input(frame.signal);
        noise_features.shift_and_filter_input(frame.noise);
        comb_features.shift_and_filter_input(frame.combined);

        // TODO: we don't need to actually compute the full frame features -- we only need the
        // Fourier transform and band energies.
        clean_features.compute_frame_features();
        noise_features.compute_frame_features();

        let silence = comb_features.compute_frame_features();
        let band_gain_cutoff = if silence { 0 } else { frame.band_gain_cutoff };
        for i in 0..band_gain_cutoff {
            gains[i] = if clean_features.ex[i] < 5e-2 && comb_features.ex[i] < 5e-2 {
                -1.0
            } else {
                ((clean_features.ex[i] + 1e-3) / (comb_features.ex[i] + 1e-3))
                    .sqrt()
                    .min(1.0)
            };
        }
        for i in band_gain_cutoff..NB_BANDS {
            gains[i] = -1.0;
        }

        for (nl, ne) in noise_level.iter_mut().zip(&noise_features.ex) {
            *nl = (*ne + 1e-2).log10();
        }

        // NOTE: the training doesn't actually seem to use the noise level. We can remove it (as
        // long as we also update the training script).
        output.extend_from_slice(comb_features.features());
        output.extend_from_slice(&gains[..]);
        output.extend_from_slice(&noise_level[..]);
        output.extend_from_slice(&[frame.vad][..]);
        dataset.write_slice(&output[..], (i, ..))?;
        output.clear();
    }
    out_file.flush()?;
    out_file.close()?;

    Ok(())
}

// The signals (both the clean signal and the noise) are spread out over lots of files, and we want
// to sample audio from many of them. This struct abstracts over that task: it holds a bunch of
// paths, and you can just ask it for the next frame of audio.
struct SignalReader {
    paths: Vec<PathBuf>,
    /// How many frames should we extract before skipping to the next file? (This is an upper bound
    /// -- if the file doesn't have enough frames, we just read the whole thing.)
    frames_per_file: usize,
    /// The index of the file we're currently reading, or the next one to read.
    cur_idx: usize,
    /// How many frames should we read from the current file?
    frames_left: usize,
    /// The current input.
    reader: Option<WavReader<BufReader<File>>>,
}

impl SignalReader {
    fn new(paths: Vec<PathBuf>, count: usize) -> SignalReader {
        assert!(!paths.is_empty(), "cannot read from an empty set of files");
        SignalReader {
            frames_per_file: ((count / paths.len()) + 1).max(100),
            paths,
            cur_idx: 0,
            frames_left: 0,
            reader: None,
        }
    }

    fn next_reader(&mut self) -> Result<()> {
        if self.cur_idx >= self.paths.len() {
            self.cur_idx = 0;
        }
        let mut reader = WavReader::open(&self.paths[self.cur_idx])?;

        let spec = reader.spec();
        if spec.channels != 1
            || spec.sample_rate != 48_000
            || spec.bits_per_sample != 16
            || spec.sample_format != hound::SampleFormat::Int
        {
            anyhow::bail!(
                "unsupported wav format {:?} in {}",
                spec,
                self.paths[self.cur_idx].to_string_lossy()
            );
        }

        // We want num_samples samples, and the file has len samples. If the file is big
        // enough, take a random slice of it.
        let len = reader.duration() as usize;
        let num_samples = nnnoiseless::FRAME_SIZE * self.frames_per_file;
        if len > num_samples {
            reader
                .seek(rand::thread_rng().gen_range(0..=(len - num_samples) as u32))
                .context(format!("failed to seek in {:?}", self.paths[self.cur_idx]))?;
            self.frames_left = self.frames_per_file;
        } else {
            self.frames_left = len / nnnoiseless::FRAME_SIZE;
        }
        if self.frames_left > 0 {
            self.reader = Some(reader);
        }
        Ok(())
    }

    fn frame(&mut self, buf: &mut [f32]) -> Result<()> {
        while self.reader.is_none() {
            self.next_reader()?;
        }

        let r = self.reader.as_mut().unwrap();
        let mut samples_read = 0;
        for (s, x) in r.samples::<i16>().zip(buf.iter_mut()) {
            samples_read += 1;
            *x = s? as f32;
        }
        if samples_read < buf.len() {
            // We ran out of samples in this file.
            // TODO: warn? This shouldn't happen if we calculated frames_left correctly...
            for x in &mut buf[samples_read..] {
                *x = 0.0;
            }
            self.frames_left = 0;
        }

        if self.frames_left <= 1 {
            self.reader = None;
            self.cur_idx += 1;
        } else {
            self.frames_left -= 1;
        }
        Ok(())
    }
}

struct NoiseSimulator {
    signal: SignalReader,
    noise: SignalReader,
    sig_filter: Biquad,
    noise_filter: Biquad,
    vad_count: i32,
    gain_change_count: u32,
    signal_gain: f32,
    noise_gain: f32,
    lowpass: usize,
    band_lp: usize,

    sig_buf: Vec<f32>,
    noise_buf: Vec<f32>,
    out_buf: Vec<f32>,

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

fn random_filter() -> Biquad {
    let r = || 0.75 * (rand::random::<f32>() - 0.5);
    Biquad {
        a: [r(), r()],
        b: [r(), r()],
    }
}

impl NoiseSimulator {
    fn new(signal: SignalReader, noise: SignalReader) -> NoiseSimulator {
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

            sig_buf: vec![0.0; nnnoiseless::FRAME_SIZE],
            noise_buf: vec![0.0; nnnoiseless::FRAME_SIZE],
            out_buf: vec![0.0; nnnoiseless::FRAME_SIZE],

            signal_resp_mem: [0.0, 0.0],
            noise_resp_mem: [0.0, 0.0],
        }
    }

    fn read_noise(&mut self) -> Result<()> {
        self.noise.frame(&mut self.noise_buf)?;
        for x in &mut self.noise_buf {
            *x *= self.noise_gain;
        }
        Ok(())
    }

    /// Returns the strength of the signal (before applying gain).
    fn read_signal(&mut self) -> Result<f32> {
        self.signal.frame(&mut self.sig_buf)?;
        let mut energy = 0.0;
        for x in &mut self.sig_buf {
            energy += *x * *x;
            *x *= self.signal_gain;
        }
        Ok(energy)
    }

    fn randomize(&mut self) {
        let mut rng = rand::thread_rng();
        self.signal_gain = 10.0_f32.powf(rng.gen_range(-40..20) as f32 / 20.0);
        self.noise_gain = 10.0_f32.powf(rng.gen_range(-20..20) as f32 / 20.0);
        self.noise_gain *= self.signal_gain;
        /*
        if rng.gen_bool(0.1) {
            self.noise_gain = 0.0;
        }
        */
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

        self.sig_filter
            .filter_in_place(&mut self.sig_buf[..], &mut self.signal_resp_mem);
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
