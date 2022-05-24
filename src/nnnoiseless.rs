use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

use anyhow::{anyhow, Context, Error};
use clap::{arg, crate_version, Command};
use dasp_interpolate::{sinc::Sinc, Interpolator};
use dasp_ring_buffer::Fixed;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

use nnnoiseless::{DenoiseState, RnnModel};

const FRAME_SIZE: usize = DenoiseState::FRAME_SIZE;

trait ReadSample {
    fn next_sample(&mut self) -> Result<Option<&[f32]>, Error>;
    fn channels(&self) -> usize;

    fn resampled(self, ratio: f64) -> Resample<Self>
    where
        Self: Sized,
    {
        Resample {
            sinc: (0..self.channels())
                .map(|_| Sinc::new(Fixed::from([0.0; 16])))
                .collect(),
            buf: vec![0.0; self.channels()],
            ratio,
            pos: 0.0,
            read: self,
        }
    }
}

// TODO: support either endianness
struct RawSampleIter<R: Read> {
    bytes: std::io::Bytes<R>,
}

struct Resample<RS: ReadSample> {
    sinc: Vec<Sinc<[f32; 16]>>,
    buf: Vec<f32>,
    ratio: f64,
    pos: f64,
    read: RS,
}

struct IterReadSample<I> {
    samples: I,
    buf: Vec<f32>,
}

impl<I: Iterator<Item = Result<f32, Error>>> IterReadSample<I> {
    fn new(iter: I, channels: usize) -> IterReadSample<I> {
        IterReadSample {
            samples: iter,
            buf: vec![0.0; channels],
        }
    }
}

impl<R: Read> Iterator for RawSampleIter<R> {
    type Item = Result<f32, Error>;

    fn next(&mut self) -> Option<Result<f32, Error>> {
        match self.bytes.next() {
            None => None,
            Some(Err(e)) => Some(Err(e.into())),
            Some(Ok(a)) => match self.bytes.next() {
                None => Some(Err(anyhow!(
                    "Unexpected end of input (expected an even number of bytes)"
                ))),
                Some(Err(e)) => Some(Err(e.into())),
                Some(Ok(b)) => Some(Ok(i16::from_le_bytes([a, b]) as f32)),
            },
        }
    }
}

impl<I: Iterator<Item = Result<f32, Error>>> ReadSample for IterReadSample<I> {
    fn next_sample(&mut self) -> Result<Option<&[f32]>, Error> {
        for (i, sample) in self.buf.iter_mut().enumerate() {
            match self.samples.next() {
                None => {
                    if i == 0 {
                        return Ok(None);
                    } else {
                        return Err(anyhow!(
                            "Unexpected end of input (expected a multiple of {} samples)",
                            self.buf.len()
                        ));
                    }
                }
                Some(Err(e)) => return Err(e),
                Some(Ok(x)) => *sample = x,
            }
        }
        Ok(Some(&self.buf[..]))
    }

    fn channels(&self) -> usize {
        self.buf.len()
    }
}

impl<RS: ReadSample> ReadSample for Resample<RS> {
    fn next_sample(&mut self) -> Result<Option<&[f32]>, Error> {
        self.pos += self.ratio;
        while self.pos >= 1.0 {
            self.pos -= 1.0;

            if let Some(buf) = self.read.next_sample()? {
                for (s, &x) in self.sinc.iter_mut().zip(buf) {
                    s.next_source_frame(x);
                }
            } else {
                return Ok(None);
            }
        }

        for (s, x) in self.sinc.iter().zip(&mut self.buf) {
            *x = s.interpolate(self.pos);
        }

        Ok(Some(&self.buf[..]))
    }

    fn channels(&self) -> usize {
        self.read.channels()
    }
}

trait FrameWriter {
    fn write_frame(&mut self, buf: &[f32]) -> Result<(), Error>;
    fn finalize(&mut self) -> Result<(), Error>;
}

struct RawFrameWriter<W: Write> {
    writer: W,
    buf: Vec<u8>,
}

struct WavFrameWriter<W: Write + Seek> {
    writer: WavWriter<W>,
}

impl<W: Write> FrameWriter for RawFrameWriter<W> {
    fn write_frame(&mut self, buf: &[f32]) -> Result<(), Error> {
        assert_eq!(buf.len() * 2, self.buf.len());
        for (dst, src) in self.buf.chunks_mut(2).zip(buf) {
            let bytes =
                (src.max(i16::MIN as f32).min(i16::MAX as f32).round() as i16).to_le_bytes();
            dst[0] = bytes[0];
            dst[1] = bytes[1];
        }
        self.writer.write_all(&self.buf[..]).map_err(|e| e.into())
    }

    fn finalize(&mut self) -> Result<(), Error> {
        self.writer.flush()?;
        Ok(())
    }
}

impl<W: Write + Seek> FrameWriter for WavFrameWriter<W> {
    fn write_frame(&mut self, buf: &[f32]) -> Result<(), Error> {
        let mut w = self.writer.get_i16_writer(buf.len() as u32);
        for &x in buf {
            w.write_sample(x.max(i16::MIN as f32).min(i16::MAX as f32).round() as i16);
        }
        w.flush().map_err(|e| e.into())
    }

    fn finalize(&mut self) -> Result<(), Error> {
        self.writer.flush().map_err(|e| e.into())
    }
}

fn raw_samples<R: Read + 'static>(r: R, channels: usize, sample_rate: f64) -> Box<dyn ReadSample> {
    let raw = IterReadSample::new(RawSampleIter { bytes: r.bytes() }, channels);

    if sample_rate != 48_000.0 {
        Box::new(raw.resampled(sample_rate / 48_000.0))
    } else {
        Box::new(raw)
    }
}

fn wav_samples<R: Read + 'static>(wav: WavReader<R>) -> Box<dyn ReadSample> {
    let sample_rate = wav.spec().sample_rate as f64;
    let channels = wav.spec().channels as usize;
    match wav.spec().sample_format {
        SampleFormat::Int => {
            let bits_per_sample = wav.spec().bits_per_sample;
            assert!(bits_per_sample <= 32);

            let iter = wav.into_samples::<i32>().map(move |s| {
                s.map(|s| {
                    if bits_per_sample < 16 {
                        (s << (16 - bits_per_sample)) as f32
                    } else {
                        (s >> (bits_per_sample - 16)) as f32
                    }
                })
                .map_err(|e| e.into())
            });

            let read_sample = IterReadSample::new(iter, channels);
            if sample_rate != 48_000.0 {
                Box::new(read_sample.resampled(sample_rate / 48_000.0))
            } else {
                Box::new(read_sample)
            }
        }
        SampleFormat::Float => {
            let iter = wav
                .into_samples::<f32>()
                .map(|s| s.map(|s| s * 32767.0).map_err(|e| e.into()));

            let read_sample = IterReadSample::new(iter, channels);
            if sample_rate != 48_000.0 {
                Box::new(read_sample.resampled(sample_rate / 48_000.0))
            } else {
                Box::new(read_sample)
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches =
        Command::new("nnnoiseless")
            .version(crate_version!())
            .about("Remove noise from audio files")
            .arg(arg!(<INPUT> "input audio file"))
            .arg(arg!(<OUTPUT> "output audio file"))
            .arg(arg!(--"wav-in" "the input is a wav file (default is to detect wav files by their filename"))
            .arg(arg!(--"wav-out" "the output is a wav file (default is to detect wav files by their filename)"))
            .arg(arg!(--"sample-rate" <RATE> "for raw input, the sample rate of the input (defaults to 48kHz)").required(false)
                    .validator(|s| s.parse::<f64>()),
            )
            .arg(
                arg!(--channels <CHANNELS> "for raw input, the number of channels (defaults to 1)")
                    .required(false)
                    .validator(|s| s.parse::<u16>()),
            )
            .arg(arg!(--model <PATH> "path to a custom model file").required(false))
            .get_matches();

    let in_name = matches.value_of("INPUT").unwrap();
    let out_name = matches.value_of("OUTPUT").unwrap();
    let in_file = BufReader::new(
        File::open(in_name)
            .with_context(|| format!("Failed to open input file \"{}\"", in_name))?,
    );
    let out_file = BufWriter::new(
        File::create(out_name)
            .with_context(|| format!("Failed to open output file \"{}\"", out_name))?,
    );
    let in_wav =
        matches.is_present("wav-in") || Path::new(in_name).extension() == Some("wav".as_ref());
    let out_wav =
        matches.is_present("wav-out") || Path::new(out_name).extension() == Some("wav".as_ref());

    let (mut samples, channels) = if in_wav {
        let wav_reader = WavReader::new(in_file)?;
        let channels = wav_reader.spec().channels;
        (wav_samples(wav_reader), channels)
    } else {
        let sample_rate = matches.value_of_t("sample-rate").unwrap_or(48_000.0);
        let channels = matches.value_of_t("channels").unwrap_or(1);
        (
            raw_samples(in_file, channels as usize, sample_rate),
            channels,
        )
    };

    let mut frame_writer: Box<dyn FrameWriter> = if out_wav {
        let spec = WavSpec {
            channels,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let writer = WavWriter::new(out_file, spec)?;
        Box::new(WavFrameWriter { writer })
    } else {
        Box::new(RawFrameWriter {
            writer: out_file,
            buf: vec![0; FRAME_SIZE * 2],
        })
    };

    let model = if let Some(model_path) = matches.value_of("model") {
        let data = std::fs::read(model_path).context("Failed to open model file")?;
        RnnModel::from_bytes(&data).context("Failed to parse model file")?
    } else {
        RnnModel::default()
    };

    let channels = channels as usize;
    let mut in_bufs = vec![vec![0.0; FRAME_SIZE]; channels];
    let mut out_bufs = vec![vec![0.0; FRAME_SIZE]; channels];
    let mut out_buf = vec![0.0; FRAME_SIZE * channels];
    let mut states = vec![DenoiseState::with_model(&model); channels];
    let mut first = true;
    'outer: loop {
        for i in 0..FRAME_SIZE {
            if let Some(buf) = samples.next_sample()? {
                for j in 0..channels {
                    in_bufs[j][i] = buf[j];
                }
            } else {
                break 'outer;
            }
        }

        for j in 0..channels {
            states[j].process_frame(&mut out_bufs[j], &in_bufs[j]);
        }
        if !first {
            for i in 0..FRAME_SIZE {
                for j in 0..channels {
                    out_buf[i * channels + j] = out_bufs[j][i];
                }
            }
            frame_writer.write_frame(&out_buf[..])?;
        }
        first = false;
    }
    frame_writer.finalize()?;

    Ok(())
}
