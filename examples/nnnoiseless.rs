use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Error};
use clap::{crate_version, App, Arg};
use dasp_interpolate::sinc::Sinc;
use dasp_ring_buffer::Fixed;
use dasp_signal::{interpolate::Converter, Signal};
use hound::{SampleFormat, WavIntoSamples, WavReader, WavSpec, WavWriter};
use itertools::Itertools;

use nnnoiseless::DenoiseState;

const FRAME_SIZE: usize = DenoiseState::FRAME_SIZE;

struct RawReadSignal<R: Read> {
    bytes: itertools::Tuples<std::io::Bytes<R>, (std::io::Result<u8>, std::io::Result<u8>)>,
    finished: bool,
}

impl<R: Read> RawReadSignal<R> {
    fn new(r: R) -> Self {
        RawReadSignal {
            bytes: r.bytes().tuples(),
            finished: false,
        }
    }
}

struct WavReadIntSignal<R> {
    bits_per_sample: u16,
    finished: bool,
    samples: WavIntoSamples<R, i32>,
}

impl<R: Read> Signal for RawReadSignal<R> {
    type Frame = f32;
    fn next(&mut self) -> f32 {
        if self.is_exhausted() {
            return 0.0;
        }

        match self.bytes.next() {
            None => {
                self.finished = true;
                0.0
            }
            Some((Err(_), _)) | Some((_, Err(_))) => {
                eprintln!("Encountered an error while reading the input file; the output might be truncated");
                self.finished = true;
                0.0
            }
            Some((Ok(x), Ok(y))) => i16::from_le_bytes([x, y]) as f32,
        }
    }

    fn is_exhausted(&self) -> bool {
        self.finished
    }
}

impl<R: Read> Signal for WavReadIntSignal<R> {
    type Frame = f32;
    fn next(&mut self) -> f32 {
        if self.finished {
            return 0.0;
        }
        match self.samples.next() {
            Some(Ok(x)) => (x >> (32 - self.bits_per_sample)) as f32,
            Some(Err(_)) => {
                eprintln!("Encountered an error while reading the input file; the output might be truncated");
                self.finished = true;
                0.0
            }
            None => {
                self.finished = true;
                0.0
            }
        }
    }
}

trait FrameReader {
    fn read_frame(&mut self, buf: &mut [f32]);
    fn finished(&self) -> bool;
}

trait FrameWriter {
    fn write_frame(&mut self, buf: &[f32]) -> Result<(), Box<dyn std::error::Error>>;
    fn finalize(&mut self) -> Result<(), Box<dyn std::error::Error>>;
}

struct RawFrameWriter<W: Write> {
    writer: W,
    buf: Vec<u8>,
}

struct WavFrameWriter<W: Write + Seek> {
    writer: WavWriter<W>,
    buf: Vec<i16>,
}

impl<W: Write> FrameWriter for RawFrameWriter<W> {
    fn write_frame(&mut self, buf: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(buf.len() * 2, self.buf.len());
        for (dst, src) in self.buf.chunks_mut(2).zip(buf) {
            let bytes =
                (src.max(i16::MIN as f32).min(i16::MAX as f32).round() as i16).to_le_bytes();
            dst[0] = bytes[0];
            dst[1] = bytes[1];
        }
        self.writer
            .write_all(&self.buf[..])
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }

    fn finalize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: maybe should wrap around File instead of Write, to catch errors on close
        Ok(())
    }
}

impl<W: Write + Seek> FrameWriter for WavFrameWriter<W> {
    fn write_frame(&mut self, buf: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(buf.len(), self.buf.len());
        for (dst, src) in self.buf.iter_mut().zip(buf) {
            *dst = src.max(i16::MIN as f32).min(i16::MAX as f32).round() as i16;
        }
        let mut w = self.writer.get_i16_writer(self.buf.len() as u32);
        for &s in &self.buf {
            w.write_sample(s);
        }
        w.flush()
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }

    fn finalize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.writer
            .flush()
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }
}

struct SignalFrameReader<S: Signal> {
    //signal: Converter<S, Sinc<[f32; 16]>>,
    signal: S,
}

impl<S: Signal<Frame = f32>> FrameReader for SignalFrameReader<S> {
    fn read_frame(&mut self, buf: &mut [f32]) {
        for y in buf {
            *y = self.signal.next();
        }
    }

    fn finished(&self) -> bool {
        self.signal.is_exhausted()
    }
}

impl<R: Read> SignalFrameReader<WavReadIntSignal<R>> {
    fn from_wav_reader(wav: WavReader<R>) -> Self {
        assert_eq!(wav.spec().sample_format, hound::SampleFormat::Int);

        let sample_rate = wav.spec().sample_rate;
        let s = WavReadIntSignal {
            bits_per_sample: wav.spec().bits_per_sample,
            finished: false,
            samples: wav.into_samples(),
        };
        SignalFrameReader {
            signal: todo!(),
            /*
            signal: s.from_hz_to_hz(
                Sinc::new(Fixed::from([0.0; 16])),
                sample_rate as f64,
                48_000.0,
            ),
            */
        }
    }
}

impl<R: Read> SignalFrameReader<RawReadSignal<R>> {
    fn from_raw(r: R, sample_rate: f64) -> Self {
        SignalFrameReader {
            signal: RawReadSignal::new(r), /*
                                           signal: RawReadSignal::new(r).from_hz_to_hz(
                                               Sinc::new(Fixed::from([0.0; 16])),
                                               sample_rate,
                                               48_000.0,
                                           ),
                                           */
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = App::new("nnnoiseless")
        .version(crate_version!())
        .about("Remove noise from audio files")
        .arg(
            Arg::with_name("INPUT")
                .help("input audio file")
                .required(true),
        )
        .arg(
            Arg::with_name("OUTPUT")
                .help("output audio file")
                .required(true),
        )
        .arg(Arg::with_name("wav-in").long("wav-in").help(
            "if set, the input is a wav file (default is to detect wav files by their filename)",
        ))
        .arg(Arg::with_name("wav-out").long("wav-out").help(
            "if set, the output is a wav file (default is to detect wav files by their filename",
        ))
        .arg(
            Arg::with_name("sample-rate")
                .long("sample-rate")
                .help("sample rate of the input (defaults to 48kHz for raw input)")
                .takes_value(true),
        )
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

    let mut state = DenoiseState::new();
    let mut in_buf = [0.0; FRAME_SIZE];
    let mut out_buf = [0.0; FRAME_SIZE];
    let mut first = true;

    let mut frame_reader: Box<dyn FrameReader> = if in_wav {
        let wav_reader = WavReader::new(in_file)?;
        match wav_reader.spec().sample_format {
            SampleFormat::Int => Box::new(SignalFrameReader::from_wav_reader(wav_reader)),
            SampleFormat::Float => unimplemented!(),
        }
    } else {
        let sample_rate = matches
            .value_of("sample-rate")
            .and_then(|s| f64::from_str(s).ok())
            .unwrap_or(48_000.0);
        Box::new(SignalFrameReader::from_raw(in_file, sample_rate))
    };

    let mut frame_writer: Box<dyn FrameWriter> = if out_wav {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let writer = WavWriter::new(out_file, spec)?;
        Box::new(WavFrameWriter {
            writer,
            buf: vec![0; FRAME_SIZE],
        })
    } else {
        Box::new(RawFrameWriter {
            writer: out_file,
            buf: vec![0; FRAME_SIZE * 2],
        })
    };

    while !frame_reader.finished() {
        frame_reader.read_frame(&mut in_buf);
        state.process_frame(&mut out_buf[..], &in_buf[..]);
        if !first {
            frame_writer.write_frame(&out_buf[..])?;
        }
        first = false;
    }
    frame_writer.finalize()?;

    Ok(())
}
