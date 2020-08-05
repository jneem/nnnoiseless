const MAX_NEURONS: usize = 128;

const TANSIG_TABLE: [f32; 201] = [
    0.000000, 0.039979, 0.079830, 0.119427, 0.158649, 0.197375, 0.235496, 0.272905, 0.309507,
    0.345214, 0.379949, 0.413644, 0.446244, 0.477700, 0.507977, 0.537050, 0.564900, 0.591519,
    0.616909, 0.641077, 0.664037, 0.685809, 0.706419, 0.725897, 0.744277, 0.761594, 0.777888,
    0.793199, 0.807569, 0.821040, 0.833655, 0.845456, 0.856485, 0.866784, 0.876393, 0.885352,
    0.893698, 0.901468, 0.908698, 0.915420, 0.921669, 0.927473, 0.932862, 0.937863, 0.942503,
    0.946806, 0.950795, 0.954492, 0.957917, 0.961090, 0.964028, 0.966747, 0.969265, 0.971594,
    0.973749, 0.975743, 0.977587, 0.979293, 0.980869, 0.982327, 0.983675, 0.984921, 0.986072,
    0.987136, 0.988119, 0.989027, 0.989867, 0.990642, 0.991359, 0.992020, 0.992631, 0.993196,
    0.993718, 0.994199, 0.994644, 0.995055, 0.995434, 0.995784, 0.996108, 0.996407, 0.996682,
    0.996937, 0.997172, 0.997389, 0.997590, 0.997775, 0.997946, 0.998104, 0.998249, 0.998384,
    0.998508, 0.998623, 0.998728, 0.998826, 0.998916, 0.999000, 0.999076, 0.999147, 0.999213,
    0.999273, 0.999329, 0.999381, 0.999428, 0.999472, 0.999513, 0.999550, 0.999585, 0.999617,
    0.999646, 0.999673, 0.999699, 0.999722, 0.999743, 0.999763, 0.999781, 0.999798, 0.999813,
    0.999828, 0.999841, 0.999853, 0.999865, 0.999875, 0.999885, 0.999893, 0.999902, 0.999909,
    0.999916, 0.999923, 0.999929, 0.999934, 0.999939, 0.999944, 0.999948, 0.999952, 0.999956,
    0.999959, 0.999962, 0.999965, 0.999968, 0.999970, 0.999973, 0.999975, 0.999977, 0.999978,
    0.999980, 0.999982, 0.999983, 0.999984, 0.999986, 0.999987, 0.999988, 0.999989, 0.999990,
    0.999990, 0.999991, 0.999992, 0.999992, 0.999993, 0.999994, 0.999994, 0.999994, 0.999995,
    0.999995, 0.999996, 0.999996, 0.999996, 0.999997, 0.999997, 0.999997, 0.999997, 0.999997,
    0.999998, 0.999998, 0.999998, 0.999998, 0.999998, 0.999998, 0.999999, 0.999999, 0.999999,
    0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999,
    0.999999, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
    1.000000, 1.000000, 1.000000,
];

fn tansig_approx(x: f32) -> f32 {
    // Tests are reversed to catch NaNs
    if !(x < 8.0) {
        return 1.0;
    }
    if !(x > -8.0) {
        return -1.0;
    }

    let (mut x, sign) = if x < 0.0 { (-x, -1.0) } else { (x, 1.0) };
    let i = (0.5 + 25.0 * x).floor();
    x -= 0.04 * i;
    let y = TANSIG_TABLE[i as usize];
    let dy = 1.0 - y * y;
    let y = y + x * dy * (1.0 - y * x);
    sign * y
}

fn sigmoid_approx(x: f32) -> f32 {
    0.5 + 0.5 * tansig_approx(0.5 * x)
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Tanh = 0,
    Sigmoid = 1,
    Relu = 2,
}

const WEIGHTS_SCALE: f32 = 1.0 / 256.0;

#[derive(Copy, Clone)]
pub struct DenseLayer {
    /// An array of length `nb_neurons`.
    pub bias: &'static [i8],
    /// An array of length `nb_inputs * nb_neurons`.
    pub input_weights: &'static [i8],
    pub nb_inputs: usize,
    pub nb_neurons: usize,
    pub activation: Activation,
}

#[derive(Copy, Clone)]
pub struct GruLayer {
    /// An array of length `3 * nb_neurons`.
    pub bias: &'static [i8],
    /// An array of length `3 * nb_inputs * nb_neurons`.
    pub input_weights: &'static [i8],
    /// An array of length `3 * nb_neurons^2`.
    pub recurrent_weights: &'static [i8],
    pub nb_inputs: usize,
    pub nb_neurons: usize,
    pub activation: Activation,
}

#[derive(Clone)]
pub struct RnnModel {
    pub input_dense_size: usize,
    pub input_dense: DenseLayer,
    pub vad_gru_size: usize,
    pub vad_gru: GruLayer,
    pub noise_gru_size: usize,
    pub noise_gru: GruLayer,
    pub denoise_gru_size: usize,
    pub denoise_gru: GruLayer,
    pub denoise_output_size: usize,
    pub denoise_output: DenseLayer,
    pub vad_output_size: usize,
    pub vad_output: DenseLayer,
}

#[derive(Clone)]
pub struct RnnState {
    model: &'static RnnModel,
    vad_gru_state: Vec<f32>,
    noise_gru_state: Vec<f32>,
    denoise_gru_state: Vec<f32>,
}

impl RnnState {
    pub fn new() -> RnnState {
        let model = &crate::model::MODEL;
        let vad_gru_state = vec![0.0f32; model.vad_gru_size];
        let noise_gru_state = vec![0.0f32; model.noise_gru_size];
        let denoise_gru_state = vec![0.0f32; model.denoise_gru_size];
        RnnState {
            model,
            vad_gru_state,
            noise_gru_state,
            denoise_gru_state,
        }
    }
}

fn inner_p(xs: &[i8], ys: &[f32]) -> f32 {
    let n = xs.len();
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;

    let n_4 = n - n % 4;
    for (x, y) in xs[..n_4].chunks_exact(4).zip(ys[..n_4].chunks_exact(4)) {
        sum0 += x[0] as f32 * y[0];
        sum1 += x[1] as f32 * y[1];
        sum2 += x[2] as f32 * y[2];
        sum3 += x[3] as f32 * y[3];
    }

    let mut sum = sum0 + sum1 + sum2 + sum3;
    for (&x, &y) in xs[n_4..n].iter().zip(&ys[n_4..n]) {
        sum += x as f32 * y;
    }
    sum
}

fn compute_dense(layer: &DenseLayer, output: &mut [f32], input: &[f32]) {
    let m = layer.nb_inputs;
    let n = layer.nb_neurons;

    for i in 0..n {
        // Compute update gate.
        let sum =
            layer.bias[i] as f32 + inner_p(&layer.input_weights[(i * m)..((i + 1) * m)], input);
        output[i] = WEIGHTS_SCALE * sum;
    }
    match layer.activation {
        Activation::Sigmoid => {
            for i in 0..n {
                output[i] = sigmoid_approx(output[i]);
            }
        }
        Activation::Tanh => {
            for i in 0..n {
                output[i] = tansig_approx(output[i]);
            }
        }
        Activation::Relu => {
            for i in 0..n {
                output[i] = relu(output[i]);
            }
        }
    }
}

fn compute_gru(gru: &GruLayer, state: &mut [f32], input: &[f32]) {
    let mut z = [0.0; MAX_NEURONS];
    let mut r = [0.0; MAX_NEURONS];
    let m = gru.nb_inputs;
    let n = gru.nb_neurons;

    for i in 0..n {
        // Compute update gate.
        let sum = gru.bias[i] as f32
            + inner_p(&gru.input_weights[(i * m)..((i + 1) * m)], input)
            + inner_p(&gru.recurrent_weights[(i * n)..((i + 1) * n)], state);
        z[i] = sigmoid_approx(WEIGHTS_SCALE * sum);
    }
    for i in 0..n {
        // Compute reset gate.
        let sum = gru.bias[n + i] as f32
            + inner_p(&gru.input_weights[((i + n) * m)..((i + n + 1) * m)], input)
            + inner_p(
                &gru.recurrent_weights[((i + n) * n)..((i + n + 1) * n)],
                state,
            );
        // NOTE: our r[i] differs from the one in rnnoise because we're premultiplying it by
        // state[i].
        r[i] = state[i] * sigmoid_approx(WEIGHTS_SCALE * sum);
    }
    for i in 0..n {
        // Compute output.
        let sum = gru.bias[2 * n + i] as f32
            + inner_p(
                &gru.input_weights[((i + 2 * n) * m)..((i + 2 * n + 1) * m)],
                input,
            )
            + inner_p(
                &gru.recurrent_weights[((i + 2 * n) * n)..((i + 2 * n + 1) * n)],
                &r[..],
            );
        let sum = match gru.activation {
            Activation::Sigmoid => sigmoid_approx(WEIGHTS_SCALE * sum),
            Activation::Tanh => tansig_approx(WEIGHTS_SCALE * sum),
            Activation::Relu => relu(WEIGHTS_SCALE * sum),
        };
        state[i] = z[i] * state[i] + (1.0 - z[i]) * sum;
    }
}

const INPUT_SIZE: usize = 42;

fn copy(dst: &mut [f32], src: &[f32]) {
    for (x, y) in dst.iter_mut().zip(src) {
        *x = *y;
    }
}

pub fn compute_rnn(rnn: &mut RnnState, gains: &mut [f32], vad: &mut [f32], input: &[f32]) {
    assert_eq!(input.len(), INPUT_SIZE);

    let mut buf = [0.0; MAX_NEURONS * 3];
    let mut denoise_buf = [0.0; MAX_NEURONS * 3];
    let model = &rnn.model;

    let vad_gru_state = &mut rnn.vad_gru_state[..];
    let noise_gru_state = &mut rnn.noise_gru_state[..];
    let denoise_gru_state = &mut rnn.denoise_gru_state[..];
    compute_dense(
        &model.input_dense,
        &mut buf[0..model.input_dense_size],
        input,
    );
    compute_gru(
        &model.vad_gru,
        vad_gru_state,
        &buf[0..model.input_dense_size],
    );
    compute_dense(&model.vad_output, vad, vad_gru_state);

    copy(&mut buf[model.input_dense_size..], vad_gru_state);
    copy(
        &mut buf[(model.input_dense_size + model.vad_gru_size)..],
        input,
    );
    compute_gru(&model.noise_gru, noise_gru_state, &buf);

    copy(&mut denoise_buf, vad_gru_state);
    copy(&mut denoise_buf[model.vad_gru_size..], noise_gru_state);
    copy(
        &mut denoise_buf[(model.vad_gru_size + model.noise_gru_size)..],
        input,
    );
    compute_gru(&model.denoise_gru, denoise_gru_state, &denoise_buf);
    compute_dense(&model.denoise_output, gains, denoise_gru_state);
}
