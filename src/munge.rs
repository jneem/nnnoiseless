#![allow(dead_code)]

//! This is a binary for shuffling the order of neural network weights, the goal being to increase
//! memory locality and eliminate bounds checks.
//!
//! The access pattern of these neural net weights essentially follows that of matrix
//! multiplication: besides the weights, you have a slice of length m. You take various subsets of
//! m elements of the weights and compute their inner product with the slice.
//!
//! To be fast, the weights should be stored in row-major order, but in rnnoise they were stored in
//! column-major order. This script was used to swap the order, and I've kept in around in case
//! anyone wants to re-train the model and needs to swap the orders again.
mod model;
mod rnn;
mod util;

use rnn::Activation;

fn reorder_weights(m: usize, n: usize, weights: &[i8]) -> Vec<i8> {
    assert_eq!(weights.len(), m * n);
    let mut ret = vec![0; weights.len()];

    for i in 0..n {
        for j in 0..m {
            assert_eq!(0, ret[i * m + j]);
            ret[i * m + j] = weights[i + n * j];
        }
    }
    ret
}

fn act_string(a: Activation) -> &'static str {
    match a {
        Activation::Sigmoid => "Activation::Sigmoid",
        Activation::Tanh => "Activation::Tanh",
        Activation::Relu => "Activation::Relu",
    }
}

fn print_array(name: String, data: &[i8]) {
    let data: Vec<_> = data.iter().map(|i| i.to_string()).collect();
    println!(
        "const {}: [i8; {}] = [{}];",
        name,
        data.len(),
        data.join(", ")
    );
}

fn reorder_gru_layer(layer: &rnn::GruLayer, prefix: &'static str) {
    let weights = reorder_weights(layer.nb_inputs, 3 * layer.nb_neurons, layer.input_weights);
    let recurrent = reorder_weights(
        layer.nb_neurons,
        3 * layer.nb_neurons,
        layer.recurrent_weights,
    );

    print_array(format!("{}_BIAS", prefix), layer.bias);
    print_array(format!("{}_WEIGHTS", prefix), &weights);
    print_array(format!("{}_RECURRENT_WEIGHTS", prefix), &recurrent);
    println!(
        "static {}: GruLayer = GruLayer {{
        bias: &{}_BIAS,
        input_weights: &{}_WEIGHTS,
        recurrent_weights: &{}_RECURRENT_WEIGHTS,
        nb_inputs: {},
        nb_neurons: {},
        activation: {},
    }};",
        prefix,
        prefix,
        prefix,
        prefix,
        layer.nb_inputs,
        layer.nb_neurons,
        act_string(layer.activation)
    );
}

fn reorder_dense_layer(layer: &rnn::DenseLayer, prefix: &'static str) {
    let weights = reorder_weights(layer.nb_inputs, layer.nb_neurons, layer.input_weights);
    print_array(format!("{}_BIAS", prefix), layer.bias);
    print_array(format!("{}_WEIGHTS", prefix), &weights);
    println!(
        "static {}: DenseLayer = DenseLayer {{
        bias: &{}_BIAS,
        input_weights: &{}_WEIGHTS,
        nb_inputs: {},
        nb_neurons: {},
        activation: {},
    }};",
        prefix,
        prefix,
        prefix,
        layer.nb_inputs,
        layer.nb_neurons,
        act_string(layer.activation)
    );
}

fn main() {
    let m = &model::MODEL;
    println!("
        // This file was automatically generated from a Keras model, and then manually ported to rust.
        // Then the ordering of the weights was changed with `munge.rs`.
        // TODO: support generating this file in rust direction.

        use crate::rnn::{{Activation, DenseLayer, GruLayer, RnnModel}};
        ");
    reorder_dense_layer(&m.input_dense, "INPUT_DENSE");
    reorder_gru_layer(&m.vad_gru, "VAD_GRU");
    reorder_gru_layer(&m.noise_gru, "NOISE_GRU");
    reorder_gru_layer(&m.denoise_gru, "DENOISE_GRU");
    reorder_dense_layer(&m.denoise_output, "DENOISE_OUTPUT");
    reorder_dense_layer(&m.vad_output, "VAD_OUTPUT");

    println!(
        "pub static MODEL: RnnModel = RnnModel {{
    input_dense_size: 24,
    input_dense: INPUT_DENSE,
    vad_gru_size: 24,
    vad_gru: VAD_GRU,
    noise_gru_size: 48,
    noise_gru: NOISE_GRU,
    denoise_gru_size: 96,
    denoise_gru: DENOISE_GRU,
    denoise_output_size: 22,
    denoise_output: DENOISE_OUTPUT,
    vad_output_size: 1,
    vad_output: VAD_OUTPUT,
    }};
    "
    );
}
