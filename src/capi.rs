use std::boxed::Box;
use std::os::raw::{c_float, c_int};

use libc::FILE;

pub struct DenoiseState(crate::DenoiseState);

impl Default for DenoiseState {
    fn default() -> Self {
        DenoiseState(crate::DenoiseState::default())
    }
}

impl DenoiseState {
    fn boxed() -> Box<DenoiseState> {
        Box::new(DenoiseState::default())
    }
}

pub struct RNNModel(crate::rnn::RnnModel);

/// Return the number of samples processed at time
///
/// See `rnnoise_process_frame()`
#[no_mangle]
pub unsafe extern "C" fn rnnoise_get_frame_size() -> c_int {
    crate::DenoiseState::FRAME_SIZE as c_int
}

/// Return the size of DenoiseState
///
/// It should be avoided, use directly `rnnoise_create`
#[no_mangle]
pub unsafe extern "C" fn rnnoise_get_size() -> c_int {
    std::mem::size_of::<DenoiseState>() as c_int
}

/// Init a pre-allocated DenoiseState
///
/// It should be avoided, use directly `rnnoise_create`
#[no_mangle]
pub unsafe extern "C" fn rnnoise_init(st: *mut DenoiseState, model: *mut RNNModel) -> c_int {
    let state = DenoiseState::default();

    if !model.is_null() {
        todo!("Custom RnnModel model");
    }

    *st = state;

    0
}

/// Create and initialize a DenoseState
///
/// Use `rnnoise_destroy` to deallocate it
#[no_mangle]
pub unsafe extern "C" fn rnnoise_create(model: *mut RNNModel) -> *mut DenoiseState {
    if !model.is_null() {
        todo!("Custom RnnModel model");
    }

    Box::into_raw(DenoiseState::boxed())
}

/// Deallocate and destroy a DenoiseState
///
/// Use it only on pointers returned by `rnnoise_create`.
#[no_mangle]
pub unsafe extern "C" fn rnnoise_destroy(st: *mut DenoiseState) {
    let _ = Box::from_raw(st);
}

/// Processes a chunk of samples.
///
/// It processes `rnnoise_get_frame_size()` samples at time.
///
/// The current output of `process_frame` depends on the current input, but also on the
/// preceding inputs. Because of this, you might prefer to discard the very first output; it
/// will contain some fade-in artifacts.
#[no_mangle]
pub unsafe extern "C" fn rnnoise_process_frame(
    st: *mut DenoiseState,
    out: *mut c_float,
    input: *mut c_float,
) -> c_float {
    let state = st.as_mut().expect("Invalid pointer");
    let output = std::slice::from_raw_parts_mut(out, crate::DenoiseState::FRAME_SIZE);
    let input = std::slice::from_raw_parts(input, crate::DenoiseState::FRAME_SIZE);

    state.0.process_frame(output, input)
}

/// Load a Custom Model from file
///
/// TODO: unimplemented
#[no_mangle]
pub unsafe extern "C" fn rnnoise_model_from_file(file: *mut FILE) -> *mut RNNModel {
    let _ = file;
    todo!("Custom Model")
}

/// Free a Custom Model
///
/// See `rnnoise_model_from_file`
#[no_mangle]
pub unsafe extern "C" fn rnnoise_model_free(model: *mut RNNModel) {
    let _ = std::sync::Arc::from_raw(model);
}
