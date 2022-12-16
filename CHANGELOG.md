# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2022-12-16

### Changed
- Reverts to the weights from RNNoise, which work better for voice detection.

## [0.5.0] - 2022-05-26

### Added
- Adds a more user-friendly API, based on the `dasp` crate. This adds a dependency on `dasp`,
  which can be removed by disabling the (on-by-default) "dasp" feature.

### Changed
- Publically exposes some training-related functions, even when the `train` feature is disabled.

## [0.4.0] - 2022-2-22

### Added
- New `train` feature allows for generating training features.
- Documentation for training new available in `train/README.md`.
- `RnnModel::from_static_bytes` allows for loading models without allocation.

### Changed
- `RnnModel::from_read` replaced by `RnnModel::from_bytes`. *(Breaking change!)*
