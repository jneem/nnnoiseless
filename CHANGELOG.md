# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New `train` feature allows for generating training features.
- Documentation for training new available in `train/README.md`.
- `RnnModel::from_static_bytes` allows for loading models without allocation.

### Changed
- `RnnModel::from_read` replaced by `RnnModel::from_bytes`. *(Breaking change!)*
