use assert_cmd::prelude::*;
use assert_fs::prelude::*;
use std::process::Command;

#[test]
fn basic_usage() -> anyhow::Result<()> {
    let mut cmd = Command::cargo_bin("nnnoiseless")?;
    let tmp = assert_fs::TempDir::new()?;
    let input = tmp.child("input.raw");
    let output = tmp.child("output.raw");
    input.write_binary(&vec![0u8; 480 * 10])?;

    cmd.arg(input.path()).arg(output.path());
    cmd.assert().success();
    assert!(output.exists());
    Ok(())
}

#[test]
fn invalid_wav() -> anyhow::Result<()> {
    let mut cmd = Command::cargo_bin("nnnoiseless")?;
    let tmp = assert_fs::TempDir::new()?;
    let input = tmp.child("input.wav");
    let output = tmp.child("output.wav");
    input.write_binary(&vec![0u8; 480 * 10])?;

    cmd.arg(input.path()).arg(output.path());
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("no RIFF tag found"));

    let input = tmp.child("input.raw");
    input.write_binary(&vec![0u8; 480 * 10])?;
    let mut cmd = Command::cargo_bin("nnnoiseless")?;
    cmd.arg("--wav-in").arg(input.path()).arg(output.path());
    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("no RIFF tag found"));

    Ok(())
}
