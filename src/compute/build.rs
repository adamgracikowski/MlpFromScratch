//! Records the git commit of the sources, written to the training artifacts
//! (`metadata.json`) so that every run can be traced back to its code.
//!
//! Sets `COMPUTE_GIT_COMMIT` (`unknown` without git) and `COMPUTE_GIT_DIRTY`
//! (`true` when files of this crate differ from the commit, `unknown`
//! without git).

use std::process::Command;

fn main() {
    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    // Only the crate directory: unrelated changes elsewhere in the
    // repository do not change the binary.
    let dirty = match git(&["status", "--porcelain", "--untracked-files=no", "--", "."]) {
        Some(status) if status.is_empty() => "false",
        Some(_) => "true",
        None => "unknown",
    };
    println!("cargo:rustc-env=COMPUTE_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=COMPUTE_GIT_DIRTY={dirty}");

    // Rerun after commits, checkouts and staging, and when sources change.
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        for file in ["HEAD", "index", "logs/HEAD"] {
            println!("cargo:rerun-if-changed={git_dir}/{file}");
        }
    }
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
}

/// Output of a successful git command in the crate directory, trimmed.
fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_owned())
}
