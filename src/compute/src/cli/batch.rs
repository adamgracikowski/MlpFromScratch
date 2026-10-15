//! Training several experiments in parallel.

use std::fmt;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use miette::{IntoDiagnostic, miette};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

use super::progress::MetricsLine;
use super::train::{RunResult, TrainOptions, train_quiet};
use crate::training::StopReason;

/// Trains `dirs` with up to `jobs` runs at a time (default: the number of
/// CPUs), printing one line per finished run and a final summary. Runs are
/// named by their path relative to the command-line argument `roots` they
/// were found in.
///
/// A failed run does not stop the others; their errors are printed at the
/// end and make the command fail. Every run has its own seeded random
/// streams, so the results do not depend on `jobs` or on the order in which
/// runs finish.
pub(super) fn run(
    dirs: &[PathBuf],
    roots: &[PathBuf],
    options: &TrainOptions,
    jobs: Option<NonZeroUsize>,
) -> miette::Result<()> {
    let jobs = jobs
        .or_else(|| thread::available_parallelism().ok())
        .map_or(1, NonZeroUsize::get)
        .min(dirs.len());
    println!("training:   {} run(s), {jobs} at a time", dirs.len());
    let start = Instant::now();
    let results = train_all(dirs, roots, options, jobs)?;
    let counts = Counts::new(&results);
    println!(
        "finished:   {} run(s) in {:.2} s: {counts}",
        dirs.len(),
        start.elapsed().as_secs_f64()
    );

    for (dir, result) in dirs.iter().zip(results) {
        if let Err(report) = result {
            let report = report.wrap_err(format!("run `{}` failed", dir.display()));
            eprintln!("{report:?}");
        }
    }
    if counts.failed > 0 {
        return Err(miette!("{} of {} run(s) failed", counts.failed, dirs.len()));
    }
    Ok(())
}

/// Trains `dirs` on a pool of `jobs` threads and returns their results in
/// the order of `dirs`, printing a line whenever a run finishes.
pub(super) fn train_all(
    dirs: &[PathBuf],
    roots: &[PathBuf],
    options: &TrainOptions,
    jobs: usize,
) -> miette::Result<Vec<miette::Result<RunResult>>> {
    let pool = ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .into_diagnostic()?;
    let finished = AtomicUsize::new(0);
    let results = pool.install(|| {
        dirs.par_iter()
            .map(|dir| {
                let start = Instant::now();
                let result = train_quiet(dir, options);
                let index = finished.fetch_add(1, Ordering::Relaxed) + 1;
                println!(
                    "{}",
                    RunLine {
                        index,
                        total: dirs.len(),
                        dir: run_name(dir, roots),
                        result: &result,
                        elapsed: start.elapsed(),
                    }
                );
                result
            })
            .collect()
    });
    Ok(results)
}

/// Path of `dir` relative to the first of `roots` that contains it, or the
/// whole path for a root that is the experiment itself.
fn run_name<'a>(dir: &'a Path, roots: &[PathBuf]) -> &'a Path {
    roots
        .iter()
        .filter_map(|root| dir.strip_prefix(root).ok())
        .find(|relative| !relative.as_os_str().is_empty())
        .unwrap_or(dir)
}

/// One finished run, e.g.
/// `[ 3/18] <dir>: completed after 200 epoch(s), test loss 0.1234 acc 97.20% (2.31 s)`.
struct RunLine<'a> {
    index: usize,
    total: usize,
    dir: &'a Path,
    result: &'a miette::Result<RunResult>,
    elapsed: Duration,
}

impl fmt::Display for RunLine<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let width = self.total.to_string().len();
        write!(
            f,
            "[{:>width$}/{}] {}: ",
            self.index,
            self.total,
            self.dir.display()
        )?;
        let outcome = match self.result {
            Ok(RunResult::Trained(outcome)) => outcome,
            Ok(RunResult::Skipped) => return write!(f, "skipped (already trained)"),
            Err(_) => return write!(f, "failed (see below)"),
        };
        let epochs = outcome.epochs;
        match outcome.stop {
            StopReason::Completed => write!(f, "completed after {epochs} epoch(s)")?,
            StopReason::EarlyStopped => write!(f, "early stopped after {epochs} epoch(s)")?,
            StopReason::Diverged { epoch, .. } => write!(f, "diverged in epoch {epoch}")?,
        }
        if let Some(best) = outcome.best_epoch {
            write!(f, " (best {best})")?;
        }
        if !matches!(outcome.stop, StopReason::Diverged { .. }) {
            write!(f, ", test {}", MetricsLine(&outcome.metrics.test))?;
        }
        write!(f, " ({:.2} s)", self.elapsed.as_secs_f64())
    }
}

/// Number of runs per result.
#[derive(Debug, Default, PartialEq, Eq)]
struct Counts {
    completed: usize,
    early_stopped: usize,
    diverged: usize,
    skipped: usize,
    failed: usize,
}

impl Counts {
    fn new(results: &[miette::Result<RunResult>]) -> Self {
        let mut counts = Self::default();
        for result in results {
            let count = match result {
                Ok(RunResult::Trained(outcome)) => match outcome.stop {
                    StopReason::Completed => &mut counts.completed,
                    StopReason::EarlyStopped => &mut counts.early_stopped,
                    StopReason::Diverged { .. } => &mut counts.diverged,
                },
                Ok(RunResult::Skipped) => &mut counts.skipped,
                Err(_) => &mut counts.failed,
            };
            *count += 1;
        }
        counts
    }
}

/// Non-zero counts, e.g. `15 completed, 1 diverged, 2 failed`.
impl fmt::Display for Counts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = [
            (self.completed, "completed"),
            (self.early_stopped, "early stopped"),
            (self.diverged, "diverged"),
            (self.skipped, "skipped"),
            (self.failed, "failed"),
        ]
        .into_iter()
        .filter(|(count, _)| *count > 0)
        .map(|(count, name)| format!("{count} {name}"))
        .collect();
        write!(f, "{}", parts.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;
    use crate::artifacts::ARTIFACTS_DIR;
    use crate::config::{CLASSIFICATION, CONFIG_FILE_NAME};

    /// Experiment directories `run-<i>` with the given configs, whose
    /// datasets are temporary CSV files.
    fn experiments(configs: &[String]) -> (TempDir, Vec<PathBuf>) {
        let root = TempDir::new().unwrap();
        let train = root.path().join("train.csv");
        let test = root.path().join("test.csv");
        let mut rows = String::from("x,y,cls\n");
        for i in 0..24 {
            let x = f64::from(i) / 24.0;
            rows.push_str(&format!("{x},{},{}\n", 1.0 - x, 1 + i % 2));
        }
        fs::write(&train, &rows).unwrap();
        fs::write(&test, &rows).unwrap();
        let dirs = configs
            .iter()
            .enumerate()
            .map(|(index, config)| {
                let dir = root.path().join(format!("run-{index}"));
                fs::create_dir(&dir).unwrap();
                let text = config
                    .replace("\"train.csv\"", &format!("'{}'", train.display()))
                    .replace("\"test.csv\"", &format!("'{}'", test.display()));
                fs::write(dir.join(CONFIG_FILE_NAME), text).unwrap();
                dir
            })
            .collect();
        (root, dirs)
    }

    fn config(seed: u64) -> String {
        CLASSIFICATION
            .replace("epochs = 200", "epochs = 3")
            .replace("seed = 42", &format!("seed = {seed}"))
    }

    fn model(dir: &Path) -> Vec<u8> {
        fs::read(dir.join(ARTIFACTS_DIR).join("model/layer_0_weights.npy")).unwrap()
    }

    #[test]
    fn names_runs_relative_to_their_root() {
        let roots = [PathBuf::from("single"), PathBuf::from("study")];
        let name = |dir: &str| run_name(Path::new(dir), &roots).to_path_buf();
        assert_eq!(name("study/runs/a/seed-1"), Path::new("runs/a/seed-1"));
        assert_eq!(name("single"), Path::new("single"));
        assert_eq!(name("other/run"), Path::new("other/run"));
    }

    #[test]
    fn results_do_not_depend_on_jobs() {
        let configs: Vec<String> = (1..=4).map(config).collect();
        let (_sequential_root, sequential) = experiments(&configs);
        let (_parallel_root, parallel) = experiments(&configs);
        let options = TrainOptions::default();
        train_all(&sequential, &[], &options, 1).unwrap();
        train_all(&parallel, &[], &options, 4).unwrap();
        for (a, b) in sequential.iter().zip(&parallel) {
            assert_eq!(model(a), model(b));
        }
        assert_ne!(model(&sequential[0]), model(&sequential[1]));
    }

    #[test]
    fn failed_run_does_not_stop_the_others() {
        let configs = [config(1), config(2).replace("seed = 2", ""), config(3)];
        let (_root, dirs) = experiments(&configs);
        let results = train_all(&dirs, &[], &TrainOptions::default(), 2).unwrap();
        let counts = Counts::new(&results);
        assert_eq!(
            counts,
            Counts {
                completed: 2,
                failed: 1,
                ..Counts::default()
            }
        );
        assert!(results[1].is_err());
        assert!(dirs[2].join(ARTIFACTS_DIR).join("metadata.json").exists());
        assert_eq!(counts.to_string(), "2 completed, 1 failed");
    }

    #[test]
    fn skips_runs_trained_with_the_same_config() {
        let (_root, dirs) = experiments(&[config(1), config(2)]);
        let options = TrainOptions {
            skip_trained: true,
            ..TrainOptions::default()
        };
        train_all(&dirs, &[], &options, 2).unwrap();
        let path = dirs[1].join(CONFIG_FILE_NAME);
        let changed = fs::read_to_string(&path)
            .unwrap()
            .replace("batch_size = 32", "batch_size = 8");
        fs::write(&path, changed).unwrap();

        let results = train_all(&dirs, &[], &options, 2).unwrap();
        assert!(matches!(results[0], Ok(RunResult::Skipped)));
        assert!(matches!(results[1], Ok(RunResult::Trained(_))));

        let retrain = train_all(&dirs[..1], &[], &TrainOptions::default(), 1).unwrap();
        assert!(matches!(retrain[0], Ok(RunResult::Trained(_))));
    }
}
