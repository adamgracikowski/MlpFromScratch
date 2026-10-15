//! Writing the run directories of a study and their index.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::{INDEX_FILE_NAME, RUNS_DIR, Study, StudyError};
use crate::config::CONFIG_FILE_NAME;

/// What [`write_runs`] changed on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteReport {
    /// Configs written because they were new or different.
    pub written: usize,
    /// Configs left untouched because their text did not change; their
    /// artifacts stay valid.
    pub unchanged: usize,
    /// Run and variant directories that no longer belong to the study,
    /// relative to the study directory, removed with their artifacts.
    pub removed: Vec<PathBuf>,
    /// Path of `runs/index.csv`.
    pub index: PathBuf,
}

/// Writes `config.toml` of every run, removes the directories under `runs/`
/// that the study no longer generates and writes `runs/index.csv`.
///
/// Files whose text did not change are not rewritten, so that the
/// configs and the artifacts trained from them stay in sync. The index has
/// one row per run: `dir,variant,seed`, then the level of every axis.
pub fn write_runs(study: &Study) -> Result<WriteReport, StudyError> {
    let mut written = 0;
    let mut unchanged = 0;
    for run in &study.runs {
        let path = study.dir.join(&run.spec.dir).join(CONFIG_FILE_NAME);
        if write_if_changed(&path, &run.text)? {
            written += 1;
        } else {
            unchanged += 1;
        }
    }
    let removed = remove_orphans(study)?;
    let index = study.dir.join(RUNS_DIR).join(INDEX_FILE_NAME);
    write_if_changed(&index, &index_csv(study))?;
    Ok(WriteReport {
        written,
        unchanged,
        removed,
        index,
    })
}

/// Writes `text` to `path` unless the file already holds it. Returns
/// whether the file was written.
fn write_if_changed(path: &Path, text: &str) -> Result<bool, StudyError> {
    if fs::read_to_string(path).is_ok_and(|current| current == text) {
        return Ok(false);
    }
    let parent = path.parent().expect("a file path has a parent");
    fs::create_dir_all(parent).map_err(write_error(parent))?;
    fs::write(path, text).map_err(write_error(path))?;
    Ok(true)
}

/// Removes variant and seed directories under `runs/` that no run uses.
fn remove_orphans(study: &Study) -> Result<Vec<PathBuf>, StudyError> {
    let runs: HashSet<&Path> = study
        .runs
        .iter()
        .map(|run| run.spec.dir.as_path())
        .collect();
    let variants: HashSet<&str> = study
        .runs
        .iter()
        .map(|run| run.spec.variant.as_str())
        .collect();
    let mut removed = Vec::new();
    for variant in subdirectories(&study.dir.join(RUNS_DIR))? {
        let relative = Path::new(RUNS_DIR).join(&variant);
        if !variants.contains(variant.as_str()) {
            remove(&study.dir.join(&relative))?;
            removed.push(relative);
            continue;
        }
        for seed in subdirectories(&study.dir.join(&relative))? {
            let relative = relative.join(seed);
            if !runs.contains(relative.as_path()) {
                remove(&study.dir.join(&relative))?;
                removed.push(relative);
            }
        }
    }
    removed.sort();
    Ok(removed)
}

/// Names of the directories in `dir`. None when `dir` does not exist.
fn subdirectories(dir: &Path) -> Result<Vec<String>, StudyError> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(write_error(dir)(err)),
    };
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry.map_err(write_error(dir))?;
        if entry.file_type().map_err(write_error(dir))?.is_dir() {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    names.sort();
    Ok(names)
}

fn remove(dir: &Path) -> Result<(), StudyError> {
    fs::remove_dir_all(dir).map_err(write_error(dir))
}

/// Text of `runs/index.csv`. Paths use `/` on every platform.
fn index_csv(study: &Study) -> String {
    let mut writer = csv::Writer::from_writer(Vec::new());
    let mut header = vec!["dir", "variant", "seed"];
    header.extend(study.config.axes.iter().map(|axis| axis.name.as_str()));
    writer.write_record(&header).expect("writing to memory");
    for run in &study.runs {
        let spec = &run.spec;
        let dir: Vec<String> = spec
            .dir
            .iter()
            .map(|part| part.to_string_lossy().into_owned())
            .collect();
        let mut record = vec![dir.join("/"), spec.variant.clone(), spec.seed.to_string()];
        record.extend(spec.levels.iter().cloned());
        writer.write_record(&record).expect("writing to memory");
    }
    let bytes = writer.into_inner().expect("flushing to memory");
    String::from_utf8(bytes).expect("the index is built from UTF-8 strings")
}

fn write_error(path: &Path) -> impl Fn(io::Error) -> StudyError + '_ {
    move |source| StudyError::Write {
        path: path.to_path_buf(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::fixtures::{STUDY, study_dir};
    use crate::study::load_study;

    #[test]
    fn writes_configs_and_index() {
        let dir = study_dir(STUDY);
        let study = load_study(dir.path()).unwrap();
        let report = write_runs(&study).unwrap();
        assert_eq!((report.written, report.unchanged), (8, 0));
        assert!(report.removed.is_empty());
        let config = dir
            .path()
            .join("runs/tanh__8-8/seed-2")
            .join(CONFIG_FILE_NAME);
        assert_eq!(fs::read_to_string(config).unwrap(), study.runs[7].text);

        let index = fs::read_to_string(report.index).unwrap();
        let lines: Vec<&str> = index.lines().collect();
        assert_eq!(lines.len(), 9);
        assert_eq!(lines[0], "dir,variant,seed,activation,layers");
        assert_eq!(lines[1], "runs/sigmoid__0/seed-1,sigmoid__0,1,sigmoid,0");
        assert_eq!(lines[8], "runs/tanh__8-8/seed-2,tanh__8-8,2,tanh,8-8");
    }

    #[test]
    fn keeps_unchanged_configs_and_their_artifacts() {
        let dir = study_dir(STUDY);
        write_runs(&load_study(dir.path()).unwrap()).unwrap();
        let artifact = dir
            .path()
            .join("runs/sigmoid__0/seed-1/artifacts/metadata.json");
        fs::create_dir_all(artifact.parent().unwrap()).unwrap();
        fs::write(&artifact, "{}").unwrap();

        let report = write_runs(&load_study(dir.path()).unwrap()).unwrap();
        assert_eq!((report.written, report.unchanged), (0, 8));
        assert!(artifact.exists());
    }

    #[test]
    fn removes_runs_outside_the_study() {
        let dir = study_dir(STUDY);
        for orphan in ["runs/relu__0/seed-1", "runs/sigmoid__0/seed-9"] {
            fs::create_dir_all(dir.path().join(orphan)).unwrap();
        }
        let report = write_runs(&load_study(dir.path()).unwrap()).unwrap();
        let expected: Vec<PathBuf> = ["runs/relu__0", "runs/sigmoid__0/seed-9"]
            .iter()
            .map(PathBuf::from)
            .collect();
        assert_eq!(report.removed, expected);
        assert!(!dir.path().join("runs/relu__0").exists());
        assert!(dir.path().join("runs/sigmoid__0/seed-1").exists());
    }
}
