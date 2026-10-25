//! Merge exposure-bracketed raws into HDR TIFFs, headless.
//!
//! ```text
//! hdr <dir>                      every bracketed set found in the folder
//! hdr <raw> <raw> [<raw>...]     these frames, as one set
//!     [--style natural|hdr|dramatic] [--out DIR]
//! ```
//!
//! Each set becomes `<first frame>_hdr.tif` (in `--out` if given), which the
//! editor opens like a raw. Same code path as the GUI's merge.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use raw_editor::hdr::{self, Style};

const USAGE: &str = "usage: hdr <dir | raw raw [raw...]> [--style natural|hdr|dramatic] [--out DIR]";

fn main() -> Result<()> {
    let mut inputs = Vec::new();
    let mut style = Style::default();
    let mut out_dir: Option<PathBuf> = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--style" => {
                let s = it.next().context("--style needs a value")?;
                style = Style::parse(&s).with_context(|| format!("unknown style {s:?}\n{USAGE}"))?;
            }
            "--out" => out_dir = Some(PathBuf::from(it.next().context("--out needs a directory")?)),
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other if other.starts_with('-') => bail!("unknown option {other:?}\n{USAGE}"),
            other => inputs.push(PathBuf::from(other)),
        }
    }

    let sets = match inputs.as_slice() {
        [] => bail!("{USAGE}"),
        [dir] if dir.is_dir() => {
            let list = raw_editor::formats::photos_in(dir)?;
            let sets = raw_editor::bracket::sets_in(&list);
            if sets.is_empty() {
                bail!("no bracketed sets found in {}", dir.display());
            }
            sets
        }
        [_] => bail!("one frame is not a bracketed set\n{USAGE}"),
        files => vec![files.to_vec()],
    };
    if let Some(dir) = &out_dir {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }

    let mut failed = 0;
    for set in &sets {
        let names: Vec<String> =
            set.iter().map(|p| p.file_name().unwrap_or_default().to_string_lossy().into_owned()).collect();
        println!("{}", names.join(" + "));
        let out = out_dir.as_ref().map(|d| d.join(hdr::output_path(set).file_name().unwrap_or_default()));
        let started = std::time::Instant::now();
        match hdr::merge_files(set, style, out.as_deref(), &|step| println!("  {step}…")) {
            Ok((path, report)) => println!(
                "  → {} ({:.1}s): reference {}, shifts {:?}, scales {:?}, ghosts {:?}, +{:.1} stops",
                path.display(),
                started.elapsed().as_secs_f32(),
                names[report.reference],
                report.shifts,
                report.scales.iter().map(|s| format!("{s:.3}")).collect::<Vec<_>>(),
                report.ghosted.iter().map(|g| format!("{:.1}%", g * 100.0)).collect::<Vec<_>>(),
                report.extra_stops,
            ),
            Err(e) => {
                eprintln!("  failed: {e:#}");
                failed += 1;
            }
        }
    }
    if failed > 0 {
        bail!("{failed} of {} sets failed", sets.len());
    }
    Ok(())
}
