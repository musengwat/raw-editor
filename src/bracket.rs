//! Finding exposure-bracketed sets among a folder's raws.
//!
//! Cameras do not mark a bracket in standard EXIF (Sony keeps "Exposure
//! Bracketing" and the frame's place in the sequence in its maker note), so
//! sets are recognised by what every body does record: frames from the same
//! camera, fired one straight after another, at different exposures. A Sony
//! A7 IV bracket's frames are ~170 ms apart (plus the shutter time), while
//! separate presses of the shutter button are a second or more apart.
//!
//! Exposures within a set may repeat: in auto-ISO the camera brackets by ISO,
//! and when the base frame is already at ISO 100 the −2 EV frame cannot go
//! lower and comes out identical. So equal exposures do not split a set; the
//! set only has to span some range.

use std::path::{Path, PathBuf};

use crate::metadata::ExportMetadata;

/// Longest pause between two frames of one set, on top of the earlier
/// frame's shutter time, when both carry sub-second timestamps.
const MAX_GAP_S: f64 = 0.5;
/// The same without sub-second timestamps: whole seconds can make
/// back-to-back frames look a second apart.
const MAX_GAP_WHOLE_S: f64 = 1.5;
/// A set must span at least this much exposure, or it is a burst.
pub const MIN_SPREAD_EV: f32 = 0.5;
/// The most frames one set can have (cameras offer up to 9).
pub const MAX_FRAMES: usize = 9;

/// What bracket detection needs to know about one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Shot {
    pub path: PathBuf,
    /// Make and model.
    pub camera: String,
    /// Capture time in seconds (only differences matter), and whether it
    /// has sub-second precision.
    pub time: Option<(f64, bool)>,
    /// Shutter time in seconds.
    pub shutter: Option<f64>,
    /// How bright the frame was exposed, in stops ([`exposure_ev`]).
    pub ev: Option<f32>,
}

impl Shot {
    pub fn from_metadata(path: &Path, m: &ExportMetadata) -> Self {
        let rat = |v: Option<(u32, u32)>| v.filter(|r| r.1 != 0).map(|(n, d)| n as f64 / d as f64);
        let shutter = rat(m.exposure_time);
        let fnumber = rat(m.fnumber);
        let ev = match (shutter, fnumber, m.iso) {
            (Some(t), Some(n), Some(iso)) if t > 0.0 && n > 0.0 && iso > 0 => Some(exposure_ev(t, n, iso as f64)),
            _ => None,
        };
        let time = m.date_time_original.as_deref().and_then(|d| {
            let whole = parse_exif_time(d)?;
            let sub = m.sub_sec_time_original.as_deref().and_then(sub_seconds);
            Some((whole + sub.unwrap_or(0.0), sub.is_some()))
        });
        let camera = format!("{} {}", m.make.as_deref().unwrap_or(""), m.model.as_deref().unwrap_or(""));
        Self { path: path.to_owned(), camera, time, shutter, ev }
    }

    /// Read a raw's metadata (no pixels decoded).
    pub fn read(path: &Path) -> anyhow::Result<Self> {
        Ok(Self::from_metadata(path, &crate::metadata::read(path)?))
    }
}

/// How brightly a frame is exposed, in stops: shutter time × ISO over the
/// aperture's area. One more stop of shutter, ISO or aperture is +1, so two
/// frames' values differ by the exposure difference between them.
pub fn exposure_ev(shutter: f64, fnumber: f64, iso: f64) -> f32 {
    (shutter * iso / 100.0 / (fnumber * fnumber)).log2() as f32
}

/// `"2026:10:20 15:15:51"` → seconds since 1970 (time zone ignored; only
/// differences are used).
pub fn parse_exif_time(s: &str) -> Option<f64> {
    let s = s.trim();
    let num = |r: std::ops::Range<usize>| s.get(r)?.parse::<i64>().ok();
    let (y, mo, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (h, mi, se) = (num(11..13)?, num(14..16)?, num(17..19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || se > 60 {
        return None;
    }
    Some((days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + se) as f64)
}

/// `"844"` → 0.844. The digits are a decimal fraction whatever their count.
fn sub_seconds(s: &str) -> Option<f64> {
    let digits = s.trim().trim_end_matches('\0').trim();
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    format!("0.{digits}").parse().ok()
}

/// Days from 1970-01-01 to a proleptic Gregorian date (Howard Hinnant's
/// algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Whether `b` was fired straight after `a` as part of the same set.
fn follows(a: &Shot, b: &Shot) -> bool {
    let (Some((ta, sub_a)), Some((tb, sub_b))) = (a.time, b.time) else { return false };
    let gap = if sub_a && sub_b { MAX_GAP_S } else { MAX_GAP_WHOLE_S };
    let dt = tb - ta;
    a.camera == b.camera && a.ev.is_some() && b.ev.is_some() && dt >= 0.0 && dt <= gap + a.shutter.unwrap_or(0.0)
}

/// Bracketed sets among `shots` (in file order), as runs of indices. Frames
/// that belong to no set are left out.
pub fn find_sets(shots: &[Shot]) -> Vec<Vec<usize>> {
    let mut sets = Vec::new();
    let mut run: Vec<usize> = Vec::new();
    let mut close = |run: &mut Vec<usize>| {
        let evs: Vec<f32> = run.iter().filter_map(|&i| shots[i].ev).collect();
        let lo = evs.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = evs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if run.len() >= 2 && hi - lo >= MIN_SPREAD_EV {
            sets.push(std::mem::take(run));
        }
        run.clear();
    };
    for i in 0..shots.len() {
        let joins = run.last().is_some_and(|&j| follows(&shots[j], &shots[i])) && run.len() < MAX_FRAMES;
        if !joins {
            close(&mut run);
        }
        run.push(i);
    }
    close(&mut run);
    sets
}

/// The bracketed set `photo` belongs to, looking at its neighbours in
/// `folder` (sorted, as [`crate::formats::photos_in`] returns it). Only raws
/// take part. Frames whose metadata cannot be read just end a set.
pub fn set_containing(folder: &[PathBuf], photo: &Path) -> Option<Vec<PathBuf>> {
    if !crate::formats::is_raw(photo) {
        return None;
    }
    let at = folder.iter().position(|p| p == photo)?;
    let lo = at.saturating_sub(MAX_FRAMES - 1);
    let hi = (at + MAX_FRAMES).min(folder.len());
    let shots: Vec<Shot> = folder[lo..hi]
        .iter()
        .filter(|p| crate::formats::is_raw(p))
        .filter_map(|p| Shot::read(p).ok())
        .collect();
    find_sets(&shots)
        .into_iter()
        .map(|set| set.into_iter().map(|i| shots[i].path.clone()).collect::<Vec<_>>())
        .find(|set| set.iter().any(|p| p == photo))
}

/// Every bracketed set in a folder's raws.
pub fn sets_in(folder: &[PathBuf]) -> Vec<Vec<PathBuf>> {
    let shots: Vec<Shot> = folder
        .iter()
        .filter(|p| crate::formats::is_raw(p))
        .filter_map(|p| match Shot::read(p) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("brackets: skipping {}: {e:#}", p.display());
                None
            }
        })
        .collect();
    find_sets(&shots).into_iter().map(|set| set.into_iter().map(|i| shots[i].path.clone()).collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame `ms` milliseconds into the day at `ev`.
    fn shot(name: &str, ms: u64, ev: f32) -> Shot {
        Shot {
            path: PathBuf::from(name),
            camera: "Sony ILCE-7M4".into(),
            time: Some((ms as f64 / 1000.0, true)),
            shutter: Some(1.0 / 250.0),
            ev: Some(ev),
        }
    }

    #[test]
    fn exif_times_and_sub_seconds() {
        let a = parse_exif_time("2026:10:20 15:15:51").unwrap();
        let b = parse_exif_time("2026:10:21 00:00:00").unwrap();
        assert_eq!(b - a, (8 * 3600 + 44 * 60 + 9) as f64);
        assert_eq!(parse_exif_time("1970:01:01 00:00:00"), Some(0.0));
        assert_eq!(parse_exif_time("2026:13:01 00:00:00"), None);
        assert_eq!(parse_exif_time("    :  :     :  :  "), None, "cameras write blanks for unknown dates");
        assert_eq!(sub_seconds("844"), Some(0.844));
        assert_eq!(sub_seconds("05"), Some(0.05));
        assert_eq!(sub_seconds("ab"), None);
    }

    #[test]
    fn exposure_ev_counts_stops() {
        let base = exposure_ev(1.0 / 250.0, 3.5, 2500.0);
        // DSC01195..97: same shutter and aperture, ISO 2500 / 640 / 10000.
        assert!((exposure_ev(1.0 / 250.0, 3.5, 640.0) - base + 1.97).abs() < 0.01);
        assert!((exposure_ev(1.0 / 250.0, 3.5, 10000.0) - base - 2.0).abs() < 0.01);
        // A stop of aperture is a factor √2 in f-number.
        assert!((exposure_ev(1.0 / 250.0, 3.5 * 2f64.sqrt(), 2500.0) - base + 1.0).abs() < 1e-4);
    }

    #[test]
    fn three_frame_sets_back_to_back() {
        // Two sets a few seconds apart, and a lone frame between them.
        let shots = vec![
            shot("a1", 1_000, 0.0),
            shot("a2", 1_166, -2.0),
            shot("a3", 1_332, 2.0),
            shot("lone", 5_000, 0.0),
            shot("b1", 9_000, 0.0),
            shot("b2", 9_166, -2.0),
            shot("b3", 9_332, 2.0),
        ];
        assert_eq!(find_sets(&shots), vec![vec![0, 1, 2], vec![4, 5, 6]]);
    }

    #[test]
    fn sets_split_on_a_pause_even_when_close() {
        // DSC01350–54: a two-frame set, 1.36 s pause, then a full set.
        let shots = vec![
            shot("1350", 29_204, 0.0),
            shot("1351", 29_370, -2.0),
            shot("1352", 30_733, 0.0),
            shot("1353", 30_899, -2.0),
            shot("1354", 31_065, 2.0),
        ];
        assert_eq!(find_sets(&shots), vec![vec![0, 1], vec![2, 3, 4]]);
    }

    #[test]
    fn repeated_exposures_stay_in_the_set() {
        // DSC01303–05: ISO 100 floor, so 0 and −2 EV came out the same.
        let shots = vec![shot("a", 0, 0.0), shot("b", 166, 0.0), shot("c", 332, 1.3)];
        assert_eq!(find_sets(&shots), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn a_burst_is_not_a_bracket() {
        let shots: Vec<Shot> = (0..5).map(|i| shot(&format!("{i}"), i * 100, 0.1 * i as f32)).collect();
        assert!(find_sets(&shots).is_empty(), "0.4 EV of drift is metering, not bracketing");
    }

    #[test]
    fn long_shutter_widens_the_gap() {
        // DSC01248–50: 0.5 s exposures, frames ~0.62 s apart.
        let mut shots = vec![shot("a", 0, 0.0), shot("b", 622, -2.0), shot("c", 1_245, 2.0)];
        for s in &mut shots {
            s.shutter = Some(0.5);
        }
        assert_eq!(find_sets(&shots), vec![vec![0, 1, 2]]);
        for s in &mut shots {
            s.shutter = Some(1.0 / 1000.0);
        }
        assert!(find_sets(&shots).is_empty());
    }

    #[test]
    fn different_cameras_or_missing_data_never_join() {
        let mut other = shot("b", 166, -2.0);
        other.camera = "Panasonic DMC-GF1".into();
        assert!(find_sets(&[shot("a", 0, 0.0), other]).is_empty());
        let mut blind = shot("b", 166, -2.0);
        blind.time = None;
        assert!(find_sets(&[shot("a", 0, 0.0), blind]).is_empty());
        let mut unknown = shot("b", 166, -2.0);
        unknown.ev = None;
        assert!(find_sets(&[shot("a", 0, 0.0), unknown]).is_empty());
    }

    #[test]
    fn whole_second_times_still_group() {
        let mut shots = vec![shot("a", 51_000, 0.0), shot("b", 52_000, -2.0), shot("c", 52_000, 2.0)];
        for s in &mut shots {
            s.time = s.time.map(|(t, _)| (t, false));
        }
        assert_eq!(find_sets(&shots), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn sets_are_capped() {
        let shots: Vec<Shot> = (0..12).map(|i| shot(&format!("{i}"), i * 150, (i % 3) as f32 - 1.0)).collect();
        let sets = find_sets(&shots);
        assert_eq!(sets[0].len(), MAX_FRAMES);
        assert_eq!(sets[1], vec![9, 10, 11]);
    }

    #[test]
    fn shot_from_metadata() {
        let m = ExportMetadata {
            make: Some("SONY".into()),
            model: Some("ILCE-7M4".into()),
            date_time_original: Some("2026:10:20 15:15:51".into()),
            sub_sec_time_original: Some("844".into()),
            exposure_time: Some((1, 250)),
            fnumber: Some((35, 10)),
            iso: Some(2500),
            ..Default::default()
        };
        let s = Shot::from_metadata(Path::new("DSC01195.ARW"), &m);
        assert_eq!(s.camera, "SONY ILCE-7M4");
        let (t, sub) = s.time.unwrap();
        assert!(sub && (t - parse_exif_time("2026:10:20 15:15:51").unwrap() - 0.844).abs() < 1e-6);
        assert!((s.ev.unwrap() - exposure_ev(0.004, 3.5, 2500.0)).abs() < 1e-6);
        let none = Shot::from_metadata(Path::new("x"), &ExportMetadata { iso: None, ..m });
        assert_eq!(none.ev, None);
    }

    #[test]
    fn real_sony_brackets_are_found() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/hdr");
        let Ok(list) = crate::formats::photos_in(&dir) else { return };
        if list.is_empty() {
            return;
        }
        let name = |p: &PathBuf| p.file_stem().unwrap().to_string_lossy().into_owned();
        let sets: Vec<Vec<String>> = sets_in(&list).iter().map(|s| s.iter().map(name).collect()).collect();
        let want = |names: &[&str]| sets.iter().any(|s| s == names);
        assert!(want(&["DSC01195", "DSC01196", "DSC01197"]), "{sets:?}");
        assert!(want(&["DSC01303", "DSC01304", "DSC01305"]), "{sets:?}");
        assert!(want(&["DSC01204", "DSC01205"]), "{sets:?}");
        let middle = dir.join("DSC01196.arw");
        let set = set_containing(&list, &middle).unwrap();
        assert_eq!(set.iter().map(name).collect::<Vec<_>>(), ["DSC01195", "DSC01196", "DSC01197"]);
    }
}
