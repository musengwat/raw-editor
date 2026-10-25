//! Which files the editor opens. One list, so the open dialog, drag and drop,
//! batch export and the CLI agree.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Raw formats rawler decodes. DNG is technically a TIFF but belongs here: it
/// is a raw mosaic and goes through the raw decoder.
pub const RAW_EXTENSIONS: &[&str] = &[
    "arw", "cr2", "cr3", "crw", "dng", "erf", "iiq", "kdc", "mef", "mos", "mrw", "nef", "nrw", "orf", "pef",
    "raf", "rw2", "srw", "3fr", "dcr", "x3f",
];

/// Finished images, opened as linear sRGB (see `rgb_input::open_tiff`).
pub const TIFF_EXTENSIONS: &[&str] = &["tif", "tiff"];

fn extension(path: &Path) -> Option<String> {
    path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase())
}

pub fn is_raw(path: &Path) -> bool {
    extension(path).is_some_and(|e| RAW_EXTENSIONS.contains(&e.as_str()))
}

pub fn is_tiff(path: &Path) -> bool {
    extension(path).is_some_and(|e| TIFF_EXTENSIONS.contains(&e.as_str()))
}

/// Anything the editor can open.
pub fn is_supported(path: &Path) -> bool {
    is_raw(path) || is_tiff(path)
}

/// One of this editor's own exports (`<name>_edit.tif`). Batch export skips
/// them, or a second run over a folder would re-edit its own output.
pub fn is_own_export(path: &Path) -> bool {
    is_tiff(path) && path.file_stem().and_then(|s| s.to_str()).is_some_and(|s| s.ends_with("_edit"))
}

/// An HDR merge written by this editor (`<name>_hdr.tif`, see
/// `hdr::output_path`). Unlike other TIFFs it is opened like a raw, in the
/// default look: the merge is tone mapped but has no tone curve yet.
pub fn is_hdr_merge(path: &Path) -> bool {
    is_tiff(path) && path.file_stem().and_then(|s| s.to_str()).is_some_and(|s| s.ends_with("_hdr"))
}

/// Extensions for a file dialog filter, lower and upper case (GTK's filter is
/// case-sensitive).
pub fn dialog_extensions() -> Vec<String> {
    RAW_EXTENSIONS
        .iter()
        .chain(TIFF_EXTENSIONS)
        .flat_map(|e| [e.to_string(), e.to_ascii_uppercase()])
        .collect()
}

/// Every supported photo directly inside `dir`, sorted, matched
/// case-insensitively, leaving out the editor's own TIFF exports.
///
/// Subdirectories are not walked: batch export is a "do this folder" action,
/// and quietly recursing into a photo library would be a nasty surprise.
pub fn photos_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if path.is_file() && is_supported(&path) && !is_own_export(&path) {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

/// The photo `step` places after (or, negative, before) `current` in
/// `list`, without wrapping round. A `current` missing from the list (just
/// deleted, say) steps from where it would sort.
pub fn neighbour(list: &[PathBuf], current: &Path, step: isize) -> Option<PathBuf> {
    let at = match list.binary_search_by(|p| p.as_path().cmp(current)) {
        Ok(i) => i as isize,
        // Not in the list: "next" is the one at the insertion point.
        Err(i) => i as isize - (step > 0) as isize,
    };
    let j = at + step;
    (0..list.len() as isize).contains(&j).then(|| list[j as usize].clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_extensions_match_case_insensitively() {
        for name in ["a.rw2", "a.RW2", "DSC06674.ARW", "x.Nef", "x.cr3", "x.dng"] {
            assert!(is_raw(Path::new(name)), "{name}");
        }
        for name in ["a.jpg", "a", "a.rw2.edit.json", "arw"] {
            assert!(!is_raw(Path::new(name)), "{name}");
        }
    }

    #[test]
    fn discovery_is_sorted_skips_other_files_and_does_not_recurse() {
        let dir = std::env::temp_dir().join(format!("raw-editor-discover-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["b.rw2", "a.RW2", "c.ARW", "note.txt", "old_edit.jpg", "a.RW2.edit.json", "scan.TIF", "a_edit.tif"] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }
        std::fs::create_dir_all(dir.join("nested")).unwrap();
        std::fs::write(dir.join("nested/deep.rw2"), b"x").unwrap();

        let found = photos_in(&dir).unwrap();
        let names: Vec<String> = found.iter().map(|p| p.file_name().unwrap().to_string_lossy().into()).collect();
        assert_eq!(names, vec!["a.RW2", "b.rw2", "c.ARW", "scan.TIF"], "own exports skipped");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn tiffs_are_supported_but_dng_is_a_raw() {
        assert!(is_tiff(Path::new("scan.tiff")) && is_supported(Path::new("scan.TIF")));
        assert!(is_raw(Path::new("x.dng")) && !is_tiff(Path::new("x.dng")));
        assert!(is_own_export(Path::new("P1_edit.tif")) && !is_own_export(Path::new("P1_edit.jpg")));
        assert!(is_hdr_merge(Path::new("DSC01195_hdr.tif")) && !is_own_export(Path::new("DSC01195_hdr.tif")));
        assert!(!is_hdr_merge(Path::new("DSC01195_hdr.arw")) && !is_hdr_merge(Path::new("scan.tif")));
    }

    #[test]
    fn dialog_lists_both_cases() {
        let e = dialog_extensions();
        assert!(e.contains(&"arw".to_string()) && e.contains(&"ARW".to_string()));
    }

    #[test]
    fn neighbour_steps_without_wrapping() {
        let list: Vec<PathBuf> = ["/p/a.ARW", "/p/b.rw2", "/p/c.tif"].iter().map(PathBuf::from).collect();
        let b = Path::new("/p/b.rw2");
        assert_eq!(neighbour(&list, b, 1), Some(PathBuf::from("/p/c.tif")));
        assert_eq!(neighbour(&list, b, -1), Some(PathBuf::from("/p/a.ARW")));
        assert_eq!(neighbour(&list, Path::new("/p/c.tif"), 1), None, "no wrap at the end");
        assert_eq!(neighbour(&list, Path::new("/p/a.ARW"), -1), None, "no wrap at the start");
        // The current file vanished from the folder: still step sensibly.
        assert_eq!(neighbour(&list, Path::new("/p/bb.rw2"), 1), Some(PathBuf::from("/p/c.tif")));
        assert_eq!(neighbour(&list, Path::new("/p/bb.rw2"), -1), Some(PathBuf::from("/p/b.rw2")));
        assert_eq!(neighbour(&[], b, 1), None);
    }

}
