//! Carry the camera's own metadata into exported files.
//!
//! rawler already parses everything we need out of the raw — camera, lens,
//! exposure, aperture, ISO, date, orientation, artist, copyright — and until
//! now the export path discarded all of it, so exports had no provenance at all
//! and nothing told a colour-managed viewer the pixels were sRGB.
//!
//! Reading is a structure-only parse (no pixel decoding), so it is cheap enough
//! to do at export time from the raw's path rather than threading state around.
//! Writing happens *after* the image encoder has produced the file: `little_exif`
//! adds the tags to the finished JPEG or TIFF in place.

use std::path::Path;

use anyhow::{Context, Result};
use little_exif::exif_tag::ExifTag;
use little_exif::metadata::Metadata;
use little_exif::rational::uR64;
use rawler::decoders::RawDecodeParams;
use rawler::rawsource::RawSource;

/// What we copy from a raw into an export. Everything is optional because
/// bodies and lenses vary in what they record.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExportMetadata {
    pub make: Option<String>,
    pub model: Option<String>,
    pub lens_make: Option<String>,
    pub lens_model: Option<String>,
    pub date_time_original: Option<String>,
    /// Fraction of a second for `date_time_original`, as the digits the
    /// camera wrote ("844" = 0.844 s). Bracket detection needs it: a burst's
    /// frames are ~170 ms apart, inside one whole second.
    pub sub_sec_time_original: Option<String>,
    pub exposure_time: Option<(u32, u32)>,
    pub fnumber: Option<(u32, u32)>,
    pub focal_length: Option<(u32, u32)>,
    pub iso: Option<u16>,
    pub orientation: Option<u16>,
    pub artist: Option<String>,
    pub copyright: Option<String>,
}

fn non_empty(s: &str) -> Option<String> {
    let t = s.trim();
    (!t.is_empty()).then(|| t.to_owned())
}

/// The same fields from a TIFF's own tags (its IFD0 and EXIF sub-IFD) —
/// rawler only reads raws. A TIFF with no tags at all is fine: everything
/// comes back empty.
fn read_tiff(path: &Path) -> Result<ExportMetadata> {
    let meta = Metadata::new_from_path(path).map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("reading tags from {}", path.display()))?;
    let mut m = ExportMetadata::default();
    let rat = |v: &[uR64]| v.first().map(|r| (r.nominator, r.denominator));
    for tag in &meta {
        match tag {
            ExifTag::Make(v) => m.make = non_empty(v),
            ExifTag::Model(v) => m.model = non_empty(v),
            ExifTag::LensMake(v) => m.lens_make = non_empty(v),
            ExifTag::LensModel(v) => m.lens_model = non_empty(v),
            ExifTag::DateTimeOriginal(v) => m.date_time_original = non_empty(v),
            ExifTag::SubSecTimeOriginal(v) => m.sub_sec_time_original = non_empty(v),
            ExifTag::ExposureTime(v) => m.exposure_time = rat(v),
            ExifTag::FNumber(v) => m.fnumber = rat(v),
            ExifTag::FocalLength(v) => m.focal_length = rat(v),
            ExifTag::ISO(v) => m.iso = v.first().copied().map(Into::into),
            ExifTag::Orientation(v) => m.orientation = v.first().copied(),
            ExifTag::Artist(v) => m.artist = non_empty(v),
            ExifTag::Copyright(v) => m.copyright = non_empty(v),
            _ => {}
        }
    }
    Ok(m)
}

impl ExportMetadata {
    /// Clockwise quarter turns that show the frame upright, from the EXIF
    /// orientation (6 = 90° clockwise, 3 = 180°, 8 = 90° anticlockwise).
    /// rawler leaves `RawImage::orientation` at Normal for RW2 files, so this
    /// — the EXIF block — is the value to trust.
    pub fn turns(&self) -> u8 {
        match self.orientation {
            Some(6) => 1,
            Some(3) => 2,
            Some(8) => 3,
            _ => 0,
        }
    }
}

/// Decode and develop a raw file, with the camera's orientation applied from
/// its EXIF (see [`ExportMetadata::turns`]). Every caller that renders a file
/// should come through here, or portrait shots export sideways.
///
/// A decoder panic (rawler indexes past the end of some truncated or unusual
/// files) comes back as an error: in the GUI it would otherwise kill the
/// loader thread silently and leave the window on "Loading…" for good.
pub fn develop_file(path: &Path) -> Result<crate::develop::DevelopedImage> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| develop_file_inner(path))).unwrap_or_else(|panic| {
        let why = panic
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown error".into());
        anyhow::bail!("{} could not be decoded ({why})", path.display())
    })
}

fn develop_file_inner(path: &Path) -> Result<crate::develop::DevelopedImage> {
    if crate::formats::is_tiff(path) {
        let mut img = crate::rgb_input::open_tiff(path)?;
        if let Ok(m) = read(path) {
            img.colour.turns = m.turns();
        }
        return Ok(img);
    }
    let raw = rawler::decode_file(path).with_context(|| format!("decoding {}", path.display()))?;
    let mut img = crate::develop::develop(&raw)?;
    match read(path) {
        Ok(m) if m.orientation.is_some() => img.colour.turns = m.turns(),
        Ok(_) => {}
        Err(e) => eprintln!("orientation: {e:#}"),
    }
    Ok(img)
}

/// Read the metadata out of a raw file without decoding any pixels.
pub fn read(path: &Path) -> Result<ExportMetadata> {
    if crate::formats::is_tiff(path) {
        return read_tiff(path);
    }
    let src = RawSource::new(path).with_context(|| format!("opening {}", path.display()))?;
    let loader = rawler::RawLoader::new();
    let decoder = loader
        .get_decoder(&src)
        .map_err(|e| anyhow::anyhow!("{e:?}"))
        .with_context(|| format!("no decoder for {}", path.display()))?;
    let meta = decoder
        .raw_metadata(&src, &RawDecodeParams::default())
        .map_err(|e| anyhow::anyhow!("{e:?}"))
        .with_context(|| format!("reading metadata from {}", path.display()))?;

    let e = &meta.exif;
    let rat = |r: &rawler::formats::tiff::Rational| (r.n, r.d);
    Ok(ExportMetadata {
        make: non_empty(&meta.make),
        model: non_empty(&meta.model),
        lens_make: meta.lens.as_ref().and_then(|l| non_empty(&l.lens_make)),
        lens_model: meta.lens.as_ref().and_then(|l| non_empty(&l.lens_model)),
        date_time_original: e.date_time_original.as_deref().and_then(non_empty),
        sub_sec_time_original: e.sub_sec_time_original.as_deref().and_then(non_empty),
        exposure_time: e.exposure_time.as_ref().map(rat),
        fnumber: e.fnumber.as_ref().map(rat),
        focal_length: e.focal_length.as_ref().map(rat),
        iso: e.iso_speed_ratings,
        orientation: e.orientation,
        artist: e.artist.as_deref().and_then(non_empty),
        copyright: e.copyright.as_deref().and_then(non_empty),
    })
}

impl ExportMetadata {
    /// Build the tag set written into an export.
    ///
    /// `orientation` holds the camera's value, but it is written as 1 ("as
    /// stored"): the render applies it (`ImageColour::turns`, set from
    /// [`ExportMetadata::turns`]) plus any user rotation, so the pixels are
    /// already upright. Writing the camera's value would make viewers rotate
    /// a portrait frame a second time.
    ///
    /// `ColorSpace = 1` marks the file sRGB so viewers stop guessing. A real ICC
    /// profile would be better and is a separate job.
    pub fn tags(&self) -> Vec<ExifTag> {
        let mut tags = vec![ExifTag::Software("raw-editor".to_string())];
        let r = |(n, d): (u32, u32)| vec![uR64 { nominator: n, denominator: d }];

        if let Some(v) = &self.make {
            tags.push(ExifTag::Make(v.clone()));
        }
        if let Some(v) = &self.model {
            tags.push(ExifTag::Model(v.clone()));
        }
        if let Some(v) = &self.lens_make {
            tags.push(ExifTag::LensMake(v.clone()));
        }
        if let Some(v) = &self.lens_model {
            tags.push(ExifTag::LensModel(v.clone()));
        }
        if let Some(v) = &self.date_time_original {
            tags.push(ExifTag::DateTimeOriginal(v.clone()));
        }
        if let Some(v) = self.exposure_time {
            tags.push(ExifTag::ExposureTime(r(v)));
        }
        if let Some(v) = self.fnumber {
            tags.push(ExifTag::FNumber(r(v)));
        }
        if let Some(v) = self.focal_length {
            tags.push(ExifTag::FocalLength(r(v)));
        }
        if let Some(v) = self.iso {
            tags.push(ExifTag::ISO(vec![v]));
        }
        if self.orientation.is_some() {
            tags.push(ExifTag::Orientation(vec![1]));
        }
        if let Some(v) = &self.artist {
            tags.push(ExifTag::Artist(v.clone()));
        }
        if let Some(v) = &self.copyright {
            tags.push(ExifTag::Copyright(v.clone()));
        }
        tags.push(ExifTag::ColorSpace(vec![1]));
        tags
    }

    /// Add these tags to an already-written JPEG or TIFF.
    pub fn write_into(&self, path: &Path) -> Result<()> {
        let mut meta = Metadata::new();
        for tag in self.tags() {
            meta.set_tag(tag);
        }
        meta.write_to_file(path)
            .with_context(|| format!("writing metadata into {}", path.display()))
    }
}

/// What identifies the camera body and lens a raw came from: the keys for
/// per-body looks (IR-3) and per-lens flat fields (IR-4).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CameraIds {
    /// Body serial. Panasonic keeps it only in its maker note
    /// (`InternalSerialNumber`, e.g. `F921004200220`), not in standard EXIF.
    pub serial: Option<String>,
    /// Lens model as the camera reports it, e.g. `LUMIX G 20/F1.7`.
    pub lens: Option<String>,
}

/// How much of the file to scan for the maker note. Panasonic writes it near
/// the start (inside the embedded preview JPEG's EXIF, a few KB in).
const MAKER_NOTE_SCAN: u64 = 4 << 20;

/// Read the body serial and lens without decoding pixels. Uses the Panasonic
/// maker note where there is one and falls back to standard EXIF fields.
pub fn camera_ids(path: &Path) -> Result<CameraIds> {
    use std::io::Read;
    let mut head = Vec::new();
    std::fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .take(MAKER_NOTE_SCAN)
        .read_to_end(&mut head)
        .with_context(|| format!("reading {}", path.display()))?;
    let mut ids = panasonic_ids(&head);
    if ids.serial.is_none() || ids.lens.is_none() {
        if let Ok(src) = RawSource::new(path) {
            let loader = rawler::RawLoader::new();
            if let Ok(dec) = loader.get_decoder(&src) {
                if let Ok(meta) = dec.raw_metadata(&src, &RawDecodeParams::default()) {
                    ids.serial = ids.serial.or_else(|| meta.exif.serial_number.as_deref().and_then(non_empty));
                    ids.lens = ids.lens.or_else(|| meta.lens.as_ref().and_then(|l| non_empty(&l.lens_model)));
                }
            }
        }
    }
    Ok(ids)
}

/// Pull `InternalSerialNumber` (0x25) and `LensType` (0x51) out of a
/// Panasonic maker note found anywhere in `bytes`.
///
/// The note is a bare IFD after a 12-byte `Panasonic\0\0\0` signature, with
/// offsets relative to the TIFF header of the EXIF block around it — the
/// nearest `Exif\0\0` before it, or the RW2's own header at byte 0. Every
/// read is bounds-checked: this parses untrusted files.
pub fn panasonic_ids(bytes: &[u8]) -> CameraIds {
    const SIG: &[u8] = b"Panasonic\0\0\0";
    let mut ids = CameraIds::default();
    let Some(sig) = bytes.windows(SIG.len()).position(|w| w == SIG) else { return ids };
    let base = bytes[..sig].windows(6).rposition(|w| w == b"Exif\0\0").map_or(0, |p| p + 6);
    let little = match bytes.get(base..base + 2) {
        Some(b"II") => true,
        Some(b"MM") => false,
        _ => return ids,
    };
    let u16_at = |o: usize| bytes.get(o..o + 2).map(|b| {
        let b = [b[0], b[1]];
        if little { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) }
    });
    let u32_at = |o: usize| bytes.get(o..o + 4).map(|b| {
        let b = [b[0], b[1], b[2], b[3]];
        if little { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) }
    });
    let ifd = sig + SIG.len();
    let Some(count) = u16_at(ifd) else { return ids };
    for k in 0..count.min(512) as usize {
        let e = ifd + 2 + k * 12;
        let (Some(tag), Some(typ), Some(n)) = (u16_at(e), u16_at(e + 2), u32_at(e + 4)) else { break };
        // Only byte-sized types (ASCII = 2, UNDEFINED = 7) are read here.
        if !(tag == 0x25 || tag == 0x51) || !(typ == 2 || typ == 7) || n > 256 {
            continue;
        }
        let n = n as usize;
        let data = if n <= 4 {
            bytes.get(e + 8..e + 8 + n)
        } else {
            u32_at(e + 8).and_then(|off| bytes.get(base + off as usize..base + off as usize + n))
        };
        let Some(text) = data.map(ascii_field) else { continue };
        match tag {
            0x25 => ids.serial = text,
            _ => ids.lens = text,
        }
    }
    ids
}

/// A NUL-padded ASCII field, or `None` if it is empty, all zeros (stock
/// GF1 files sometimes are) or not printable text.
fn ascii_field(b: &[u8]) -> Option<String> {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    let b = &b[..end];
    if b.is_empty() || !b.iter().all(|c| c.is_ascii_graphic() || *c == b' ') {
        return None;
    }
    non_empty(std::str::from_utf8(b).ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ExportMetadata {
        ExportMetadata {
            make: Some("Panasonic".into()),
            model: Some("DMC-G85".into()),
            lens_model: Some("LUMIX G 14-42".into()),
            date_time_original: Some("2026:05:02 00:53:16".into()),
            exposure_time: Some((1, 250)),
            fnumber: Some((4, 1)),
            iso: Some(100),
            orientation: Some(6),
            ..Default::default()
        }
    }

    #[test]
    fn tags_cover_the_populated_fields() {
        let tags = sample().tags();
        let names: Vec<String> = tags.iter().map(|t| format!("{t:?}")).collect();
        let joined = names.join(" ");
        for want in ["Make", "Model", "LensModel", "DateTimeOriginal", "ExposureTime", "ISO"] {
            assert!(joined.contains(want), "{want} missing from {joined}");
        }
    }

    /// Every export must be marked sRGB and stamped with the software name,
    /// even for a raw that recorded nothing else.
    #[test]
    fn empty_metadata_still_tags_colour_space_and_software() {
        let tags = ExportMetadata::default().tags();
        let joined = tags.iter().map(|t| format!("{t:?}")).collect::<Vec<_>>().join(" ");
        assert!(joined.contains("ColorSpace"), "no ColorSpace: {joined}");
        assert!(joined.contains("Software"), "no Software: {joined}");
        assert_eq!(tags.len(), 2, "unset fields should not be written: {joined}");
    }

    /// Pixels leave upright, so the tag says "as stored" whatever the camera
    /// recorded; the camera's value drives the rotation instead.
    #[test]
    fn orientation_is_written_upright() {
        let joined = sample().tags().iter().map(|t| format!("{t:?}")).collect::<Vec<_>>().join(" ");
        assert!(joined.contains("Orientation([1])"), "expected Orientation 1: {joined}");
        let none = ExportMetadata { orientation: None, ..sample() };
        let joined = none.tags().iter().map(|t| format!("{t:?}")).collect::<Vec<_>>().join(" ");
        assert!(!joined.contains("Orientation"));
    }

    #[test]
    fn exif_orientation_becomes_clockwise_turns() {
        for (tag, turns) in [(None, 0), (Some(1), 0), (Some(6), 1), (Some(3), 2), (Some(8), 3), (Some(2), 0)] {
            assert_eq!(ExportMetadata { orientation: tag, ..sample() }.turns(), turns, "tag {tag:?}");
        }
    }

    #[test]
    fn blank_strings_are_treated_as_absent() {
        assert_eq!(non_empty("  "), None);
        assert_eq!(non_empty(" Panasonic "), Some("Panasonic".to_string()));
    }

    // --- camera ids (IR-3) ----------------------------------------------

    /// A minimal EXIF block holding a Panasonic maker note, laid out like the
    /// GF1's: serial out of line (16 bytes), lens out of line, offsets relative
    /// to the TIFF header after `Exif\0\0`.
    fn maker_note(little: bool, serial: &[u8], lens: &[u8]) -> Vec<u8> {
        let w16 = |v: u16| if little { v.to_le_bytes() } else { v.to_be_bytes() };
        let w32 = |v: u32| if little { v.to_le_bytes() } else { v.to_be_bytes() };
        let mut b = b"junk\xff\xd8Exif\0\0".to_vec();
        let base = b.len();
        b.extend_from_slice(if little { b"II*\0" } else { b"MM\0*" });
        b.extend_from_slice(&[0; 4]);
        b.extend_from_slice(b"Panasonic\0\0\0");
        let entries = 3u16;
        let data_at = (b.len() - base) as u32 + 2 + 12 * entries as u32 + 4;
        b.extend_from_slice(&w16(entries));
        // An unrelated SHORT tag first, which must be skipped.
        b.extend_from_slice(&w16(0x02));
        b.extend_from_slice(&w16(3));
        b.extend_from_slice(&w32(1));
        b.extend_from_slice(&[0, 1, 0, 0]);
        b.extend_from_slice(&w16(0x25));
        b.extend_from_slice(&w16(7));
        b.extend_from_slice(&w32(serial.len() as u32));
        b.extend_from_slice(&w32(data_at));
        b.extend_from_slice(&w16(0x51));
        b.extend_from_slice(&w16(2));
        b.extend_from_slice(&w32(lens.len() as u32));
        b.extend_from_slice(&w32(data_at + serial.len() as u32));
        b.extend_from_slice(&[0; 4]);
        b.extend_from_slice(serial);
        b.extend_from_slice(lens);
        b
    }

    #[test]
    fn panasonic_ids_read_serial_and_lens_in_both_byte_orders() {
        for little in [true, false] {
            let b = maker_note(little, b"F921004200220\0\0\0", b"LUMIX G 20/F1.7\0");
            let ids = panasonic_ids(&b);
            assert_eq!(ids.serial.as_deref(), Some("F921004200220"), "little={little}");
            assert_eq!(ids.lens.as_deref(), Some("LUMIX G 20/F1.7"), "little={little}");
        }
    }

    #[test]
    fn blank_or_garbage_fields_are_not_ids() {
        let ids = panasonic_ids(&maker_note(true, &[0; 16], b"\x01\x02\x03\x04\x05\0"));
        assert_eq!(ids, CameraIds::default());
    }

    /// Truncation at every length must never panic — these are untrusted files.
    #[test]
    fn truncated_maker_notes_are_handled() {
        let b = maker_note(true, b"F921004200220\0\0\0", b"LUMIX G 20/F1.7\0");
        for n in 0..b.len() {
            let _ = panasonic_ids(&b[..n]);
        }
        assert_eq!(panasonic_ids(b"no maker note here"), CameraIds::default());
    }

    /// The real files, when present (testdata is gitignored).
    #[test]
    fn camera_ids_from_real_files() {
        let gf1 = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/ir/P1320167.rw2");
        let g85 = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/P1050019.rw2");
        if !gf1.exists() || !g85.exists() {
            eprintln!("SKIP: testdata not present");
            return;
        }
        let ids = camera_ids(&gf1).unwrap();
        assert_eq!(ids.serial.as_deref(), Some("F921004200220"));
        assert_eq!(ids.lens.as_deref(), Some("LUMIX G 20/F1.7"));
        let ids = camera_ids(&g85).unwrap();
        assert_eq!(ids.serial.as_deref(), Some("XEL2112060047"));
        assert_eq!(ids.lens.as_deref(), Some("LUMIX G VARIO 7-14/F4.0"));
    }
}
