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

/// Read the metadata out of a raw file without decoding any pixels.
pub fn read(path: &Path) -> Result<ExportMetadata> {
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
    /// Orientation is deliberately carried through: the develop stage does not
    /// rotate pixels, so they sit in sensor order, and passing the camera's
    /// orientation on is what makes a portrait frame display upright — the same
    /// thing the in-camera JPEG relies on. Dropping it would silently leave
    /// every portrait export on its side.
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
        if let Some(v) = self.orientation {
            tags.push(ExifTag::Orientation(vec![v]));
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

    /// Orientation must survive, or portrait exports display on their side.
    #[test]
    fn orientation_is_carried() {
        let joined = sample().tags().iter().map(|t| format!("{t:?}")).collect::<Vec<_>>().join(" ");
        assert!(joined.contains("Orientation"), "orientation dropped: {joined}");
        let none = ExportMetadata { orientation: None, ..sample() };
        let joined = none.tags().iter().map(|t| format!("{t:?}")).collect::<Vec<_>>().join(" ");
        assert!(!joined.contains("Orientation"));
    }

    #[test]
    fn blank_strings_are_treated_as_absent() {
        assert_eq!(non_empty("  "), None);
        assert_eq!(non_empty(" Panasonic "), Some("Panasonic".to_string()));
    }
}
