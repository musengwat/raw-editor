#!/bin/sh
# Builds a universal (Intel + Apple Silicon) raw-editor.app and zips it.
# Runs on macOS only: it needs lipo, codesign, plutil and ditto.
#
#   packaging/macos/bundle.sh            # -> dist/raw-editor-macos-universal.zip
#
# The deployment target is what lets this run on an old Intel MacBook Air:
# without it the binary inherits the build machine's macOS version and
# refuses to launch anywhere older.
set -eu

MIN_MACOS=10.13
export MACOSX_DEPLOYMENT_TARGET=$MIN_MACOS

cd "$(dirname "$0")/../.."
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n1)

for target in x86_64-apple-darwin aarch64-apple-darwin; do
    cargo build --release --bin raw-editor --target "$target"
done

APP=dist/raw-editor.app
rm -rf dist
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

lipo -create \
    target/x86_64-apple-darwin/release/raw-editor \
    target/aarch64-apple-darwin/release/raw-editor \
    -output "$APP/Contents/MacOS/raw-editor"

sed -e "s/@VERSION@/$VERSION/g" -e "s/@MIN_MACOS@/$MIN_MACOS/g" \
    packaging/macos/Info.plist > "$APP/Contents/Info.plist"
plutil -lint "$APP/Contents/Info.plist"

# --- checks: fail the build rather than ship something that will not launch.

archs=$(lipo -archs "$APP/Contents/MacOS/raw-editor")
case "$archs" in
    *x86_64*arm64*|*arm64*x86_64*) ;;
    *) echo "expected a universal binary, got: $archs" >&2; exit 1 ;;
esac

# Below 10.14 the linker records LC_VERSION_MIN_MACOSX, from 10.14 on
# LC_BUILD_VERSION; either way the Intel slice must say $MIN_MACOS.
minos=$(otool -arch x86_64 -l "$APP/Contents/MacOS/raw-editor" \
    | awk '/LC_VERSION_MIN_MACOSX|LC_BUILD_VERSION/ {f=1} f && /^ *(version|minos) / {print $2; exit}')
if [ "$minos" != "$MIN_MACOS" ]; then
    echo "Intel slice targets macOS '$minos', expected $MIN_MACOS" >&2
    exit 1
fi
echo "universal binary ($archs), Intel slice targets macOS $minos"

# Ad-hoc signature: not a developer ID, so Gatekeeper still asks on first
# launch, but Apple Silicon will not run an unsigned binary at all and lipo
# invalidates the per-slice signatures the linker applied.
codesign --force --sign - "$APP"
codesign --verify --strict --verbose=2 "$APP"

cp README.md dist/
(cd dist && ditto -c -k --keepParent raw-editor.app raw-editor-macos-universal.zip)
echo "dist/raw-editor-macos-universal.zip"
