#!/usr/bin/env bash
#
# build-tex.sh - compile a LaTeX document to PDF and clean up build artifacts.
#
# Usage:
#   scripts/build-tex.sh <file.tex> [output.pdf] [--keep]
#
# Arguments:
#   <file.tex>    source document (relative or absolute path)
#   [output.pdf]  name of the resulting PDF, written next to the .tex file
#                 (default: the .tex file name with a .pdf extension;
#                 the .pdf extension is added if missing)
#   --keep        keep the .latexbuild/ directory (useful for debugging)
#   -h, --help    show this help
#
# Examples:
#   scripts/build-tex.sh docs/report.tex
#   scripts/build-tex.sh docs/report.tex report.pdf
#
# How it works:
#   1. Changes into the directory of the .tex file, so relative paths to the
#      bibliography (.bib) and images (\includegraphics) resolve as usual.
#   2. Runs latexmk with -outdir=.latexbuild. latexmk runs biber/bibtex and
#      repeats pdflatex until cross-references and citations are stable.
#   3. Copies the PDF next to the .tex file under the requested name and
#      removes .latexbuild/, so only the sources and the PDF remain.
#
# On failure the script prints the LaTeX errors, saves the full log as
# <output>.log next to the .tex file and exits with a non-zero code.
# Undefined citations/references and biber warnings are reported but do not
# fail the build.
#
# Requirements: bash (Git Bash on Windows), latexmk with pdflatex and biber
# on PATH (e.g. MiKTeX or TeX Live), perl (needed by latexmk).

set -euo pipefail

BUILD_DIR=".latexbuild"

usage() {
    sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-2}"
}

keep=0
positional=()
for arg in "$@"; do
    case "$arg" in
        --keep) keep=1 ;;
        -h|--help) usage 0 ;;
        -*) echo "Unknown option: $arg" >&2; usage ;;
        *) positional+=("$arg") ;;
    esac
done

[[ ${#positional[@]} -ge 1 && ${#positional[@]} -le 2 ]] || usage

tex_path="${positional[0]}"
if [[ ! -f "$tex_path" ]]; then
    echo "Error: file '$tex_path' not found." >&2
    exit 1
fi
if [[ "$tex_path" != *.tex ]]; then
    echo "Error: '$tex_path' does not have a .tex extension." >&2
    exit 1
fi
if ! command -v latexmk >/dev/null 2>&1; then
    echo "Error: latexmk not found on PATH (install MiKTeX or TeX Live)." >&2
    exit 1
fi

tex_dir="$(cd "$(dirname "$tex_path")" && pwd)"
tex_file="$(basename "$tex_path")"
job="${tex_file%.tex}"

pdf_name="${positional[1]:-$job.pdf}"
[[ "$pdf_name" == *.pdf ]] || pdf_name="$pdf_name.pdf"
if [[ "$pdf_name" == */* ]]; then
    echo "Error: give a bare PDF file name (no directory); it is written next to '$tex_file'." >&2
    exit 1
fi

cd "$tex_dir"
rm -rf "$BUILD_DIR"

echo "Compiling: $tex_dir/$tex_file"
status=0
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
    -outdir="$BUILD_DIR" "$tex_file" >"$BUILD_DIR.out" 2>&1 || status=$?

log="$BUILD_DIR/$job.log"

if [[ $status -ne 0 || ! -f "$BUILD_DIR/$job.pdf" ]]; then
    echo "Compilation failed (exit code $status)." >&2
    if [[ -f "$log" ]]; then
        # Errors in file:line:error format and classic '!' lines.
        grep -n -A3 -E '^(!|.*:[0-9]+: )' "$log" | head -40 >&2 || true
        cp "$log" "${pdf_name%.pdf}.log"
        echo "Full log: $tex_dir/${pdf_name%.pdf}.log" >&2
    else
        tail -20 "$BUILD_DIR.out" >&2
    fi
    rm -f "$BUILD_DIR.out"
    [[ $keep -eq 1 ]] || rm -rf "$BUILD_DIR"
    exit 1
fi

# Warnings that do not stop the build but leave the document broken.
if grep -q -E "Citation .* undefined|There were undefined references|Reference .* undefined" "$log"; then
    echo "Warning: undefined citations or references:" >&2
    grep -E "Citation .* undefined|Reference .* undefined" "$log" | sort -u | head -20 >&2 || true
fi
if [[ -f "$BUILD_DIR/$job.blg" ]] && grep -q -E "^(ERROR|WARN)" "$BUILD_DIR/$job.blg"; then
    echo "Warning: biber messages:" >&2
    grep -E "^(ERROR|WARN)" "$BUILD_DIR/$job.blg" | head -20 >&2 || true
fi

cp "$BUILD_DIR/$job.pdf" "$pdf_name"
rm -f "$BUILD_DIR.out" "${pdf_name%.pdf}.log"
[[ $keep -eq 1 ]] || rm -rf "$BUILD_DIR"

echo "Done: $tex_dir/$pdf_name"
