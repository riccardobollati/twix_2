#!/usr/bin/env bash
# Process source PDFs in data/ and data/<document>/; skip nested generated PDFs.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${DATA_DIR:-$repo_root/data}"
output_root="${OUTPUT_ROOT:-$repo_root/test-sorter-lines}"
python_bin="${PYTHON:-$repo_root/env/bin/python}"
list_only=false
if [[ "${1:-}" == "--list" ]]; then
    list_only=true
    shift
fi

if [[ ! -d "$data_dir" ]]; then
    printf 'Data directory not found: %s\n' "$data_dir" >&2
    exit 1
fi
if [[ "$list_only" == false && ! -x "$python_bin" ]]; then
    printf 'Python not executable: %s (set PYTHON to another interpreter)\n' "$python_bin" >&2
    exit 1
fi

count=0
failures=0
while IFS= read -r -d '' pdf; do
    parent="$(basename "$(dirname "$pdf")")"
    filename="$(basename "$pdf")"
    if [[ "$(dirname "$pdf")" == "$data_dir" ]]; then
        result_name="${filename%.*}"
    else
        result_name="$parent"
    fi
    count=$((count + 1))
    if [[ "$list_only" == true ]]; then
        printf '%s -> %s/%s/results/\n' "$pdf" "$output_root" "$result_name"
    else
        printf '[%d] Detecting lines in %s -> %s/results/\n' "$count" "$pdf" "$result_name"
        if ! "$python_bin" "$repo_root/test-sorter-lines/detect_lines.py" "$pdf" \
                --output-root "$output_root" --result-name "$result_name" "$@"; then
            failures=$((failures + 1))
        fi
    fi
done < <(find "$data_dir" -maxdepth 2 -type f -iname '*.pdf' -print0)

if [[ "$count" -eq 0 ]]; then
    printf 'No source PDFs found under %s\n' "$data_dir" >&2
    exit 1
fi
if [[ "$list_only" == false ]]; then
    printf 'Processed %d PDF(s); %d failed. Results: %s\n' "$count" "$failures" "$output_root"
fi
[[ "$failures" -eq 0 ]]
