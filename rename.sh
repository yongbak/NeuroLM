#!/usr/bin/env bash
set -euo pipefail

# 사용법:
#   ./rename_pkl.sh /path/to/X
# 미지정 시 현재 디렉토리를 X로 간주
X_DIR="${1:-.}"

DRY_RUN="${DRY_RUN:-1}"   # 기본은 드라이런(출력만). 실제 변경하려면 DRY_RUN=0

for split in train test val; do
  dir="$X_DIR/$split"
  [[ -d "$dir" ]] || continue

  find "$dir" -type f -name '*.pkl' -print0 | while IFS= read -r -d '' f; do
    name="$(basename -- "$f")"

    # 기대 패턴: <part1>_<part2>-<part3>
    # 1) 마지막 '-' 기준으로 part3 분리 (ex: 2.pkl)
    if [[ "$name" != *-* ]]; then
      continue
    fi
    left="${name%-*}"     # ex: 999620_gaussian_noise_2_s6_b_2024_03
    part3="${name##*-}"   # ex: 2.pkl

    # 2) left에서 "_s..." 시작점 기준으로 part1/part2 분리
    #    part2는 s로 시작한다고 가정 (예: s6_b_2024_03)
    if [[ "$left" != *_s* ]]; then
      continue
    fi
    part1="${left%%_s*}"        # ex: 999620_gaussian_noise_2
    part2="s${left#*_s}"        # ex: s6_b_2024_03

    new="${part1}-${part2}-${part3}"
    [[ "$new" == "$name" ]] && continue

    src="$f"
    dst="$(dirname -- "$f")/$new"

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[DRY] $src -> $dst"
    else
      # 덮어쓰기 방지(-n). 이미 존재하면 스킵.
      mv -n -- "$src" "$dst"
      echo "[OK]  $src -> $dst"
    fi
  done
done
