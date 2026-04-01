#!/usr/bin/env bash
set -Eeuo pipefail

WATCH_DIR="${WATCH_DIR:-/home/sftpuser/inbox}"
WORKERS="${WORKERS:-8}"
SCAN_INTERVAL="${SCAN_INTERVAL:-5}"
RUN_ONCE="${RUN_ONCE:-0}"

MONGO_URI="${MONGO_URI:-mongodb+srv://christoloisel:rose@cluster0.ppyauvl.mongodb.net/physical_data}"
MONGO_DB="${MONGO_DB:-physical_data}"
MONGO_COLLECTION="${MONGO_COLLECTION:-cameras}"

LOCK_ROOT="${LOCK_ROOT:-/run/camera_renamer}"
LOG_FILE="${LOG_FILE:-/var/log/camera_renamer.log}"

mkdir -p "$LOCK_ROOT"
touch "$LOG_FILE"

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

require_bin() {
  command -v "$1" >/dev/null 2>&1 || {
    log "ERREUR: binaire manquant: $1"
    exit 1
  }
}

require_bin jq
require_bin mongosh
require_bin find
require_bin mv
require_bin mktemp
require_bin xargs
require_bin flock
require_bin chown
require_bin cp
require_bin tee

mongo_position_by_serial() {
  local serial="$1"
  local out

  out="$(
    mongosh "$MONGO_URI" --quiet --eval "
      const dbObj = db.getSiblingDB('$MONGO_DB');
      const doc = dbObj.getCollection('$MONGO_COLLECTION').findOne(
        { serial_number: '$serial' },
        { _id: 0, position: 1 }
      );
      if (doc && doc.position) {
        print(String(doc.position));
      }
    " 2>/dev/null || true
  )"

  printf '%s\n' "$out" | tr -d '\r' | sed '/^[[:space:]]*$/d' | tail -n 1
}

process_session() {
  local session_dir="$1"
  local session_name metadata videos_dir session_lock
  local tmp_json new_tmp changed owner_ref
  local cam_files_count final_files_count remaining_cam_files
  local tmp_prefix

  session_name="$(basename "$session_dir")"
  metadata="$session_dir/metadata.json"
  videos_dir="$session_dir/videos"
  session_lock="$LOCK_ROOT/${session_name}.lock"
  changed=0
  tmp_prefix=".rename_tmp_$$"

  [[ -d "$session_dir" ]] || return 0
  [[ -f "$metadata" ]] || { log "SKIP [$session_name] metadata.json absent"; return 0; }
  [[ -d "$videos_dir" ]] || { log "SKIP [$session_name] videos absent"; return 0; }

  exec 9>"$session_lock"
  flock -n 9 || {
    log "SKIP [$session_name] deja en cours"
    return 0
  }

  if find "$videos_dir" -maxdepth 1 -type f -name '*.part' | grep -q .; then
    log "SKIP [$session_name] capture en cours"
    return 0
  fi

  cam_files_count="$(find "$videos_dir" -maxdepth 1 -type f \( -name 'cam*.mp4' -o -name 'cam*.jsonl' \) | wc -l)"
  final_files_count="$(find "$videos_dir" -maxdepth 1 -type f \( -name 'left.mp4' -o -name 'left.jsonl' -o -name 'right.mp4' -o -name 'right.jsonl' -o -name 'head.mp4' -o -name 'head.jsonl' \) | wc -l)"

  if [[ "$cam_files_count" -gt 0 && "$final_files_count" -gt 0 ]]; then
    log "ERREUR [$session_name] etat mixte detecte: fichiers cam* et left/right/head presents en meme temps"
    return 1
  fi

  log "SCAN [$session_name]"

  tmp_json="$(mktemp)"
  cp "$metadata" "$tmp_json"

  mapfile -t cam_ids < <(jq -r '.cameras | keys[]' "$tmp_json" 2>/dev/null || true)

  if [[ "${#cam_ids[@]}" -eq 0 ]]; then
    rm -f "$tmp_json"
    log "NOOP [$session_name] aucune camera"
    return 0
  fi

  declare -A SRC_POS_BY_ID=()
  declare -A DST_POS_BY_ID=()
  declare -A SERIAL_BY_ID=()

  local cam_id serial src_pos dst_pos

  for cam_id in "${cam_ids[@]}"; do
    serial="$(jq -r --arg id "$cam_id" '.cameras[$id].serial // empty' "$tmp_json")"
    src_pos="$(jq -r --arg id "$cam_id" '.cameras[$id].position // empty' "$tmp_json")"

    if [[ -z "$serial" || -z "$src_pos" ]]; then
      log "WARN [$session_name] camera id=$cam_id incomplete"
      continue
    fi

    dst_pos="$(mongo_position_by_serial "$serial")"

    if [[ -z "$dst_pos" ]]; then
      log "WARN [$session_name] serial=$serial absent MongoDB"
      continue
    fi

    case "$dst_pos" in
      left|right|head) ;;
      *)
        log "WARN [$session_name] serial=$serial position invalide: $dst_pos"
        continue
        ;;
    esac

    SRC_POS_BY_ID["$cam_id"]="$src_pos"
    DST_POS_BY_ID["$cam_id"]="$dst_pos"
    SERIAL_BY_ID["$cam_id"]="$serial"
  done

  local old_mp4 old_jsonl tmp_mp4 tmp_jsonl final_mp4 final_jsonl

  for cam_id in "${cam_ids[@]}"; do
    src_pos="${SRC_POS_BY_ID[$cam_id]:-}"
    dst_pos="${DST_POS_BY_ID[$cam_id]:-}"
    [[ -n "$src_pos" && -n "$dst_pos" ]] || continue
    [[ "$src_pos" != "$dst_pos" ]] || continue

    old_mp4="$videos_dir/${src_pos}.mp4"
    old_jsonl="$videos_dir/${src_pos}.jsonl"
    tmp_mp4="$videos_dir/${tmp_prefix}_${cam_id}.mp4"
    tmp_jsonl="$videos_dir/${tmp_prefix}_${cam_id}.jsonl"

    if [[ -e "$old_mp4" ]]; then
      mv -- "$old_mp4" "$tmp_mp4"
      log "TMP  [$session_name] $(basename "$old_mp4") -> $(basename "$tmp_mp4")"
      changed=1
    fi

    if [[ -e "$old_jsonl" ]]; then
      mv -- "$old_jsonl" "$tmp_jsonl"
      log "TMP  [$session_name] $(basename "$old_jsonl") -> $(basename "$tmp_jsonl")"
      changed=1
    fi
  done

  for cam_id in "${cam_ids[@]}"; do
    src_pos="${SRC_POS_BY_ID[$cam_id]:-}"
    dst_pos="${DST_POS_BY_ID[$cam_id]:-}"
    serial="${SERIAL_BY_ID[$cam_id]:-}"
    [[ -n "$src_pos" && -n "$dst_pos" ]] || continue

    if [[ "$src_pos" != "$dst_pos" ]]; then
      tmp_mp4="$videos_dir/${tmp_prefix}_${cam_id}.mp4"
      tmp_jsonl="$videos_dir/${tmp_prefix}_${cam_id}.jsonl"
      final_mp4="$videos_dir/${dst_pos}.mp4"
      final_jsonl="$videos_dir/${dst_pos}.jsonl"

      if [[ -e "$final_mp4" || -e "$final_jsonl" ]]; then
        log "ERREUR [$session_name] destination finale existe deja pour $dst_pos"
        rm -f "$tmp_json"
        return 1
      fi

      [[ -e "$tmp_mp4" ]] && mv -- "$tmp_mp4" "$final_mp4"
      [[ -e "$tmp_jsonl" ]] && mv -- "$tmp_jsonl" "$final_jsonl"

      log "FIX  [$session_name] serial=$serial $src_pos -> $dst_pos"
    fi

    new_tmp="$(mktemp)"
    jq --arg id "$cam_id" --arg newpos "$dst_pos" \
      '.cameras[$id].position = $newpos' \
      "$tmp_json" > "$new_tmp"
    mv "$new_tmp" "$tmp_json"
  done

  if jq -e '.camera_anchors' "$tmp_json" >/dev/null 2>&1; then
    for cam_id in "${cam_ids[@]}"; do
      src_pos="${SRC_POS_BY_ID[$cam_id]:-}"
      dst_pos="${DST_POS_BY_ID[$cam_id]:-}"
      [[ -n "$src_pos" && -n "$dst_pos" ]] || continue
      [[ "$src_pos" != "$dst_pos" ]] || continue

      if jq -e --arg old "$src_pos" '.camera_anchors | has($old)' "$tmp_json" >/dev/null 2>&1; then
        new_tmp="$(mktemp)"
        jq --arg old "$src_pos" --arg new "$dst_pos" '
          .camera_anchors[$new] = .camera_anchors[$old]
          | del(.camera_anchors[$old])
        ' "$tmp_json" > "$new_tmp"
        mv "$new_tmp" "$tmp_json"
      fi
    done
  fi

  if [[ "$changed" -eq 1 ]]; then
    cp "$metadata" "$metadata.bak.$(date +%s)"
    mv "$tmp_json" "$metadata"

    owner_ref="$session_dir/gripper_left_data.csv"
    if [[ -e "$owner_ref" ]]; then
      chown --reference="$owner_ref" "$metadata" 2>/dev/null || true
      find "$videos_dir" -maxdepth 1 -type f \( -name 'left.*' -o -name 'right.*' -o -name 'head.*' \) -exec chown --reference="$owner_ref" {} \; 2>/dev/null || true
    fi

    log "UPDATE [$session_name]"
  else
    rm -f "$tmp_json"
    log "NOOP [$session_name]"
  fi

  remaining_cam_files="$(find "$videos_dir" -maxdepth 1 -type f \( -name 'cam*.mp4' -o -name 'cam*.jsonl' \) | sort || true)"

  if [[ -n "$remaining_cam_files" ]]; then
    log "ERREUR [$session_name] fichiers cam* restants apres traitement:"
    while IFS= read -r f; do
      [[ -n "$f" ]] && log "ERREUR [$session_name] - $(basename "$f")"
    done <<< "$remaining_cam_files"
    return 1
  fi

  return 0
}

export WATCH_DIR WORKERS SCAN_INTERVAL RUN_ONCE MONGO_URI MONGO_DB MONGO_COLLECTION LOCK_ROOT LOG_FILE
export -f log require_bin mongo_position_by_serial process_session

run_scan() {
  find "$WATCH_DIR" -mindepth 1 -maxdepth 1 -type d -name 'session_*' -print0 \
    | xargs -0 -n 1 -P "$WORKERS" bash -c 'process_session "$1"' _
}

main() {
  log "SERVICE START watch_dir=$WATCH_DIR workers=$WORKERS"

  if [[ "$RUN_ONCE" == "1" ]]; then
    run_scan
    log "TERMINE"
    exit 0
  fi

  while true; do
    run_scan || true
    sleep "$SCAN_INTERVAL"
  done
}

main "$@"
