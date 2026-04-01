#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_mac.sh — Lance SyncML Studio en local sur macOS
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Valeurs par défaut ────────────────────────────────────────────────────────
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
BRONZE_DIR="${BRONZE_DIR:-/Users/christopher/Downloads/Video ok 2}"
SILVER_DIR="${SILVER_DIR:-${HOME}/silver}"
VENV_DIR="${SCRIPT_DIR}/.venv"
OPEN_BROWSER="${OPEN_BROWSER:-true}"

# ── Aide ──────────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $(basename "$0") [COMMANDE] [OPTIONS]

Commandes :
  server        Lance le serveur web FastAPI (défaut)
  run           Lance la pipeline en CLI
  gui           Lance l'interface graphique PyQt6
  install       Installe les dépendances dans .venv
  help          Affiche cette aide

Options (server) :
  --host HOST          Adresse d'écoute       (défaut: 127.0.0.1)
  --port PORT          Port                   (défaut: 8000)
  --bronze-dir DIR     Dossier de travail     (défaut: /Users/christopher/Downloads/Video ok 2)
  --silver-dir DIR     Dossier de sortie      (défaut: ~/silver)
  --no-browser         Ne pas ouvrir le navigateur

Options (run) :
  --session NAME       Traiter une session spécifique
  --all                Traiter toutes les sessions non traitées
  --bronze-dir DIR     Dossier de travail     (défaut: /Users/christopher/Downloads/Video ok 2)
  --write              Copier vers silver après validation
  --delete-after-store Supprimer de bronze après copie (requiert --write)
  Tout autre argument est transmis tel quel à pipeline.py.

Variables d'environnement :
  PORT, HOST, BRONZE_DIR, SILVER_DIR, OPEN_BROWSER (true/false)

Exemples :
  ./run_mac.sh install
  ./run_mac.sh server --bronze-dir ~/Desktop/sessions --port 9000
  ./run_mac.sh run --all --bronze-dir ~/Desktop/sessions
  ./run_mac.sh run --session ma_session --write
  ./run_mac.sh gui
EOF
  exit 0
}

# ── Couleurs ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERREUR]${NC} $*" >&2; }
die()     { error "$*"; exit 1; }

# ── Python ────────────────────────────────────────────────────────────────────
find_python() {
  for py in python3.12 python3.11 python3.10 python3; do
    if command -v "$py" &>/dev/null; then
      echo "$py"; return
    fi
  done
  die "Python 3.10+ introuvable. Installez-le via: brew install python@3.12"
}

# ── Venv ──────────────────────────────────────────────────────────────────────
ensure_venv() {
  local py
  py=$(find_python)
  if [ ! -f "${VENV_DIR}/bin/python" ]; then
    info "Création du venv Python dans ${VENV_DIR}…"
    "$py" -m venv "${VENV_DIR}"
  fi
}

install_deps() {
  ensure_venv
  info "Installation des dépendances web…"
  "${VENV_DIR}/bin/pip" install --quiet --upgrade pip
  "${VENV_DIR}/bin/pip" install --quiet -r "${SCRIPT_DIR}/server/requirements_web.txt"
  ok "Dépendances web installées."
}

install_deps_gui() {
  ensure_venv
  info "Installation des dépendances GUI…"
  "${VENV_DIR}/bin/pip" install --quiet -r "${SCRIPT_DIR}/server/requirements_gui.txt"
  ok "Dépendances GUI installées."
}

# ── Préparation du dossier bronze ─────────────────────────────────────────────
prepare_bronze() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    warn "Dossier bronze introuvable : ${dir}"
    read -r -p "  Créer le dossier maintenant ? [o/N] " ans
    case "$ans" in
      [oOyY]*) mkdir -p "$dir"; ok "Créé : ${dir}" ;;
      *) die "Annulé. Spécifiez un dossier existant avec --bronze-dir." ;;
    esac
  fi
}

# ══════════════════════════════════════════════════════════════════════════════
# Parsing de la commande principale
# ══════════════════════════════════════════════════════════════════════════════
CMD="${1:-server}"
shift || true

case "$CMD" in
  help|--help|-h) usage ;;
  server|run|gui|install) ;;  # valides
  *) die "Commande inconnue : ${CMD}. Lancez '$(basename "$0") help' pour l'aide." ;;
esac

# ── Commande : install ────────────────────────────────────────────────────────
if [ "$CMD" = "install" ]; then
  install_deps
  install_deps_gui
  ok "Installation terminée. Vous pouvez maintenant lancer :"
  echo "  ./run_mac.sh server"
  echo "  ./run_mac.sh gui"
  exit 0
fi

# ── Parsing des options spécifiques ──────────────────────────────────────────
EXTRA_ARGS=()
NO_BROWSER=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)         HOST="$2";       shift 2 ;;
    --port)         PORT="$2";       shift 2 ;;
    --bronze-dir)   BRONZE_DIR="$2"; shift 2 ;;
    --silver-dir)   SILVER_DIR="$2"; shift 2 ;;
    --no-browser)   NO_BROWSER=true; shift ;;
    *)              EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# ── Vérifications communes ────────────────────────────────────────────────────
ensure_venv

# Vérifie que les dépendances sont présentes
if ! "${VENV_DIR}/bin/python" -c "import fastapi, uvicorn" 2>/dev/null; then
  warn "Dépendances manquantes — installation automatique…"
  install_deps
fi

# ══════════════════════════════════════════════════════════════════════════════
# Commande : server
# ══════════════════════════════════════════════════════════════════════════════
if [ "$CMD" = "server" ]; then
  prepare_bronze "$BRONZE_DIR"

  info "Démarrage du serveur SyncML Studio"
  echo "  Bronze   : ${BRONZE_DIR}"
  echo "  Silver   : ${SILVER_DIR}"
  echo "  URL      : http://${HOST}:${PORT}"
  echo ""

  if [ "$NO_BROWSER" = "false" ] && [ "$OPEN_BROWSER" = "true" ] && command -v open &>/dev/null; then
    (sleep 2 && open "http://127.0.0.1:${PORT}") &
  fi

  BRONZE_DIR="$BRONZE_DIR" SILVER_DIR="$SILVER_DIR" \
    "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/main.py" server \
      --host "$HOST" \
      --port "$PORT" \
      "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
  exit $?
fi

# ══════════════════════════════════════════════════════════════════════════════
# Commande : run (pipeline CLI)
# ══════════════════════════════════════════════════════════════════════════════
if [ "$CMD" = "run" ]; then
  prepare_bronze "$BRONZE_DIR"

  info "Lancement de la pipeline"
  echo "  Bronze : ${BRONZE_DIR}"
  echo ""

  "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/main.py" run \
    --bronze-dir "$BRONZE_DIR" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
  exit $?
fi

# ══════════════════════════════════════════════════════════════════════════════
# Commande : gui
# ══════════════════════════════════════════════════════════════════════════════
if [ "$CMD" = "gui" ]; then
  if ! "${VENV_DIR}/bin/python" -c "import PyQt6" 2>/dev/null; then
    warn "PyQt6 manquant — installation automatique…"
    install_deps_gui
  fi

  prepare_bronze "$BRONZE_DIR"

  info "Lancement de l'interface graphique"
  echo "  Bronze : ${BRONZE_DIR}"
  echo ""

  BRONZE_DIR="$BRONZE_DIR" SILVER_DIR="$SILVER_DIR" \
    "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/main.py" gui \
      "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
  exit $?
fi
