#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="treatment"
USER_NAME="ia"
GROUP_NAME="ia"

APP_DIR="/Users/christopher/Downloads/sync_test_1/treatment"
WORK_DIR="/mnt/storage/mistral"
PYTHON_BIN="/usr/bin/python3"
PORT="8000"
HOST="0.0.0.0"

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "[1/6] Vérification des chemins"
if [ ! -d "$APP_DIR" ]; then
  echo "Erreur: dossier introuvable: $APP_DIR"
  exit 1
fi

if [ ! -d "$WORK_DIR" ]; then
  echo "Création du dossier de travail: $WORK_DIR"
  mkdir -p "$WORK_DIR"
  chown "$USER_NAME:$GROUP_NAME" "$WORK_DIR"
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Erreur: Python introuvable: $PYTHON_BIN"
  exit 1
fi

echo "[2/6] Création du fichier systemd: $SERVICE_FILE"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Treatment FastAPI Server
After=network.target
Wants=network.target

[Service]
Type=simple
User=$USER_NAME
Group=$GROUP_NAME
WorkingDirectory=$APP_DIR
ExecStart=$PYTHON_BIN $APP_DIR/server/server.py --root $WORK_DIR --host $HOST --port $PORT
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

# Sécurité minimale
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

echo "[3/6] Rechargement de systemd"
systemctl daemon-reload

echo "[4/6] Activation au démarrage"
systemctl enable "$SERVICE_NAME"

echo "[5/6] Démarrage immédiat"
systemctl restart "$SERVICE_NAME"

echo "[6/6] Statut"
systemctl --no-pager --full status "$SERVICE_NAME" || true

echo
echo "Service installé."
echo "Commandes utiles :"
echo "  sudo systemctl status $SERVICE_NAME"
echo "  sudo journalctl -u $SERVICE_NAME -f"
echo "  sudo systemctl restart $SERVICE_NAME"
echo "  sudo systemctl stop $SERVICE_NAME"
