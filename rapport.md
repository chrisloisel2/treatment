# Rapport d'Audit — Problèmes Résiduels Post-Correction

**Date :** 2026-04-03  
**Périmètre :** 103 sessions (206 côtés) dans `data/`  
**Objet :** Audit exhaustif des 9 côtés signalés comme non corrigeables + découverte de problèmes systémiques

---

## Résumé Exécutif

Le verdict initial « impossible à corriger par logiciel » était **incorrect**. L'audit révèle :

1. **Tous les gaps capteur signalés sont corrigeables** par interpolation — la pince ne bouge pas pendant les gaps.
2. **Un bug systémique affecte 100 % des sessions** (202/202 CSV) : les timestamps gripper ne sont pas triés, causant de faux gaps dans les calculs de latence.
3. **194351** est la seule session véritablement irrécupérable (fichiers sources absents).

Résultat attendu après correction logicielle : **de 9 côtés problématiques à 2** (copies de 194351, structurellement incomplètes).

---

## 1. Bug Systémique — CSV Gripper Non Triés (100 % des sessions)

### 1.1 Détection

Sur 202 fichiers `gripper_{side}_data.csv` analysés dans `data/`, **202 présentent des timestamps non triés**.

Pattern identique sur tous les fichiers :

```
[0]  ts=1774630997230.403ms   ← buffer flush de l'initialisation
[1]  ts=1774630997246.880ms
...
[11] ts=1774630997407.046ms
[12] ts=1774630997167.106ms   ← SAUT NÉGATIF -239.9ms : vrai t0 session
[13] ts=1774630997191.784ms
...
```

Les premières ~12 lignes du CSV sont des **échantillons en retard du buffer d'acquisition** (flush au démarrage). Ils arrivent dans le fichier *après* les premiers vrais échantillons de session, mais leurs timestamps sont **postérieurs** au vrai démarrage de ~60–240ms. Le vrai début de session est à `row 12`, distance du trigger = 3ms.

### 1.2 Mesure

| Métrique | Valeur |
|---|---|
| Sessions avec CSV non triés | 202/202 (100%) |
| Amplitude du saut négatif | −60ms à −240ms (typiquement ~−200ms) |
| Nombre de lignes hors-ordre | 12 lignes en début de chaque fichier |
| Impact actuel sur session_pinces | Latences calculées incorrectement → faux gaps |

### 1.3 Impact sur les Métriques Actuelles

Le code `session_pinces.py` utilise `np.searchsorted` sur le tableau de timestamps **non trié**, ce qui donne des résultats incorrects. Dans la plupart des sessions les données s'avèrent correctes car les 12 premières lignes tombent avant la fenêtre vidéo, mais pour `192016/right` et `174827/right` elles tombent dans la fenêtre et génèrent un faux gap de ~115–571ms.

### 1.4 Quantification

**192016/right** (détecté comme ERROR, lat_max=44.7ms) :
- Sans tri : gap apparent = 115ms à row 6→7 → lat_max=44.7ms, 2 frames > 25ms
- Avec tri : gap réel = 115ms mais à la vraie position → lat_max=12.5ms après interpolation

**174827/right** (détecté OK par chance) :
- Sans tri : gap apparent = 571ms → mais lat_max=12.8ms car session_pinces utilise `sort` implicitement via `np.searchsorted` sur tableau non trié (comportement fortuit correct)

### 1.5 Résolution

**Fix immédiat dans `session_pinces.py`** :
```python
# Dans load_gripper_data() ou compute_alignment() :
ts_ns = np.sort(df['timestamp_ns'].values)   # ← ajouter ce sort
```

**Fix permanent dans la pipeline d'acquisition** : écrire les échantillons dans l'ordre chronologique (flush du buffer avant le trigger, pas après).

---

## 2. Gaps Capteur — Session 180317 (× 2 copies)

### 2.1 Description du Problème

| Attribut | Valeur |
|---|---|
| Sessions impactées | `session_20260327_180317` et `session_20260327_180317 copy` |
| Côtés | left ET right |
| Gap left | 733.7ms (row 316→317) à t=+5.1s de session |
| Gap right | 736.5ms (row 316→317) à t=+5.1s de session |
| lat_max actuel | left=355ms, right=364ms |
| Frames video dans le gap | 22 frames (idx 93–114) |

### 2.2 Analyse de la Cause

Le gap est **simultané** sur left, right **et** le tracker VIVE :

```
Gripper left  gap: 1774631002236ms → 1774631002970ms  (733.7ms)
Gripper right gap: 1774631002234ms → 1774631002971ms  (736.5ms)
Tracker VIVE  gap: 1774631002312ms → 1774631002975ms  (662.4ms)
```

Il s'agit d'une **déconnexion USB/Bluetooth simultanée** affectant tous les capteurs. La pince était **immobile** pendant toute la durée du gap :

```
Before gap: opening = 4.4mm (stable)
After gap:  opening = 4.4mm (stable)
Change:     0.0mm (aucun mouvement)
```

### 2.3 Évaluation de Corrigibilité

| Condition | Valeur | Seuil | Corrigible ? |
|---|---|---|---|
| Changement d'ouverture | 0.0mm | < 2.0mm | ✅ OUI |
| Pince en mouvement | Non | — | ✅ OUI |
| Données avant disponibles | Oui | — | ✅ OUI |
| Données après disponibles | Oui | — | ✅ OUI |

**Méthode de correction : interpolation constante**

La pince est immobile → l'ouverture durant le gap est exactement `4.4mm`. On génère des échantillons synthétiques à intervalles nominaux (17ms @ 60Hz) avec la valeur constante.

**Résultat simulé post-correction :**

| Métrique | Avant | Après |
|---|---|---|
| lat_max left | 355ms | **10.2ms** |
| lat_max right | 364ms | **10.2ms** |
| Frames > 25ms | 22 | **0** |
| Status | ERROR | **OK** |

### 2.4 Approche IA

L'interpolation constante est **optimale** ici car le signal est plat. Une IA (LSTM, GP) apporterait de la complexité sans bénéfice pour ce cas. En revanche, pour des gaps dynamiques (pince en mouvement), un modèle entraîné sur les trajectoires de pince pourrait interpoler plus précisément. Sur les 203 gaps > 50ms du dataset, 93% sont statiques → interpolation constante couvre l'essentiel.

---

## 3. Gap Capteur — Session 183527 / left

### 3.1 Description

| Attribut | Valeur |
|---|---|
| Session | `session_20260327_183527` |
| Côté | left uniquement (right : max gap = 20ms, normal) |
| Gap | 276.2ms (row 1250→1251) à t=+20.1s |
| lat_max actuel | 25.3ms (LATENCY_MAX dépassé de 0.3ms) |
| Frames dans le gap | 3 frames (idx 694–696) |

### 3.2 Analyse

Pince **complètement immobile** pendant et autour du gap :

```
Avant: [19.5, 19.5, 19.5, 19.5, 19.5]mm
Après: [19.5, 19.5, 19.5, 19.5, 19.5]mm
Changement: 0.0mm
```

Le gap n'affecte que le capteur gauche — le capteur droit est parfaitement continu (gap max = 20ms). Cause probable : déconnexion temporaire du seul capteur left.

### 3.3 Corrigibilité

**Résultat simulé post-correction :**

| Métrique | Avant | Après |
|---|---|---|
| lat_max left | 25.3ms | **8.4ms** |
| Frames > 25ms | 2 | **0** |
| Status | WARNING | **OK** |

---

## 4. Gap + CSV Non Trié — Session 192016 (× 3 copies)

### 4.1 Description

| Attribut | Valeur |
|---|---|
| Sessions impactées | `192016`, `192016 copy`, `192016 copy 2` |
| Côté | right uniquement |
| Gap réel (après tri) | 115.7ms au début de session (row 0→1 après tri) |
| lat_max actuel | 44.7ms |
| Frames dans le gap | 4 frames (idx 11–14) |

### 4.2 Double Problème

**Problème 1 — CSV non trié (systémique)** :  
Les 6 premières lignes du CSV right sont des échantillons du buffer d'initialisation. Leurs timestamps (1774635616210ms–1774635616290ms) sont postérieurs au vrai t0 (1774635615920ms). Cela crée un faux saut de ~290ms en début de fichier.

**Problème 2 — Gap hardware réel** :  
Après tri, il reste un gap de **115.7ms** entre row 0 et row 1 (1774635615920ms → 1774635616036ms). Ce gap est à l'initialisation du capteur, pas en cours de session.

**Anomalie supplémentaire dans `copy`** :  
Le `tracker_positions.csv` de `192016 copy` a les colonnes `tracker_left_*` et `tracker_right_*` **interverties** par rapport à l'original. Tous les calculs utilisant ce fichier pour le côté left/right seront incorrects.

### 4.3 Corrigibilité

Ouverture pendant le gap : 26.7mm → 26.6mm (Δ = 0.1mm, pince immobile).

**Résultat simulé post-correction (tri + interpolation) :**

| Métrique | Avant | Après |
|---|---|---|
| lat_max right | 44.7ms | **12.5ms** |
| Frames > 25ms | 2 | **0** |
| Status | WARNING | **OK** |

---

## 5. Frames Sans Capteur — Session 193649 / left

### 5.1 Description

| Attribut | Valeur |
|---|---|
| Session | `session_20260327_193649` |
| Durée totale | 6.94s (session courte) |
| Côté | left (et head, right partiellement) |
| Problème | 6 frames avant le premier sample capteur |
| Écart max | 84.6ms avant sensor t0 |

### 5.2 Cause Racine

La correction `fix_camera_offset` a décalé les timestamps vidéo de **-3556ms** (trigger à 1774636609423ms, vidéo originale commençait à 1774636612989ms). Après correction, les 6 premières frames tombent **avant** le premier échantillon capteur (sensor t0 = 1774636609517ms, trigger t0 = 1774636609423ms).

Le capteur démarre 94ms après le trigger — les 6 premières frames (32–84ms après trigger) tombent dans cette fenêtre aveugle.

### 5.3 Corrigibilité

Le capteur démarre avec opening = 0.6mm et reste stable :

```
[0] ts=1774636609517ms opening=0.60mm
[1] ts=1774636609534ms opening=0.60mm
[2] ts=1774636609549ms opening=0.50mm
```

**Fix : extrapolation constante vers l'arrière** — assigner `opening = 0.6mm` aux 6 frames antérieures. Erreur maximale ≈ 0.1mm.

| Métrique | Avant | Après |
|---|---|---|
| FRAMES_NO_SENSOR | 6 frames | **0** |
| lat_max | 8.3ms | **8.3ms** (inchangé) |
| Status | WARNING | **OK** |

---

## 6. Session 194351 — Incomplète (Irrécupérable)

### 6.1 Inventaire des Fichiers

| Fichier | Original | Copy | Status |
|---|---|---|---|
| `metadata.json` | ✗ ABSENT | ✗ ABSENT | **Irrécupérable** |
| `gripper_left_data.csv` | ✗ ABSENT | ✗ ABSENT | **Irrécupérable** |
| `gripper_right_data.csv` | ✗ ABSENT | ✗ ABSENT | **Irrécupérable** |
| `tracker_positions.csv` | ✗ ABSENT | ✗ ABSENT | **Irrécupérable** |
| `videos/left.mp4` | ✗ ABSENT | ✗ ABSENT | **Irrécupérable** |
| `videos/left.jsonl` | ✗ ABSENT | ✗ ABSENT | **Irrécupérable** |
| `videos/head.mp4` | ✗ ABSENT | ✗ ABSENT | **Irrécupérable** |
| `videos/head.jsonl` | ✗ ABSENT | ✗ ABSENT | **Irrécupérable** |
| `videos/right.mp4` | ✓ 21.4 MB | ✓ 21.4 MB | Exploitable partiellement |
| `videos/right.jsonl` | ✓ 1057 frames | ✓ 1057 frames | Exploitable partiellement |

### 6.2 Analyse du right.jsonl

```
Frames disponibles : idx 86 → 1142  (1 057 frames valides)
Frames manquantes  : idx 1 → 85     (85 frames perdues)
Dernière ligne     : tronquée ("{"index":11" — crash en écriture)
Durée couverte     : 35.25s @ 30.0 fps
```

### 6.3 Cause Probable

Crash ou interruption brutale du processus d'acquisition après démarrage. Le fichier right.jsonl a été coupé en cours d'écriture (dernière ligne tronquée = `{"index":11`). Les fichiers left, head, metadata et CSV n'ont jamais été créés.

### 6.4 Ce Qui Est Récupérable

- La vidéo right.mp4 est **complète** et exploitable (h264, 1280×720, 1057 frames, 35.23s)
- Les timestamps right.jsonl permettent un horodatage partiel

**Ce qui n'est pas récupérable** :
- Données gripper (aucune source alternative)
- Positions tracker 3D
- Vidéos left et head
- Métadonnées de session

**Verdict : session à exclure du dataset**, les copies sont identiques.

---

## 7. Tableau de Bord — Récapitulatif

### 7.1 État Avant → Après Corrections Proposées

| Session/Côté | Problème | lat_max avant | lat_max après | Status avant | Status après |
|---|---|---|---|---|---|
| 180317/left | Gap 734ms (capteur immobile) | 355ms | **10ms** | ERROR | **OK** |
| 180317/right | Gap 737ms (capteur immobile) | 364ms | **10ms** | ERROR | **OK** |
| 180317 copy/left | Idem | 355ms | **10ms** | ERROR | **OK** |
| 180317 copy/right | Idem | 364ms | **10ms** | ERROR | **OK** |
| 183527/left | Gap 276ms (capteur immobile) | 25ms | **8ms** | WARNING | **OK** |
| 192016/right | CSV non trié + gap 116ms | 45ms | **13ms** | WARNING | **OK** |
| 192016 copy/right | Idem + colonnes tracker swappées | 45ms | **13ms** | WARNING | **OK** |
| 192016 copy2/right | CSV non trié + gap 116ms | 45ms | **13ms** | WARNING | **OK** |
| 193649/left | 6 frames avant sensor | N/A | N/A | WARNING | **OK** |
| 194351/left | Fichiers absents | FAILED | FAILED | FAILED | **FAILED** (irrécupérable) |
| 194351/right | Fichiers absents | FAILED | FAILED | FAILED | **FAILED** (irrécupérable) |
| 194351 copy/* | Idem | FAILED | FAILED | FAILED | **FAILED** (irrécupérable) |

### 7.2 Bilan Global Post-Corrections

```
206 côtés total
├── OK          : 193 (93.7%)
├── Corrigibles : 9+  (voir ci-dessus)  →  202+ après fix (97.9%)
├── FAILED      : 4   (194351 ×2, copies — irrécupérables)
└── Systémique  : CSV non triés sur 100% des sessions — fix immédiat dans session_pinces.py
```

---

## 8. Rôle Possible d'une IA

### 8.1 Ce Que l'IA Peut Faire (au-delà de l'interpolation simple)

**Cas 1 — Gaps statiques (93% des cas) :**  
L'interpolation constante est optimale. Erreur = 0mm. Aucune IA nécessaire.

**Cas 2 — Gaps dynamiques (7% des cas, pince en mouvement) :**  
Exemple : `174827/right` — gap 571ms avec Δopening = 16mm. L'interpolation linéaire introduit une erreur sur la forme du mouvement. Un modèle entraîné sur les trajectoires de pince pourrait reconstruire la cinématique manquante avec une erreur bien inférieure à 16mm.

Architecture suggérée :
```
Entrée  : N échantillons avant gap + N après gap (fenêtre glissante)
Modèle  : TCN (Temporal Convolutional Network) ou LSTM bidirectionnel
Sortie  : séquence interpolée dans le gap
Dataset : 203 gaps dont 189 statiques (supervision parfaite) + 14 dynamiques
```

**Cas 3 — Détection automatique de la validité d'un gap :**  
Un classificateur (gradient boosting ou simple threshold) sur :
- `|opening_after - opening_before|` (principal)
- `velocity_before` (mm/s dans les 200ms précédant le gap)
- `gap_duration_ms`

Règle actuelle suffisante : `|Δopening| < 2mm AND velocity_before < 50mm/s → interpoler`.

### 8.2 Ce Que l'IA Ne Peut Pas Faire

- **194351** : pas de données source → impossible de reconstruire gripper, tracker, ou caméras manquantes
- **Réordonner des données absentes** : le bug CSV non trié est corrigeable par un simple `sort`, pas par IA
- **Vérification de validité physique** : une IA ne peut pas certifier que l'interpolation est correcte sans données de référence

### 8.3 Recommandation Pratique

Pour ce dataset spécifique :

1. **Court terme** : implémenter `fix_sensor_gaps()` — tri CSV + interpolation constante si Δopening < 2mm — résout 9/9 cas en quelques lignes de code.
2. **Moyen terme** : logger un signal de "santé capteur" (heartbeat) permettant de distinguer gap hardware vs données perdues.
3. **Long terme** : si le projet génère davantage de données avec des gaps dynamiques, entraîner un TCN de reconstruction sur les sessions propres.

---

## 9. Plan de Correction Logicielle

### 9.1 Fix 1 — Tri des CSV Gripper (systémique, impact 100% sessions)

Dans `session_pinces.py`, fonction de chargement des données gripper :

```python
df = pd.read_csv(path)
df = df.sort_values('timestamp_ns').reset_index(drop=True)  # fix unsorted buffer
ts_ns = df['timestamp_ns'].values
opening = df['opening_mm'].values
```

**Impact :** supprime les faux gaps sur 192016 (et potentiellement d'autres sessions).

### 9.2 Fix 2 — Interpolation des Gaps Statiques

Nouvelle fonction `fill_sensor_gaps()` à appeler dans `fix` mode :

```python
def fill_sensor_gaps(ts_ns, opening_mm, dt_nominal_ms=17.0, 
                     max_opening_change_mm=2.0, max_gap_ms=1500.0):
    """
    Detects and fills gaps in sensor data where the gripper is static.
    Returns augmented (ts_ns, opening_mm) arrays.
    
    Conditions for filling:
    - gap_duration < max_gap_ms
    - |opening_after - opening_before| < max_opening_change_mm
    """
    diffs = np.diff(ts_ns) / 1e6  # ms
    filled_ts = list(ts_ns)
    filled_op = list(opening_mm)
    
    for i, gap in enumerate(diffs):
        if gap > 4 * dt_nominal_ms:  # gap > 4 intervals
            op_before = opening_mm[i]
            op_after = opening_mm[i + 1]
            if gap <= max_gap_ms and abs(op_after - op_before) <= max_opening_change_mm:
                t_start = ts_ns[i]
                t_end = ts_ns[i + 1]
                n = max(1, int(gap / dt_nominal_ms) - 1)
                synth_ts = np.linspace(t_start + dt_nominal_ms * 1e6, 
                                       t_end - dt_nominal_ms * 1e6, n, dtype=np.int64)
                synth_op = np.linspace(op_before, op_after, n)
                filled_ts.extend(synth_ts.tolist())
                filled_op.extend(synth_op.tolist())
    
    # Re-sort after insertions
    order = np.argsort(filled_ts)
    return np.array(filled_ts)[order], np.array(filled_op)[order]
```

### 9.3 Fix 3 — Extrapolation vers l'Arrière (193649)

Dans `compute_alignment()`, avant le calcul de latence :

```python
# Si des frames tombent avant le premier sample capteur, étendre le capteur
vid_start_ns = frame_ts_ns[0]
if ts_sensor_ns[0] > vid_start_ns:
    # Extrapolation constante : opening = première valeur connue
    synth_ts = np.arange(vid_start_ns, ts_sensor_ns[0], dt_nominal_ns)
    synth_op = np.full(len(synth_ts), opening_mm[0])
    ts_sensor_ns = np.concatenate([synth_ts, ts_sensor_ns])
    opening_mm = np.concatenate([synth_op, opening_mm])
```

### 9.4 Fix 4 — Signalement de la Copie 192016 avec Tracker Swappé

Dans `verify.py`, ajouter une vérification de cohérence des headers tracker :

```python
def check_tracker_column_order(session_path):
    """Détecte si les colonnes left/right sont dans un ordre inhabituel."""
    csv = Path(session_path) / 'tracker_positions.csv'
    if not csv.exists(): return
    cols = pd.read_csv(csv, nrows=0).columns.tolist()
    # Ordre attendu: head, left, right (ou head, right, left selon les sessions)
    # Détecter les inversions par rapport aux autres sessions du même lot
    ...
```

---

## 10. Conclusions

### Ce qui était considéré hardware est en réalité logiciel

| Problème | Cause réelle | Solution |
|---|---|---|
| lat_max 355ms (180317) | Gap USB 734ms, pince immobile | Interpolation constante |
| lat_max 45ms (192016) | CSV non trié + gap 115ms | Sort + interpolation |
| lat_max 25ms (183527) | Gap BT 276ms, pince immobile | Interpolation constante |
| FRAMES_NO_SENSOR (193649) | Capteur démarre 94ms après trigger | Extrapolation arrière |

### Découverte critique non signalée auparavant

**100% des CSV gripper ont des timestamps non triés.** Ce bug systémique dans le code d'acquisition (buffer flush tardif) crée des faux sauts temporels en début de chaque fichier (~12 lignes hors-ordre, amplitude ~200ms). L'impact sur les latences calculées est limité dans la plupart des cas car `session_pinces.py` utilise `np.searchsorted` qui requiert des données triées — les résultats actuels sont donc partiellement incorrects.

**Action requise :** corriger la pipeline d'acquisition ET ajouter un tri systématique dans `session_pinces.py`.

### Seul 194351 est véritablement irrécupérable

Les deux copies de `session_20260327_194351` n'ont que `videos/right.mp4` et `videos/right.jsonl`. Sans metadata, gripper CSV, tracker CSV, et caméras left/head : la session ne peut pas être intégrée au pipeline. À exclure du dataset.
