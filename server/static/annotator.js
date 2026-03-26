// ══════════════════════════════════════════════════════════════════
// ANNOTATEUR — Vue episode_subtitle.json
// ══════════════════════════════════════════════════════════════════

const AN = {
	sessPath: null,
	data: null,     // contenu brut de episode_subtitle.json
};

// Palette de couleurs pour les segments (par texte unique)
const AN_PALETTE = [
	'#5b7df5', '#22d386', '#f59e0b', '#f04444',
	'#8b5cf6', '#06b6d4', '#ec4899', '#10b981',
	'#f97316', '#6366f1',
];

// ── Liste de sessions ──────────────────────────────────────────────

function anRefreshSessions() {
	const list = ge('an-sess-list');
	const empty = ge('an-sess-empty');
	if (!S.sessions.length) {
		empty.style.display = '';
		list.querySelectorAll('.an-sess-item').forEach(el => el.remove());
		return;
	}
	const withSub = S.sessions.filter(s => s.has_subtitle);
	const total = S.sessions.length;

	if (!withSub.length) {
		empty.style.display = '';
		empty.textContent = total
			? `Aucune session avec episode_subtitle.json (${total} scannées)`
			: 'Cliquez « Scanner »\npour charger les sessions';
		list.querySelectorAll('.an-sess-item').forEach(el => el.remove());
		return;
	}
	empty.style.display = 'none';
	list.querySelectorAll('.an-sess-item').forEach(el => el.remove());

	withSub.forEach(s => {
		const scenario = s.meta?.scenario || '';
		const div = document.createElement('div');
		div.className = 'an-sess-item' + (AN.sessPath === s.path ? ' active' : '');
		div.title = s.path;
		div.innerHTML = `
			<div class="an-sess-item-name">${esc(s.name)}</div>
			${scenario ? `<div class="an-sess-item-meta">${esc(scenario)}</div>` : ''}
		`;
		div.onclick = () => anSelectSession(s.path, s.name);
		list.appendChild(div);
	});

	// Compteur en bas
	const counter = document.createElement('div');
	counter.style.cssText = 'padding:6px 12px;font-size:.65rem;color:var(--text3);border-top:1px solid var(--border);margin-top:4px';
	counter.textContent = `${withSub.length} / ${total} sessions annotées`;
	list.appendChild(counter);
}

function anScanSessions() {
	const btn = ge('an-sess-scan-btn');
	btn.classList.add('scanning');
	btn.disabled = true;
	scanAndShow().catch(() => {});
	setTimeout(() => { btn.classList.remove('scanning'); btn.disabled = false; }, 3000);
}

// ── Sélection de session ───────────────────────────────────────────

async function anSelectSession(path, name) {
	AN.sessPath = path;
	AN.data = null;

	// Mettre en évidence dans la liste
	document.querySelectorAll('.an-sess-item').forEach(el => {
		el.classList.toggle('active', el.title === path);
	});

	ge('an-sess-name').textContent = name || path.split('/').at(-1);
	ge('an-loading').style.display = 'flex';
	ge('an-inner').style.display = 'none';

	try {
		const res = await fetch(
			`/api/session/file?session_path=${encodeURIComponent(path)}&filename=episode_subtitle.json`
		);
		if (!res.ok) throw new Error(`HTTP ${res.status} — fichier introuvable`);
		AN.data = await res.json();
		ge('an-loading').style.display = 'none';
		ge('an-inner').style.display = 'block';
		anRender(AN.data);
	} catch (e) {
		ge('an-loading').textContent = `⚠ ${e.message}`;
		ge('an-loading').style.color = 'var(--red)';
	}
}

// ── Rendu des pistes ───────────────────────────────────────────────

function anRender(data) {
	const container = ge('an-tracks');
	container.innerHTML = '';

	// Durée totale = max de tous les ends
	let totalFrames = 0;
	for (const segs of Object.values(data)) {
		for (const seg of segs) {
			if (seg.end > totalFrames) totalFrames = seg.end;
		}
	}
	if (!totalFrames) { container.textContent = 'Aucune donnée.'; return; }

	// Construire la palette texte→couleur (globale sur toutes les pistes)
	const textSet = new Set();
	for (const segs of Object.values(data)) segs.forEach(s => textSet.add(s.text));
	const textList = [...textSet];
	const colorMap = {};
	textList.forEach((t, i) => { colorMap[t] = AN_PALETTE[i % AN_PALETTE.length]; });

	// Légende globale
	const legend = document.createElement('div');
	legend.className = 'an-legend';
	legend.innerHTML = textList.map(t => `
		<div class="an-legend-item">
			<div class="an-legend-dot" style="background:${colorMap[t]}"></div>
			<span style="color:var(--text2)">${esc(t)}</span>
		</div>`).join('');

	// Une piste par clé
	const tracks = document.createElement('div');
	for (const [key, segs] of Object.entries(data)) {
		const track = document.createElement('div');
		track.className = 'an-track';

		const label = document.createElement('div');
		label.className = 'an-track-label';
		label.textContent = key;

		const timeline = document.createElement('div');
		timeline.className = 'an-timeline';

		segs.forEach(seg => {
			const left = (seg.start / totalFrames) * 100;
			const width = ((seg.end - seg.start) / totalFrames) * 100;
			const color = colorMap[seg.text] || '#888';
			const el = document.createElement('div');
			el.className = 'an-segment';
			el.style.left = left + '%';
			el.style.width = width + '%';
			el.style.background = color;
			el.title = `${seg.text}\n[${seg.start} → ${seg.end}]`;
			// N'afficher le texte que si le segment est assez large (>5%)
			el.textContent = width > 5 ? seg.text : '';
			timeline.appendChild(el);
		});

		// Règle
		const ruler = document.createElement('div');
		ruler.className = 'an-ruler';
		const TICKS = 10;
		for (let i = 0; i <= TICKS; i++) {
			const pct = (i / TICKS) * 100;
			const frame = Math.round((i / TICKS) * totalFrames);
			const tick = document.createElement('div');
			tick.className = 'an-ruler-tick';
			tick.style.left = pct + '%';
			tick.textContent = frame;
			ruler.appendChild(tick);
		}

		track.appendChild(label);
		track.appendChild(timeline);
		track.appendChild(ruler);
		tracks.appendChild(track);
	}

	container.appendChild(tracks);
	container.appendChild(legend);
}
