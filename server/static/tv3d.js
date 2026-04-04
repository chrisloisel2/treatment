// ══════════════════════════════════════════════════════════════════
// VUE 3D — Three.js
// ══════════════════════════════════════════════════════════════════

const TV3D = {
	renderer: null,
	scene: null,
	camera: null,
	animId: null,
	isDragging: false,
	isRightDrag: false,
	lastMouse: { x: 0, y: 0 },
	spherical: { theta: 0.5, phi: 1.0, r: 3.0 },
	target: { x: 0, y: 0, z: 0 },
	trajLines: {},
	trajPts: {},
	cursors: {},
	mode: 'both',
	axesMode: 'xyz',
	initialized: false,
};

// Dérivé de SLOT_COLORS (index.html) — même palette head/left/right
const TV3D_COLORS = typeof SLOT_COLORS !== 'undefined'
	? Object.fromEntries(Object.entries(SLOT_COLORS).map(([k, v]) => [k, parseInt(v.slice(1), 16)]))
	: { head: 0x5b7df5, left: 0x22d386, right: 0xf59e0b };

function tv3dInitIfNeeded() {
	if (!TV3D.initialized) tv3dInit();
}

function tv3dInit() {
	const canvas = ge('tv3d-canvas');
	const wrap = ge('tv-3d-block');
	if (!canvas || !wrap) return;

	TV3D.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
	TV3D.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
	TV3D.renderer.setClearColor(0x13132a, 1);

	TV3D.scene = new THREE.Scene();

	const w = wrap.clientWidth || 600;
	const h = wrap.clientHeight || 300;
	TV3D.camera = new THREE.PerspectiveCamera(50, w / h, 0.001, 1000);

	TV3D.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
	const dir = new THREE.DirectionalLight(0xffffff, 0.8);
	dir.position.set(2, 4, 3);
	TV3D.scene.add(dir);

	const axH = new THREE.AxesHelper(0.3);
	axH.name = 'axesHelper';
	TV3D.scene.add(axH);

	const grid = new THREE.GridHelper(2, 20, 0x252545, 0x252545);
	grid.name = 'grid';
	TV3D.scene.add(grid);

	// Camera par défaut (avant que les données arrivent)
	TV3D.spherical = { theta: 0.6, phi: 0.9, r: 2.5 };
	TV3D.target = { x: 0, y: 0, z: 0 };
	tv3dApplyCamera();

	tv3dAttachControls(canvas, wrap);
	TV3D.initialized = true;

	// Lancer le rendu
	tv3dAnimate();

	// Resize initial + observer
	requestAnimationFrame(tv3dResize);
	new ResizeObserver(() => tv3dResize()).observe(wrap);
}

function tv3dResize() {
	const wrap = ge('tv-3d-block');
	if (!wrap || !TV3D.renderer || !TV3D.camera) return;
	const w = wrap.clientWidth, h = wrap.clientHeight;
	if (w < 10 || h < 10) return;
	TV3D.renderer.setSize(w, h, false);
	TV3D.camera.aspect = w / h;
	TV3D.camera.updateProjectionMatrix();
}

function tv3dResetCamera() {
	if (!TV3D.initialized) return;
	TV3D.spherical = { theta: 0.6, phi: 0.9, r: 2.5 };
	TV3D.target = { x: 0, y: 0, z: 0 };
	tv3dApplyCamera();
}

function tv3dApplyCamera() {
	if (!TV3D.camera) return;
	const { theta, phi, r } = TV3D.spherical;
	const { x: tx, y: ty, z: tz } = TV3D.target;
	TV3D.camera.position.set(
		tx + r * Math.sin(phi) * Math.sin(theta),
		ty + r * Math.cos(phi),
		tz + r * Math.sin(phi) * Math.cos(theta)
	);
	TV3D.camera.lookAt(tx, ty, tz);
}

function tv3dAttachControls(canvas) {
	canvas.addEventListener('mousedown', e => {
		TV3D.isDragging = true;
		TV3D.isRightDrag = e.button === 2;
		TV3D.lastMouse = { x: e.clientX, y: e.clientY };
	});
	window.addEventListener('mousemove', e => {
		if (!TV3D.isDragging || !TV3D.camera) return;
		const dx = e.clientX - TV3D.lastMouse.x;
		const dy = e.clientY - TV3D.lastMouse.y;
		TV3D.lastMouse = { x: e.clientX, y: e.clientY };
		if (TV3D.isRightDrag) {
			const right = new THREE.Vector3(), up = new THREE.Vector3();
			TV3D.camera.matrix.extractBasis(right, up, new THREE.Vector3());
			const f = TV3D.spherical.r * 0.0015;
			TV3D.target.x -= right.x * dx * f;
			TV3D.target.y -= up.y * dy * f;
			TV3D.target.z -= right.z * dx * f;
		} else {
			TV3D.spherical.theta -= dx * 0.005;
			TV3D.spherical.phi = Math.max(0.05, Math.min(Math.PI - 0.05, TV3D.spherical.phi + dy * 0.005));
		}
		tv3dApplyCamera();
	});
	window.addEventListener('mouseup', () => { TV3D.isDragging = false; });
	canvas.addEventListener('wheel', e => {
		e.preventDefault();
		TV3D.spherical.r = Math.max(0.01, TV3D.spherical.r * (1 + e.deltaY * 0.001));
		tv3dApplyCamera();
	}, { passive: false });
	canvas.addEventListener('contextmenu', e => e.preventDefault());
}

function tv3dAnimate() {
	requestAnimationFrame(tv3dAnimate);
	if (TV3D.renderer && TV3D.scene && TV3D.camera) {
		TV3D.renderer.render(TV3D.scene, TV3D.camera);
	}
}

// ── Construit les trajectoires à partir de TV.data.tracker ─────────
function tv3dBuild() {
	if (!TV3D.initialized) return;

	const raw = TV.data?.tracker;
	if (!raw?.t_ms?.length) {
		ge('tv3d-empty-msg').style.display = '';
		ge('tv3d-empty-msg').textContent = 'Pas de données tracker';
		return;
	}

	// Nettoyer anciens objets
	for (const role of ['head', 'left', 'right']) {
		if (TV3D.trajLines[role]) { TV3D.scene.remove(TV3D.trajLines[role]); TV3D.trajLines[role] = null; }
		if (TV3D.cursors[role]) { TV3D.scene.remove(TV3D.cursors[role]); TV3D.cursors[role] = null; }
		TV3D.trajPts[role] = null;
	}

	// Quels rôles ont des XYZ ?
	const roles = ['head', 'left', 'right'].filter(r =>
		Array.isArray(raw[`${r}_x`]) && raw[`${r}_x`].length > 0
	);
	if (!roles.length) {
		ge('tv3d-empty-msg').style.display = '';
		ge('tv3d-empty-msg').textContent = `Colonnes XYZ manquantes (clés: ${Object.keys(raw).join(', ')})`;
		return;
	}
	ge('tv3d-empty-msg').style.display = 'none';

	// Centroïde global
	let cx = 0, cy = 0, cz = 0, n = 0;
	for (const r of roles) {
		const xs = raw[`${r}_x`], ys = raw[`${r}_y`], zs = raw[`${r}_z`];
		for (let i = 0; i < xs.length; i++) { cx += xs[i]; cy += ys[i]; cz += zs[i]; n++; }
	}
	if (n) { cx /= n; cy /= n; cz /= n; }
	TV3D.target = { x: cx, y: cy, z: cz };

	// Rayon max (pour grille + caméra)
	let R = 0.05;
	for (const r of roles) {
		const xs = raw[`${r}_x`], ys = raw[`${r}_y`], zs = raw[`${r}_z`];
		for (let i = 0; i < xs.length; i++) {
			const d = Math.sqrt((xs[i] - cx) ** 2 + (ys[i] - cy) ** 2 + (zs[i] - cz) ** 2);
			if (d > R) R = d;
		}
	}
	TV3D.spherical.r = R * 3.5;

	// Grille centrée, sous les données
	TV3D.scene.children.filter(c => c.name === 'grid').forEach(g => TV3D.scene.remove(g));
	const grid = new THREE.GridHelper(R * 4, 20, 0x252545, 0x1a1a36);
	grid.position.set(cx, cy - R, cz);
	grid.name = 'grid';
	TV3D.scene.add(grid);

	// Axes helper centré sur les données
	TV3D.scene.children.filter(c => c.name === 'axesHelper').forEach(a => TV3D.scene.remove(a));
	const axH = new THREE.AxesHelper(R * 0.5);
	axH.position.set(cx, cy, cz);
	axH.name = 'axesHelper';
	TV3D.scene.add(axH);

	// Trajectoires + sphères
	for (const role of roles) {
		const [ax1, ax2, ax3] = tv3dGetAxes(role, raw);
		if (!ax1?.length) continue;

		// Downsample (max 3000 pts par ligne pour perf)
		const stride = Math.max(1, Math.ceil(ax1.length / 3000));
		const pts = [];
		for (let i = 0; i < ax1.length; i += stride) pts.push(new THREE.Vector3(ax1[i], ax2[i], ax3[i]));
		if (pts.at(-1) !== undefined) {
			const last = new THREE.Vector3(ax1.at(-1), ax2.at(-1), ax3.at(-1));
			if (!pts.at(-1).equals(last)) pts.push(last);
		}

		// Stocker les points pour mise à jour progressive du drawRange
		TV3D.trajPts[role] = pts;

		// Créer la géométrie avec tous les points mais drawRange = 0 au départ
		const positions = new Float32Array(pts.length * 3);
		for (let i = 0; i < pts.length; i++) {
			positions[i * 3]     = pts[i].x;
			positions[i * 3 + 1] = pts[i].y;
			positions[i * 3 + 2] = pts[i].z;
		}
		const geo = new THREE.BufferGeometry();
		geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
		geo.setDrawRange(0, 0);
		const mat = new THREE.LineBasicMaterial({ color: TV3D_COLORS[role], linewidth: 2, opacity: 0.8, transparent: true });
		const line = new THREE.Line(geo, mat);
		TV3D.scene.add(line);
		TV3D.trajLines[role] = line;

		// Sphère curseur (taille proportionnelle, minimum visible)
		const sr = Math.max(0.008, R * 0.04);
		const sphere = new THREE.Mesh(
			new THREE.SphereGeometry(sr, 10, 10),
			new THREE.MeshPhongMaterial({ color: TV3D_COLORS[role], emissive: TV3D_COLORS[role], emissiveIntensity: 0.6 })
		);
		sphere.position.set(ax1[0], ax2[0], ax3[0]);
		TV3D.scene.add(sphere);
		TV3D.cursors[role] = sphere;
	}

	tv3dApplyCamera();
	tv3dApplyMode();
	tv3dUpdateLegend();
}

// Retourne les 3 axes selon axesMode
function tv3dGetAxes(role, data) {
	const xs = data[`${role}_x`], ys = data[`${role}_y`], zs = data[`${role}_z`];
	if (!xs) return [null, null, null];
	return TV3D.axesMode === 'xzy' ? [xs, zs, ys] : [xs, ys, zs];
}

// Déplace les sphères à la position t_ms (recherche binaire)
function tv3dUpdateCursor(t_ms) {
	if (!TV3D.initialized) return;
	const data = TV.data?.tracker;
	if (!data?.t_ms?.length) return;
	const ts = data.t_ms;

	// Recherche binaire
	let lo = 0, hi = ts.length - 1;
	while (lo < hi) { const mid = (lo + hi) >> 1; if (ts[mid] < t_ms) lo = mid + 1; else hi = mid; }
	const best = (lo > 0 && Math.abs(ts[lo - 1] - t_ms) < Math.abs(ts[lo] - t_ms)) ? lo - 1 : lo;

	const parts = [];
	for (const role of ['head', 'left', 'right']) {
		const sphere = TV3D.cursors[role];
		const line = TV3D.trajLines[role];
		const pts = TV3D.trajPts[role];
		const [ax1, ax2, ax3] = tv3dGetAxes(role, data);
		if (!sphere || !ax1 || best >= ax1.length) continue;
		sphere.position.set(ax1[best], ax2[best], ax3[best]);
		parts.push(`${role}: (${ax1[best].toFixed(3)}, ${ax2[best].toFixed(3)}, ${ax3[best].toFixed(3)})`);

		// Mettre à jour le drawRange : afficher seulement les points déjà parcourus
		if (line && pts) {
			const stride = Math.max(1, Math.ceil(ax1.length / pts.length));
			const ptIdx = Math.min(Math.floor(best / stride) + 1, pts.length);
			line.geometry.setDrawRange(0, ptIdx);
		}
	}
	const coordEl = ge('tv3d-coords');
	if (coordEl && parts.length) { coordEl.textContent = parts.join('  |  '); coordEl.style.display = ''; }
}

function tv3dApplyMode() {
	const mode = TV3D.mode;
	for (const role of ['head', 'left', 'right']) {
		if (TV3D.trajLines[role]) TV3D.trajLines[role].visible = (mode !== 'curseur');
		if (TV3D.cursors[role]) TV3D.cursors[role].visible = (mode !== 'trajectoire');
	}
}

function tv3dSetMode(val) { TV3D.mode = val; tv3dApplyMode(); }
function tv3dSetAxes(val) { TV3D.axesMode = val; tv3dBuild(); tv3dUpdateCursor(TV.t_ms); }

function tv3dUpdateLegend() {
	const leg = ge('tv3d-legend');
	if (!leg) return;
	const roles = ['head', 'left', 'right'].filter(r => TV3D.cursors[r]);
	leg.innerHTML = roles.map(role => {
		const hex = '#' + TV3D_COLORS[role].toString(16).padStart(6, '0');
		return `<div style="display:flex;align-items:center;gap:5px;background:#00000099;padding:2px 8px;border-radius:99px">
      <div style="width:10px;height:10px;border-radius:50%;background:${hex}"></div>
      <span style="color:#dde0f5">${role}</span>
    </div>`;
	}).join('');
}
