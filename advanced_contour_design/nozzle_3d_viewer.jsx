import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import * as THREE from 'three';

const computeNozzleGeometry = (params) => {
  const { R_t, CR, L_star, ER, L_percent } = params;
  const A_t = Math.PI * R_t ** 2;
  const R_c = R_t * Math.sqrt(CR);
  const R_e = R_t * Math.sqrt(ER);
  const V_c = L_star * A_t;
  const R_up = 1.5 * R_t, R_down = 0.382 * R_t;
  const theta_conv = 35 * Math.PI / 180;
  const dx_up = R_up * Math.sin(theta_conv);
  const dx_down = R_down * Math.sin(theta_conv);
  let dy_cone = Math.max((R_c - R_up * (1 - Math.cos(theta_conv))) - (R_t + R_down * (1 - Math.cos(theta_conv))), 0);
  const dx_cone = dy_cone > 0 ? dy_cone / Math.tan(theta_conv) : 0;
  const L_conv = dx_up + dx_cone + dx_down;
  const V_conv_approx = (Math.PI / 3) * L_conv * (R_c ** 2 + R_c * R_t + R_t ** 2);
  const L_c = Math.max((V_c - V_conv_approx * 0.7) / (Math.PI * R_c ** 2), 2 * R_c);
  const L_15_cone = (R_e - R_t) / Math.tan(15 * Math.PI / 180);
  const L_div = (L_percent / 100) * L_15_cone;
  return { R_c, R_e, L_c, L_conv, L_div, L_total: L_c + L_conv + L_div, throat_x: L_c + L_conv };
};

const linspace = (start, end, n) => Array.from({length: n}, (_, i) => start + (end - start) * i / (n - 1));

const cubicBezier = (P0, P1, P2, P3, t) => {
  const mt = 1 - t;
  return [mt**3*P0[0] + 3*mt**2*t*P1[0] + 3*mt*t**2*P2[0] + t**3*P3[0],
          mt**3*P0[1] + 3*mt**2*t*P1[1] + 3*mt*t**2*P2[1] + t**3*P3[1]];
};

const generate2DContour = (params, nPoints = 150) => {
  const geom = computeNozzleGeometry(params);
  const { R_t, theta_n, theta_e } = params;
  const { R_c, R_e, L_c, L_conv, L_div } = geom;
  const theta_n_rad = theta_n * Math.PI / 180, theta_e_rad = theta_e * Math.PI / 180;
  const theta_conv = 35 * Math.PI / 180;
  const R_up = 1.5 * R_t, R_down = 0.382 * R_t;
  const x = [], r = [];
  
  linspace(0, L_c, nPoints/5|0).forEach(xi => { x.push(xi); r.push(R_c); });
  linspace(Math.PI/2, Math.PI/2 - theta_conv, nPoints/6|0).slice(1).forEach(p => {
    x.push(L_c + R_up * Math.cos(p)); r.push(R_c - R_up + R_up * Math.sin(p));
  });
  
  const xThroat = L_c + L_conv;
  const xCS = x[x.length-1], rCS = r[r.length-1];
  const xCE = xThroat - R_down * Math.sin(theta_conv), rCE = R_t + R_down * (1 - Math.cos(theta_conv));
  linspace(0, 1, nPoints/8|0).slice(1).forEach(t => { x.push(xCS + t*(xCE-xCS)); r.push(rCS + t*(rCE-rCS)); });
  linspace(theta_conv, 0, nPoints/6|0).slice(1).forEach(p => {
    x.push(xThroat - R_down * Math.sin(p)); r.push(R_t + R_down * (1 - Math.cos(p)));
  });
  linspace(0, theta_n_rad, nPoints/8|0).slice(1).forEach(p => {
    x.push(xThroat + R_down * Math.sin(p)); r.push(R_t + R_down * (1 - Math.cos(p)));
  });
  
  const xN = x[x.length-1], rN = r[r.length-1], xE = xThroat + L_div;
  const P0 = [xN, rN], P3 = [xE, R_e], LB = xE - xN;
  const P1 = [P0[0] + 0.35*LB*Math.cos(theta_n_rad), P0[1] + 0.35*LB*Math.sin(theta_n_rad)];
  const P2 = [P3[0] - 0.35*LB*Math.cos(theta_e_rad), P3[1] - 0.35*LB*Math.sin(theta_e_rad)];
  linspace(0, 1, nPoints/3|0).slice(1).forEach(t => { const pt = cubicBezier(P0, P1, P2, P3, t); x.push(pt[0]); r.push(pt[1]); });
  
  return { x, r, geom };
};

const computeHelixAngle = (xi, geom, hp) => {
  const { L_c, L_conv, L_div, throat_x } = geom;
  const toRad = d => d * Math.PI / 180;
  const smooth = t => t * t * (3 - 2 * t);
  if (xi <= L_c) return toRad(hp.angle_chamber);
  if (xi <= throat_x) return toRad(hp.angle_chamber + smooth((xi - L_c) / L_conv) * (hp.angle_throat - hp.angle_chamber));
  return toRad(hp.angle_throat + smooth(Math.min((xi - throat_x) / L_div, 1)) * (hp.angle_exit - hp.angle_throat));
};

const computeHelixPath = (xArr, rArr, chIdx, nCh, geom, hp, cp) => {
  const theta = [chIdx * 2 * Math.PI / nCh];
  for (let i = 1; i < xArr.length; i++) {
    const dx = xArr[i] - xArr[i-1], rAvg = (rArr[i] + rArr[i-1]) / 2;
    const ha = computeHelixAngle(xArr[i], geom, hp);
    theta.push(theta[i-1] + (rAvg > 0 ? hp.direction * Math.tan(ha) * dx / rAvg : 0));
  }
  const rCh = rArr.map(ri => ri + cp.inner_wall_thickness + cp.channel_height_throat / 2);
  return { x: xArr, y: rCh.map((rc, i) => rc * Math.cos(theta[i])), z: rCh.map((rc, i) => rc * Math.sin(theta[i])), theta };
};

const computeChannelGeometry = (xi, ri, params, cp, geom) => {
  const { R_t } = params;
  const circ = 2 * Math.PI * R_t;
  const nCh = cp.n_channels || Math.max((circ / (cp.channel_width_throat + cp.rib_width_throat))|0, 12);
  let wr = cp.channel_width_ratio_chamber;
  if (xi > geom.L_c && xi <= geom.throat_x) wr = cp.channel_width_ratio_chamber + (xi - geom.L_c) / geom.L_conv * (1 - cp.channel_width_ratio_chamber);
  else if (xi > geom.throat_x) wr = 1 + Math.min((xi - geom.throat_x) / geom.L_div, 1) * (cp.channel_width_ratio_exit - 1);
  const chH = cp.channel_height_throat * Math.min(Math.max(0.8 + 0.2 * ri / R_t, 0.7), 1.3);
  return { nChannels: nCh, channelHeight: chH };
};

const ThreeJSViewer = ({ nozzleParams, coolingParams, helixParams, viewMode, showCrossSection, crossSectionAngle }) => {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const ctrlRef = useRef({ drag: false, px: 0, py: 0, ry: 0.5, rx: 0.3, zoom: 1 });
  const meshesRef = useRef([]);
  const frameRef = useRef(null);

  const contour = useMemo(() => generate2DContour(nozzleParams, 100), [nozzleParams]);

  const buildScene = useCallback((scene) => {
    meshesRef.current.forEach(m => scene.remove(m));
    meshesRef.current = [];
    const { x, r, geom } = contour;
    const { inner_wall_thickness: iwt, outer_wall_thickness: owt, channel_height_throat: cht } = coolingParams;
    const scale = 1000, nTh = 64;
    const maxTh = (showCrossSection ? 360 - crossSectionAngle : 360) * Math.PI / 180;
    const thStep = maxTh / nTh;

    const mkSurf = (rProf, col, op) => {
      const geo = new THREE.BufferGeometry(), verts = [], idx = [];
      for (let i = 0; i < x.length; i++) {
        const rad = rProf[i] * scale;
        for (let j = 0; j <= nTh; j++) {
          const th = j * thStep;
          verts.push(x[i] * scale, rad * Math.cos(th), rad * Math.sin(th));
        }
        if (i > 0) {
          const c = i * (nTh + 1), p = (i - 1) * (nTh + 1);
          for (let j = 0; j < nTh; j++) idx.push(p+j, c+j, c+j+1, p+j, c+j+1, p+j+1);
        }
      }
      geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
      geo.setIndex(idx);
      geo.computeVertexNormals();
      return new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ color: col, opacity: op, transparent: op < 1, side: THREE.DoubleSide, shininess: 80 }));
    };

    const rOut = r.map((ri, i) => ri + iwt + computeChannelGeometry(x[i], ri, nozzleParams, coolingParams, geom).channelHeight + owt);
    const rChB = r.map(ri => ri + iwt);
    const rChT = r.map((ri, i) => ri + iwt + computeChannelGeometry(x[i], ri, nozzleParams, coolingParams, geom).channelHeight);

    if (viewMode === 'full' || viewMode === 'inner') { const m = mkSurf(r, 0xE74C3C, viewMode === 'full' ? 0.6 : 1); scene.add(m); meshesRef.current.push(m); }
    if (viewMode === 'full' || viewMode === 'outer') { const m = mkSurf(rOut, 0x3498DB, viewMode === 'full' ? 0.4 : 0.7); scene.add(m); meshesRef.current.push(m); }
    if (viewMode === 'channels' || viewMode === 'full') {
      const mB = mkSurf(rChB, 0x2ECC71, viewMode === 'full' ? 0.3 : 0.6);
      const mT = mkSurf(rChT, 0x9B59B6, viewMode === 'full' ? 0.3 : 0.6);
      scene.add(mB); scene.add(mT); meshesRef.current.push(mB, mT);
    }

    const nCh = computeChannelGeometry(geom.throat_x, nozzleParams.R_t, nozzleParams, coolingParams, geom).nChannels;
    const nShow = Math.min(nCh, viewMode === 'channels' ? nCh : 8);
    
    if (viewMode === 'channels' || viewMode === 'full') {
      const mat = new THREE.LineBasicMaterial({ color: helixParams.enabled ? 0x00FFFF : 0xF39C12, linewidth: 2 });
      for (let c = 0; c < nShow; c++) {
        const ci = (c * nCh / nShow)|0;
        let px, py, pz;
        if (helixParams.enabled) {
          const p = computeHelixPath(x, r, ci, nCh, geom, helixParams, coolingParams);
          px = p.x; py = p.y; pz = p.z;
        } else {
          const thC = ci * 2 * Math.PI / nCh;
          const rM = r.map((ri, i) => (rChB[i] + rChT[i]) / 2);
          px = x; py = rM.map(rm => rm * Math.cos(thC)); pz = rM.map(rm => rm * Math.sin(thC));
        }
        const pts = px.map((pxi, i) => new THREE.Vector3(pxi * scale, py[i] * scale, pz[i] * scale));
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat);
        scene.add(line); meshesRef.current.push(line);
      }
    }

    const tRing = new THREE.Mesh(new THREE.RingGeometry(nozzleParams.R_t * scale - 2, nozzleParams.R_t * scale + 2, 64),
      new THREE.MeshBasicMaterial({ color: 0xFFFF00, side: THREE.DoubleSide }));
    tRing.rotation.y = Math.PI / 2; tRing.position.x = geom.throat_x * scale;
    scene.add(tRing); meshesRef.current.push(tRing);
    const ax = new THREE.AxesHelper(geom.L_total * scale * 0.25);
    scene.add(ax); meshesRef.current.push(ax);
  }, [contour, coolingParams, nozzleParams, helixParams, viewMode, showCrossSection, crossSectionAngle]);

  useEffect(() => {
    if (!containerRef.current) return;
    const w = containerRef.current.clientWidth, h = containerRef.current.clientHeight;
    const scene = new THREE.Scene(); scene.background = new THREE.Color(0x1a1a2e); sceneRef.current = scene;
    const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 10000); camera.position.set(400, 200, 300); cameraRef.current = camera;
    const renderer = new THREE.WebGLRenderer({ antialias: true }); renderer.setSize(w, h); renderer.setPixelRatio(devicePixelRatio);
    containerRef.current.appendChild(renderer.domElement); rendererRef.current = renderer;
    scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const l1 = new THREE.DirectionalLight(0xffffff, 0.8); l1.position.set(200, 200, 200); scene.add(l1);
    const l2 = new THREE.DirectionalLight(0x4ECDC4, 0.4); l2.position.set(-200, -100, -200); scene.add(l2);
    buildScene(scene);
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      const c = ctrlRef.current, cx = contour.geom.L_total * 500, d = 400 / c.zoom;
      camera.position.set(cx + d * Math.sin(c.ry) * Math.cos(c.rx), d * Math.sin(c.rx), d * Math.cos(c.ry) * Math.cos(c.rx));
      camera.lookAt(cx, 0, 0); renderer.render(scene, camera);
    };
    animate();
    const onResize = () => { if (!containerRef.current) return; const nw = containerRef.current.clientWidth, nh = containerRef.current.clientHeight;
      camera.aspect = nw / nh; camera.updateProjectionMatrix(); renderer.setSize(nw, nh); };
    window.addEventListener('resize', onResize);
    return () => { window.removeEventListener('resize', onResize); cancelAnimationFrame(frameRef.current); renderer.dispose();
      if (containerRef.current?.contains(renderer.domElement)) containerRef.current.removeChild(renderer.domElement); };
  }, []);

  useEffect(() => { if (sceneRef.current) buildScene(sceneRef.current); }, [buildScene]);

  const onMD = e => { ctrlRef.current.drag = true; ctrlRef.current.px = e.clientX; ctrlRef.current.py = e.clientY; };
  const onMM = e => { if (!ctrlRef.current.drag) return; const c = ctrlRef.current;
    c.ry += (e.clientX - c.px) * 0.01; c.rx = Math.max(-1.47, Math.min(1.47, c.rx + (e.clientY - c.py) * 0.01));
    c.px = e.clientX; c.py = e.clientY; };
  const onMU = () => { ctrlRef.current.drag = false; };
  const onWh = e => { e.preventDefault(); ctrlRef.current.zoom = Math.max(0.2, Math.min(5, ctrlRef.current.zoom * (1 - e.deltaY * 0.001))); };

  return <div ref={containerRef} style={{ width: '100%', height: '100%', cursor: 'grab' }}
    onMouseDown={onMD} onMouseMove={onMM} onMouseUp={onMU} onMouseLeave={onMU} onWheel={onWh} />;
};

export default function Nozzle3DViewer() {
  const [np, setNp] = useState({ R_t: 0.015, CR: 5, L_star: 1, ER: 10, theta_n: 32, theta_e: 8, L_percent: 80 });
  const [cp, setCp] = useState({ inner_wall_thickness: 0.001, outer_wall_thickness: 0.002, channel_width_throat: 0.003,
    channel_height_throat: 0.004, rib_width_throat: 0.002, channel_width_ratio_chamber: 1.5, channel_width_ratio_exit: 1.8, n_channels: null });
  const [hp, setHp] = useState({ enabled: false, angle_chamber: 10, angle_throat: 20, angle_exit: 15, direction: 1 });
  const [vm, setVm] = useState('full');
  const [cs, setCs] = useState(false);
  const [csa, setCsa] = useState(90);

  const geom = useMemo(() => computeNozzleGeometry(np), [np]);
  const nCh = useMemo(() => { const c = 2 * Math.PI * np.R_t; return cp.n_channels || Math.max((c / (cp.channel_width_throat + cp.rib_width_throat))|0, 12); }, [np.R_t, cp]);

  const ss = { width: '100%', height: 4, borderRadius: 2, background: 'linear-gradient(90deg, #4ECDC4, #45B7D1)', appearance: 'none', cursor: 'pointer' };
  const uNp = (k, v) => setNp(p => ({ ...p, [k]: +v }));
  const uCp = (k, v) => setCp(p => ({ ...p, [k]: +v }));
  const uHp = (k, v) => setHp(p => ({ ...p, [k]: typeof v === 'boolean' ? v : +v }));

  return (
    <div style={{ display: 'flex', height: '100vh', background: 'linear-gradient(135deg, #0f0f1a, #1a1a2e)', fontFamily: 'JetBrains Mono, monospace', color: '#fff' }}>
      <div style={{ width: 320, padding: 14, overflowY: 'auto', background: 'rgba(0,0,0,0.3)', borderRight: '1px solid rgba(255,255,255,0.1)' }}>
        <h1 style={{ fontSize: 14, margin: '0 0 4px', color: '#4ECDC4' }}>🚀 3D NOZZLE + COOLING</h1>
        <p style={{ fontSize: 9, color: '#888', margin: '0 0 10px' }}>Straight & Helical Channels</p>

        <div style={{ marginBottom: 10, padding: 8, background: 'rgba(255,255,255,0.05)', borderRadius: 6 }}>
          <h3 style={{ fontSize: 9, color: '#4ECDC4', margin: '0 0 6px' }}>VIEW</h3>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            {['full', 'inner', 'outer', 'channels'].map(m => (
              <button key={m} onClick={() => setVm(m)} style={{ padding: '4px 8px', fontSize: 8, border: 'none', borderRadius: 3, cursor: 'pointer',
                background: vm === m ? '#4ECDC4' : 'rgba(255,255,255,0.1)', color: vm === m ? '#000' : '#fff' }}>{m.toUpperCase()}</button>
            ))}
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 5, marginTop: 6, fontSize: 9, color: '#aaa' }}>
            <input type="checkbox" checked={cs} onChange={e => setCs(e.target.checked)} /> Cross-Section
          </label>
          {cs && <><span style={{ fontSize: 8, color: '#888' }}>Angle: {csa}°</span>
            <input type="range" min={30} max={180} value={csa} onChange={e => setCsa(+e.target.value)} style={ss} /></>}
        </div>

        <div style={{ marginBottom: 10, padding: 8, background: 'rgba(255,255,255,0.05)', borderRadius: 6 }}>
          <h3 style={{ fontSize: 9, color: '#FF6B6B', margin: '0 0 8px' }}>NOZZLE</h3>
          {[['R_t', 'Throat R', 'mm', 5, 50, 1, 1000], ['CR', 'CR', '', 2, 10, 0.5, 1], ['ER', 'ER', '', 3, 50, 1, 1],
            ['L_star', 'L*', 'm', 0.3, 2, 0.1, 1], ['L_percent', 'Bell%', '%', 60, 100, 5, 1]].map(([k, l, u, mn, mx, st, sc]) => (
            <div key={k} style={{ marginBottom: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 8 }}>
                <span style={{ color: '#aaa' }}>{l}</span><span style={{ color: '#45B7D1' }}>{(np[k] * sc).toFixed(1)} {u}</span>
              </div>
              <input type="range" min={mn} max={mx} step={st} value={np[k] * sc} onChange={e => uNp(k, e.target.value / sc)} style={ss} />
            </div>
          ))}
        </div>

        <div style={{ marginBottom: 10, padding: 8, background: 'rgba(255,255,255,0.05)', borderRadius: 6 }}>
          <h3 style={{ fontSize: 9, color: '#2ECC71', margin: '0 0 8px' }}>COOLING</h3>
          {[['inner_wall_thickness', 'Inner Wall', 'mm', 0.5, 3, 0.1, 1000], ['channel_width_throat', 'Width', 'mm', 1, 8, 0.5, 1000],
            ['channel_height_throat', 'Height', 'mm', 1, 10, 0.5, 1000], ['rib_width_throat', 'Rib', 'mm', 0.5, 5, 0.25, 1000],
            ['outer_wall_thickness', 'Outer Wall', 'mm', 1, 5, 0.5, 1000]].map(([k, l, u, mn, mx, st, sc]) => (
            <div key={k} style={{ marginBottom: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 8 }}>
                <span style={{ color: '#aaa' }}>{l}</span><span style={{ color: '#2ECC71' }}>{(cp[k] * sc).toFixed(1)} {u}</span>
              </div>
              <input type="range" min={mn} max={mx} step={st} value={cp[k] * sc} onChange={e => uCp(k, e.target.value / sc)} style={ss} />
            </div>
          ))}
        </div>

        <div style={{ marginBottom: 10, padding: 8, background: hp.enabled ? 'rgba(0,255,255,0.1)' : 'rgba(255,255,255,0.05)', borderRadius: 6,
          border: hp.enabled ? '1px solid rgba(0,255,255,0.3)' : 'none' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <h3 style={{ fontSize: 9, color: '#00FFFF', margin: 0 }}>HELICAL</h3>
            <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 9 }}>
              <input type="checkbox" checked={hp.enabled} onChange={e => uHp('enabled', e.target.checked)} />
              <span style={{ color: hp.enabled ? '#00FFFF' : '#888' }}>{hp.enabled ? 'ON' : 'OFF'}</span>
            </label>
          </div>
          {hp.enabled && <>
            {[['angle_chamber', 'Chamber', 0, 45], ['angle_throat', 'Throat', 0, 45], ['angle_exit', 'Exit', 0, 45]].map(([k, l, mn, mx]) => (
              <div key={k} style={{ marginBottom: 6 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 8 }}>
                  <span style={{ color: '#aaa' }}>{l}</span><span style={{ color: '#00FFFF' }}>{hp[k]}°</span>
                </div>
                <input type="range" min={mn} max={mx} step={1} value={hp[k]} onChange={e => uHp(k, e.target.value)}
                  style={{ ...ss, background: 'linear-gradient(90deg, #006666, #00FFFF)' }} />
              </div>
            ))}
            <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
              {[[1, '↻ Right'], [-1, '↺ Left']].map(([d, lbl]) => (
                <button key={d} onClick={() => uHp('direction', d)} style={{ flex: 1, padding: 5, fontSize: 8, border: 'none', borderRadius: 3,
                  cursor: 'pointer', background: hp.direction === d ? '#00FFFF' : 'rgba(255,255,255,0.1)', color: hp.direction === d ? '#000' : '#fff' }}>{lbl}</button>
              ))}
            </div>
          </>}
        </div>

        <div style={{ padding: 8, background: 'rgba(78,205,196,0.1)', borderRadius: 6, border: '1px solid rgba(78,205,196,0.3)' }}>
          <h3 style={{ fontSize: 9, color: '#4ECDC4', margin: '0 0 6px' }}>GEOMETRY</h3>
          {[['Length', (geom.L_total * 1000).toFixed(1), 'mm'], ['Chamber R', (geom.R_c * 1000).toFixed(1), 'mm'],
            ['Exit R', (geom.R_e * 1000).toFixed(1), 'mm'], ['Channels', nCh, ''], ['Type', hp.enabled ? 'Helical' : 'Straight', '']].map(([l, v, u]) => (
            <div key={l} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 8, marginBottom: 2 }}>
              <span style={{ color: '#888' }}>{l}</span><span style={{ color: '#fff' }}>{v} {u}</span>
            </div>
          ))}
        </div>

        <div style={{ marginTop: 10, padding: 8, background: 'rgba(255,255,255,0.03)', borderRadius: 6 }}>
          <h3 style={{ fontSize: 9, color: '#888', margin: '0 0 4px' }}>LEGEND</h3>
          {[['Inner Wall', '#E74C3C'], ['Ch. Bottom', '#2ECC71'], ['Ch. Top', '#9B59B6'], ['Outer Wall', '#3498DB'],
            ['Throat', '#FFFF00'], [hp.enabled ? 'Helix Path' : 'Channel', hp.enabled ? '#00FFFF' : '#F39C12']].map(([l, c]) => (
            <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 8, marginBottom: 2 }}>
              <div style={{ width: 8, height: 8, background: c, borderRadius: 2 }} /><span style={{ color: '#aaa' }}>{l}</span>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 8, fontSize: 7, color: '#555', textAlign: 'center' }}>🖱️ Drag rotate • Scroll zoom</div>
      </div>

      <div style={{ flex: 1, position: 'relative' }}>
        <ThreeJSViewer nozzleParams={np} coolingParams={cp} helixParams={hp} viewMode={vm} showCrossSection={cs} crossSectionAngle={csa} />
        <div style={{ position: 'absolute', top: 10, right: 10, padding: 8, background: 'rgba(0,0,0,0.7)', borderRadius: 5, fontSize: 9 }}>
          <div style={{ color: '#4ECDC4', fontWeight: 'bold' }}>{vm.toUpperCase()}</div>
          <div style={{ color: '#888' }}>{cs ? `Section: ${csa}°` : 'Full'}</div>
          <div style={{ color: hp.enabled ? '#00FFFF' : '#F39C12', marginTop: 3 }}>
            {hp.enabled ? `Helix ${hp.angle_throat}° @ throat` : 'Straight'}
          </div>
        </div>
      </div>
    </div>
  );
}
