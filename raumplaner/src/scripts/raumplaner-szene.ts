/**
 * Raumplaner – parametrische 3D-Szene (three.js).
 *
 * Alles wird aus Quadern erzeugt, nichts importiert: Breite und Tiefe sind
 * stufenlos, ein fertiges Modell müsste man dafür verzerren. Holz, Fronten,
 * Glas, Metall und Licht sind reine Material- bzw. Sichtbarkeits-Schalter.
 *
 * Achsen: x = Breite (0 .. breite), z = Tiefe (0 .. tiefe), y = Höhe.
 * Rückwand liegt bei z = 0, die linke Wand bei x = 0.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export type RaumTyp = 'hotelzimmer' | 'kueche' | 'wohnzimmer' | 'laden';
export type HolzArt = 'eiche-hell' | 'altholz' | 'nussbaum' | 'fichte-hell';
export type FrontArt = 'holz' | 'weiss' | 'anthrazit' | 'betonoptik';

export interface Konfiguration {
  raum: RaumTyp;
  breite: number;
  tiefe: number;
  holz: HolzArt;
  front: FrontArt;
  glas: boolean;
  metall: boolean;
  licht: boolean;
}

export const STANDARD: Konfiguration = {
  raum: 'kueche',
  breite: 4.4,
  tiefe: 4.2,
  holz: 'eiche-hell',
  front: 'weiss',
  glas: false,
  metall: false,
  licht: true,
};

/** Grundton je Holzart – die Maserung wird darüber gezeichnet. */
const HOLZ_TON: Record<HolzArt, { grund: number; maser: number }> = {
  'eiche-hell': { grund: 0xd3a05a, maser: 0xb5813d },
  altholz: { grund: 0x7d6a56, maser: 0x5d4c3c },
  nussbaum: { grund: 0x5a3a26, maser: 0x412617 },
  'fichte-hell': { grund: 0xe8d6ae, maser: 0xd0b989 },
};

const FRONT_TON: Record<FrontArt, number> = {
  holz: 0x000000, // wird durch die Holztextur ersetzt
  weiss: 0xf1efe9,
  anthrazit: 0x35383b,
  betonoptik: 0x9d9a93,
};

const WAND = 0xe9e3d9;
const ARBEITSPLATTE = 0x1c1c1c;

/* ---------------------------------------------------------------- Texturen */

function leinwand(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return [c, c.getContext('2d')!];
}

/** Holzmaserung: Grundton, Dielenfugen, ein paar Faserstreifen. */
function holzTextur(art: HolzArt, dielen = 6): THREE.CanvasTexture {
  const { grund, maser } = HOLZ_TON[art];
  const [c, g] = leinwand(512, 512);
  g.fillStyle = '#' + grund.toString(16).padStart(6, '0');
  g.fillRect(0, 0, 512, 512);

  const maserHex = '#' + maser.toString(16).padStart(6, '0');
  // Faserstreifen längs, leicht wellig
  g.strokeStyle = maserHex;
  g.lineWidth = 1;
  for (let i = 0; i < 160; i++) {
    const y = Math.random() * 512;
    g.globalAlpha = 0.04 + Math.random() * 0.1;
    g.beginPath();
    g.moveTo(0, y);
    for (let x = 0; x <= 512; x += 32) {
      g.lineTo(x, y + Math.sin((x + i * 40) / 90) * 3.5);
    }
    g.stroke();
  }
  // Astlöcher nur bei den rustikalen Sorten
  if (art === 'altholz' || art === 'fichte-hell') {
    for (let i = 0; i < 5; i++) {
      const x = Math.random() * 512;
      const y = Math.random() * 512;
      g.globalAlpha = 0.25;
      for (let r = 10; r > 0; r -= 2) {
        g.beginPath();
        g.ellipse(x, y, r * 1.6, r, 0, 0, Math.PI * 2);
        g.stroke();
      }
    }
  }
  // Dielenfugen quer
  g.globalAlpha = 0.5;
  g.strokeStyle = 'rgba(0,0,0,0.35)';
  g.lineWidth = 2;
  const schritt = 512 / dielen;
  for (let i = 1; i < dielen; i++) {
    g.beginPath();
    g.moveTo(0, i * schritt);
    g.lineTo(512, i * schritt);
    g.stroke();
  }
  g.globalAlpha = 1;

  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 8;
  return t;
}

/** Betonoptik: feine Sprenkel und ein paar Schlieren. */
function betonTextur(): THREE.CanvasTexture {
  const [c, g] = leinwand(256, 256);
  g.fillStyle = '#9d9a93';
  g.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 6000; i++) {
    g.fillStyle = `rgba(${Math.random() > 0.5 ? 255 : 0},${Math.random() > 0.5 ? 255 : 0},${
      Math.random() > 0.5 ? 255 : 0
    },0.035)`;
    g.fillRect(Math.random() * 256, Math.random() * 256, 2, 2);
  }
  for (let i = 0; i < 14; i++) {
    g.strokeStyle = 'rgba(255,255,255,0.05)';
    g.lineWidth = 2 + Math.random() * 8;
    g.beginPath();
    g.moveTo(Math.random() * 256, Math.random() * 256);
    g.lineTo(Math.random() * 256, Math.random() * 256);
    g.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

/** Textbeschriftung für die Maßlinien als Sprite. */
function beschriftung(text: string): THREE.Sprite {
  const [c, g] = leinwand(256, 64);
  g.clearRect(0, 0, 256, 64);
  g.fillStyle = '#4a463e';
  g.font = '600 34px "DM Mono", ui-monospace, monospace';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(text, 128, 34);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthTest: false }));
  s.scale.set(0.9, 0.225, 1);
  s.renderOrder = 10;
  return s;
}

/* -------------------------------------------------------------- Bausteine */

/** Quader mit Ursprung in der unteren vorderen linken Ecke statt im Zentrum. */
function quader(
  b: number,
  h: number,
  t: number,
  material: THREE.Material | THREE.Material[],
  x = 0,
  y = 0,
  z = 0,
): THREE.Mesh {
  const m = new THREE.Mesh(new THREE.BoxGeometry(b, h, t), material);
  m.position.set(x + b / 2, y + h / 2, z + t / 2);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}

/* -------------------------------------------------------------- Raumplaner */

export class Raumplaner {
  private renderer: THREE.WebGLRenderer;
  private szene = new THREE.Scene();
  private kamera: THREE.PerspectiveCamera;
  private steuerung: OrbitControls;
  private huelle: HTMLElement;

  /** Alles Konfigurationsabhängige hängt hier drunter und wird neu gebaut. */
  private inhalt = new THREE.Group();

  private texturen: THREE.Texture[] = [];
  private materialien: THREE.Material[] = [];
  private geometrien: THREE.BufferGeometry[] = [];

  private konfig: Konfiguration = { ...STANDARD };
  private laeuft = false;
  private beobachter?: ResizeObserver;

  constructor(huelle: HTMLElement) {
    this.huelle = huelle;

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      // Für den Screenshot, der mit der Anfrage mitgeht.
      preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    huelle.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.touchAction = 'none';
    this.renderer.domElement.style.display = 'block';
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';

    this.szene.background = null;

    this.kamera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
    this.steuerung = new OrbitControls(this.kamera, this.renderer.domElement);
    this.steuerung.enableDamping = true;
    this.steuerung.dampingFactor = 0.07;
    this.steuerung.enablePan = false;
    // Nicht unter den Boden und nicht von oben senkrecht herab.
    this.steuerung.minPolarAngle = 0.35;
    this.steuerung.maxPolarAngle = Math.PI / 2 - 0.06;
    // Nur der offene Quadrant vor dem Raum – sonst schaut man von hinten
    // durch die Wände.
    this.steuerung.minAzimuthAngle = -0.15;
    this.steuerung.maxAzimuthAngle = Math.PI / 2 + 0.15;

    this.licht();
    this.szene.add(this.inhalt);

    this.beobachter = new ResizeObserver(() => this.groesse());
    this.beobachter.observe(huelle);
    this.groesse();
  }

  /* --------------------------------------------------------- Beleuchtung */

  private licht(): void {
    this.szene.add(new THREE.HemisphereLight(0xffffff, 0xd8cfc0, 1.6));

    const sonne = new THREE.DirectionalLight(0xfff4e2, 2.4);
    sonne.position.set(-4, 6.5, 6);
    sonne.castShadow = true;
    sonne.shadow.mapSize.set(2048, 2048);
    sonne.shadow.camera.near = 1;
    sonne.shadow.camera.far = 24;
    const a = 7;
    sonne.shadow.camera.left = -a;
    sonne.shadow.camera.right = a;
    sonne.shadow.camera.top = a;
    sonne.shadow.camera.bottom = -a;
    sonne.shadow.bias = -0.0012;
    sonne.shadow.normalBias = 0.02;
    this.szene.add(sonne);
    this.szene.add(sonne.target);

    // Aufheller von vorne, damit die Fronten nicht absaufen.
    const fuell = new THREE.DirectionalLight(0xffffff, 0.5);
    fuell.position.set(6, 3, 8);
    this.szene.add(fuell);
  }

  /* ------------------------------------------------------------ Material */

  private merke<T extends THREE.Material>(m: T): T {
    this.materialien.push(m);
    return m;
  }

  private merkeTextur<T extends THREE.Texture>(t: T): T {
    this.texturen.push(t);
    return t;
  }

  private holzMaterial(wiederholung = 1, dielen = 6): THREE.MeshStandardMaterial {
    const t = this.merkeTextur(holzTextur(this.konfig.holz, dielen));
    t.repeat.set(wiederholung, wiederholung);
    return this.merke(new THREE.MeshStandardMaterial({ map: t, roughness: 0.72, metalness: 0.02 }));
  }

  /** Material der Möbelfronten – Holz erbt die gewählte Sorte. */
  private frontMaterial(): THREE.MeshStandardMaterial {
    const f = this.konfig.front;
    if (f === 'holz') return this.holzMaterial(1, 3);
    if (f === 'betonoptik') {
      const t = this.merkeTextur(betonTextur());
      return this.merke(new THREE.MeshStandardMaterial({ map: t, roughness: 0.88, metalness: 0.0 }));
    }
    return this.merke(
      new THREE.MeshStandardMaterial({
        color: FRONT_TON[f],
        roughness: f === 'weiss' ? 0.58 : 0.5,
        metalness: 0.03,
      }),
    );
  }

  private korpusMaterial(): THREE.MeshStandardMaterial {
    return this.merke(new THREE.MeshStandardMaterial({ color: 0xf4f2ec, roughness: 0.8 }));
  }

  private plattenMaterial(): THREE.MeshStandardMaterial {
    return this.merke(
      new THREE.MeshStandardMaterial({ color: ARBEITSPLATTE, roughness: 0.35, metalness: 0.12 }),
    );
  }

  private metallMaterial(): THREE.MeshStandardMaterial {
    return this.merke(
      new THREE.MeshStandardMaterial({ color: 0xb9bcc0, roughness: 0.28, metalness: 0.92 }),
    );
  }

  private glasMaterial(): THREE.MeshStandardMaterial {
    return this.merke(
      new THREE.MeshStandardMaterial({
        color: 0xcfe0e3,
        roughness: 0.06,
        metalness: 0.1,
        transparent: true,
        opacity: 0.32,
      }),
    );
  }

  /* --------------------------------------------------------------- Aufbau */

  /** Alte Szene wegräumen: three gibt GPU-Speicher nicht von selbst frei. */
  private leeren(): void {
    this.inhalt.clear();
    for (const m of this.materialien) m.dispose();
    for (const t of this.texturen) t.dispose();
    for (const g of this.geometrien) g.dispose();
    this.materialien = [];
    this.texturen = [];
    this.geometrien = [];
  }

  private bauen(): void {
    this.leeren();
    const { breite: B, tiefe: T } = this.konfig;

    this.raumhuelle(B, T);
    this.masslinien(B, T);

    switch (this.konfig.raum) {
      case 'kueche':
        this.kueche(B, T);
        break;
      case 'hotelzimmer':
        this.hotelzimmer(B, T);
        break;
      case 'wohnzimmer':
        this.wohnzimmer(B, T);
        break;
      case 'laden':
        this.laden(B, T);
        break;
    }

    // Jede Geometrie einsammeln, damit leeren() sie freigeben kann.
    this.inhalt.traverse((o) => {
      if ((o as THREE.Mesh).isMesh) this.geometrien.push((o as THREE.Mesh).geometry);
    });

    this.kameraSetzen(B, T);
  }

  /** Boden, zwei Wände, Sockelleiste, Fenster in der linken Wand. */
  private raumhuelle(B: number, T: number): void {
    const H = 2.7;

    const boden = this.holzMaterial(1, 5);
    (boden.map as THREE.Texture).repeat.set(Math.max(1, B / 1.15), Math.max(1, T / 1.15));
    const b = new THREE.Mesh(new THREE.PlaneGeometry(B, T), boden);
    b.rotation.x = -Math.PI / 2;
    b.position.set(B / 2, 0, T / 2);
    b.receiveShadow = true;
    this.inhalt.add(b);

    const wand = this.merke(new THREE.MeshStandardMaterial({ color: WAND, roughness: 0.95, side: THREE.DoubleSide }));

    const hinten = new THREE.Mesh(new THREE.PlaneGeometry(B, H), wand);
    hinten.position.set(B / 2, H / 2, 0);
    hinten.receiveShadow = true;
    this.inhalt.add(hinten);

    const links = new THREE.Mesh(new THREE.PlaneGeometry(T, H), wand);
    links.rotation.y = Math.PI / 2;
    links.position.set(0, H / 2, T / 2);
    links.receiveShadow = true;
    this.inhalt.add(links);

    // Fenster: helle Fläche knapp vor der linken Wand.
    const scheibe = this.merke(
      new THREE.MeshStandardMaterial({ color: 0xdfe8ec, emissive: 0xaecad6, emissiveIntensity: 0.55, roughness: 0.1 }),
    );
    const fb = Math.min(1.4, T * 0.32);
    const f = new THREE.Mesh(new THREE.PlaneGeometry(fb, 1.25), scheibe);
    f.rotation.y = Math.PI / 2;
    f.position.set(0.02, 1.45, T * 0.78);
    this.inhalt.add(f);

    const rahmen = this.merke(new THREE.MeshStandardMaterial({ color: 0xf6f4ef, roughness: 0.7 }));
    this.inhalt.add(quader(0.05, 1.33, fb + 0.08, rahmen, 0.0, 1.45 - 1.25 / 2 - 0.04, T * 0.78 - fb / 2 - 0.04));

    // Sockelleiste an beiden Wänden.
    const sockel = this.merke(new THREE.MeshStandardMaterial({ color: 0xf2efe8, roughness: 0.85 }));
    this.inhalt.add(quader(B, 0.08, 0.02, sockel, 0, 0, 0));
    this.inhalt.add(quader(0.02, 0.08, T, sockel, 0, 0, 0));
  }

  /** Maßlinien mit Beschriftung, wie in der bisherigen Zeichnung. */
  private masslinien(B: number, T: number): void {
    const farbe = 0x9a9488;
    const mat = this.merke(new THREE.LineBasicMaterial({ color: farbe, transparent: true, opacity: 0.85 }));
    const abstand = 0.42;

    const linie = (a: THREE.Vector3, b: THREE.Vector3) => {
      const g = new THREE.BufferGeometry().setFromPoints([a, b]);
      this.geometrien.push(g);
      this.inhalt.add(new THREE.Line(g, mat));
    };
    const anschlag = (p: THREE.Vector3, richtung: 'x' | 'z') => {
      const d = richtung === 'x' ? new THREE.Vector3(0, 0, 0.09) : new THREE.Vector3(0.09, 0, 0);
      linie(p.clone().sub(d), p.clone().add(d));
    };

    // Breite, vor dem Raum
    const zB = T + abstand;
    linie(new THREE.Vector3(0, 0.005, zB), new THREE.Vector3(B, 0.005, zB));
    anschlag(new THREE.Vector3(0, 0.005, zB), 'x');
    anschlag(new THREE.Vector3(B, 0.005, zB), 'x');
    const lb = beschriftung(`${B.toFixed(1).replace('.', ',')} m`);
    lb.position.set(B / 2, 0.16, zB + 0.16);
    this.inhalt.add(lb);

    // Tiefe, rechts daneben
    const xT = B + abstand;
    linie(new THREE.Vector3(xT, 0.005, 0), new THREE.Vector3(xT, 0.005, T));
    anschlag(new THREE.Vector3(xT, 0.005, 0), 'z');
    anschlag(new THREE.Vector3(xT, 0.005, T), 'z');
    const lt = beschriftung(`${T.toFixed(1).replace('.', ',')} m`);
    lt.position.set(xT + 0.18, 0.16, T / 2);
    this.inhalt.add(lt);
  }

  /** Warmer Streifen unter Hängeschränken bzw. Regalen. */
  private lichtleiste(x: number, y: number, z: number, laenge: number): void {
    if (!this.konfig.licht) return;
    const mat = this.merke(new THREE.MeshBasicMaterial({ color: 0xffe9c4 }));
    const streifen = new THREE.Mesh(new THREE.BoxGeometry(laenge, 0.015, 0.05), mat);
    streifen.position.set(x + laenge / 2, y, z);
    this.inhalt.add(streifen);

    // Ein paar Punktlichter statt RectAreaLight – das spart die
    // zusätzliche Uniform-Bibliothek und reicht optisch völlig.
    const anzahl = Math.max(2, Math.round(laenge / 0.8));
    for (let i = 0; i < anzahl; i++) {
      const l = new THREE.PointLight(0xffd9a0, 0.75, 1.9, 2.2);
      l.position.set(x + (laenge * (i + 0.5)) / anzahl, y - 0.03, z + 0.04);
      this.inhalt.add(l);
    }
  }

  /** Griffleiste, nur wenn Metall gewählt ist. */
  private griffe(x: number, y: number, z: number, laenge: number, stueck: number): void {
    if (!this.konfig.metall) return;
    const mat = this.metallMaterial();
    const g = new THREE.CylinderGeometry(0.008, 0.008, Math.min(0.26, (laenge / stueck) * 0.5), 8);
    this.geometrien.push(g);
    for (let i = 0; i < stueck; i++) {
      const m = new THREE.Mesh(g, mat);
      m.rotation.z = Math.PI / 2;
      m.position.set(x + (laenge * (i + 0.5)) / stueck, y, z);
      m.castShadow = true;
      this.inhalt.add(m);
    }
  }

  /* ------------------------------------------------------ Raumvarianten */

  private kueche(B: number, T: number): void {
    const korpus = this.korpusMaterial();
    const front = this.frontMaterial();
    const platte = this.plattenMaterial();

    const zeile = Math.min(B - 0.9, B * 0.78); // Unterschrankzeile an der Rückwand
    const x0 = B - zeile;
    const tiefeUS = 0.62;
    const hoeheUS = 0.88;

    this.inhalt.add(quader(zeile, hoeheUS, tiefeUS, korpus, x0, 0, 0));
    // Fronten leicht vorgesetzt, damit man sie vom Korpus unterscheidet.
    const felder = Math.max(3, Math.round(zeile / 0.62));
    for (let i = 0; i < felder; i++) {
      const fb = zeile / felder - 0.012;
      this.inhalt.add(quader(fb, hoeheUS - 0.09, 0.02, front, x0 + (zeile * i) / felder + 0.006, 0.06, tiefeUS));
    }
    this.griffe(x0, hoeheUS - 0.14, tiefeUS + 0.035, zeile, felder);
    this.inhalt.add(quader(zeile, 0.04, tiefeUS + 0.02, platte, x0, hoeheUS, 0));

    // Hängeschränke mit Rückwandstreifen aus Holz dazwischen
    const hoeheOS = 0.68;
    const yOS = 1.5;
    this.inhalt.add(quader(zeile, hoeheOS, 0.36, korpus, x0, yOS, 0));
    const osFelder = Math.max(2, Math.round(zeile / 0.8));
    for (let i = 0; i < osFelder; i++) {
      const fb = zeile / osFelder - 0.012;
      const mat = this.konfig.glas && i % 2 === 1 ? this.glasMaterial() : front;
      this.inhalt.add(quader(fb, hoeheOS - 0.03, 0.02, mat, x0 + (zeile * i) / osFelder + 0.006, yOS + 0.015, 0.36));
    }
    this.lichtleiste(x0, yOS - 0.03, 0.3, zeile);

    // Holzstreifen als Rückwand zwischen Arbeitsplatte und Hängeschrank
    const rueck = this.holzMaterial(1, 3);
    this.inhalt.add(quader(zeile, yOS - hoeheUS - 0.04, 0.02, rueck, x0, hoeheUS + 0.04, 0.0));

    // Hochschrank links neben der Zeile
    const hsB = Math.min(0.7, x0 - 0.05);
    if (hsB > 0.3) {
      this.inhalt.add(quader(hsB, 2.2, 0.64, korpus, Math.max(0.05, x0 - hsB - 0.04), 0, 0));
      this.inhalt.add(quader(hsB - 0.03, 2.14, 0.02, front, Math.max(0.05, x0 - hsB - 0.04) + 0.015, 0.03, 0.64));
    }

    // Kochfeld als dunkle Platte, Dunstabzug darüber
    this.inhalt.add(
      quader(0.6, 0.012, 0.5, this.merke(new THREE.MeshStandardMaterial({ color: 0x101010, roughness: 0.18 })),
        x0 + zeile * 0.55, hoeheUS + 0.04, 0.07),
    );

    // Insel, sofern der Raum tief genug ist
    if (T > 3.0) {
      const iB = Math.min(1.9, zeile * 0.62);
      const iT = 0.92;
      const ix = x0 + 0.1;
      const iz = Math.min(T - iT - 0.7, tiefeUS + 0.95);
      this.inhalt.add(quader(iB, hoeheUS, iT, korpus, ix, 0, iz));
      this.inhalt.add(quader(iB + 0.14, 0.05, iT + 0.14, platte, ix - 0.07, hoeheUS, iz - 0.07));
      const iFelder = Math.max(2, Math.round(iB / 0.6));
      for (let i = 0; i < iFelder; i++) {
        const fb = iB / iFelder - 0.012;
        this.inhalt.add(quader(fb, hoeheUS - 0.09, 0.02, front, ix + (iB * i) / iFelder + 0.006, 0.06, iz + iT));
      }
      this.griffe(ix, hoeheUS - 0.14, iz + iT + 0.035, iB, iFelder);
    }
  }

  private hotelzimmer(B: number, T: number): void {
    const holz = this.holzMaterial(1, 4);
    const front = this.frontMaterial();
    const stoff = this.merke(new THREE.MeshStandardMaterial({ color: 0xe6e1d5, roughness: 1 }));
    const decke = this.merke(new THREE.MeshStandardMaterial({ color: 0xcfd4cb, roughness: 1 }));

    // Kopfteil aus Holz an der Rückwand
    const bettB = Math.min(1.8, B * 0.5);
    const bx = Math.max(0.35, B * 0.22);
    this.inhalt.add(quader(bettB + 0.6, 1.1, 0.06, holz, bx - 0.3, 0.3, 0.01));

    // Bett
    this.inhalt.add(quader(bettB, 0.32, 2.0, holz, bx, 0, 0.1));
    this.inhalt.add(quader(bettB - 0.06, 0.22, 1.94, stoff, bx + 0.03, 0.32, 0.13));
    this.inhalt.add(quader(bettB - 0.06, 0.06, 1.1, decke, bx + 0.03, 0.54, 0.95));
    // Kissen
    for (let i = 0; i < 2; i++) {
      this.inhalt.add(quader(bettB / 2 - 0.12, 0.12, 0.42, stoff, bx + 0.06 + i * (bettB / 2), 0.52, 0.22));
    }

    // Nachttische mit Leseleisten
    for (const nx of [bx - 0.5, bx + bettB + 0.08]) {
      if (nx < 0.05 || nx + 0.42 > B) continue;
      this.inhalt.add(quader(0.42, 0.42, 0.4, holz, nx, 0, 0.12));
      this.inhalt.add(quader(0.38, 0.02, 0.02, front, nx + 0.02, 0.22, 0.52));
      this.lichtleiste(nx, 1.25, 0.06, 0.42);
    }

    // Schrank an der linken Wand
    const schrankT = Math.min(1.2, T * 0.3);
    this.inhalt.add(quader(0.62, 2.15, schrankT, this.korpusMaterial(), 0.06, 0, 0.85));
    this.inhalt.add(quader(0.02, 2.09, schrankT - 0.06, front, 0.68, 0.03, 0.88));

    // Schreibtisch rechts – nur wenn hinter dem Nachttisch wirklich Platz ist.
    const dx = bx + bettB + 0.08 + 0.42 + 0.25;
    const db = Math.min(1.2, B - 0.05 - dx);
    if (db >= 0.6) {
      this.inhalt.add(quader(db, 0.04, 0.5, holz, dx, 0.74, 0.05));
      this.inhalt.add(quader(0.05, 0.74, 0.46, holz, dx + 0.04, 0, 0.07));
      this.inhalt.add(quader(0.05, 0.74, 0.46, holz, dx + db - 0.09, 0, 0.07));
      this.lichtleiste(dx, 1.5, 0.12, db);
    }
  }

  private wohnzimmer(B: number, T: number): void {
    const holz = this.holzMaterial(1, 4);
    const front = this.frontMaterial();
    const stoff = this.merke(new THREE.MeshStandardMaterial({ color: 0x8d9384, roughness: 1 }));

    // Regalwand an der Rückwand
    const rB = Math.min(B - 0.6, B * 0.72);
    const rx = (B - rB) / 2;
    const rH = 2.1;
    this.inhalt.add(quader(rB, 0.04, 0.34, holz, rx, rH, 0));
    this.inhalt.add(quader(0.04, rH, 0.34, holz, rx, 0, 0));
    this.inhalt.add(quader(0.04, rH, 0.34, holz, rx + rB - 0.04, 0, 0));
    const boeden = 4;
    for (let i = 1; i <= boeden; i++) {
      const y = (rH / (boeden + 1)) * i;
      this.inhalt.add(quader(rB - 0.08, 0.03, 0.34, holz, rx + 0.04, y, 0));
      this.lichtleiste(rx + 0.06, y - 0.03, 0.28, rB - 0.12);
    }
    // Geschlossene Fächer unten, ggf. mit Glas
    const faecher = Math.max(2, Math.round(rB / 0.9));
    for (let i = 0; i < faecher; i++) {
      const fb = (rB - 0.08) / faecher - 0.012;
      const mat = this.konfig.glas && i % 2 === 0 ? this.glasMaterial() : front;
      this.inhalt.add(quader(fb, rH / (boeden + 1) - 0.06, 0.02, mat, rx + 0.04 + ((rB - 0.08) * i) / faecher, 0.03, 0.34));
    }
    this.griffe(rx + 0.04, 0.2, 0.375, rB - 0.08, faecher);

    // Sofa
    const sB = Math.min(2.2, B * 0.55);
    const sx = (B - sB) / 2;
    const sz = Math.min(T - 1.2, 2.1);
    this.inhalt.add(quader(sB, 0.34, 0.9, stoff, sx, 0.08, sz));
    this.inhalt.add(quader(sB, 0.5, 0.18, stoff, sx, 0.42, sz + 0.72));
    this.inhalt.add(quader(0.18, 0.28, 0.9, stoff, sx, 0.42, sz));
    this.inhalt.add(quader(0.18, 0.28, 0.9, stoff, sx + sB - 0.18, 0.42, sz));
    for (const fx of [sx + 0.08, sx + sB - 0.14]) {
      for (const fz of [sz + 0.08, sz + 0.76]) {
        this.inhalt.add(quader(0.06, 0.08, 0.06, holz, fx, 0, fz));
      }
    }

    // Couchtisch aus Holz mit Metallgestell
    const tB = Math.min(1.1, sB * 0.5);
    const tx = (B - tB) / 2;
    const tz = sz - 0.85;
    if (tz > 0.5) {
      this.inhalt.add(quader(tB, 0.05, 0.6, holz, tx, 0.36, tz));
      const beinMat = this.konfig.metall ? this.metallMaterial() : holz;
      for (const px of [tx + 0.05, tx + tB - 0.09]) {
        for (const pz of [tz + 0.05, tz + 0.51]) {
          this.inhalt.add(quader(0.04, 0.36, 0.04, beinMat, px, 0, pz));
        }
      }
    }
  }

  private laden(B: number, T: number): void {
    const holz = this.holzMaterial(1, 4);
    const front = this.frontMaterial();
    const platte = this.plattenMaterial();

    // Rückwandregal
    const rB = Math.min(B - 0.4, B * 0.85);
    const rx = (B - rB) / 2;
const rH = 2.3;
    const rT = 0.38;
    const korpus = this.korpusMaterial();
    // Offenes Regal: Rueckwand, Wangen, Deckel - kein geschlossener Kasten,
    // sonst verschwinden die Boeden darin.
    this.inhalt.add(quader(rB, rH, 0.03, korpus, rx, 0, 0));
    this.inhalt.add(quader(0.04, rH, rT, korpus, rx, 0, 0));
    this.inhalt.add(quader(0.04, rH, rT, korpus, rx + rB - 0.04, 0, 0));
    this.inhalt.add(quader(rB, 0.04, rT, korpus, rx, rH, 0));
    const felderR = Math.max(2, Math.round(rB / 1.1));
    for (let i = 1; i < felderR; i++) {
      this.inhalt.add(quader(0.03, rH, rT - 0.02, korpus, rx + (rB * i) / felderR, 0, 0.02));
    }
    for (let i = 1; i <= 5; i++) {
      const y = (rH / 6) * i;
      this.inhalt.add(quader(rB - 0.08, 0.035, rT - 0.02, holz, rx + 0.04, y, 0.02));
      this.lichtleiste(rx + 0.06, y - 0.035, rT - 0.06, rB - 0.12);
    }
    if (this.konfig.metall) {
      for (let i = 1; i < felderR; i++) {
        this.inhalt.add(quader(0.02, rH, 0.02, this.metallMaterial(), rx + (rB * i) / felderR + 0.005, 0, rT));
      }
    }

    // Theke: L-förmig, Korpus Holz, Platte dunkel
    const tB = Math.min(B * 0.6, 2.8);
    const tx = Math.max(0.3, (B - tB) / 2);
    const tz = Math.min(T - 1.4, 1.9);
    const tH = 1.06;
    this.inhalt.add(quader(tB, tH, 0.68, holz, tx, 0, tz));
    this.inhalt.add(quader(tB + 0.1, 0.06, 0.8, platte, tx - 0.05, tH, tz - 0.06));
    const felder = Math.max(3, Math.round(tB / 0.7));
    for (let i = 0; i < felder; i++) {
      const fb = tB / felder - 0.014;
      const mat = this.konfig.glas && i % 2 === 0 ? this.glasMaterial() : front;
      this.inhalt.add(quader(fb, tH - 0.14, 0.02, mat, tx + (tB * i) / felder + 0.007, 0.08, tz + 0.68));
    }
    this.lichtleiste(tx, tH - 0.16, tz + 0.72, tB);

    // Seitlicher Schenkel der Theke
    const sT = Math.min(1.2, T - tz - 0.6);
    if (sT > 0.5) {
      this.inhalt.add(quader(0.68, tH, sT, holz, tx, 0, tz + 0.68));
      this.inhalt.add(quader(0.8, 0.06, sT, platte, tx - 0.06, tH, tz + 0.68));
    }
  }

  /* --------------------------------------------------------------- Kamera */

  private kameraSetzen(B: number, T: number): void {
    const ziel = new THREE.Vector3(B / 2, 1.05, T / 2);
    this.steuerung.target.copy(ziel);
    // Abstand am größeren Raummaß ausrichten, damit nie etwas abgeschnitten wird.
    const d = Math.max(B, T) * 1.3 + 2.0;
    this.steuerung.minDistance = d * 0.55;
    this.steuerung.maxDistance = d * 1.7;
    if (!this.laeuft) {
      // Startblick: von schräg vorne rechts, ähnlich der bisherigen Isometrie.
      const winkel = Math.PI * 0.27;
      this.kamera.position.set(
        ziel.x + Math.sin(winkel) * d,
        ziel.y + d * 0.55,
        ziel.z + Math.cos(winkel) * d,
      );
    }
    this.steuerung.update();
  }

  private groesse(): void {
    const b = this.huelle.clientWidth || 1;
    const h = this.huelle.clientHeight || 1;
    this.renderer.setSize(b, h, false);
    this.kamera.aspect = b / h;
    this.kamera.updateProjectionMatrix();
  }

  /* ------------------------------------------------------------- Öffentlich */

  aktualisieren(teil: Partial<Konfiguration>): void {
    const vorher = this.konfig;
    this.konfig = { ...vorher, ...teil };
    this.bauen();
  }

  start(): void {
    if (this.laeuft) return;
    this.bauen();
    this.laeuft = true;
    const schleife = () => {
      if (!this.laeuft) return;
      this.steuerung.update();
      this.renderer.render(this.szene, this.kamera);
      requestAnimationFrame(schleife);
    };
    schleife();
  }

  /**
   * Einzelbild rendern – für Tests und den Screenshot der Anfrage.
   * JPEG, weil ein PNG dieser Ansicht rund 470 kB wiegt und damit das
   * Formular unnötig aufbläht.
   */
  bild(qualitaet = 0.85): string {
    this.renderer.render(this.szene, this.kamera);
    return this.renderer.domElement.toDataURL('image/jpeg', qualitaet);
  }

  aufraeumen(): void {
    this.laeuft = false;
    this.beobachter?.disconnect();
    this.leeren();
    this.steuerung.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

export function unterstuetzt(): boolean {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch {
    return false;
  }
}
