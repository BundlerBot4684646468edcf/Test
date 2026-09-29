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
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';

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
  /** Maßlinien zeigen – im vorab gerenderten Standbild stören sie. */
  masse?: boolean;
  /** Deko: Brett, Schale, Pflanze. Kostet Geometrie, bringt viel Bild. */
  deko?: boolean;
  /**
   * 'plan' – Puppenhausansicht von außen, zum Planen.
   * 'innen' – Augenhöhe im Raum, wie eine Innenaufnahme. Dafür braucht es
   * die rechte Wand und eine Decke, sonst schaut man ins Leere.
   * 'rundum' – wie 'innen', aber zusätzlich mit der vierten Wand: beim
   * 360-Grad-Blick dreht man sich auch dorthin um.
   */
  blick?: 'plan' | 'innen' | 'rundum';
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

const WAND = 0xefe9df;
const ARBEITSPLATTE = 0x1c1c1c;

/* ---------------------------------------------------------------- Texturen */

function leinwand(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return [c, c.getContext('2d')!];
}

/**
 * Holzmaserung. Jede Diele bekommt einen eigenen Helligkeitsversatz, eine
 * eigene Maserungsphase und eine Fase an der Fuge – ohne diesen Kontrast
 * zwischen den Dielen sieht der Boden aus wie bedrucktes Plastik.
 */
function holzLeinwand(art: HolzArt, dielen = 6): HTMLCanvasElement {
  const { grund, maser } = HOLZ_TON[art];
  const [c, g] = leinwand(1024, 1024);
  const N = 1024;
  const hoehe = N / dielen;

  const kanal = (farbe: number, i: number) => (farbe >> (16 - i * 8)) & 0xff;
  const mische = (farbe: number, f: number) =>
    `rgb(${Math.min(255, Math.max(0, Math.round(kanal(farbe, 0) * f)))},` +
    `${Math.min(255, Math.max(0, Math.round(kanal(farbe, 1) * f)))},` +
    `${Math.min(255, Math.max(0, Math.round(kanal(farbe, 2) * f)))})`;

  const rustikal = art === 'altholz' || art === 'fichte-hell';

  for (let d = 0; d < dielen; d++) {
    const y0 = d * hoehe;
    // Helligkeit von Diele zu Diele – das Wichtigste am ganzen Bild.
    const versatz = 0.8 + Math.random() * 0.4;
    g.fillStyle = mische(grund, versatz);
    g.fillRect(0, y0, N, hoehe);

    // Längsmaserung, an der Diele ausgerichtet
    const phase = Math.random() * 100;
    const dichte = rustikal ? 55 : 38;
    for (let i = 0; i < dichte; i++) {
      const y = y0 + Math.random() * hoehe;
      const staerke = 0.06 + Math.random() * 0.16;
      g.strokeStyle = mische(maser, 0.75 + Math.random() * 0.5);
      g.globalAlpha = staerke;
      g.lineWidth = 1.2 + Math.random() * 4.5;
      g.beginPath();
      g.moveTo(0, y);
      for (let x = 0; x <= N; x += 24) {
        g.lineTo(x, y + Math.sin((x + phase * 60) / 160) * (hoehe * 0.06));
      }
      g.stroke();
    }

    // Kathedralfigur: ein paar lange, spitz zulaufende Bögen
    g.globalAlpha = rustikal ? 0.16 : 0.09;
    for (let i = 0; i < 3; i++) {
      const mx = Math.random() * N;
      const my = y0 + hoehe * (0.3 + Math.random() * 0.4);
      g.strokeStyle = mische(maser, 0.85);
      for (let r = 0; r < 7; r++) {
        g.lineWidth = 1.1;
        g.beginPath();
        g.ellipse(mx, my, 80 + r * 26, hoehe * 0.1 + r * 3, 0, 0, Math.PI * 2);
        g.stroke();
      }
    }

    if (rustikal) {
      for (let i = 0; i < 2; i++) {
        const x = Math.random() * N;
        const y = y0 + hoehe * (0.25 + Math.random() * 0.5);
        g.globalAlpha = 0.5;
        g.strokeStyle = mische(maser, 0.55);
        for (let r = 12; r > 0; r -= 2.5) {
          g.beginPath();
          g.ellipse(x, y, r * 1.7, r * 0.8, 0, 0, Math.PI * 2);
          g.stroke();
        }
      }
    }

    // Fuge: dunkler Schatten oben, heller Grat darunter
    g.globalAlpha = 1;
    g.fillStyle = 'rgba(0,0,0,0.45)';
    g.fillRect(0, y0, N, 2.5);
    g.fillStyle = 'rgba(255,255,255,0.14)';
    g.fillRect(0, y0 + 2.5, N, 1.5);
  }

  g.globalAlpha = 1;
  return c;
}

/** Aus einer Vorlage eine Textur bauen; Datenkarten dürfen nicht sRGB sein. */
function ausLeinwand(c: HTMLCanvasElement, daten = false): THREE.CanvasTexture {
  const t = new THREE.CanvasTexture(c);
  if (!daten) t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 8;
  return t;
}

/**
 * Normalmap aus der Helligkeit einer Vorlage (Sobel).
 * Ohne sie sieht Holz aus wie bedrucktes Papier – mit ihr bekommt es Poren.
 */
function normalAusHoehe(quelle: HTMLCanvasElement, staerke = 3.5): HTMLCanvasElement {
  const b = quelle.width;
  const h = quelle.height;
  const q = quelle.getContext('2d')!.getImageData(0, 0, b, h).data;
  const [ziel, zg] = leinwand(b, h);
  const aus = zg.createImageData(b, h);

  const hell = (x: number, y: number) => {
    const i = (((y + h) % h) * b + ((x + b) % b)) * 4;
    return (q[i] * 0.299 + q[i + 1] * 0.587 + q[i + 2] * 0.114) / 255;
  };

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < b; x++) {
      const dx =
        hell(x - 1, y - 1) + 2 * hell(x - 1, y) + hell(x - 1, y + 1) -
        (hell(x + 1, y - 1) + 2 * hell(x + 1, y) + hell(x + 1, y + 1));
      const dy =
        hell(x - 1, y - 1) + 2 * hell(x, y - 1) + hell(x + 1, y - 1) -
        (hell(x - 1, y + 1) + 2 * hell(x, y + 1) + hell(x + 1, y + 1));
      // Normieren, damit der Vektor wirklich Länge 1 hat.
      const nx = dx * staerke;
      const ny = dy * staerke;
      const nz = 1;
      const l = Math.hypot(nx, ny, nz);
      const i = (y * b + x) * 4;
      aus.data[i] = ((nx / l) * 0.5 + 0.5) * 255;
      aus.data[i + 1] = ((ny / l) * 0.5 + 0.5) * 255;
      aus.data[i + 2] = ((nz / l) * 0.5 + 0.5) * 255;
      aus.data[i + 3] = 255;
    }
  }
  zg.putImageData(aus, 0, 0);
  return ziel;
}

/** Rauheitsmap: dunkle Maserung ist offenporiger, also matter. */
function rauheitAusHoehe(quelle: HTMLCanvasElement, min = 0.45, max = 0.9): HTMLCanvasElement {
  const b = quelle.width;
  const h = quelle.height;
  const q = quelle.getContext('2d')!.getImageData(0, 0, b, h);
  const d = q.data;
  for (let i = 0; i < d.length; i += 4) {
    const hell = (d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114) / 255;
    const r = (max - (max - min) * hell) * 255;
    d[i] = d[i + 1] = d[i + 2] = r;
    d[i + 3] = 255;
  }
  const [ziel, zg] = leinwand(b, h);
  zg.putImageData(q, 0, 0);
  return ziel;
}

/**
 * Die Vorlagen sind teuer (Sobel über 262k Pixel) und hängen nur an der
 * Holzart – einmal zeichnen reicht, auch wenn am Regler gezogen wird.
 */
const vorlagen = new Map<string, { farbe: HTMLCanvasElement; normal: HTMLCanvasElement; rauheit: HTMLCanvasElement }>();

/**
 * Fotografierte Hölzer einsetzen. Die Normal- und Rauheitskarte werden aus
 * dem Foto selbst abgeleitet, es braucht also nur das eine Bild je Sorte:
 * flach ausgeleuchtet, nahtlos kachelbar, ohne Schatten und Glanzlichter.
 *
 *   await holzFotos({ 'eiche-hell': '/holz/eiche-hell.jpg', … });
 *
 * Schlägt ein Bild fehl, bleibt die gezeichnete Maserung stehen – die Seite
 * darf an einem fehlenden Foto nicht scheitern.
 */
export async function holzFotos(
  quellen: Partial<Record<HolzArt, string>>,
  kante = 1024,
): Promise<HolzArt[]> {
  const geladen: HolzArt[] = [];
  await Promise.all(
    (Object.keys(quellen) as HolzArt[]).map(async (art) => {
      const url = quellen[art];
      if (!url) return;
      try {
        const bild = new Image();
        bild.crossOrigin = 'anonymous';
        bild.decoding = 'async';
        await new Promise<void>((fertig, fehler) => {
          bild.onload = () => fertig();
          bild.onerror = () => fehler(new Error(url));
          bild.src = url;
        });
        const [c, g] = leinwand(kante, kante);
        g.drawImage(bild, 0, 0, kante, kante);
        // Gleiche Ableitung wie bei der gezeichneten Vorlage.
        const vorlage = { farbe: c, normal: normalAusHoehe(c, 2.6), rauheit: rauheitAusHoehe(c) };
        for (const dielen of [3, 4, 5, 6]) vorlagen.set(`${art}|${dielen}`, vorlage);
        geladen.push(art);
      } catch {
        // Bewusst still: die gezeichnete Vorlage bleibt gültig.
      }
    }),
  );
  return geladen;
}

function holzVorlage(art: HolzArt, dielen: number) {
  const schluessel = `${art}|${dielen}`;
  let v = vorlagen.get(schluessel);
  if (!v) {
    const farbe = holzLeinwand(art, dielen);
    v = { farbe, normal: normalAusHoehe(farbe), rauheit: rauheitAusHoehe(farbe) };
    vorlagen.set(schluessel, v);
  }
  return v;
}

/** Betonoptik: feine Sprenkel und ein paar Schlieren. */
function betonLeinwand(): HTMLCanvasElement {
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
  return c;
}

/**
 * Bergpanorama hinter der Verglasung. Gezeichnet, nicht fotografiert –
 * aber ein Fenster, hinter dem etwas liegt, macht aus dem Kasten einen Raum.
 */
function landschaftLeinwand(): HTMLCanvasElement {
  const [c, g] = leinwand(1600, 900);

  const himmel = g.createLinearGradient(0, 0, 0, 640);
  himmel.addColorStop(0, '#b9d4e6');
  himmel.addColorStop(0.55, '#d8e6ee');
  himmel.addColorStop(1, '#eceee9');
  g.fillStyle = himmel;
  g.fillRect(0, 0, 1600, 900);

  // Drei Bergketten, nach hinten heller – das ergibt die Tiefe.
  const kette = (grund: number, farbe: string, zacken: number, hoehe: number) => {
    g.fillStyle = farbe;
    g.beginPath();
    g.moveTo(0, 900);
    g.lineTo(0, grund);
    let x = 0;
    let auf = true;
    while (x < 1600) {
      const schritt = 1600 / zacken / 2;
      x += schritt;
      g.lineTo(x, auf ? grund - hoehe * (0.6 + Math.random() * 0.6) : grund + hoehe * 0.15);
      auf = !auf;
    }
    g.lineTo(1600, grund);
    g.lineTo(1600, 900);
    g.closePath();
    g.fill();
  };
  kette(560, '#9fb4c4', 6, 200);
  kette(625, '#7d95a3', 8, 155);
  kette(678, '#63796b', 11, 115);

  // Nadelwald am Hang
  g.fillStyle = '#5d7757';
  for (let i = 0; i < 420; i++) {
    const x = Math.random() * 1600;
    const y = 660 + Math.random() * 120;
    const h = 14 + Math.random() * 22;
    g.beginPath();
    g.moveTo(x, y);
    g.lineTo(x - h * 0.28, y + h);
    g.lineTo(x + h * 0.28, y + h);
    g.closePath();
    g.fill();
  }

  // Wiese
  g.fillStyle = '#7d9550';
  g.fillRect(0, 760, 1600, 140);

  // Etwas Dunst über dem Tal, aber nicht so viel, dass die Ketten verschwinden
  g.fillStyle = 'rgba(255,255,255,0.14)';
  g.fillRect(0, 590, 1600, 90);

  return c;
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

  private umgebung?: THREE.Texture;
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
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    huelle.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.touchAction = 'none';
    this.renderer.domElement.style.display = 'block';
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';

    this.szene.background = null;
    // Prozedurale Umgebung für Spiegelungen: kein Download, aber Glas,
    // Metall und die Arbeitsplatte bekommen etwas zum Spiegeln.
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.umgebung = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.szene.environment = this.umgebung;
    this.szene.environmentIntensity = 0.35;
    pmrem.dispose();

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
    // Die Umgebung liefert schon Grundhelligkeit, deshalb weniger Himmelslicht.
    this.szene.add(new THREE.HemisphereLight(0xffffff, 0xd8cfc0, 0.8));

    const sonne = new THREE.DirectionalLight(0xfff4e2, 2.2);
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
    const fuell = new THREE.DirectionalLight(0xffffff, 0.3);
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
    const v = holzVorlage(this.konfig.holz, dielen);
    const map = this.merkeTextur(ausLeinwand(v.farbe));
    const normalMap = this.merkeTextur(ausLeinwand(v.normal, true));
    const roughnessMap = this.merkeTextur(ausLeinwand(v.rauheit, true));
    for (const t of [map, normalMap, roughnessMap]) t.repeat.set(wiederholung, wiederholung);
    return this.merke(
      new THREE.MeshStandardMaterial({
        map,
        normalMap,
        normalScale: new THREE.Vector2(0.55, 0.55),
        roughnessMap,
        roughness: 1,
        metalness: 0,
        envMapIntensity: 0.25,
      }),
    );
  }

  /** Material der Möbelfronten – Holz erbt die gewählte Sorte. */
  private frontMaterial(): THREE.MeshStandardMaterial {
    const f = this.konfig.front;
    if (f === 'holz') return this.holzMaterial(1, 3);
    if (f === 'betonoptik') {
      const c = betonLeinwand();
      const map = this.merkeTextur(ausLeinwand(c));
      const normalMap = this.merkeTextur(ausLeinwand(normalAusHoehe(c, 1.1), true));
      return this.merke(
        new THREE.MeshStandardMaterial({
          map,
          normalMap,
          normalScale: new THREE.Vector2(0.3, 0.3),
          roughness: 0.85,
          metalness: 0,
          envMapIntensity: 0.25,
        }),
      );
    }
    return this.merke(
      new THREE.MeshPhysicalMaterial({
        color: FRONT_TON[f],
        roughness: f === 'weiss' ? 0.45 : 0.38,
        metalness: 0,
        clearcoat: 0.35,
        clearcoatRoughness: 0.35,
        envMapIntensity: 0.8,
      }),
    );
  }

  /**
   * Korpus im Ton der Fronten. Ein weißer Korpus hinter dunklen Fronten
   * zeichnet helle Linien um jede Tür – das sieht nach Modell aus, nicht
   * nach Möbel.
   */
  private korpusMaterial(): THREE.MeshStandardMaterial {
    const nach: Record<FrontArt, number> = {
      weiss: 0xf1efe9,
      anthrazit: 0x2b2e30,
      betonoptik: 0x8b8882,
      holz: 0x6d5642,
    };
    return this.merke(
      new THREE.MeshStandardMaterial({
        color: nach[this.konfig.front],
        roughness: 0.8,
        envMapIntensity: 0.3,
      }),
    );
  }

  private plattenMaterial(): THREE.MeshPhysicalMaterial {
    return this.merke(
      new THREE.MeshPhysicalMaterial({
        color: ARBEITSPLATTE,
        roughness: 0.22,
        metalness: 0.1,
        clearcoat: 0.6,
        clearcoatRoughness: 0.2,
        envMapIntensity: 2.6,
      }),
    );
  }

  private metallMaterial(): THREE.MeshStandardMaterial {
    return this.merke(
      new THREE.MeshStandardMaterial({
        color: 0xc6c9cd,
        roughness: 0.22,
        metalness: 1,
        envMapIntensity: 3.2,
      }),
    );
  }

  private glasMaterial(): THREE.MeshPhysicalMaterial {
    // Kein transmission: das kostet auf dem Handy einen eigenen Renderdurchgang.
    // Spiegelung aus der Umgebung reicht optisch völlig.
    return this.merke(
      new THREE.MeshPhysicalMaterial({
        color: 0xdbe7ea,
        roughness: 0.03,
        metalness: 0,
        transparent: true,
        opacity: 0.28,
        clearcoat: 1,
        clearcoatRoughness: 0.02,
        envMapIntensity: 4,
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
    if (this.konfig.masse !== false) this.masslinien(B, T);

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

    const wand = this.merke(
      new THREE.MeshStandardMaterial({ color: WAND, roughness: 0.95, envMapIntensity: 0.3, side: THREE.DoubleSide }),
    );
    const sockelVorn = this.merke(
      new THREE.MeshStandardMaterial({ color: 0xf2efe8, roughness: 0.85, envMapIntensity: 0.3 }),
    );

    // Das Wohnzimmer bekommt statt der halben Rückwand eine Verglasung.
    const verglast = this.konfig.raum === 'wohnzimmer';
    const massivAb = verglast ? B * 0.52 : 0;
    const massivB = B - massivAb;
    const hinten = new THREE.Mesh(new THREE.PlaneGeometry(massivB, H), wand);
    hinten.position.set(massivAb + massivB / 2, H / 2, 0);
    hinten.receiveShadow = true;
    this.inhalt.add(hinten);
    if (verglast) this.verglasung(massivAb, H, B, T);

    // Beim verglasten Wohnzimmer bleibt links nur der hintere Rest stehen.
    const linksAb = verglast ? T * 0.55 : 0;
    const linksT = T - linksAb;
    const links = new THREE.Mesh(new THREE.PlaneGeometry(linksT, H), wand);
    links.rotation.y = Math.PI / 2;
    links.position.set(0, H / 2, linksAb + linksT / 2);
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

    const geschlossen = this.konfig.blick === 'innen' || this.konfig.blick === 'rundum';
    if (geschlossen) {
      const rechts = new THREE.Mesh(new THREE.PlaneGeometry(T, H), wand);
      rechts.rotation.y = -Math.PI / 2;
      rechts.position.set(B, H / 2, T / 2);
      rechts.receiveShadow = true;
      this.inhalt.add(rechts);

      if (this.konfig.blick === 'rundum') {
        // Vierte Wand: beim Rundumblick steht man sonst vor einem Loch.
        const vorne = new THREE.Mesh(new THREE.PlaneGeometry(B, H), wand);
        vorne.rotation.y = Math.PI;
        vorne.position.set(B / 2, H / 2, T);
        vorne.receiveShadow = true;
        this.inhalt.add(vorne);
        this.inhalt.add(quader(B, 0.08, 0.02, sockelVorn, 0, 0, T - 0.02));
      }

      if (this.konfig.raum === 'wohnzimmer') {
        this.holzdecke(B, T, H);
      } else {
        const deckeMat = this.merke(
          new THREE.MeshStandardMaterial({ color: 0xf7f4ee, roughness: 1, envMapIntensity: 0.2 }),
        );
        const decke = new THREE.Mesh(new THREE.PlaneGeometry(B, T), deckeMat);
        decke.rotation.x = Math.PI / 2;
        decke.position.set(B / 2, H, T / 2);
        this.inhalt.add(decke);
      }
    }

    // Sockelleiste an beiden Wänden.
    const sockel = sockelVorn;
    this.inhalt.add(quader(B, 0.08, 0.02, sockel, 0, 0, 0));
    this.inhalt.add(quader(0.02, 0.08, T, sockel, 0, 0, 0));
  }

  /**
   * Raumhohe Verglasung in der Rückwand, mit Bergpanorama dahinter.
   * Die Landschaft hängt weit hinter der Scheibe, damit die Fensterpfosten
   * Tiefe bekommen statt wie aufgeklebt zu wirken.
   */
  private verglasung(bisX: number, H: number, B: number, T: number): void {
    const landschaft = this.merkeTextur(ausLeinwand(landschaftLeinwand()));
    const panorama = this.merke(
      new THREE.MeshBasicMaterial({ map: landschaft, toneMapped: false }),
    );
    const weite = Math.max(B, T) * 2.6;
    const fern = new THREE.Mesh(new THREE.PlaneGeometry(weite, weite * 0.56), panorama);
    fern.position.set(B / 2, H * 0.6, -weite * 0.26);
    this.inhalt.add(fern);

    // Tageslicht von draußen herein
    const tag = new THREE.DirectionalLight(0xf2f7ff, 1.5);
    tag.position.set(bisX * 0.4, 3.2, -6);
    tag.target.position.set(B * 0.6, 0.8, T * 0.5);
    this.inhalt.add(tag);
    this.inhalt.add(tag.target);

    const scheibe = this.merke(
      new THREE.MeshPhysicalMaterial({
        color: 0xdce7ea,
        roughness: 0.02,
        metalness: 0,
        transparent: true,
        opacity: 0.12,
        envMapIntensity: 2.4,
      }),
    );
    const g = new THREE.Mesh(new THREE.PlaneGeometry(bisX, H), scheibe);
    g.position.set(bisX / 2, H / 2, 0.01);
    this.inhalt.add(g);

    // Zweite Panoramafläche und Verglasung über Eck, wie in der Referenz.
    const fern2 = new THREE.Mesh(new THREE.PlaneGeometry(weite, weite * 0.56), panorama);
    fern2.rotation.y = Math.PI / 2;
    fern2.position.set(-weite * 0.26, H * 0.6, T / 2);
    this.inhalt.add(fern2);

    const linksTiefe = T * 0.55;
    const gl = new THREE.Mesh(new THREE.PlaneGeometry(linksTiefe, H), scheibe);
    gl.rotation.y = Math.PI / 2;
    gl.position.set(0.01, H / 2, linksTiefe / 2);
    this.inhalt.add(gl);

    // Schwarze Rahmenprofile, wie im Bild
    const profil = this.merke(
      new THREE.MeshStandardMaterial({ color: 0x24262a, roughness: 0.42, metalness: 0.5 }),
    );
    const felder = Math.max(2, Math.round(bisX / 1.5));
    for (let i = 0; i <= felder; i++) {
      const x = (bisX * i) / felder;
      this.inhalt.add(quader(0.055, H, 0.09, profil, Math.min(x, bisX - 0.055), 0, -0.04));
    }
    this.inhalt.add(quader(bisX, 0.07, 0.09, profil, 0, H - 0.07, -0.04));
    this.inhalt.add(quader(bisX, 0.05, 0.09, profil, 0, 0, -0.04));

    const lFelder = Math.max(2, Math.round(linksTiefe / 1.5));
    for (let i = 0; i <= lFelder; i++) {
      const z = (linksTiefe * i) / lFelder;
      this.inhalt.add(quader(0.09, H, 0.055, profil, -0.04, 0, Math.min(z, linksTiefe - 0.055)));
    }
    this.inhalt.add(quader(0.09, 0.07, linksTiefe, profil, -0.04, H - 0.07, 0));
    this.inhalt.add(quader(0.09, 0.05, linksTiefe, profil, -0.04, 0, 0));
  }

  /** Holzdecke mit eingelassenen Lichtlinien – das Stück der Tischlerei. */
  private holzdecke(B: number, T: number, H: number): void {
    const holz = this.holzMaterial(1, 9);
    const map = holz.map as THREE.Texture;
    // Schmale Leisten quer, deutlich feiner als beim Boden.
    map.repeat.set(Math.max(1, B / 2.4), Math.max(1, T / 0.85));
    for (const k of [holz.normalMap, holz.roughnessMap]) if (k) k.repeat.copy(map.repeat);

    const decke = new THREE.Mesh(new THREE.PlaneGeometry(B, T), holz);
    decke.rotation.x = Math.PI / 2;
    decke.position.set(B / 2, H, T / 2);
    decke.receiveShadow = true;
    this.inhalt.add(decke);

    if (!this.konfig.licht) return;
    const leuchte = this.merke(new THREE.MeshBasicMaterial({ color: 0xffe6b8, toneMapped: false }));
    const bahnen = 3;
    for (let i = 0; i < bahnen; i++) {
      const z = (T * (i + 0.7)) / (bahnen + 0.4);
      const laenge = B * (i % 2 ? 0.42 : 0.62);
      const x = i % 2 ? B * 0.3 : B * 0.12;
      const streifen = new THREE.Mesh(new THREE.BoxGeometry(laenge, 0.012, 0.045), leuchte);
      streifen.position.set(x + laenge / 2, H - 0.008, z);
      this.inhalt.add(streifen);
      const l = new THREE.PointLight(0xffdca8, 0.7, 5.0, 2);
      l.position.set(x + laenge / 2, H - 0.25, z);
      this.inhalt.add(l);
    }
    // Lichtvoute an der Rückwand
    const voute = new THREE.Mesh(new THREE.BoxGeometry(B * 0.92, 0.012, 0.05), leuchte);
    voute.position.set(B / 2, H - 0.02, 0.22);
    this.inhalt.add(voute);
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
      if (this.konfig.deko) this.kuechenDeko(ix, iB, hoeheUS + 0.05, iz, iT);
    }
  }

  /**
   * Deko auf der Insel. Ein leerer Raum sieht immer nach Modell aus –
   * Brett, Schale und Pflanze machen daraus eine Küche.
   */
  private kuechenDeko(x: number, b: number, y: number, z: number, t: number): void {
    const holz = this.holzMaterial(1, 2);
    const keramik = this.merke(
      new THREE.MeshPhysicalMaterial({ color: 0xf0ece3, roughness: 0.35, clearcoat: 0.5 }),
    );

    // Schneidebrett, leicht schräg
    const brett = quader(0.42, 0.028, 0.3, holz, x + b * 0.08, y, z + t * 0.34);
    brett.rotation.y = -0.16;
    this.inhalt.add(brett);

    // Schale
    const schale = new THREE.Mesh(new THREE.SphereGeometry(0.13, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2), keramik);
    schale.rotation.x = Math.PI;
    schale.position.set(x + b * 0.52, y + 0.065, z + t * 0.45);
    schale.castShadow = true;
    this.inhalt.add(schale);
    const inhalt = this.merke(new THREE.MeshStandardMaterial({ color: 0x86a04a, roughness: 0.6 }));
    for (let i = 0; i < 5; i++) {
      const o = new THREE.Mesh(new THREE.SphereGeometry(0.037, 12, 8), inhalt);
      const w = (i / 5) * Math.PI * 2;
      o.position.set(x + b * 0.52 + Math.cos(w) * 0.055, y + 0.055, z + t * 0.45 + Math.sin(w) * 0.055);
      o.castShadow = true;
      this.inhalt.add(o);
    }

    // Topfpflanze
    const topfMat = this.merke(new THREE.MeshStandardMaterial({ color: 0x8d6a52, roughness: 0.8 }));
    const topf = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.058, 0.13, 18), topfMat);
    topf.position.set(x + b * 0.82, y + 0.065, z + t * 0.4);
    topf.castShadow = true;
    this.inhalt.add(topf);
    const blattMat = this.merke(
      new THREE.MeshStandardMaterial({ color: 0x4e7a3c, roughness: 0.65, side: THREE.DoubleSide }),
    );
    for (let i = 0; i < 9; i++) {
      const blatt = new THREE.Mesh(new THREE.SphereGeometry(0.075, 8, 5), blattMat);
      blatt.scale.set(0.42, 1.5, 0.12);
      const w = (i / 9) * Math.PI * 2;
      blatt.position.set(
        x + b * 0.82 + Math.cos(w) * 0.05,
        y + 0.2 + Math.sin(i * 1.7) * 0.05,
        z + t * 0.4 + Math.sin(w) * 0.05,
      );
      blatt.rotation.set(Math.cos(w) * 0.5, w, Math.sin(w) * 0.5);
      blatt.castShadow = true;
      this.inhalt.add(blatt);
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

  /**
   * Wohnzimmer nach der Referenz: Verglasung mit Panorama in der Rückwand,
   * Hängekamin davor, Ecksofa mit Récamiere auf einem Teppich, Lowboard mit
   * Fernseher an der rechten Wand. Decke und Boden sind das Holz der Wahl.
   */
  private wohnzimmer(B: number, T: number): void {
    const holz = this.holzMaterial(1, 3);
    const front = this.frontMaterial();
    const stoff = this.merke(new THREE.MeshStandardMaterial({ color: 0x6f6a63, roughness: 1 }));
    const dunkel = this.merke(
      new THREE.MeshStandardMaterial({ color: 0x24262a, roughness: 0.45, metalness: 0.35 }),
    );

    this.haengekamin(B * 0.38, T * 0.3, 2.7, B, T);

    // Teppich unter der Sitzgruppe
    const teppich = this.merke(new THREE.MeshStandardMaterial({ color: 0x4f4a44, roughness: 1 }));
    const tB = Math.min(B * 0.72, 3.4);
    const tT = Math.min(T * 0.46, 2.5);
    const tx = B * 0.06;
    const tz = T * 0.42;
    this.inhalt.add(quader(tB, 0.014, tT, teppich, tx, 0.001, tz));

    // Ecksofa: Längsteil zur Verglasung, Récamiere nach rechts
    const sB = Math.min(2.5, B * 0.55);
    const sx = tx + 0.14;
    const sz = tz + 0.2;
    const sitzH = 0.34;
    this.inhalt.add(quader(sB, sitzH, 0.98, stoff, sx, 0.08, sz));
    this.inhalt.add(quader(sB, 0.42, 0.22, stoff, sx, sitzH + 0.08, sz + 0.76)); // Rückenlehne
    this.inhalt.add(quader(0.2, 0.26, 0.98, stoff, sx, sitzH + 0.08, sz)); // Armlehne links
    // Récamiere
    const rB = Math.min(1.5, B * 0.3);
    this.inhalt.add(quader(rB, sitzH, 1.5, stoff, sx + sB, 0.08, sz - 0.52));
    this.inhalt.add(quader(0.2, 0.26, 1.5, stoff, sx + sB + rB - 0.2, sitzH + 0.08, sz - 0.52));
    // Kissen
    for (let i = 0; i < 3; i++) {
      const k = quader(0.42, 0.14, 0.4, stoff, sx + 0.28 + i * 0.5, sitzH + 0.08, sz + 0.5);
      k.rotation.x = -0.22;
      this.inhalt.add(k);
    }
    // Plaid auf der Récamiere
    const plaid = this.merke(new THREE.MeshStandardMaterial({ color: 0x8a8378, roughness: 1 }));
    this.inhalt.add(quader(rB * 0.8, 0.05, 0.5, plaid, sx + sB + 0.1, sitzH + 0.08, sz - 0.3));

    // Beistelltisch: Steinplatte auf dünnem schwarzem Gestell
    const stein = this.merke(
      new THREE.MeshPhysicalMaterial({ color: 0xe8e4dc, roughness: 0.18, clearcoat: 0.7 }),
    );
    const btx = sx + sB * 0.25;
    const btz = sz - 0.62;
    const platte = new THREE.Mesh(new THREE.CylinderGeometry(0.29, 0.29, 0.035, 28), stein);
    platte.position.set(btx, 0.5, btz);
    platte.castShadow = true;
    this.inhalt.add(platte);
    for (let i = 0; i < 3; i++) {
      const w = (i / 3) * Math.PI * 2;
      const bein = quader(0.022, 0.5, 0.022, dunkel, btx + Math.cos(w) * 0.2, 0, btz + Math.sin(w) * 0.2);
      this.inhalt.add(bein);
    }
    // Tischleuchte, Kugel
    const kugel = new THREE.Mesh(
      new THREE.SphereGeometry(0.085, 20, 14),
      this.merke(new THREE.MeshBasicMaterial({ color: 0xffe9c0, toneMapped: false })),
    );
    kugel.position.set(btx + 0.05, 0.6, btz);
    this.inhalt.add(kugel);
    if (this.konfig.licht) {
      const l = new THREE.PointLight(0xffd9a0, 1.4, 2.8, 2);
      l.position.copy(kugel.position);
      this.inhalt.add(l);
    }

    // Lowboard mit Fernseher auf dem massiven Teil der Rückwand. Dort sieht
    // die Kamera es auch – an der rechten Wand läge es außerhalb des Bildes,
    // und die Frontenwahl wäre in diesem Raum unsichtbar.
    const lwX = B * 0.58;
    const lwB = Math.min(B * 0.38, 2.6);
    this.inhalt.add(quader(lwB, 0.4, 0.4, holz, lwX, 0.42, 0.02));
    // Griffmulde als dunkler Schlitz
    this.inhalt.add(quader(lwB * 0.42, 0.05, 0.02, dunkel, lwX + lwB * 0.08, 0.62, 0.42));
    this.inhalt.add(
      quader(Math.min(1.3, lwB * 0.68), 0.74, 0.05, dunkel, lwX + lwB * 0.14, 1.18, 0.03),
    );

    // Holzscheite neben dem Kamin
    if (this.konfig.deko) {
      const korb = this.merke(new THREE.MeshStandardMaterial({ color: 0x2c2e31, roughness: 0.8 }));
      const kx = B * 0.62;
      const kz = T * 0.16;
      const schale = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.22, 0.3, 20, 1, true), korb);
      schale.position.set(kx, 0.15, kz);
      schale.castShadow = true;
      this.inhalt.add(schale);
      for (let i = 0; i < 7; i++) {
        const scheit = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.34, 9), holz);
        scheit.position.set(kx + (Math.random() - 0.5) * 0.24, 0.32 + Math.random() * 0.06, kz + (Math.random() - 0.5) * 0.2);
        scheit.rotation.set(Math.PI / 2, 0, Math.random() * Math.PI);
        scheit.castShadow = true;
        this.inhalt.add(scheit);
      }
    }

    // Hochschrank neben dem Lowboard in der gewählten Front – sonst hätte die
    // Frontenwahl in diesem Raum keine sichtbare Fläche.
    const hsB = Math.min(0.62, B - (lwX + lwB) - 0.06);
    if (hsB > 0.25) {
      this.inhalt.add(quader(hsB, 2.05, 0.42, this.korpusMaterial(), lwX + lwB + 0.04, 0, 0.02));
      this.inhalt.add(quader(hsB - 0.03, 1.99, 0.02, front, lwX + lwB + 0.055, 0.03, 0.44));
    }
  }

  /** Hängekamin: Rohr von der Decke, runder Korpus, offene Feuerseite. */
  private haengekamin(x: number, z: number, H: number, B: number, T: number): void {
    const stahl = this.merke(
      new THREE.MeshStandardMaterial({ color: 0x1f2124, roughness: 0.5, metalness: 0.45 }),
    );

    const rohr = new THREE.Mesh(new THREE.CylinderGeometry(0.085, 0.085, H - 1.72, 18), stahl);
    rohr.position.set(x, 1.72 + (H - 1.72) / 2, z);
    rohr.castShadow = true;
    this.inhalt.add(rohr);

    const haube = new THREE.Mesh(new THREE.ConeGeometry(0.44, 0.34, 26), stahl);
    haube.position.set(x, 1.62, z);
    haube.castShadow = true;
    this.inhalt.add(haube);

    // Die Feuerseite bleibt offen, sonst sieht man von schräg oben nur den
    // geschlossenen Topf – und das Feuer ist der Punkt, auf den alles zuläuft.
    // Die Öffnung zeigt dorthin, wo die Innenkamera steht.
    const zurKamera = Math.atan2(B * 0.86 - x, T * 0.97 - z);
    const luecke = 1.15; // Radiant, gut 65 Grad
    const korpus = new THREE.Mesh(
      new THREE.CylinderGeometry(
        0.44, 0.3, 0.42, 26, 1, true,
        zurKamera + luecke / 2,
        Math.PI * 2 - luecke,
      ),
      this.merke(
        new THREE.MeshStandardMaterial({
          color: 0x1f2124,
          roughness: 0.5,
          metalness: 0.45,
          side: THREE.DoubleSide,
        }),
      ),
    );
    korpus.position.set(x, 1.26, z);
    korpus.castShadow = true;
    this.inhalt.add(korpus);

    // Bodenplatte und Scheite im Feuerraum
    const platte = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 0.025, 26), stahl);
    platte.position.set(x, 1.06, z);
    this.inhalt.add(platte);
    const glut = this.merke(new THREE.MeshBasicMaterial({ color: 0xff8a2b, toneMapped: false }));
    const asche = this.merke(new THREE.MeshStandardMaterial({ color: 0x2a2521, roughness: 1 }));
    for (let i = 0; i < 4; i++) {
      const sch = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.3, 8), asche);
      sch.rotation.set(Math.PI / 2, 0, (i / 4) * Math.PI);
      sch.position.set(x + (i - 1.5) * 0.045, 1.11, z);
      this.inhalt.add(sch);
    }
    const feuer = new THREE.Mesh(new THREE.SphereGeometry(0.15, 16, 10), glut);
    feuer.scale.set(1.05, 0.95, 0.75);
    feuer.position.set(x, 1.2, z);
    this.inhalt.add(feuer);
    const flamme = new THREE.PointLight(0xff9c40, 2.6, 4.5, 2);
    flamme.position.set(x, 1.3, z + 0.1);
    this.inhalt.add(flamme);
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
    const tz = Math.min(T * 0.34, 1.5);
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
    const sT = Math.min(1.0, T - tz - 1.9);
    if (sT > 0.5) {
      this.inhalt.add(quader(0.68, tH, sT, holz, tx, 0, tz + 0.68));
      this.inhalt.add(quader(0.8, 0.06, sT, platte, tx - 0.06, tH, tz + 0.68));
    }
  }

  /* --------------------------------------------------------------- Kamera */

  /**
   * Abstand, bei dem der ganze Raum ins Bild passt – waagrecht wie senkrecht.
   * Am Handy ist das Bild fast quadratisch, da reicht der vertikale
   * Blickwinkel allein nicht und die Zeile wird rechts abgeschnitten.
   */
  private noetigerAbstand(B: number, T: number): number {
    const radius = Math.hypot(B, T, 2.7) / 2 + 0.5;
    const vFov = (this.kamera.fov * Math.PI) / 180;
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * this.kamera.aspect);
    return Math.max(radius / Math.sin(vFov / 2), radius / Math.sin(hFov / 2));
  }

  private kameraSetzen(B: number, T: number): void {
    if (this.konfig.blick === 'innen' || this.konfig.blick === 'rundum') {
      // Je Raum ein eigener Standpunkt: eine gemeinsame Ecke funktioniert
      // nicht, weil die Möbel unterschiedlich stehen – im Wohnzimmer säße
      // die Kamera sonst mitten im Sofa.
      const blicke: Record<RaumTyp, { pos: [number, number, number]; ziel: [number, number, number]; fov: number }> = {
        kueche:      { pos: [0.12, 1.58, 0.90], ziel: [0.68, 1.00, 0.14], fov: 48 },
        wohnzimmer:  { pos: [0.86, 1.64, 0.97], ziel: [0.30, 1.15, 0.04], fov: 52 },
        hotelzimmer: { pos: [0.88, 1.60, 0.93], ziel: [0.30, 1.00, 0.12], fov: 50 },
        laden:       { pos: [0.90, 1.66, 0.95], ziel: [0.34, 1.02, 0.12], fov: 52 },
      };
      if (this.konfig.blick === 'rundum') {
        // Standpunkt je Raum in eine freie Fläche gelegt – in der Raummitte
        // stünde man im Wohnzimmer mitten im Sofa und in der Küche in der
        // Insel. Die Blickrichtung setzt der Betrachter selbst.
        const stand: Record<RaumTyp, [number, number]> = {
          kueche: [0.5, 0.26], // zwischen Zeile und Insel
          wohnzimmer: [0.62, 0.3], // zwischen Kamin und Sitzgruppe
          hotelzimmer: [0.5, 0.72], // am Fußende des Betts
          laden: [0.55, 0.85], // vor der Theke, wo der Kunde steht
        };
        const [fx, fz] = stand[this.konfig.raum];
        this.kamera.fov = 70;
        this.kamera.near = 0.05;
        this.kamera.updateProjectionMatrix();
        this.kamera.position.set(B * fx, 1.62, T * fz);
        this.steuerung.target.set(B * fx, 1.5, T * (fz - 0.4));
        this.steuerung.update();
        return;
      }
      const w = blicke[this.konfig.raum];
      this.kamera.fov = w.fov;
      this.kamera.near = 0.05;
      this.kamera.updateProjectionMatrix();
      this.kamera.position.set(B * w.pos[0], w.pos[1], T * w.pos[2]);
      this.steuerung.target.set(B * w.ziel[0], w.ziel[1], T * w.ziel[2]);
      this.steuerung.minDistance = 0.3;
      this.steuerung.maxDistance = Math.hypot(B, T) * 1.2;
      this.steuerung.update();
      return;
    }
    this.kamera.fov = 34;
    this.kamera.updateProjectionMatrix();
    const ziel = new THREE.Vector3(B / 2, 1.05, T / 2);
    this.steuerung.target.copy(ziel);
    const d = this.noetigerAbstand(B, T);
    this.steuerung.minDistance = d * 0.5;
    this.steuerung.maxDistance = d * 1.6;
    if (!this.laeuft) {
      // Startblick: von schräg vorne rechts, ähnlich der bisherigen Isometrie.
      const winkel = Math.PI * 0.27;
      this.kamera.position.set(
        ziel.x + Math.sin(winkel) * d * 0.78,
        ziel.y + d * 0.52,
        ziel.z + Math.cos(winkel) * d * 0.78,
      );
    } else {
      this.abstandNachziehen(d);
    }
    this.steuerung.update();
  }

  /** Nach Drehen des Geräts oder größeren Maßen wieder alles ins Bild holen. */
  private abstandNachziehen(d: number): void {
    const richtung = this.kamera.position.clone().sub(this.steuerung.target);
    if (richtung.length() < d) {
      this.kamera.position.copy(this.steuerung.target).add(richtung.setLength(d));
    }
  }

  private groesse(): void {
    const b = this.huelle.clientWidth || 1;
    const h = this.huelle.clientHeight || 1;
    this.renderer.setSize(b, h, false);
    this.kamera.aspect = b / h;
    this.kamera.updateProjectionMatrix();
    if (this.laeuft && this.konfig.blick === 'plan') {
      const d = this.noetigerAbstand(this.konfig.breite, this.konfig.tiefe);
      this.steuerung.minDistance = d * 0.5;
      this.steuerung.maxDistance = d * 1.6;
      this.abstandNachziehen(d);
      this.steuerung.update();
    }
  }

  /* ------------------------------------------------------------- Öffentlich */

  /** Deckende Hintergrundfarbe statt Transparenz – nötig fürs JPEG. */
  hintergrund(farbe: number | null): void {
    this.szene.background = farbe === null ? null : new THREE.Color(farbe);
  }

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

  /** Nur fürs Vorab-Rendern: Zugriff auf die Innereien der Szene. */
  intern() {
    return {
      szene: this.szene,
      kamera: this.kamera,
      renderer: this.renderer,
      steuerung: this.steuerung,
    };
  }

  aufraeumen(): void {
    this.laeuft = false;
    this.beobachter?.disconnect();
    this.leeren();
    this.steuerung.dispose();
    this.umgebung?.dispose();
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
