/**
 * 360-Grad-Betrachter für die vorab gerenderten Panoramen.
 *
 * Die Kugelprojektion wird von innen auf eine Kugel gelegt; die Kamera sitzt
 * in deren Mittelpunkt und dreht sich nur. Das ist alles, was es braucht –
 * kein Laufen durch den Raum, sondern Umschauen vom Standpunkt aus.
 */
import * as THREE from 'three';

export class Rundgang {
  private renderer: THREE.WebGLRenderer;
  private szene = new THREE.Scene();
  private kamera: THREE.PerspectiveCamera;
  private kugel: THREE.Mesh;
  private material: THREE.MeshBasicMaterial;
  private huelle: HTMLElement;
  private lader = new THREE.TextureLoader();
  private beobachter?: ResizeObserver;
  private laeuft = false;

  /** Blickrichtung in Grad; wird beim Wechsel der Ansicht beibehalten. */
  private laenge = 0;
  private breite = 0;
  private zielLaenge = 0;
  private zielBreite = 0;
  private zieht = false;
  private letzteZeige = { x: 0, y: 0 };
  private aktuelleQuelle = '';

  constructor(huelle: HTMLElement) {
    this.huelle = huelle;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    const c = this.renderer.domElement;
    c.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;cursor:grab';
    huelle.appendChild(c);

    this.kamera = new THREE.PerspectiveCamera(72, 1, 0.1, 100);

    // Kugel nach innen gestülpt, damit die Textur von innen sichtbar ist.
    this.material = new THREE.MeshBasicMaterial({ side: THREE.BackSide });
    this.kugel = new THREE.Mesh(new THREE.SphereGeometry(10, 60, 40), this.material);
    this.szene.add(this.kugel);

    this.zeigerBinden(c);
    this.beobachter = new ResizeObserver(() => this.groesse());
    this.beobachter.observe(huelle);
    this.groesse();
  }

  private zeigerBinden(c: HTMLCanvasElement): void {
    const start = (e: PointerEvent) => {
      this.zieht = true;
      this.letzteZeige = { x: e.clientX, y: e.clientY };
      c.setPointerCapture(e.pointerId);
      c.style.cursor = 'grabbing';
    };
    const zug = (e: PointerEvent) => {
      if (!this.zieht) return;
      const dx = e.clientX - this.letzteZeige.x;
      const dy = e.clientY - this.letzteZeige.y;
      this.letzteZeige = { x: e.clientX, y: e.clientY };
      // Empfindlichkeit an das Blickfeld koppeln: beim Hineinzoomen soll
      // dieselbe Fingerbewegung weniger drehen.
      const takt = this.kamera.fov / 72;
      this.zielLaenge -= dx * 0.12 * takt;
      this.zielBreite = Math.max(-85, Math.min(85, this.zielBreite + dy * 0.12 * takt));
    };
    const ende = (e: PointerEvent) => {
      this.zieht = false;
      if (c.hasPointerCapture(e.pointerId)) c.releasePointerCapture(e.pointerId);
      c.style.cursor = 'grab';
    };
    c.addEventListener('pointerdown', start);
    c.addEventListener('pointermove', zug);
    c.addEventListener('pointerup', ende);
    c.addEventListener('pointercancel', ende);

    c.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        this.kamera.fov = Math.max(32, Math.min(88, this.kamera.fov + e.deltaY * 0.05));
        this.kamera.updateProjectionMatrix();
      },
      { passive: false },
    );

    // Zwei Finger zum Zoomen
    let startAbstand = 0;
    let startFov = 72;
    c.addEventListener('touchstart', (e) => {
      if (e.touches.length !== 2) return;
      startAbstand = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY,
      );
      startFov = this.kamera.fov;
    });
    c.addEventListener(
      'touchmove',
      (e) => {
        if (e.touches.length !== 2 || !startAbstand) return;
        e.preventDefault();
        const jetzt = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY,
        );
        this.kamera.fov = Math.max(32, Math.min(88, startFov * (startAbstand / jetzt)));
        this.kamera.updateProjectionMatrix();
      },
      { passive: false },
    );
  }

  private groesse(): void {
    const b = this.huelle.clientWidth || 1;
    const h = this.huelle.clientHeight || 1;
    this.renderer.setSize(b, h, false);
    this.kamera.aspect = b / h;
    this.kamera.updateProjectionMatrix();
  }

  /**
   * Panorama wechseln. Gibt ein Versprechen zurück, das erfüllt ist, sobald
   * das Bild steht – die Seite kann so lange ihren Ladehinweis zeigen.
   */
  laden(quelle: string): Promise<void> {
    this.aktuelleQuelle = quelle;
    return new Promise((fertig, fehler) => {
      this.lader.load(
        quelle,
        (textur) => {
          // Zwischenzeitlich weitergeklickt? Dann gilt die neuere Auswahl.
          if (this.aktuelleQuelle !== quelle) {
            textur.dispose();
            return fertig();
          }
          textur.colorSpace = THREE.SRGBColorSpace;
          // Die Kugelgeometrie läuft andersherum als die Projektion.
          textur.wrapS = THREE.RepeatWrapping;
          textur.repeat.x = -1;
          const alt = this.material.map;
          this.material.map = textur;
          this.material.needsUpdate = true;
          alt?.dispose();
          fertig();
        },
        undefined,
        () => fehler(new Error(quelle)),
      );
    });
  }

  start(): void {
    if (this.laeuft) return;
    this.laeuft = true;
    const schleife = () => {
      if (!this.laeuft) return;
      // Weich nachziehen, damit das Drehen nicht hakt.
      this.laenge += (this.zielLaenge - this.laenge) * 0.12;
      this.breite += (this.zielBreite - this.breite) * 0.12;
      const phi = THREE.MathUtils.degToRad(90 - this.breite);
      const theta = THREE.MathUtils.degToRad(this.laenge);
      this.kamera.lookAt(
        Math.sin(phi) * Math.cos(theta),
        Math.cos(phi),
        Math.sin(phi) * Math.sin(theta),
      );
      this.renderer.render(this.szene, this.kamera);
      requestAnimationFrame(schleife);
    };
    schleife();
  }

  /** Blick zurück auf den Ausgangspunkt. */
  zuruecksetzen(): void {
    this.zielLaenge = 0;
    this.zielBreite = 0;
    this.kamera.fov = 72;
    this.kamera.updateProjectionMatrix();
  }

  aufraeumen(): void {
    this.laeuft = false;
    this.beobachter?.disconnect();
    this.material.map?.dispose();
    this.material.dispose();
    this.kugel.geometry.dispose();
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

declare global {
  interface Window {
    Rundgang: typeof Rundgang;
    rundgangUnterstuetzt: typeof unterstuetzt;
  }
}
window.Rundgang = Rundgang;
window.rundgangUnterstuetzt = unterstuetzt;
