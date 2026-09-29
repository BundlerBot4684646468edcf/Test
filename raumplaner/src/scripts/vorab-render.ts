/**
 * Vorab-Rendern: erzeugt die Standbilder, die der Konfigurator später nur
 * noch umschaltet. Läuft NICHT im Browser des Besuchers, sondern einmal
 * hier über ein kopfloses Chromium – deshalb darf es teuer sein.
 *
 * Gegenüber der Live-Ansicht kommt dazu:
 *   - Supersampling (SSAA) für saubere Kanten
 *   - GTAO, also Verschattung in Ecken und Fugen
 *   - Tiefenunschärfe, wie bei einer echten Aufnahme
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { SSAARenderPass } from 'three/examples/jsm/postprocessing/SSAARenderPass.js';
import { GTAOPass } from 'three/examples/jsm/postprocessing/GTAOPass.js';
import { BokehPass } from 'three/examples/jsm/postprocessing/BokehPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { Raumplaner, holzFotos, type Konfiguration } from './raumplaner-szene';

export interface RenderOptionen {
  breite: number;
  hoehe: number;
  /** 0 = aus, 2 = 4 Abtastungen, 3 = 8. Kostet linear Zeit. */
  ssaa: number;
  ao: boolean;
  unschaerfe: boolean;
}

export class VorabRenderer {
  private planer: Raumplaner;
  private komponist!: EffectComposer;
  private ssaaPass!: SSAARenderPass;
  private gtao?: GTAOPass;
  private bokeh?: BokehPass;
  private opt: RenderOptionen;

  constructor(huelle: HTMLElement, opt: RenderOptionen) {
    this.opt = opt;
    this.planer = new Raumplaner(huelle);
    this.planer.hintergrund(0xede7dc);
    this.aufbauen();
  }

  private aufbauen(): void {
    const { szene, kamera, renderer } = this.planer.intern();
    const { breite: b, hoehe: h } = this.opt;

    renderer.setPixelRatio(1);
    renderer.setSize(b, h, false);
    kamera.aspect = b / h;
    kamera.updateProjectionMatrix();

    this.komponist = new EffectComposer(renderer);
    this.komponist.setSize(b, h);

    this.ssaaPass = new SSAARenderPass(szene, kamera);
    this.ssaaPass.sampleLevel = this.opt.ssaa;
    this.ssaaPass.unbiased = true;
    this.komponist.addPass(this.ssaaPass);

    if (this.opt.ao) {
      this.gtao = new GTAOPass(szene, kamera, b, h);
      // Kleiner Radius: die Verschattung soll in Fugen und Ecken sitzen,
      // nicht den halben Raum abdunkeln.
      this.gtao.updateGtaoMaterial({ radius: 0.18, distanceExponent: 1.4, scale: 1.1, thickness: 0.4 });
      this.gtao.blendIntensity = 0.85;
      this.komponist.addPass(this.gtao);
    }

    if (this.opt.unschaerfe) {
      this.bokeh = new BokehPass(szene, kamera, { focus: 6, aperture: 0.00055, maxblur: 0.008 });
      this.komponist.addPass(this.bokeh);
    }

    this.komponist.addPass(new OutputPass());
  }

  /** Eine Konfiguration aufbauen und als JPEG zurückgeben. */
  bild(konfig: Partial<Konfiguration>, guete = 0.9): string {
    this.planer.aktualisieren({ masse: false, deko: true, blick: 'innen', ...konfig });
    const { kamera, renderer } = this.planer.intern();

    if (this.bokeh) {
      // Auf die Raummitte scharfstellen.
      const ziel = this.planer.intern().steuerung.target;
      (this.bokeh.uniforms as Record<string, { value: number }>).focus.value =
        kamera.position.distanceTo(ziel);
    }

    this.komponist.render();
    return renderer.domElement.toDataURL('image/jpeg', guete);
  }
}

/**
 * 360-Grad-Panorama. Die Würfelkamera nimmt sechs Richtungen auf, ein
 * Shader rechnet den Würfel in die Kugelprojektion um, die jeder
 * Panorama-Betrachter erwartet (equirectangular, Seitenverhältnis 2:1).
 */
export class PanoramaRenderer {
  private planer: Raumplaner;
  private breite: number;
  private hoehe: number;
  private wuerfelZiel: THREE.WebGLCubeRenderTarget;
  private wuerfelKamera: THREE.CubeCamera;
  private flachZiel: THREE.WebGLRenderTarget;
  private flachSzene = new THREE.Scene();
  private flachKamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private material: THREE.ShaderMaterial;
  private ausgabe: HTMLCanvasElement;

  constructor(huelle: HTMLElement, kante = 1024) {
    this.breite = kante * 2;
    this.hoehe = kante;
    this.planer = new Raumplaner(huelle);
    this.planer.hintergrund(0xede7dc);

    const { renderer } = this.planer.intern();
    renderer.setPixelRatio(1);
    renderer.setSize(kante, kante, false);

    this.wuerfelZiel = new THREE.WebGLCubeRenderTarget(kante, {
      generateMipmaps: false,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      colorSpace: THREE.SRGBColorSpace,
    });
    this.wuerfelKamera = new THREE.CubeCamera(0.05, 100, this.wuerfelZiel);

    this.flachZiel = new THREE.WebGLRenderTarget(this.breite, this.hoehe, {
      colorSpace: THREE.SRGBColorSpace,
    });

    this.material = new THREE.ShaderMaterial({
      uniforms: { wuerfel: { value: this.wuerfelZiel.texture } },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: `
        uniform samplerCube wuerfel;
        varying vec2 vUv;
        #define PI 3.141592653589793
        void main() {
          // Bildkoordinate -> Längen- und Breitengrad -> Richtungsvektor
          float laenge = (vUv.x - 0.5) * 2.0 * PI;
          float breite = (vUv.y - 0.5) * PI;
          vec3 richtung = vec3(
            cos(breite) * sin(laenge),
            sin(breite),
            cos(breite) * cos(laenge)
          );
          gl_FragColor = textureCube(wuerfel, richtung);
        }
      `,
    });
    this.flachSzene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material));

    this.ausgabe = document.createElement('canvas');
    this.ausgabe.width = this.breite;
    this.ausgabe.height = this.hoehe;
  }

  /** Eine Konfiguration als Kugelpanorama, JPEG als Data-URL. */
  bild(konfig: Partial<Konfiguration>, guete = 0.84): string {
    this.planer.aktualisieren({ masse: false, deko: true, blick: 'rundum', ...konfig });
    const { szene, kamera, renderer } = this.planer.intern();

    // Die Würfelkamera steht dort, wo die Szene den Standpunkt gesetzt hat.
    this.wuerfelKamera.position.copy(kamera.position);
    this.wuerfelKamera.update(renderer, szene);

    renderer.setRenderTarget(this.flachZiel);
    renderer.render(this.flachSzene, this.flachKamera);

    const pixel = new Uint8Array(this.breite * this.hoehe * 4);
    renderer.readRenderTargetPixels(this.flachZiel, 0, 0, this.breite, this.hoehe, pixel);
    renderer.setRenderTarget(null);

    // readRenderTargetPixels liefert die unterste Zeile zuerst – umdrehen.
    const g = this.ausgabe.getContext('2d')!;
    const bild = g.createImageData(this.breite, this.hoehe);
    const zeile = this.breite * 4;
    for (let y = 0; y < this.hoehe; y++) {
      const von = (this.hoehe - 1 - y) * zeile;
      bild.data.set(pixel.subarray(von, von + zeile), y * zeile);
    }
    g.putImageData(bild, 0, 0);
    return this.ausgabe.toDataURL('image/jpeg', guete);
  }
}

/** Vom Render-Skript aus aufrufbar machen. */
declare global {
  interface Window {
    VorabRenderer: typeof VorabRenderer;
    PanoramaRenderer: typeof PanoramaRenderer;
    holzFotos: typeof holzFotos;
  }
}
window.VorabRenderer = VorabRenderer;
window.PanoramaRenderer = PanoramaRenderer;
window.holzFotos = holzFotos;
