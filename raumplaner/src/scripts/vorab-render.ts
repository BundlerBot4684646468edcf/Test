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

/** Vom Render-Skript aus aufrufbar machen. */
declare global {
  interface Window {
    VorabRenderer: typeof VorabRenderer;
    holzFotos: typeof holzFotos;
  }
}
window.VorabRenderer = VorabRenderer;
window.holzFotos = holzFotos;
