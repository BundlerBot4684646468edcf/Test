import { createCanvas, loadImage, GlobalFonts } from '@napi-rs/canvas';
import path from 'path';

// Register the bundled fonts once (ship with the project, so they render
// identically on every machine — no reliance on OS-installed fonts, which
// @napi-rs/canvas may not find at all on some systems, e.g. showing tofu
// boxes for "sans-serif" on some Windows setups).
let fontReady = false;
function ensureFont() {
  if (fontReady) return;
  const handwriting = require.resolve(
    '@fontsource/caveat/files/caveat-latin-400-normal.woff'
  );
  GlobalFonts.registerFromPath(handwriting, 'Caveat');

  const bodyRegular = require.resolve(
    '@fontsource/inter/files/inter-latin-400-normal.woff'
  );
  GlobalFonts.registerFromPath(bodyRegular, 'Inter');
  const bodyBold = require.resolve(
    '@fontsource/inter/files/inter-latin-700-normal.woff'
  );
  GlobalFonts.registerFromPath(bodyBold, 'Inter Bold');

  fontReady = true;
}

export interface SignBox {
  // All values relative to the image (0..1), so they survive any resolution.
  x: number; // center of the sign, horizontal
  y: number; // center of the sign, vertical
  width: number; // sign width as fraction of image width
  height: number; // sign height as fraction of image height
  rotation: number; // degrees, e.g. -3 for a slight tilt
}

export const DEFAULT_SIGN_BOX: SignBox = {
  x: 0.5,
  y: 0.62,
  width: 0.55,
  height: 0.22,
  rotation: -3,
};

/**
 * Draws the customer's first name in handwriting onto the (blank) sign
 * area of the owner photo. Returns a PNG buffer.
 */
export async function personalizePhoto(
  baseImage: Buffer,
  firstName: string,
  box: SignBox = DEFAULT_SIGN_BOX
): Promise<Buffer> {
  ensureFont();

  const img = await loadImage(baseImage);
  const W = img.width;
  const H = img.height;

  const canvas = createCanvas(W, H);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, W, H);

  const cx = box.x * W;
  const cy = box.y * H;
  const boxW = box.width * W;
  const boxH = box.height * H;
  const rot = (box.rotation * Math.PI) / 180;

  // Auto-fit the font: start big, shrink until the name fits the sign.
  const name = firstName.trim();
  let fontSize = Math.floor(boxH * 0.72);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (; fontSize > 12; fontSize -= 4) {
    ctx.font = `${fontSize}px Caveat`;
    if (ctx.measureText(name).width <= boxW * 0.86) break;
  }

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(rot);
  ctx.font = `${fontSize}px Caveat`;
  ctx.fillStyle = '#1f2937';
  ctx.fillText(name, 0, boxH * 0.04);
  ctx.restore();

  return canvas.toBuffer('image/png');
}

/**
 * Generates a demo photo (red wall + white sign, like the salon video) with
 * the name already written on it — used for previewing without a real photo.
 */
export async function demoPhoto(firstName: string): Promise<Buffer> {
  ensureFont();

  const W = 800;
  const H = 600;
  const canvas = createCanvas(W, H);
  const ctx = canvas.getContext('2d');

  // Wall
  ctx.fillStyle = '#b91c1c';
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(255,255,255,0.06)';
  ctx.fillRect(0, 0, W, 180);

  // "Logo" text on the wall
  ctx.fillStyle = '#ffffff';
  ctx.font = '44px "Inter Bold"';
  ctx.textAlign = 'center';
  ctx.fillText('Mundpost Demo', W / 2, 110);
  ctx.font = '22px Inter';
  ctx.fillText('Dein Betrieb · Südtirol', W / 2, 150);

  // Sign with shadow
  ctx.save();
  ctx.translate(W / 2, 400);
  ctx.rotate(-0.04);
  ctx.shadowColor = 'rgba(0,0,0,0.35)';
  ctx.shadowBlur = 18;
  ctx.shadowOffsetY = 8;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(-270, -110, 540, 220);
  ctx.restore();

  // Name on the sign
  const name = firstName.trim();
  let fontSize = 130;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (; fontSize > 12; fontSize -= 4) {
    ctx.font = `${fontSize}px Caveat`;
    if (ctx.measureText(name).width <= 460) break;
  }
  ctx.save();
  ctx.translate(W / 2, 400);
  ctx.rotate(-0.04);
  ctx.fillStyle = '#1f2937';
  ctx.font = `${fontSize}px Caveat`;
  ctx.fillText(name, 0, 8);
  ctx.restore();

  return canvas.toBuffer('image/png');
}
