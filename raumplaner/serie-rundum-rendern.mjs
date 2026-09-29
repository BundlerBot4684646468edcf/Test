import { chromium } from 'playwright';
import fs from 'node:fs';
const raeume  = ['kueche','hotelzimmer','wohnzimmer','laden'];
const hoelzer = ['eiche-hell','altholz','nussbaum','fichte-hell'];
const fronten = ['holz','weiss','anthrazit','betonoptik'];
fs.mkdirSync('out/pano', { recursive: true });

const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  args: ['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
const p = await b.newPage({ viewport: { width: 1200, height: 900 } });
p.on('pageerror', e => console.log('FEHLER:', e.message));
await p.goto('http://127.0.0.1:8324/vorab.html');
await p.waitForTimeout(1200);
await p.evaluate(() => {
  const el = document.getElementById('buehne'); el.innerHTML = '';
  window.PR = new window.PanoramaRenderer(el, 1024);
});
let n = 0; const t0 = Date.now();
const liste = [];
for (const raum of raeume) for (const holz of hoelzer) for (const front of fronten) {
  const d = await p.evaluate((k) => window.PR.bild(k, 0.82), { raum, holz, front });
  const name = `${raum}_${holz}_${front}.jpg`;
  fs.writeFileSync('out/pano/' + name, Buffer.from(d.split(',')[1], 'base64'));
  liste.push({ raum, holz, front, datei: name, kb: Math.round(d.length/1365) });
  n++;
  if (n % 8 === 0) console.log(`${n}/64  ${((Date.now()-t0)/1000/n).toFixed(1)}s je Bild`);
}
fs.writeFileSync('out/pano/index.json', JSON.stringify(liste, null, 1));
console.log('fertig:', n, 'Panoramen,', liste.reduce((s,x)=>s+x.kb,0), 'kB,', ((Date.now()-t0)/1000/60).toFixed(1), 'min');
await b.close();
