// Generates bewerber-bot-mindmap.html (amstudio branding, A2 landscape) and renders PNG + PDF.
// Usage: node mindmap/bewerber-bot.js
const fs = require('fs');
const path = require('path');

const OUT = __dirname;
const NAME = 'bewerber-bot-mindmap';
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const chip = (label, cls = '') => `<div class="chip ${cls}">${esc(label)}</div>`;
const chips = (id, parent, dir, items, style) =>
  `<div class="chips" id="${id}" data-parent="${parent}" data-dir="${dir}" style="${style}">${items
    .map(i => (Array.isArray(i) ? chip(i[0], i[1]) : chip(i)))
    .join('')}</div>`;

const node = ({ id, num, eyebrow, title, big, sub, style }) => `
<div class="node" id="${id}" style="${style}">
  <div class="eyebrow"><span class="badge">${num}</span>${esc(eyebrow)}</div>
  <h3>${esc(title)}</h3>
  <div class="big">${esc(big)}</div>
  ${sub ? `<div class="sub">${esc(sub)}</div>` : ''}
</div>`;

const branches = [
  {
    id: 'n1', side: 'left', cy: 232, num: '01', eyebrow: 'Meta', title: 'Instant-Formular', big: 'In der App', sub: 'Instagram & Facebook',
    items: ['Typ „Höhere Kaufabsicht“', 'Nummer bestätigen lassen', ['Häkchen: Einwilligung Anruf', 'key'],
      'Frage: Wann erreichbar?', 'Abschluss: „Anruf in Minuten“', 'Button „Anrufen“ mit Nummer'],
  },
  {
    id: 'n2', side: 'left', cy: 560, num: '02', eyebrow: 'Wann', title: 'Anrufregeln', big: '8–20 Uhr', sub: 'Mo–Fr · Sa 9–13 Uhr',
    items: ['Mo–Fr 8–20 Uhr', 'Samstag 9–13 Uhr', ['Sonntag & Feiertage: nie', 'x'], 'Südtiroler Feiertagskalender',
      'Außerhalb: sofort SMS', ['„Nicht anrufen“ = sofort Stopp', 'key']],
  },
  {
    id: 'n3', side: 'left', cy: 880, num: '03', eyebrow: 'Nicht erreicht', title: 'Wiederholung', big: 'max. 3×', sub: 'dann nur noch SMS',
    items: ['1. Anruf nach 5 Minuten', '2. Anruf nach 2 Stunden', '3. Anruf am nächsten Tag', 'Andere Uhrzeit wählen',
      'Danach nur SMS mit Link'],
  },
  {
    id: 'n4', side: 'right', cy: 232, num: '04', eyebrow: 'Im Gespräch', title: 'Bot fragt', big: '7 Infos', sub: 'kurz, freundlich, DE / IT',
    items: ['Berufserfahrung', 'Führerschein', 'Anfahrtsweg', 'Gehaltswunsch', 'Verfügbar ab', 'Wechselgrund', 'Sprache DE / IT'],
  },
  {
    id: 'n5', side: 'right', cy: 572, num: '05', eyebrow: 'An den Chef', title: 'Nachricht', big: '1 Klick', sub: 'WhatsApp oder E-Mail',
    items: ['Name, Ort, Nummer', 'Alle Antworten kompakt', 'Wunschtermin', ['Bestätigen · Absagen · Rückruf', 'key'],
      'Termin direkt im Kalender'],
  },
  {
    id: 'n6', side: 'right', cy: 890, num: '06', eyebrow: 'Grenzen', title: 'Spielregeln', big: 'Mensch', sub: 'entscheidet, nicht der Bot',
    items: [['Bot sortiert nicht aus', 'key'], 'Sagt „digitale Assistentin“', 'Südtiroler Nummer', 'DSGVO + AV-Vertrag',
      ['Vor Verkauf rechtlich prüfen', 'plan']],
  },
];

let branchHtml = '';
for (const b of branches) {
  const left = b.side === 'left';
  branchHtml += node({ ...b, style: `left:${left ? 420 : 1694}px;top:${b.cy}px;width:266px;transform:translateY(-50%)` });
  branchHtml += chips(`${b.id}c`, b.id, left ? 'left' : 'right', b.items,
    `left:${left ? 36 : 2024}px;top:${b.cy}px;width:320px;transform:translateY(-50%)`);
}

const step = (n, s, cls = '') => `<span class="step ${cls}"><b>${n}</b>${esc(s)}</span>`;
const arr = '<span class="arr"></span>';
const row = parts => `<div class="flow">${parts.join(arr)}</div>`;

const bottom = `
<div class="bcard" id="bF" style="left:36px;width:1500px">
  <div class="bhead"><span class="num">07</span><h2>Ablauf</h2><span class="hint">vom Klick auf die Anzeige bis zum Termin</span></div>
  ${row([step(1, 'Anzeige Instagram / Facebook'), step(2, 'Instant-Formular'), step(3, 'Abschluss: „Anruf in Minuten“'), step(4, 'Lead sofort per Webhook')])}
  ${row([step(5, 'Anruf nach 5 Minuten', 'hot'), step(6, 'Infos + Wunschtermin'), step(7, 'Nachricht an Chef'), step(8, 'Chef bestätigt'), step(9, 'Termin im Kalender', 'done')])}
  <div class="alt">
    <span class="lbl">Nicht abgenommen</span>
    ${row([step('↺', 'SMS mit Buchungslink', 'warn'), step('↺', 'Wiederholung nach Regel 03', 'warn')])}
    <span class="lbl">Außerhalb 8–20 Uhr</span>
    ${row([step('☾', 'Sofort SMS', 'warn'), step('☀', 'Anruf am Morgen', 'warn')])}
  </div>
</div>
<div class="bcard" id="bP" style="left:1566px;width:778px">
  <div class="bhead"><span class="num">08</span><h2>Im Pilot messen</h2></div>
  <div class="kpis">
    <div class="kpi"><span>Zeit bis zum 1. Anruf</span><b>? Min</b></div>
    <div class="kpi"><span>Bewerber erreicht</span><b>? %</b></div>
    <div class="kpi"><span>Termin ausgemacht</span><b>? %</b></div>
    <div class="kpi"><span>Zum Termin erschienen</span><b>? %</b></div>
    <div class="kpi"><span>Eingestellt</span><b>?</b></div>
  </div>
  <p class="note">Erst messen, dann im Pitch nennen. Keine Zahl versprechen, die wir nicht belegt haben.</p>
</div>`;

const html = `<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>amstudio Bewerber-Bot</title>
<style>
@page { size: 594mm 420mm; margin: 0; }
:root{ --bg:#111216; --card:#1A1C22; --line:#34373F; --ink:#F2F2F0; --muted:#A3A7AF; --y:#F4C500; --yl:#FFE27A; }
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,'Liberation Sans',Inter,sans-serif;-webkit-font-smoothing:antialiased}
body{min-height:100vh;display:grid;place-items:center;overflow:auto}
#fit{width:2380px;height:1684px;transform-origin:0 0}
#stage{position:relative;width:2380px;height:1684px;overflow:hidden;
  background:radial-gradient(900px 620px at 88% 0%, rgba(244,197,0,.13), transparent 70%),
             radial-gradient(800px 560px at 50% 34%, rgba(244,197,0,.06), transparent 70%),
             radial-gradient(rgba(255,255,255,.035) 1px, transparent 1.2px) 0 0/28px 28px, var(--bg)}
@media print{ body{display:block} #fit{transform:none!important} }
.black{font-family:'Arial Black','Inter Display',Inter,sans-serif;font-weight:900}

svg#links{position:absolute;inset:0;width:2380px;height:1684px;z-index:0}
.layer{position:absolute;inset:0;z-index:1}
.layer > *{position:absolute}

.logo{left:36px;top:30px;display:flex;align-items:center;gap:14px;font:800 40px/1 'Inter Display',Inter,sans-serif;letter-spacing:-.01em}
.logo i{width:22px;height:22px;background:var(--y);transform:rotate(45deg);border-radius:4px}
.top{left:790px;top:40px;width:800px;text-align:center;font:700 14px Arial,'Liberation Sans',sans-serif;letter-spacing:.42em;color:var(--muted);text-transform:uppercase}

#glow{left:840px;top:250px;width:700px;height:560px;border-radius:50%;background:radial-gradient(closest-side, rgba(244,197,0,.22), transparent);filter:blur(8px)}
#hub{left:950px;top:378px;width:480px;height:300px;border-radius:34px;background:#16171C;border:2.5px solid var(--y);
  box-shadow:0 0 0 8px rgba(244,197,0,.08),0 30px 80px -20px rgba(0,0,0,.8),0 0 80px -10px rgba(244,197,0,.35);
  display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:24px}
#hub .e{font:700 14px Arial,'Liberation Sans',sans-serif;letter-spacing:.32em;color:var(--y);text-transform:uppercase}
#hub h1{margin:16px 0 18px;font-size:56px;line-height:1.02;text-transform:uppercase;color:var(--ink)}
#hub .t{font:700 22px Arial,'Liberation Sans',sans-serif;color:var(--muted)}
#hub .t b{color:var(--yl)}

.node{border-radius:22px;padding:20px 22px;background:linear-gradient(150deg,#26241A 0%,#1A1C22 70%);border:1.5px solid rgba(244,197,0,.6);
  box-shadow:0 24px 60px -26px rgba(244,197,0,.45)}
.node .eyebrow{display:flex;align-items:center;gap:10px;font:700 12px/1 Arial,'Liberation Sans',sans-serif;letter-spacing:.16em;color:var(--yl);text-transform:uppercase;white-space:nowrap}
.node .badge{flex:none;width:32px;height:32px;border-radius:50%;background:var(--y);color:#111216;display:grid;place-items:center;font:900 13px 'Inter Display',Inter,sans-serif;letter-spacing:0}
.node h3{margin:14px 0 8px;font:700 23px/1.15 Arial,'Liberation Sans',sans-serif;color:var(--ink);white-space:nowrap}
.node .big{font-family:'Arial Black','Inter Display',Inter,sans-serif;font-weight:900;font-size:33px;line-height:1.05;color:var(--y);white-space:nowrap}
.node .sub{margin-top:8px;font:500 14px Arial,'Liberation Sans',sans-serif;color:var(--muted);white-space:nowrap}

.chips{display:flex;flex-direction:column;gap:9px}
.chip{display:flex;align-items:center;gap:10px;height:34px;padding:0 14px;border-radius:17px;white-space:nowrap;overflow:hidden;
  font:500 15.5px/1 Arial,'Liberation Sans',sans-serif;color:#E6E6E8;background:#1A1C22;border:1.2px solid var(--line)}
.chip::before{content:"";flex:none;width:7px;height:7px;transform:rotate(45deg);background:var(--y)}
.chip.key{font-weight:700;color:#fff;border-color:rgba(244,197,0,.75);background:linear-gradient(90deg,rgba(244,197,0,.16),#1A1C22)}
.chip.plan{border-style:dashed;border-color:rgba(255,226,122,.8);color:var(--yl)}
.chip.x{color:#C9CCD2;border-color:rgba(255,120,120,.5);background:linear-gradient(90deg,rgba(255,110,110,.12),#1A1C22)}
.chip.x::before{content:"×";transform:none;width:auto;height:auto;background:none;color:#FF8C8C;font:800 18px/1 Arial,sans-serif;margin-top:-2px}

.bcard{top:1150px;height:470px;border-radius:26px;padding:26px 32px;background:#16171C;border:1.5px solid var(--line);box-shadow:0 30px 70px -30px rgba(0,0,0,.8)}
.bhead{display:flex;align-items:baseline;gap:16px;margin-bottom:22px}
.bhead .num{font:900 16px 'Inter Display',Inter,sans-serif;letter-spacing:.2em;color:var(--y)}
.bhead h2{margin:0;font-family:'Arial Black','Inter Display',Inter,sans-serif;font-weight:900;font-size:32px;text-transform:uppercase;color:var(--ink)}
.bhead .hint{font:500 15px Arial,'Liberation Sans',sans-serif;color:var(--muted)}
.flow{display:flex;align-items:center;gap:8px;flex-wrap:nowrap}
.flow + .flow{margin-top:14px}
.step{display:inline-flex;align-items:center;gap:10px;height:46px;padding:0 18px 0 8px;border-radius:12px;white-space:nowrap;
  font:600 16.5px Arial,'Liberation Sans',sans-serif;color:var(--ink);background:#1F2128;border:1.2px solid var(--line)}
.step b{display:grid;place-items:center;width:30px;height:30px;border-radius:8px;background:#2B2E36;color:var(--yl);font:900 14px 'Inter Display',Inter,sans-serif}
.step.hot{border-color:var(--y);background:linear-gradient(90deg,rgba(244,197,0,.22),#1F2128);font-weight:800}
.step.hot b,.step.done b{background:var(--y);color:#111216}
.step.done{border-color:rgba(244,197,0,.75)}
.step.warn{border-style:dashed;border-color:#4A4E58;color:#C9CCD2;height:40px;font-size:15px}
.step.warn b{background:transparent;color:var(--muted);font-size:17px}
.arr{flex:none;width:26px;height:12px;position:relative}
.arr::before{content:"";position:absolute;left:0;right:6px;top:5px;height:2px;background:var(--y);opacity:.8}
.arr::after{content:"";position:absolute;right:2px;top:1px;width:8px;height:8px;border-top:2px solid var(--y);border-right:2px solid var(--y);transform:rotate(45deg);opacity:.8}
.alt{margin-top:26px;padding-top:20px;border-top:1px dashed #3A3D45;display:grid;grid-template-columns:auto 1fr;gap:12px 18px;align-items:center}
.alt .flow + .flow{margin-top:0}
.alt .lbl{font:700 12px Arial,'Liberation Sans',sans-serif;letter-spacing:.2em;text-transform:uppercase;color:var(--muted)}

.kpis{display:flex;flex-direction:column;gap:10px}
.kpi{display:flex;justify-content:space-between;align-items:center;height:50px;padding:0 20px;border-radius:12px;background:#1F2128;border:1.2px solid var(--line)}
.kpi span{font:600 17px Arial,'Liberation Sans',sans-serif;color:#E6E6E8}
.kpi b{font-family:'Arial Black','Inter Display',Inter,sans-serif;font-weight:900;font-size:22px;color:var(--y)}
.note{margin:18px 0 0;font:500 15px/1.45 Arial,'Liberation Sans',sans-serif;color:var(--muted)}

.foot{left:36px;top:1642px;width:2308px;display:flex;justify-content:space-between;font:500 13px Arial,'Liberation Sans',sans-serif;color:#6E727B;letter-spacing:.06em;text-transform:uppercase}
</style>
</head>
<body>
<div id="fit"><div id="stage">
<svg id="links" viewBox="0 0 2380 1684" aria-hidden="true">
  <defs><filter id="glowf" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="5"/></filter></defs>
  <g id="lg"></g><g id="lm"></g><g id="ld"></g>
</svg>
<div class="layer">
  <div class="logo"><i></i>amstudio</div>
  <div class="top">Recruiting · Bewerber-Bot · Konzept</div>
  <div id="glow"></div>
  <div id="hub">
    <div class="e">Bewerber-Bot</div>
    <h1 class="black">Anruf in<br>5 Minuten.</h1>
    <div class="t">Bot fragt. <b>Chef entscheidet.</b></div>
  </div>
  ${branchHtml}
  ${bottom}
  <div class="foot"><span>Konzept · Wirkung erst im Pilot belegen · rechtliche Prüfung (AI Act, DSGVO) vor Verkauf</span><span>amstudio.ink</span></div>
</div>
</div></div>
<script>
(function(){
  const Y='#F4C500', NS='http://www.w3.org/2000/svg', stage=document.getElementById('stage'), $=id=>document.getElementById(id);
  function rel(el){const s=stage.getBoundingClientRect(),r=el.getBoundingClientRect(),k=s.width/2380;return {x:(r.left-s.left)/k,y:(r.top-s.top)/k,w:r.width/k,h:r.height/k};}
  function pt(el,side,f){f=f==null?.5:f;const r=rel(el);return side==='l'?[r.x,r.y+r.h*f]:side==='r'?[r.x+r.w,r.y+r.h*f]:side==='t'?[r.x+r.w*f,r.y]:[r.x+r.w*f,r.y+r.h];}
  function add(g,tag,a){const e=document.createElementNS(NS,tag);for(const k in a)e.setAttribute(k,a[k]);$(g).appendChild(e);}
  function h(a,b){const m=(a[0]+b[0])/2;return 'M'+a[0]+' '+a[1]+' C'+m+' '+a[1]+' '+m+' '+b[1]+' '+b[0]+' '+b[1];}
  function v(a,b){const c=b[1]-90;return 'M'+a[0]+' '+a[1]+' C'+a[0]+' '+c+' '+b[0]+' '+c+' '+b[0]+' '+b[1];}
  function main(d,end){add('lg','path',{d,stroke:Y,'stroke-width':14,fill:'none',opacity:.22,filter:'url(#glowf)'});
    add('lm','path',{d,stroke:Y,'stroke-width':4.5,fill:'none','stroke-linecap':'round'});
    add('ld','circle',{cx:end[0],cy:end[1],r:7,fill:Y,stroke:'#111216','stroke-width':3});}
  function draw(){
    const hub=$('hub');
    [['n1',.28],['n2',.5],['n3',.72]].forEach(([id,f])=>{const a=pt(hub,'l',f),b=pt($(id),'r');main(h(a,b),b);});
    [['n4',.28],['n5',.5],['n6',.72]].forEach(([id,f])=>{const a=pt(hub,'r',f),b=pt($(id),'l');main(h(a,b),b);});
    [['bF',.42,.62],['bP',.58,.35]].forEach(([id,fa,fb])=>{const a=pt(hub,'b',fa),b=pt($(id),'t',fb);main(v(a,b),b);});
    document.querySelectorAll('.chips[data-parent]').forEach(box=>{const p=$(box.dataset.parent),L=box.dataset.dir==='left';
      [...box.children].forEach(k=>{const a=pt(p,L?'l':'r'),b=pt(k,L?'r':'l');
        add('lm','path',{d:h(a,b),stroke:Y,'stroke-width':1.8,opacity:.5,fill:'none'});
        add('ld','circle',{cx:b[0],cy:b[1],r:3.2,fill:Y});});});
  }
  function fit(){const k=Math.min(innerWidth/2380,innerHeight/1684,1);$('fit').style.transform='scale('+k+')';
    document.body.style.width=(2380*k)+'px';document.body.style.height=(1684*k)+'px';document.body.style.margin='0 auto';}
  (document.fonts&&document.fonts.ready?document.fonts.ready:Promise.resolve()).then(()=>{fit();draw();window.__ready=true;});
  addEventListener('resize',fit);
})();
</script>
</body>
</html>`;

fs.writeFileSync(path.join(OUT, `${NAME}.html`), html);
console.log('HTML geschrieben');
if (process.argv.includes('--no-render')) process.exit(0);

(async () => {
  const { chromium } = require('playwright');
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 2380, height: 1684 }, deviceScaleFactor: 2 });
  await page.goto('file://' + path.join(OUT, `${NAME}.html`));
  await page.waitForFunction(() => window.__ready === true);
  const issues = await page.evaluate(() => {
    const out = [];
    document.querySelectorAll('.chip,.node h3,.node .big,.node .sub,.step,.kpi').forEach(e => {
      if (e.scrollWidth > e.clientWidth + 1) out.push('Überlauf: ' + e.textContent.trim());
    });
    document.querySelectorAll('.bcard').forEach(c => {
      if (c.scrollHeight > c.clientHeight + 1 || c.scrollWidth > c.clientWidth + 1) out.push('Karte zu voll: ' + c.id);
    });
    const boxes = [...document.querySelectorAll('.node,.chips,.bcard,#hub,.logo,.top,.foot')].map(e => ({ n: e.id || e.className, r: e.getBoundingClientRect() }));
    for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i].r, b = boxes[j].r;
      if (a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom) out.push('Überlappung: ' + boxes[i].n + ' / ' + boxes[j].n);
    }
    return out;
  });
  console.log(issues.length ? issues.join('\n') : 'Keine Überläufe/Überlappungen gefunden');
  await page.screenshot({ path: path.join(OUT, `${NAME}.png`) });
  await page.emulateMedia({ media: 'print' });
  await page.pdf({ path: path.join(OUT, `${NAME}-A2.pdf`), width: '594mm', height: '420mm', printBackground: true, scale: 0.9426, pageRanges: '1' });
  await browser.close();
  console.log('PNG + PDF gerendert');
})();
