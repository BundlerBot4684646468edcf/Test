// Generates am-studio-mindmap.html (A2 landscape strategy board) and renders PNG + PDF.
// Usage: node mindmap/build.js
const fs = require('fs');
const path = require('path');

const OUT = __dirname;
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const chip = (label, cls = '') => `<div class="chip ${cls}">${esc(label)}</div>`;
const chips = (id, parent, dir, items, style, cls = '') =>
  `<div class="chips ${cls}" id="${id}" data-parent="${parent}" data-dir="${dir}" style="${style}">${items
    .map(i => (Array.isArray(i) ? chip(i[0], i[1]) : chip(i)))
    .join('')}</div>`;

const node = ({ id, color, letter, eyebrow, title, price, priceCls = '', sub, tag, style }) => `
<div class="node ${color}" id="${id}" style="${style}">
  <div class="eyebrow"><span class="badge">${letter}</span>${esc(eyebrow)}</div>
  <h3>${title.map(esc).join('<br>')}</h3>
  <div class="price ${priceCls}">${price}</div>
  ${sub ? `<div class="sub">${esc(sub)}</div>` : ''}
  ${tag ? `<div class="tag">${esc(tag)}</div>` : ''}
</div>`;

// ---------- 01 WEBDESIGN (links) ----------
const web = [
  {
    id: 'wA', cy: 222, letter: 'A', eyebrow: 'Website', title: ['Normale Website'],
    price: 'Nach Umfang', priceCls: 'web', sub: 'Einmalprojekt · individuell',
    items: ['Modern & mobiloptimiert', 'DE / IT / EN nach Bedarf', 'Leistungs- & Unternehmensseiten',
      'Kontakt, Anruf & WhatsApp', 'Technische SEO-Basis', ['Optional: Rechner, Projekte, KI', 'opt']],
  },
  {
    id: 'wB', cy: 540, letter: 'B', eyebrow: 'Website + Local SEO', title: ['Website +', 'Local SEO'],
    price: 'Nach Umfang', priceCls: 'web', sub: 'Einmalprojekt · individuell',
    items: [['Alles aus Website A', 'inherit'], 'Service × Location Strategie', 'Leistungs- & Ortsseiten',
      'Lokale Keyword-Optimierung', 'Interne Verlinkung', 'Conversion-Optimierung'],
  },
  {
    id: 'wC', cy: 858, letter: 'C', eyebrow: 'Betreuung', title: ['SEO/GEO', 'Visibility Care'],
    price: 'Monatlich', sub: 'Preis nach Umfang',
    items: [['Nur für AM-Studio-Websites', 'key'], 'Laufende SEO & GEO', 'Google-Unternehmensprofil',
      'Content & Technik', 'Reporting & Sichtbarkeit', ['Kein Standalone-SEO', 'x']],
  },
];

let webHtml = '';
for (const w of web) {
  webHtml += node({ ...w, color: 'blue', style: `left:420px;top:${w.cy}px;width:272px;transform:translateY(-50%)` });
  webHtml += chips(`${w.id}c`, w.id, 'left', w.items, `left:36px;top:${w.cy}px;width:318px;transform:translateY(-50%)`, 'blue');
}

// ---------- 02 RECRUITING (rechts) ----------
const pilot = node({
  id: 'rP', color: 'pilot', letter: 'A', eyebrow: 'Pilot', title: ['Recruiting Pilot'],
  price: '4.900–5.900&nbsp;€', priceCls: 'mid', sub: 'Referenz-Paket', tag: 'Preisvorschlag',
  style: 'left:1600px;top:212px;width:290px;transform:translateY(-50%)',
});
const pilotChips = chips('rPc', 'rP', 'right', [
  ['Nur erste 2–3 Kunden', 'key'], 'Ziel: Recruiting-Referenzen', 'Videodreh, Schnitt, Landingpage',
  'Meta Ads, Formular, Dashboard', 'Reminder & Kampagnenbetreuung',
], 'left:1926px;top:212px;width:414px;transform:translateY(-50%)', 'pilot');

const ladder = [
  {
    id: 'rB', color: 'dwy', x: 1600, top: 482, letter: 'B', eyebrow: 'DWY', title: ['Recruiting', 'Growth'],
    price: '5.900&nbsp;€', sub: '90 Tage · pro Stelle', tag: 'Preisvorschlag',
    items: ['Videodreh + 2–3 Creatives', 'Landingpage mit Video', 'Facebook & Instagram Ads',
      ['Ads-Budget 700 €/Mon.*', 'plan'], 'Formular & Dashboard', 'Automatische Reminder',
      ['Kunde: Kontakt & Interviews', 'key tight']],
  },
  {
    id: 'rC', color: 'dfy', x: 1858, top: 452, letter: 'C', eyebrow: 'DFY · 360', title: ['Recruiting', 'Partner 360'],
    price: '8.900&nbsp;€', sub: '90 Tage · pro Stelle', tag: 'Preisvorschlag',
    items: [['Alles aus DWY', 'inherit'], 'Telefonische Vorauswahl', 'Bewerber-Follow-ups',
      'Check Mindestqualifikation', 'Kalenderintegration', 'Interviews organisiert',
      'Wöchentliches Reporting', ['Kunde: nur Finalgespräche', 'key'], ['+30 Tage ohne Honorar**', 'plan']],
  },
  {
    id: 'rD', color: 'prem', x: 2116, top: 422, letter: 'D', eyebrow: 'Premium', title: ['Recruiting', 'Authority'],
    price: '10.900&nbsp;€', sub: '90 Tage · pro Stelle', tag: 'Preisvorschlag',
    items: [['Alles aus DFY', 'inherit'], 'Employer-Branding-Strategie', 'Arbeitgeber-Imagefilm',
      '3 Mitarbeiter-Testimonials', '6 Social-Media-Reels', '15 professionelle Fotos',
      'Erweiterte Karriereseite', 'Interview- & Onboarding-Kit'],
  },
];
let ladderHtml = '';
for (const l of ladder) {
  ladderHtml += `<div class="col" style="left:${l.x}px;top:${l.top}px;width:224px">
    ${node({ ...l, style: 'position:relative' })}
    ${chips(`${l.id}c`, l.id, 'spine', l.items, '', `${l.color} small`)}
  </div>`;
}

// ---------- Untere Karten ----------
const icon = {
  shield: '<path d="M12 3l7 3v6c0 4.5-3 7.7-7 9-4-1.3-7-4.5-7-9V6l7-3z"/><path d="M8.5 12.2l2.4 2.4 4.6-4.8"/>',
  bolt: '<path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z"/>',
  up: '<path d="M7 17L17 7"/><path d="M8 7h9v9"/>',
  team: '<circle cx="9" cy="8" r="3.2"/><circle cx="17" cy="9.5" r="2.4"/><path d="M3.5 19c.6-3.3 3-5 5.5-5s4.9 1.7 5.5 5"/><path d="M15 14.4c2.6-.4 4.8 1 5.4 4.1"/>',
};
const head = (num, title, ic) => `<div class="bhead"><span class="ico"><svg viewBox="0 0 24 24">${icon[ic]}</svg></span><div><div class="num">${num}</div><h2>${title}</h2></div></div>`;
const grp = (label, inner, cls = '') => `<div class="grp ${cls}"><div class="lbl">${label}</div><div class="wrap">${inner}</div></div>`;
const flow = (steps, cont = '') => `<div class="flow">${cont === 'start' ? '<span class="arr ret"></span>' : ''}${steps.map(s => `<span class="step">${esc(s)}</span>`).join('<span class="arr"></span>')}${cont === 'end' ? '<span class="arr"></span>' : ''}</div>`;

const bottom = `
<div class="bcard gold" id="bG" style="left:36px">
  ${head('03', 'Garantien', 'shield')}
  ${grp('Webdesign', chip('Funktions- & Qualitätsabnahme') + chip('Behebung vereinbarter Mängel'))}
  ${grp('Recruiting', chip('90 Tage Betreuung') + chip('DFY: +30 Tage ohne Honorar**') + chip('Budget Verlängerung separat', 'plan'))}
  ${grp('Nicht garantiert', chip('Rankings', 'x') + chip('Bewerberzahlen', 'x') + chip('Einstellungen', 'x'), 'no')}
</div>
<div class="bcard gray" id="bA" style="left:620px">
  ${head('04', 'Automationen', 'bolt')}
  ${grp('Webdesign-Funnel', flow(['Google', 'Website', 'Anfrage', 'Kontakt']))}
  ${grp('Recruiting-Funnel', flow(['Meta Ads', 'Landingpage', 'Formular', 'Dashboard'], 'end') + flow(['Qualifizierung', 'Kalender', 'Interview'], 'start'))}
  ${grp('Technologien', ['CRM', 'Kalender', 'Reminder', 'Reporting', 'Bewerbermanagement'].map(t => chip(t, 'tech')).join(''))}
</div>
<div class="bcard teal" id="bU" style="left:1204px">
  ${head('05', 'Upsells', 'up')}
  ${grp('Cross-Sell', `<div class="xsell"><span>Website-Kunde</span><i></i><b>Recruiting</b></div><div class="xsell"><span>Recruiting-Kunde</span><i></i><b>Webdesign</b></div>`)}
  ${grp('Ausbau', chip('SEO/GEO nur eigene Websites', 'key') + chip('Weitere Recruiting-Stellen') + chip('Employer Branding') + chip('Zusätzliche Videoproduktion') + chip('Laufende Betreuung & Content'))}
</div>
<div class="bcard violet" id="bM" style="left:1788px">
  ${head('06', 'Geschäftsmodell', 'team')}
  ${grp('Team', `<div class="role"><b>Gründer 1</b><span>Fulfillment · Web · Ops</span></div><div class="role"><b>Gründer 2</b><span>Sales · Closing · Follow-ups</span></div><div class="role"><b>Setter</b><span>Calls & Termine</span></div>`)}
  ${grp('Setter-Ziele', `<div class="kpi"><b>200</b><span>Calls<br>pro Tag</span></div><div class="kpi"><b>15</b><span>qualifizierte Termine<br>pro Woche</span></div>`)}
  ${grp('Umsatzlogik', chip('Web: Einmalumsatz + Betreuung') + chip('Recruiting: 90 Tage + Folgeaufträge'))}
</div>`;

const html = `<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AM STUDIO Strategie-Board</title>
<style>
@page { size: 594mm 420mm; margin: 0; }
:root{
  --bg:#0B1220; --card:#0F1829; --ink:#EAF0F8; --muted:#8E9BB3; --dim:#5D6A82;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,sans-serif;-webkit-font-smoothing:antialiased}
body{min-height:100vh;display:grid;place-items:center;overflow:auto}
#fit{width:2380px;height:1684px;transform-origin:0 0}
#stage{position:relative;width:2380px;height:1684px;overflow:hidden;
  background:
    radial-gradient(900px 600px at 50% 32%, rgba(79,140,255,.10), transparent 70%),
    radial-gradient(700px 500px at 88% 40%, rgba(91,212,127,.06), transparent 70%),
    radial-gradient(700px 500px at 10% 40%, rgba(79,140,255,.06), transparent 70%),
    radial-gradient(rgba(255,255,255,.045) 1px, transparent 1.2px) 0 0/28px 28px,
    var(--bg);}
@media print{ body{display:block} #fit{transform:none!important} }

.blue{--c:#4F8CFF;--l:#A9C8FF}
.pilot{--c:#FF7A45;--l:#FFB494}
.dwy{--c:#FFA13D;--l:#FFCB8C}
.dfy{--c:#5BD47F;--l:#A8F0BF}
.prem{--c:#20C997;--l:#8DF0D3}
.gold{--c:#F2C14E;--l:#FFE29A}
.gray{--c:#9AA8BF;--l:#D3DCEA}
.teal{--c:#22D3C5;--l:#93F3EA}
.violet{--c:#A684FF;--l:#D2C1FF}

svg#links{position:absolute;inset:0;width:2380px;height:1684px;z-index:0;overflow:visible}
.layer{position:absolute;inset:0;z-index:1}
.layer > *{position:absolute}

/* Kopf */
.board{left:790px;top:34px;width:800px;text-align:center}
.board .eye{font:700 13px/1 Inter;letter-spacing:.42em;color:var(--muted);text-transform:uppercase}
.legend{display:flex;justify-content:center;gap:26px;margin-top:20px}
.legend span{display:flex;align-items:center;gap:9px;font:600 13px Inter;color:#B9C4D6;letter-spacing:.04em}
.legend i{width:11px;height:11px;border-radius:50%;background:var(--c);box-shadow:0 0 10px var(--c)}
.legend i.split{background:linear-gradient(90deg,#FF7A45 50%,#5BD47F 50%);box-shadow:0 0 10px #FFA13D}

.zone{top:34px}
.zone .n{font:800 15px 'Inter Display',Inter;color:var(--c);letter-spacing:.2em}
.zone h1{margin:6px 0 0;font:800 34px/1 'Inter Display',Inter;letter-spacing:.06em;text-transform:uppercase;color:#fff}
.zone .zs{margin-top:9px;font:500 14px Inter;color:var(--muted)}
.zone.rec h1 span{background:linear-gradient(90deg,#FF7A45,#FFA13D 45%,#5BD47F);-webkit-background-clip:text;background-clip:text;color:transparent}
.zone .warn{display:inline-block;margin-left:14px;vertical-align:6px;font:700 11px Inter;letter-spacing:.14em;text-transform:uppercase;color:#FFE29A;border:1px dashed rgba(255,226,154,.7);padding:6px 11px;border-radius:99px}

/* Zentrum */
#hubglow{left:840px;top:250px;width:700px;height:580px;border-radius:50%;
  background:radial-gradient(closest-side, rgba(79,140,255,.28), rgba(34,211,197,.10) 55%, transparent);filter:blur(10px)}
#hub{left:975px;top:398px;width:430px;height:284px;border-radius:36px;padding:2px;
  background:conic-gradient(from 200deg,#4F8CFF,#22D3C5,#5BD47F,#FFA13D,#FF7A45,#F2C14E,#A684FF,#4F8CFF);
  box-shadow:0 30px 80px -20px rgba(0,0,0,.7),0 0 70px -10px rgba(79,140,255,.45)}
#hub .in{width:100%;height:100%;border-radius:34px;background:radial-gradient(120% 90% at 50% 0%, #17233D, #0C1424 70%);
  display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center}
#hub .e{font:700 12px Inter;letter-spacing:.34em;color:var(--muted);text-transform:uppercase}
#hub .logo{font:900 68px/1 'Inter Display',Inter;letter-spacing:.02em;margin:16px 0 18px;color:#fff}
#hub .logo span{background:linear-gradient(90deg,#9CC2FF,#fff 50%,#A8F0BF);-webkit-background-clip:text;background-clip:text;color:transparent}
#hub .rule{width:64px;height:3px;border-radius:3px;background:linear-gradient(90deg,#4F8CFF,#FFA13D,#22D3C5);margin-bottom:18px}
#hub .tg{font:700 22px/1.4 'Inter Display',Inter;color:#fff}
#hub .tg .b{color:#A9C8FF} #hub .tg .o{color:#FFCB8C} #hub .tg .t{color:#93F3EA}

/* Angebots-Knoten */
.node{border-radius:24px;padding:20px 22px 20px;
  background:linear-gradient(150deg,color-mix(in srgb,var(--c) 30%,#0F1829) 0%,#0F1829 72%);
  border:1.5px solid color-mix(in srgb,var(--c) 60%,transparent);
  box-shadow:0 24px 60px -24px color-mix(in srgb,var(--c) 70%,transparent),0 0 50px -14px color-mix(in srgb,var(--c) 45%,transparent)}
.node .eyebrow{display:flex;align-items:center;gap:10px;font:700 12px/1 Inter;letter-spacing:.16em;color:var(--l);text-transform:uppercase;white-space:nowrap}
.node .badge{flex:none;width:30px;height:30px;border-radius:50%;background:var(--c);color:#0B1220;display:grid;place-items:center;font:900 15px 'Inter Display',Inter;letter-spacing:0;box-shadow:0 0 16px color-mix(in srgb,var(--c) 70%,transparent)}
.node h3{margin:14px 0 10px;font:700 22px/1.15 'Inter Display',Inter;color:#fff;white-space:nowrap}
.node .price{font:900 40px/1 'Inter Display',Inter;letter-spacing:-.02em;color:var(--l);white-space:nowrap}
.node .price.mid{font-size:33px}
.node .price.web{font-size:34px}
.node .sub{margin-top:9px;font:500 13.5px Inter;color:#A4B2C8;white-space:nowrap}
.node .tag{display:inline-block;margin-top:12px;font:700 10.5px/1 Inter;letter-spacing:.14em;text-transform:uppercase;padding:6px 10px;border-radius:99px;border:1px dashed color-mix(in srgb,var(--l) 75%,transparent);color:var(--l)}

/* Unterknoten */
.chips{display:flex;flex-direction:column;gap:10px}
.chip{display:flex;align-items:center;gap:10px;height:34px;padding:0 16px 0 14px;border-radius:17px;white-space:nowrap;overflow:hidden;
  font:500 15px/1 Inter;color:#E3EAF5;
  background:linear-gradient(90deg,color-mix(in srgb,var(--c) 17%,#0F1829),color-mix(in srgb,var(--c) 6%,#0F1829));
  border:1.2px solid color-mix(in srgb,var(--c) 42%,transparent)}
.chip::before{content:"";flex:none;width:7px;height:7px;border-radius:50%;background:var(--c);box-shadow:0 0 8px var(--c)}
.chip.key{font-weight:700;color:#fff}
.chip.inherit{border-style:dashed;font-weight:700;color:var(--l);background:#0F1829}
.chip.inherit::before{background:transparent;box-shadow:none;border:1.5px solid var(--l);width:6px;height:6px}
.chip.opt{color:#B9C6DA;font-style:italic}
.chip.plan{border-style:dashed;border-color:rgba(255,226,154,.75);color:#FFE29A}
.chip.plan::before{background:#F2C14E;box-shadow:0 0 8px #F2C14E}
.chip.x{color:#AFB9C9;border-color:rgba(255,120,120,.45);background:linear-gradient(90deg,rgba(255,110,110,.12),#0F1829)}
.chip.x::before{content:"×";width:auto;height:auto;background:none;box-shadow:none;color:#FF8C8C;font:800 17px/1 Inter;margin-top:-2px}
.chips.small{gap:7px;margin-top:22px}
.chips.small .chip{height:31px;font-size:13.5px;padding:0 10px 0 11px;gap:8px;letter-spacing:-.005em}
.chips.small .chip.tight{letter-spacing:-.02em}

.col{display:flex;flex-direction:column}
.col .node{padding:18px 20px}
.col .node h3{font-size:21px}

/* Wertleiter */
.ladder{left:1600px;top:364px;width:740px}
.ladder .lab{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px}
.ladder .lab b{font:800 13px Inter;letter-spacing:.32em;color:#FFE29A}
.ladder .lab span{font:600 13px Inter;letter-spacing:.12em;color:#B9C4D6}
.ladder .bar{position:relative;height:6px;border-radius:6px;background:linear-gradient(90deg,#FFA13D,#5BD47F 60%,#20C997);box-shadow:0 0 18px rgba(91,212,127,.35)}
.ladder .bar::after{content:"";position:absolute;right:-4px;top:50%;width:14px;height:14px;transform:translateY(-50%) rotate(45deg);border-top:3px solid #20C997;border-right:3px solid #20C997}

/* Untere Karten */
.bcard{top:1150px;width:556px;height:482px;border-radius:28px;padding:26px 30px;
  background:linear-gradient(170deg,color-mix(in srgb,var(--c) 13%,#0F1829),#0E1626 55%);
  border:1.5px solid color-mix(in srgb,var(--c) 45%,transparent);
  box-shadow:0 30px 70px -30px rgba(0,0,0,.8),0 0 50px -20px color-mix(in srgb,var(--c) 50%,transparent)}
.bhead{display:flex;align-items:center;gap:16px;margin-bottom:20px}
.bhead .ico{width:52px;height:52px;border-radius:16px;display:grid;place-items:center;background:color-mix(in srgb,var(--c) 22%,#0F1829);border:1.5px solid color-mix(in srgb,var(--c) 60%,transparent)}
.bhead svg{width:28px;height:28px;fill:none;stroke:var(--l);stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.bhead .num{font:800 13px Inter;letter-spacing:.24em;color:var(--c)}
.bhead h2{margin:4px 0 0;font:800 30px/1 'Inter Display',Inter;letter-spacing:.05em;text-transform:uppercase;color:#fff}
.grp{margin-top:18px}
.grp .lbl{font:700 11.5px Inter;letter-spacing:.22em;text-transform:uppercase;color:var(--l);opacity:.85;margin-bottom:10px}
.grp .wrap{display:flex;flex-wrap:wrap;gap:9px}
.bcard .chip{height:33px;font-size:14.5px}
.chip.tech{height:30px;font-size:13.5px;font-weight:600}

.flow{display:flex;flex-wrap:nowrap;align-items:center;column-gap:6px}
.flow + .flow{margin-top:10px}
.step{height:34px;display:inline-flex;align-items:center;padding:0 13px;border-radius:10px;font:600 14.5px Inter;color:#EAF0F8;white-space:nowrap;
  background:color-mix(in srgb,var(--c) 14%,#0F1829);border:1.2px solid color-mix(in srgb,var(--c) 45%,transparent)}
.arr{width:20px;height:10px;position:relative}
.arr::before{content:"";position:absolute;left:1px;right:5px;top:4px;height:2px;background:var(--l);opacity:.75}
.arr::after{content:"";position:absolute;right:2px;top:1px;width:6px;height:6px;border-top:2px solid var(--l);border-right:2px solid var(--l);transform:rotate(45deg);opacity:.75}

.xsell{display:flex;align-items:center;gap:12px;width:100%;height:44px;padding:0 18px;border-radius:14px;font:600 16px Inter;
  background:linear-gradient(90deg,color-mix(in srgb,var(--c) 20%,#0F1829),#0F1829);border:1.3px solid color-mix(in srgb,var(--c) 55%,transparent)}
.xsell span{color:#C9D4E4}
.xsell b{color:var(--l);font-weight:800}
.xsell i{width:34px;height:2px;background:var(--l);position:relative}
.xsell i::after{content:"";position:absolute;right:0;top:-4px;width:8px;height:8px;border-top:2px solid var(--l);border-right:2px solid var(--l);transform:rotate(45deg)}

.role{flex:1;min-width:0;padding:12px 14px;border-radius:14px;background:color-mix(in srgb,var(--c) 14%,#0F1829);border:1.2px solid color-mix(in srgb,var(--c) 45%,transparent)}
.role b{display:block;font:800 16px 'Inter Display',Inter;color:#fff}
.role span{display:block;margin-top:5px;font:500 12.5px/1.35 Inter;color:#C2CCDC}
.kpi{flex:1;display:flex;align-items:center;gap:14px;padding:10px 16px;border-radius:14px;border:1.2px dashed color-mix(in srgb,var(--c) 55%,transparent)}
.kpi b{font:900 46px/1 'Inter Display',Inter;color:var(--l);letter-spacing:-.02em}
.kpi span{font:600 13px/1.3 Inter;color:#C2CCDC}

.foot{left:36px;top:1648px;width:2308px;display:flex;justify-content:space-between;align-items:center;font:500 13px Inter;color:var(--dim);letter-spacing:.02em}
.foot b{color:#C9B37A;font-weight:700}
.foot .sep{opacity:.5;margin:0 10px}
</style>
</head>
<body>
<div id="fit"><div id="stage">
<svg id="links" viewBox="0 0 2380 1684" aria-hidden="true">
  <defs>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="5"/></filter>
  </defs>
  <g id="lg"></g><g id="lm"></g><g id="ld"></g>
</svg>
<div class="layer">
  <div class="board">
    <div class="eye">Strategie-Board · Angebote · System · Wachstum</div>
    <div class="legend">
      <span class="blue"><i></i>Webdesign</span>
      <span><i class="split"></i>Recruiting</span>
      <span class="gold"><i></i>Garantien</span>
      <span class="gray"><i></i>Automationen</span>
      <span class="teal"><i></i>Upsells</span>
      <span class="violet"><i></i>Geschäftsmodell</span>
    </div>
  </div>

  <div class="zone blue" style="left:36px"><div class="n">01</div><h1>Webdesign</h1></div>
  <div class="zone rec dwy" style="left:1600px"><div class="n">02</div><h1><span>Recruiting</span><span class="warn">Preisvorschläge</span></h1></div>

  <div id="hubglow"></div>
  <div id="hub"><div class="in">
    <div class="e">Webdesign · Recruiting · Südtirol</div>
    <div class="logo"><span>AM STUDIO</span></div>
    <div class="rule"></div>
    <div class="tg"><span class="b">Mehr Kunden.</span> <span class="o">Mehr Mitarbeiter.</span><br><span class="t">Weniger Aufwand.</span></div>
  </div></div>

  ${webHtml}
  ${pilot}
  ${pilotChips}
  <div class="ladder"><div class="lab"><b>WERTSTEIGERUNG</b><span>DWY &nbsp;→&nbsp; DFY &nbsp;→&nbsp; PREMIUM</span></div><div class="bar"></div></div>
  ${ladderHtml}
  ${bottom}

  <div class="foot">
    <span><b>Hinweise</b><span class="sep">|</span>Recruiting-Preise = Preisvorschläge, keine bestätigte Preisliste<span class="sep">|</span>* Werbebudget 700 €/Monat = Planannahme – nur inklusive bei bestätigter Kalkulation<span class="sep">|</span>** Bei ausbleibender Einstellung, gemäß Vertrag; Werbebudget der Verlängerung gesondert</span>
    <span>AM STUDIO · Südtirol</span>
  </div>
</div>
</div></div>

<script>
(function(){
  const COL={blue:'#4F8CFF',pilot:'#FF7A45',dwy:'#FFA13D',dfy:'#5BD47F',prem:'#20C997',gold:'#F2C14E',gray:'#9AA8BF',teal:'#22D3C5',violet:'#A684FF'};
  const NS='http://www.w3.org/2000/svg';
  const stage=document.getElementById('stage');
  const $=id=>document.getElementById(id);
  function rel(el){const s=stage.getBoundingClientRect(),r=el.getBoundingClientRect(),k=s.width/2380;
    return {x:(r.left-s.left)/k,y:(r.top-s.top)/k,w:r.width/k,h:r.height/k};}
  function pt(el,side,f){f=f==null?.5:f;const r=rel(el);
    return side==='l'?[r.x,r.y+r.h*f]:side==='r'?[r.x+r.w,r.y+r.h*f]:side==='t'?[r.x+r.w*f,r.y]:side==='b'?[r.x+r.w*f,r.y+r.h]:[r.x+r.w/2,r.y+r.h/2];}
  function add(g,tag,attrs){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);$(g).appendChild(e);return e;}
  function hPath(a,b){const m=(a[0]+b[0])/2;return 'M'+a[0]+' '+a[1]+' C'+m+' '+a[1]+' '+m+' '+b[1]+' '+b[0]+' '+b[1];}
  function dPath(a,b){const c=b[1]-95;return 'M'+a[0]+' '+a[1]+' C'+a[0]+' '+c+' '+b[0]+' '+c+' '+b[0]+' '+b[1];}
  function main(d,color,end){
    add('lg','path',{d,stroke:color,'stroke-width':14,fill:'none',opacity:.28,filter:'url(#glow)','stroke-linecap':'round'});
    add('lm','path',{d,stroke:color,'stroke-width':5,fill:'none','stroke-linecap':'round'});
    add('ld','circle',{cx:end[0],cy:end[1],r:7,fill:color,stroke:'#0B1220','stroke-width':3});
  }
  function draw(){
    ['lg','lm','ld'].forEach(g=>$(g).innerHTML='');
    const hub=$('hub');
    // Zentrum -> Webdesign
    [['wA',.3],['wB',.5],['wC',.7]].forEach(([id,f])=>{const a=pt(hub,'l',f),b=pt($(id),'r');main(hPath(a,b),COL.blue,b);});
    // Zentrum -> Recruiting
    [['rP',.36,COL.pilot],['rB',.62,COL.dwy]].forEach(([id,f,c])=>{const a=pt(hub,'r',f),b=pt($(id),'l');main(hPath(a,b),c,b);});
    // Zentrum -> untere Bereiche
    [['bG',.3,.8,COL.gold],['bA',.43,.5,COL.gray],['bU',.57,.5,COL.teal],['bM',.7,.2,COL.violet]].forEach(([id,fa,fb,c])=>{
      const a=pt(hub,'b',fa),b=pt($(id),'t',fb);main(dPath(a,b),c,b);});
    // Knoten -> Unterknoten
    document.querySelectorAll('.chips[data-parent]').forEach(box=>{
      const p=$(box.dataset.parent),dir=box.dataset.dir,cls=[...box.classList].find(c=>COL[c]),col=COL[cls];
      const kids=[...box.children];
      if(dir==='spine'){
        const a=pt(p,'b'),last=pt(kids[kids.length-1],'c');
        add('lm','path',{d:'M'+a[0]+' '+a[1]+' L'+a[0]+' '+last[1],stroke:col,'stroke-width':2.5,opacity:.7,fill:'none'});
        return;
      }
      kids.forEach(k=>{const a=dir==='left'?pt(p,'l'):pt(p,'r'),b=dir==='left'?pt(k,'r'):pt(k,'l');
        add('lm','path',{d:hPath(a,b),stroke:col,'stroke-width':2,opacity:.6,fill:'none'});
        add('ld','circle',{cx:b[0],cy:b[1],r:3.2,fill:col});});
    });
    // Wertleiter-Pfeile DWY -> DFY -> Premium
    [['rB','rC',COL.dfy],['rC','rD',COL.prem]].forEach(([x,y,c])=>{
      const r1=rel($(x)),r2=rel($(y));const cx=(r1.x+r1.w+r2.x)/2,cy=(r1.y+r1.h/2+r2.y+r2.h/2)/2;
      add('ld','circle',{cx,cy,r:14,fill:'#0B1220',stroke:c,'stroke-width':2.5});
      add('ld','path',{d:'M'+(cx-4)+' '+(cy-6)+' L'+(cx+3)+' '+cy+' L'+(cx-4)+' '+(cy+6),stroke:c,'stroke-width':2.8,fill:'none','stroke-linecap':'round','stroke-linejoin':'round'});
    });
  }
  function fit(){
    const k=Math.min(innerWidth/2380,innerHeight/1684,1);
    const f=$('fit');f.style.transform='scale('+k+')';
    document.body.style.width=(2380*k)+'px';document.body.style.height=(1684*k)+'px';document.body.style.margin='0 auto';
  }
  function go(){fit();draw();window.__ready=true;}
  (document.fonts&&document.fonts.ready?document.fonts.ready:Promise.resolve()).then(go);
  addEventListener('resize',fit);
})();
</script>
</body>
</html>`;

fs.writeFileSync(path.join(OUT, 'am-studio-mindmap.html'), html);
console.log('HTML geschrieben');

if (process.argv.includes('--no-render')) process.exit(0);

(async () => {
  const { chromium } = require('playwright');
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 2380, height: 1684 }, deviceScaleFactor: 2 });
  await page.goto('file://' + path.join(OUT, 'am-studio-mindmap.html'));
  await page.waitForFunction(() => window.__ready === true);

  // Qualitätsprüfung: abgeschnittene Texte und überlappende Elemente
  const report = await page.evaluate(() => {
    const issues = [];
    document.querySelectorAll('.chip,.node h3,.node .price,.node .sub,.step,.role span,.xsell').forEach(e => {
      if (e.scrollWidth > e.clientWidth + 1) issues.push('Überlauf: ' + e.textContent.trim());
    });
    const boxes = [...document.querySelectorAll('.node,.chips,.bcard,#hub,.ladder,.zone,.board,.foot')].map(e => ({
      n: e.id || e.className, r: e.getBoundingClientRect(),
    }));
    for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i].r, b = boxes[j].r;
      if (a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom) {
        const nested = boxes[i].n.includes('col') || boxes[j].n.includes('col');
        if (!nested) issues.push('Überlappung: ' + boxes[i].n + ' / ' + boxes[j].n);
      }
    }
    const bcard = document.querySelectorAll('.bcard');
    bcard.forEach(c => { if (c.scrollHeight > c.clientHeight + 1) issues.push('Karte zu voll: ' + c.id); });
    return issues;
  });
  console.log(report.length ? report.join('\n') : 'Keine Überläufe/Überlappungen gefunden');

  await page.screenshot({ path: path.join(OUT, 'am-studio-mindmap.png') });
  await page.emulateMedia({ media: 'print' });
  await page.pdf({ path: path.join(OUT, 'am-studio-mindmap-A2.pdf'), width: '594mm', height: '420mm', printBackground: true, scale: 0.9426, pageRanges: '1' });
  await browser.close();
  console.log('PNG + PDF gerendert');
})();
