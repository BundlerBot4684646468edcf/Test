/* ============================================================
   AM Studio — interactivity
   ============================================================ */
(function () {
  'use strict';

  /* ---------- Navbar scroll + burger ---------- */
  const nav = document.getElementById('nav');
  const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 20);
  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });

  const burger = document.getElementById('burger');
  const links = document.querySelector('.nav-links');
  burger.addEventListener('click', () => links.classList.toggle('open'));
  links.addEventListener('click', (e) => { if (e.target.tagName === 'A') links.classList.remove('open'); });

  /* ---------- Reveal on scroll ---------- */
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); } });
  }, { threshold: 0.12 });
  document.querySelectorAll('.reveal, .card, .step, .stat-big').forEach((el) => {
    el.classList.add('reveal');
    io.observe(el);
  });

  /* ---------- Count-up stats ---------- */
  const counters = document.querySelectorAll('[data-count]');
  const cObs = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (!en.isIntersecting) return;
      const el = en.target, target = +el.dataset.count;
      let cur = 0;
      const step = Math.max(1, Math.round(target / 38));
      const tick = () => { cur += step; if (cur >= target) { el.textContent = target; } else { el.textContent = cur; requestAnimationFrame(tick); } };
      tick();
      cObs.unobserve(el);
    });
  }, { threshold: 0.5 });
  counters.forEach((c) => cObs.observe(c));

  /* ---------- Year ---------- */
  document.getElementById('year').textContent = new Date().getFullYear();

  /* ---------- Language toggle (DE / IT) ---------- */
  const i18n = {
    de: {
      'nav.sol': 'Lösungen', 'nav.demo': 'Live-Demo', 'nav.video': 'Video', 'nav.proc': 'Ablauf', 'nav.contact': 'Kontakt',
      'nav.cta': 'Jetzt testen',
      'hero.badge': 'KI-Agentur · Südtirol · Alto Adige',
      'hero.sub': 'Wir bauen Chatbots & Telefon-Agenten, die rund um die Uhr Termine annehmen, SMS-Bestätigungen mit Kalender senden, zurückrufen und automatisch Bewertungen sammeln — damit dein Team sich auf die Gäste konzentriert.',
      'hero.cta1': 'Live-Demo starten', 'hero.cta2': 'Beratung buchen',
      'demo.sub': 'Tippe unten in den Chat (oder nutz die Buttons). Der Assistent bucht, sendet die SMS mit Kalender, fragt eine Bewertung an und ruft zurück — genau wie bei einem echten Gast.',
      'chat.input': 'Nachricht schreiben…', 'chat.online': 'Online · antwortet sofort',
      'reset': '↻ Demo zurücksetzen',
      'cta.btn': 'Erstgespräch buchen',
    },
    it: {
      'nav.sol': 'Soluzioni', 'nav.demo': 'Demo dal vivo', 'nav.video': 'Video', 'nav.proc': 'Come funziona', 'nav.contact': 'Contatti',
      'nav.cta': 'Prova ora',
      'hero.badge': 'Agenzia AI · Südtirol · Alto Adige',
      'hero.sub': 'Costruiamo chatbot e agenti telefonici che accettano prenotazioni 24/7, inviano conferme SMS con calendario, richiamano e raccolgono recensioni in automatico — così il tuo team pensa solo agli ospiti.',
      'hero.cta1': 'Avvia la demo', 'hero.cta2': 'Prenota una consulenza',
      'demo.sub': 'Scrivi nella chat qui sotto (o usa i pulsanti). L\'assistente prenota, invia l\'SMS con il calendario, chiede una recensione e richiama — proprio come con un cliente reale.',
      'chat.input': 'Scrivi un messaggio…', 'chat.online': 'Online · risponde subito',
      'reset': '↻ Reimposta demo',
      'cta.btn': 'Prenota una consulenza',
    },
  };
  // Map elements → keys (kept light: only the high-traffic strings switch)
  const map = [
    ['.nav-links a:nth-child(1)', 'nav.sol'], ['.nav-links a:nth-child(2)', 'nav.demo'],
    ['.nav-links a:nth-child(3)', 'nav.video'], ['.nav-links a:nth-child(4)', 'nav.proc'],
    ['.nav-links a:nth-child(5)', 'nav.contact'], ['.nav-actions .btn-primary', 'nav.cta'],
    ['.hero .hero-sub', 'hero.sub'], ['.demo .section-head p', 'demo.sub'],
    ['#chatInput', null], ['#resetDemo', 'reset'], ['#leadForm .btn', 'cta.btn'],
  ];
  document.querySelectorAll('.lang-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.lang-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const L = btn.dataset.lang, t = i18n[L];
      document.documentElement.lang = L;
      const set = (sel, key) => { const el = document.querySelector(sel); if (el && t[key]) el.textContent = t[key]; };
      set('.nav-links a:nth-child(1)', 'nav.sol'); set('.nav-links a:nth-child(2)', 'nav.demo');
      set('.nav-links a:nth-child(3)', 'nav.video'); set('.nav-links a:nth-child(4)', 'nav.proc');
      set('.nav-links a:nth-child(5)', 'nav.contact'); set('.nav-actions .btn-primary', 'nav.cta');
      const sub = document.querySelector('.hero-sub'); if (sub) sub.textContent = t['hero.sub'];
      const dsub = document.querySelector('.demo .section-head p'); if (dsub) dsub.textContent = t['demo.sub'];
      const ci = document.getElementById('chatInput'); if (ci) ci.placeholder = t['chat.input'];
      const rd = document.getElementById('resetDemo'); if (rd) rd.textContent = t['reset'];
      const lf = document.querySelector('#leadForm .btn'); if (lf) lf.textContent = t['cta.btn'];
    });
  });

  /* ---------- Video ---------- */
  const vf = document.getElementById('videoFrame');
  document.getElementById('playBtn').addEventListener('click', () => vf.classList.add('playing'));

  /* ---------- Lead form ---------- */
  document.getElementById('leadForm').addEventListener('submit', (e) => {
    e.preventDefault();
    e.target.querySelector('.btn').disabled = true;
    document.getElementById('leadNote').hidden = false;
    e.target.reset();
  });

  /* ============================================================
     CHATBOT + PHONE DEMO
     ============================================================ */
  const chatBody = document.getElementById('chatBody');
  const phoneScreen = document.getElementById('phoneScreen');
  const callOverlay = document.getElementById('callOverlay');
  const form = document.getElementById('chatForm');
  const input = document.getElementById('chatInput');

  let busy = false;
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const scrollDown = () => { chatBody.scrollTop = chatBody.scrollHeight; };

  function addMsg(html, who) {
    const div = document.createElement('div');
    div.className = 'msg ' + who;
    div.innerHTML = '<p>' + html + '</p>';
    chatBody.appendChild(div);
    scrollDown();
  }
  async function botTyping(ms) {
    const t = document.createElement('div');
    t.className = 'typing-dots';
    t.innerHTML = '<span></span><span></span><span></span>';
    chatBody.appendChild(t); scrollDown();
    await wait(ms);
    t.remove();
  }
  async function bot(html, ms = 900) { await botTyping(ms); addMsg(html, 'bot'); }

  /* ---- phone feed helpers ---- */
  function phoneFeed() {
    let feed = phoneScreen.querySelector('.ph-feed');
    if (!feed) {
      const home = document.getElementById('phHome');
      if (home) home.remove();
      feed = document.createElement('div');
      feed.className = 'ph-feed';
      feed.innerHTML = '<div class="ph-status"><span>19:24</span><span>📶 🔋 87%</span></div>';
      phoneScreen.appendChild(feed);
    }
    return feed;
  }
  function pushNoti(node) { phoneFeed().appendChild(node); node.scrollIntoView({ behavior: 'smooth', block: 'end' }); }
  function el(html) { const d = document.createElement('div'); d.innerHTML = html.trim(); return d.firstChild; }

  function calendarCard() {
    return el(`
      <div class="cal-card">
        <div class="cal-head">📅 Kalender · Heute</div>
        <div class="cal-body">
          <div class="cal-row"><span class="cal-time">18:00</span><div class="cal-event"><strong>Tisch · Famiglia Rossi</strong><small>4 Personen</small></div></div>
          <div class="cal-row new"><span class="cal-time">19:30</span><div class="cal-event"><strong>Tisch · Neuer Gast</strong><small>2 Personen · via AI-Assistant</small></div><span class="cal-check">✔</span></div>
          <div class="cal-row"><span class="cal-time">21:00</span><div class="cal-event"><strong>Tisch · M. Gerola</strong><small>6 Personen</small></div></div>
        </div>
      </div>`);
  }
  function smsNoti() {
    return el(`
      <div class="noti">
        <div class="noti-top"><span class="noti-ic sms">✉</span> SMS · jetzt</div>
        <strong>Ristorante Gerola</strong>
        <p>Reservierung bestätigt ✅ Heute 19:30, 2 Personen. Kalender: gerola.it/r/8a2 — A presto!</p>
      </div>`);
  }
  function reviewNoti() {
    return el(`
      <div class="noti">
        <div class="noti-top"><span class="noti-ic star">★</span> SMS · Bewertung</div>
        <strong>Wie war dein Besuch? ⭐</strong>
        <p>Danke für deinen Besuch! Bewerte uns in 10 Sek: g.page/gerola/review</p>
      </div>`);
  }
  function callNoti() {
    return el(`
      <div class="noti">
        <div class="noti-top"><span class="noti-ic call">📞</span> Anruf · verpasst → Rückruf</div>
        <strong>Ristorante Gerola</strong>
        <p>+39 0473 555 018 · AI-Agent ruft zurück…</p>
      </div>`);
  }

  /* ---- the scripted appointment flow ---- */
  async function runBookingFlow() {
    await bot('Perfetto! 🎉 Ich schau kurz nach freien Tischen für heute Abend…', 1100);
    await bot('Ich hätte <b>19:30 Uhr für 2 Personen</b> frei. Soll ich das für dich buchen?', 1100);
    await wait(400);
    addMsg('Sì, perfetto 👍', 'user');

    await bot('Top! Ich trage den Termin ein… 📅', 1000);
    // 1) calendar appears on phone
    pushNoti(calendarCard());
    await wait(1300);

    await bot('Fertig! Du bekommst gleich eine <b>SMS mit dem Kalender-Link</b>. ✉️', 1000);
    // 2) SMS + review fire (almost) at the same time — "gebucht + rezension gleichzeitig"
    pushNoti(smsNoti());
    await wait(500);
    pushNoti(reviewNoti());
    await wait(900);

    await bot('Und weil wir’s ernst meinen mit Service: nach dem Besuch fragen wir automatisch nach einer <b>Bewertung</b> ⭐ — siehst du oben am Handy.', 1300);
    await bot('Eine Sache noch — der Agent <b>ruft dich kurz an</b>, um die Reservierung zu bestätigen. 📞', 1200);

    // 3) incoming call
    pushNoti(callNoti());
    await wait(700);
    callOverlay.classList.add('show');
  }

  async function handle(text) {
    if (busy) return;
    const t = text.trim();
    if (!t) return;
    addMsg(escapeHtml(t), 'user');
    busy = true;
    input.value = '';

    const low = t.toLowerCase();
    if (/appuntamento|termin|reserv|prenot|tisch|buch|booking|table/.test(low)) {
      await runBookingFlow();
    } else if (/öffnung|orari|open|hour|aperto/.test(low)) {
      await bot('Wir haben heute von <b>12:00–14:30</b> und <b>18:00–23:00</b> geöffnet. 🍝 Soll ich dir einen Tisch reservieren? Schreib einfach <b>„appuntamento"</b>.', 1100);
    } else if (/hallo|ciao|hi|hey|salve|servus|guten/.test(low)) {
      await bot('Ciao! 👋 Möchtest du einen Tisch reservieren? Tipp <b>„appuntamento"</b> und sieh, was passiert. ✨', 1000);
    } else {
      await bot('Gern! 😊 Für die Live-Demo schreib <b>„appuntamento"</b> — dann buche ich, sende dir die SMS mit Kalender und rufe zurück.', 1100);
    }
    busy = false;
  }

  function escapeHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

  form.addEventListener('submit', (e) => { e.preventDefault(); handle(input.value); });
  document.getElementById('quick').addEventListener('click', (e) => {
    const b = e.target.closest('.chip'); if (b) handle(b.dataset.send);
  });

  /* accept call */
  document.getElementById('acceptCall').addEventListener('click', async () => {
    callOverlay.classList.remove('show');
    await bot('Grazie! ✅ Deine Reservierung ist <b>bestätigt</b>. A presto im Ristorante Gerola! 🥂', 800);
  });
  callOverlay.querySelector('.decline').addEventListener('click', () => callOverlay.classList.remove('show'));

  /* reset */
  document.getElementById('resetDemo').addEventListener('click', () => {
    callOverlay.classList.remove('show');
    chatBody.innerHTML = '<div class="msg bot"><p>Ciao &amp; hallo! 👋 Ich bin der KI-Assistent vom Ristorante Gerola. Wie kann ich helfen?</p></div>';
    const feed = phoneScreen.querySelector('.ph-feed');
    if (feed) feed.remove();
    if (!phoneScreen.querySelector('.ph-home')) {
      phoneScreen.insertAdjacentHTML('afterbegin',
        '<div class="ph-home" id="phHome"><div class="ph-time">19:24</div><div class="ph-hint"><div class="ph-hint-ic">📲</div><p>Starte die Demo im Chat —<br/>hier erscheint, was passiert.</p></div></div>');
    }
    busy = false;
  });
})();
