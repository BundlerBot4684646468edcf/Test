gsap.registerPlugin(ScrollTrigger);

/* ---------- Custom cursor ---------- */
const dot = document.getElementById('cursorDot');
const ring = document.getElementById('cursorRing');
let mouseX = innerWidth/2, mouseY = innerHeight/2;
let ringX = mouseX, ringY = mouseY;

if (dot && ring) {
  window.addEventListener('mousemove', (e) => {
    mouseX = e.clientX; mouseY = e.clientY;
    dot.style.left = mouseX + 'px';
    dot.style.top = mouseY + 'px';
  });
  gsap.ticker.add(() => {
    ringX += (mouseX - ringX) * 0.18;
    ringY += (mouseY - ringY) * 0.18;
    ring.style.left = ringX + 'px';
    ring.style.top = ringY + 'px';
  });
  document.querySelectorAll('a, button, [data-magnetic], [data-service]').forEach(el => {
    el.addEventListener('mouseenter', () => ring.classList.add('is-active'));
    el.addEventListener('mouseleave', () => ring.classList.remove('is-active'));
  });
}

/* ---------- Magnetic buttons ---------- */
document.querySelectorAll('[data-magnetic]').forEach(el => {
  el.addEventListener('mousemove', (e) => {
    const r = el.getBoundingClientRect();
    const x = e.clientX - r.left - r.width/2;
    const y = e.clientY - r.top - r.height/2;
    gsap.to(el, { x: x*0.35, y: y*0.35, duration: 0.4, ease: 'power3.out' });
  });
  el.addEventListener('mouseleave', () => {
    gsap.to(el, { x: 0, y: 0, duration: 0.5, ease: 'elastic.out(1,0.4)' });
  });
});

/* ---------- Progress bar ---------- */
const progressBar = document.getElementById('progressBar');
gsap.to(progressBar, {
  scaleX: 1, ease: 'none',
  scrollTrigger: { trigger: document.body, start: 'top top', end: 'bottom bottom', scrub: 0.3 }
});
progressBar.style.transform = 'scaleX(0)';

/* ---------- Nav background on scroll ---------- */
const nav = document.getElementById('nav');
ScrollTrigger.create({
  start: 100, end: 99999,
  onUpdate: (self) => {
    nav.style.background = self.scroll() > 60
      ? 'rgba(10,10,12,0.85)'
      : 'linear-gradient(to bottom, rgba(10,10,12,.85), transparent)';
  }
});

/* ---------- Mobile menu ---------- */
const burger = document.getElementById('burger');
const mobileMenu = document.getElementById('mobileMenu');
if (burger) {
  burger.addEventListener('click', () => mobileMenu.classList.toggle('is-open'));
  mobileMenu.querySelectorAll('a').forEach(a => a.addEventListener('click', () => mobileMenu.classList.remove('is-open')));
}

/* ---------- Hero entrance ---------- */
const heroTl = gsap.timeline({ delay: 0.2 });
heroTl
  .fromTo('[data-reveal-line]', { yPercent: 120, rotate: 4 }, { yPercent: 0, rotate: 0, duration: 1.1, stagger: 0.12, ease: 'power4.out' })
  .fromTo('.hero-eyebrow', { opacity: 0, y: 16 }, { opacity: 1, y: 0, duration: 0.7, ease: 'power3.out' }, '-=0.9')
  .fromTo('.hero-sub', { opacity: 0, y: 16 }, { opacity: 1, y: 0, duration: 0.7, ease: 'power3.out' }, '-=0.7')
  .fromTo('.hero-actions', { opacity: 0, y: 16 }, { opacity: 1, y: 0, duration: 0.7, ease: 'power3.out' }, '-=0.6')
  .to('#boltPath', { strokeDashoffset: 0, duration: 1.6, ease: 'power2.inOut' }, '-=1');

/* ---------- Generic reveal-on-scroll ---------- */
document.querySelectorAll('[data-reveal]:not([data-reveal-line])').forEach(el => {
  ScrollTrigger.create({
    trigger: el, start: 'top 88%',
    onEnter: () => el.classList.add('is-in')
  });
});

/* ---------- Counters ---------- */
document.querySelectorAll('[data-count]').forEach(el => {
  const target = parseInt(el.dataset.count, 10);
  ScrollTrigger.create({
    trigger: el, start: 'top 90%', once: true,
    onEnter: () => {
      const obj = { val: 0 };
      gsap.to(obj, {
        val: target, duration: 1.6, ease: 'power2.out',
        onUpdate: () => el.textContent = Math.round(obj.val)
      });
    }
  });
});

/* ---------- Marquee infinite scroll ---------- */
gsap.to('#marquee', { xPercent: -50, duration: 14, repeat: -1, ease: 'linear' });

/* ---------- Service rows stagger ---------- */
gsap.fromTo('[data-service]', { opacity: 0, x: -30 }, {
  opacity: 1, x: 0, duration: 0.7, stagger: 0.08, ease: 'power3.out',
  scrollTrigger: { trigger: '.service-list', start: 'top 80%' }
});

/* ---------- Circuit path draw ---------- */
document.querySelectorAll('.circuit-path').forEach(path => {
  const len = path.getTotalLength();
  path.style.strokeDasharray = len;
  path.style.strokeDashoffset = len;
  gsap.to(path, {
    strokeDashoffset: 0, stroke: 'var(--accent)', duration: 1.4, ease: 'power2.inOut',
    scrollTrigger: { trigger: '.circuit-card', start: 'top 75%' }
  });
});

/* ---------- Project cards parallax reveal ---------- */
gsap.fromTo('.project-card', { opacity: 0, y: 40 }, {
  opacity: 1, y: 0, duration: 0.8, stagger: 0.12, ease: 'power3.out',
  scrollTrigger: { trigger: '.project-track', start: 'top 85%' }
});

/* ---------- Process steps stagger ---------- */
gsap.fromTo('.process-step', { opacity: 0, y: 30 }, {
  opacity: 1, y: 0, duration: 0.7, stagger: 0.12, ease: 'power3.out',
  scrollTrigger: { trigger: '.process-list', start: 'top 82%' }
});

/* ---------- Section titles subtle scale-in ---------- */
gsap.utils.toArray('.section-title').forEach(title => {
  gsap.fromTo(title, { opacity: 0, y: 30 }, {
    opacity: 1, y: 0, duration: 0.9, ease: 'power3.out',
    scrollTrigger: { trigger: title, start: 'top 88%' }
  });
});

/* ---------- Tilt on about circuit card ---------- */
const tiltCard = document.querySelector('[data-tilt]');
if (tiltCard) {
  tiltCard.addEventListener('mousemove', (e) => {
    const r = tiltCard.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width - 0.5;
    const py = (e.clientY - r.top) / r.height - 0.5;
    gsap.to(tiltCard, { rotateY: px * 14, rotateX: -py * 14, duration: 0.5, ease: 'power2.out', transformPerspective: 800 });
  });
  tiltCard.addEventListener('mouseleave', () => {
    gsap.to(tiltCard, { rotateY: 0, rotateX: 0, duration: 0.6, ease: 'power3.out' });
  });
}
