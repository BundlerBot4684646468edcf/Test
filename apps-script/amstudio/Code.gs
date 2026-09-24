/**
 * AMStudio — Google Apps Script web app page.
 *
 * Everything you need to edit is in CONFIG below. The page itself is Index.html.
 * Deploy: Deploy → New deployment → Web app → Execute as: Me, Who has access: Anyone.
 */

var CONFIG = {
  // Shown in the browser tab and used by Google as the page title. Put the search phrase first.
  title: 'Free Project Cost Calculator — AMStudio',

  company: 'AMStudio',
  website: 'https://YOUR-AMSTUDIO-DOMAIN.com',       // your real site: every button links here
  contactUrl: 'https://YOUR-AMSTUDIO-DOMAIN.com/contact',

  headline: 'How much will your project cost?',
  intro: 'Answer three quick questions and get an instant price range. ' +
         'No sign-up, no email required. Built by AMStudio.',

  // Calculator: the base price per project type, plus multipliers.
  currency: '€',
  projectTypes: [
    { label: 'Landing page',        base: 800 },
    { label: 'Business website',    base: 2500 },
    { label: 'Online shop',         base: 5000 },
    { label: 'Custom web app',      base: 9000 }
  ],
  sizes: [
    { label: 'Small (1–5 pages / screens)',    factor: 1.0 },
    { label: 'Medium (6–15 pages / screens)',  factor: 1.6 },
    { label: 'Large (16+ pages / screens)',    factor: 2.4 }
  ],
  extras: [
    { label: 'Copywriting',          price: 400 },
    { label: 'Logo & branding',      price: 900 },
    { label: 'SEO setup',            price: 500 },
    { label: 'Multilingual (per extra language)', price: 600 }
  ],
  rangeSpread: 0.2, // shows the result as ±20 %

  // The written content is what gets a page ranked. Make it specific and useful, not keyword filler.
  sections: [
    {
      heading: 'What drives the price of a website?',
      body: 'The biggest factors are the number of unique page layouts, custom functionality ' +
            '(bookings, shops, member areas), who writes the content, and how many languages you need. ' +
            'Templates are cheaper up front, while custom design pays off when your brand and conversion rate matter.'
    },
    {
      heading: 'How accurate is this estimate?',
      body: 'It is based on typical AMStudio projects and is usually within 20 % of the final quote. ' +
            'For an exact number, send us your brief and we reply within one business day.'
    }
  ],
  faq: [
    { q: 'How long does a typical project take?', a: 'Landing pages take 1–2 weeks, business websites 3–6 weeks and shops or web apps 6–12 weeks.' },
    { q: 'Do you offer maintenance?',             a: 'Yes. Monthly plans cover updates, backups, security and small content changes.' },
    { q: 'Can I pay in instalments?',             a: 'Yes. Usually 40 % at the start, 40 % at design approval and 20 % at launch.' }
  ]
};

function doGet() {
  var t = HtmlService.createTemplateFromFile('Index');
  t.cfg = CONFIG;
  return t.evaluate()
    .setTitle(CONFIG.title)
    .addMetaTag('viewport', 'width=device-width, initial-scale=1')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}
