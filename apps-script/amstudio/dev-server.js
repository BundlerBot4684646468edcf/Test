// Local preview of the Apps Script web app: node dev-server.js → http://localhost:8080
// Mimics HtmlService templates (<? ?>, <?= ?>, <?!= ?>) and runs doGet() from Code.gs. No dependencies.
const fs = require('fs');
const http = require('http');
const path = require('path');
const vm = require('vm');

const PORT = process.env.PORT || 8080;
const dir = __dirname;

const esc = (v) => String(v).replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function compile(src) {
  let code = 'let __out = "";\n';
  const re = /<\?(!=|=)?([\s\S]*?)\?>/g;
  let last = 0, m;
  while ((m = re.exec(src))) {
    code += `__out += ${JSON.stringify(src.slice(last, m.index))};\n`;
    if (m[1] === '=') code += `__out += __esc(${m[2]});\n`;
    else if (m[1] === '!=') code += `__out += (${m[2]});\n`;
    else code += m[2] + '\n';
    last = re.lastIndex;
  }
  code += `__out += ${JSON.stringify(src.slice(last))};\nreturn __out;`;
  return code;
}

function makeHtmlService() {
  const output = (html) => {
    const o = { html, title: '', metas: [] };
    o.setTitle = (t) => ((o.title = t), o);
    o.addMetaTag = (n, c) => (o.metas.push([n, c]), o);
    o.setXFrameOptionsMode = () => o;
    return o;
  };
  return {
    XFrameOptionsMode: { ALLOWALL: 'ALLOWALL', DEFAULT: 'DEFAULT' },
    createTemplateFromFile(name) {
      const src = fs.readFileSync(path.join(dir, name + '.html'), 'utf8');
      const t = {};
      t.evaluate = () => {
        const vars = Object.keys(t).filter((k) => k !== 'evaluate');
        const fn = new Function('__esc', ...vars, compile(src));
        return output(fn(esc, ...vars.map((k) => t[k])));
      };
      return t;
    },
  };
}

function render() {
  // Re-read Code.gs on every request so edits show up on refresh.
  const ctx = { HtmlService: makeHtmlService() };
  vm.runInNewContext(fs.readFileSync(path.join(dir, 'Code.gs'), 'utf8'), ctx);
  const out = ctx.doGet({ parameter: {} });
  const head = `<title>${esc(out.title)}</title>` +
    out.metas.map(([n, c]) => `<meta name="${esc(n)}" content="${esc(c)}">`).join('');
  return out.html.replace(/<head>/i, '<head>' + head);
}

http.createServer((req, res) => {
  try {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(render());
  } catch (e) {
    res.writeHead(500, { 'Content-Type': 'text/plain' });
    res.end(e.stack);
  }
}).listen(PORT, () => console.log(`Preview: http://localhost:${PORT}`));
