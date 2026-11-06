// Packaging only: copies engine.js, graph.txt, screen0.txt and every weights/*.txt
// verbatim into play.html so the game runs as one self-contained page.
const fs = require('fs'), path = require('path');
const d = __dirname;
const files = {
  'graph.txt': fs.readFileSync(path.join(d, 'graph.txt'), 'utf8'),
  'screen0.txt': fs.readFileSync(path.join(d, 'screen0.txt'), 'utf8'),
};
for (const f of fs.readdirSync(path.join(d, 'weights')).sort())
  files['weights/' + f] = fs.readFileSync(path.join(d, 'weights', f), 'utf8');
const html = fs.readFileSync(path.join(d, 'play.template.html'), 'utf8')
  .replace('/*ENGINE*/', () => fs.readFileSync(path.join(d, 'engine.js'), 'utf8'))
  .replace('/*FILES*/', () => JSON.stringify(files));
fs.writeFileSync(path.join(d, 'play.html'), html);
console.log('wrote play.html with', Object.keys(files).length, 'files');
