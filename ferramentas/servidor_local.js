// servidor local do protótipo: serve a pasta do jogo em http://localhost:8090 (sem instalar nada além do Node)
// uso: node ferramentas/servidor_local.js   |   para abrir no celular (mesmo Wi-Fi): HOST=0.0.0.0
const http = require('http'), fs = require('fs'), path = require('path');
const RAIZ = path.resolve(__dirname, '..'), PORTA = +process.env.PORT || 8090, HOST = process.env.HOST || '127.0.0.1';
const TIPOS = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.json': 'application/json', '.glb': 'model/gltf-binary',
  '.gltf': 'model/gltf+json', '.mp3': 'audio/mpeg', '.ogg': 'audio/ogg', '.wav': 'audio/wav', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.png': 'image/png', '.txt': 'text/plain; charset=utf-8', '.css': 'text/css' };

http.createServer((req, res) => {
  const rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const arq = path.join(RAIZ, rel === '/' ? 'index.html' : rel);
  if (!arq.startsWith(RAIZ + path.sep) && arq !== RAIZ) { res.writeHead(403); return res.end(); }
  fs.stat(arq, (err, st) => {
    if (err || !st.isFile()) { res.writeHead(404); return res.end('não encontrado: ' + rel); }
    const tipo = TIPOS[path.extname(arq).toLowerCase()] || 'application/octet-stream';
    const faixa = /bytes=(\d*)-(\d*)/.exec(req.headers.range || '');   // áudio pede pedaços do arquivo
    if (faixa) {
      const ini = faixa[1] ? +faixa[1] : st.size - +faixa[2], fim = faixa[1] && faixa[2] ? +faixa[2] : st.size - 1;
      res.writeHead(206, { 'Content-Type': tipo, 'Content-Range': `bytes ${ini}-${fim}/${st.size}`, 'Content-Length': fim - ini + 1,
        'Accept-Ranges': 'bytes', 'Cache-Control': 'no-cache' });
      return fs.createReadStream(arq, { start: ini, end: fim }).pipe(res);
    }
    res.writeHead(200, { 'Content-Type': tipo, 'Content-Length': st.size, 'Accept-Ranges': 'bytes', 'Cache-Control': 'no-cache' });
    fs.createReadStream(arq).pipe(res);
  });
}).listen(PORTA, HOST, () => console.log(`Aethelred em http://${HOST === '0.0.0.0' ? 'localhost' : HOST}:${PORTA}`));
