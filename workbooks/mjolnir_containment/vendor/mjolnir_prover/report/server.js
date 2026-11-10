// mjolnir/report/server.js — static report server (node, no deps)
const http = require("http");
const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const MIME = {".html": "text/html", ".json": "application/json",
               ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml"};

http.createServer((req, res) => {
  let p = decodeURIComponent(req.url.split("?")[0]);
  if (p === "/") p = "/report.html";
  const file = path.join(ROOT, p);
  if (!file.startsWith(ROOT)) { res.writeHead(403); return res.end("forbidden"); }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end("not found"); }
    res.writeHead(200, {"Content-Type": MIME[path.extname(file)] || "text/plain",
                        "Cache-Control": "no-store"});
    res.end(data);
  });
}).listen(8787, () => console.log("mjolnir report -> http://localhost:8787"));