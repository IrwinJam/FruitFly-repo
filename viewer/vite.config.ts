import { mkdirSync, writeFileSync } from "node:fs";
import { basename, resolve } from "node:path";
import { defineConfig } from "vite";

// Dev server only: lets the viewer save stills and recordings straight into the project instead of
// going through a browser download. POST /__save/<kind>/<name>:
//   still -> media/stills/ (gitignored)
//   video -> media/ (gitignored; attached to the GitHub release)
const TARGETS: Record<string, string> = { still: "../media/stills", video: "../media" };

export default defineConfig({
  plugins: [
    {
      name: "amongusfly-save",
      apply: "serve",
      configureServer(server) {
        server.middlewares.use("/__save", (req, res) => {
          const [, kind, raw] = (req.url ?? "").split("/");
          const dir = TARGETS[kind];
          const name = basename(decodeURIComponent(raw ?? ""));
          if (req.method !== "POST" || !dir || !/^[\w.-]+\.(png|webm)$/.test(name)) {
            res.statusCode = 400;
            res.end("bad request");
            return;
          }
          const chunks: Buffer[] = [];
          req.on("data", (c: Buffer) => chunks.push(c));
          req.on("end", () => {
            const out = resolve(__dirname, dir);
            mkdirSync(out, { recursive: true });
            writeFileSync(resolve(out, name), Buffer.concat(chunks));
            res.end(`saved ${name}`);
          });
        });
      },
    },
  ],
});
