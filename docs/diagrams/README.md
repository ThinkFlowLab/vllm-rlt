# Architecture diagrams

Open `index.html` in a browser to switch between the three diagrams and zoom.
No network connection is required.

- `.mmd`: editable Mermaid sources, mirrored in the Mermaid blocks in
  [`../unified_execution.md`](../unified_execution.md).
- `.svg`: vector exports for zooming.
- `.png`: exports at 2x resolution for previews and sharing.
- `mermaid-config.json`: shared theme and layout configuration.

With Mermaid CLI installed, regenerate from this directory:

```bash
for name in stage-architecture unified-forward pd-handoff; do
  for ext in svg png; do
    mmdc -i "$name.mmd" -o "$name.$ext" \
      -c mermaid-config.json -b white -w 1600 -s 2
  done
done
```

Exports use Mermaid 11.17.2 and Chrome 153. To select a custom browser, pass a
Puppeteer configuration file with Mermaid CLI's `-p` option.
