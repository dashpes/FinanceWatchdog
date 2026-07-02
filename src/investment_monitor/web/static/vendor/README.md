# Vendored frontend libraries

Committed to the repo so installs are deterministic and offline-safe (the Pi
install path supports scp/clone with no CDN access).

| File | Library | Version | Source | SHA-256 |
|---|---|---|---|---|
| `echarts.min.js` | Apache ECharts | 5.5.1 | https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js | `e84270bd0cd5bdf60fefc26d00c2a391cb2e81f4d26a7a9ee16185a54773a3cf` |

To bump: download the new pinned version, update this table, and re-test the
Overview equity curve + Charts candlestick pages.
