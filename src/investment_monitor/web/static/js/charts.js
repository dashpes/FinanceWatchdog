/* ECharts helpers themed to the firm's stationery (reads the CSS variables). */

window.ArchieCharts = (function () {
  "use strict";

  const registry = []; // {node, build} — rebuilt on theme change

  function palette() {
    const css = getComputedStyle(document.documentElement);
    const v = function (name) { return css.getPropertyValue(name).trim(); };
    return {
      paper: v("--paper"), raised: v("--paper-raised"), ink: v("--ink"),
      muted: v("--ink-muted"), green: v("--green"), navy: v("--navy"),
      brass: v("--brass"), hairline: v("--hairline"),
      gain: v("--gain"), loss: v("--loss"),
      serif: v("--serif") || "Georgia, serif",
    };
  }

  function base(p) {
    return {
      backgroundColor: "transparent",
      textStyle: { fontFamily: p.serif, color: p.ink },
      grid: { left: 64, right: 18, top: 24, bottom: 42 },
      tooltip: {
        trigger: "axis",
        backgroundColor: p.raised,
        borderColor: p.hairline,
        textStyle: { color: p.ink, fontFamily: p.serif, fontSize: 12 },
      },
      axisPointer: { lineStyle: { color: p.brass } },
    };
  }

  function axis(p, opts) {
    return Object.assign({
      axisLine: { lineStyle: { color: p.hairline } },
      axisTick: { show: false },
      axisLabel: { color: p.muted, fontFamily: p.serif, fontSize: 11 },
      splitLine: { lineStyle: { color: p.hairline, opacity: 0.6 } },
    }, opts || {});
  }

  function mount(node, build) {
    registry.push({ node: node, build: build });
    build();
  }

  document.addEventListener("archie:theme", function () {
    registry.forEach(function (r) {
      const inst = echarts.getInstanceByDom(r.node);
      if (inst) inst.dispose();
      r.build();
    });
  });

  window.addEventListener("resize", function () {
    registry.forEach(function (r) {
      const inst = echarts.getInstanceByDom(r.node);
      if (inst) inst.resize();
    });
  });

  /* Equity curve / any single line over dates. */
  function line(node, dates, values, name, formatter) {
    mount(node, function () {
      const p = palette();
      const chart = echarts.init(node);
      chart.setOption(Object.assign(base(p), {
        xAxis: axis(p, { type: "category", data: dates, boundaryGap: false }),
        yAxis: axis(p, {
          type: "value", scale: true,
          axisLabel: { color: p.muted, fontFamily: p.serif, fontSize: 11, formatter: formatter },
        }),
        series: [{
          name: name, type: "line", data: values,
          symbol: "none", lineStyle: { color: p.green, width: 1.8 },
          itemStyle: { color: p.green },
          areaStyle: { color: p.green, opacity: 0.06 },
        }],
      }));
    });
  }

  /* Conviction trajectory (0..1). */
  function conviction(node, points) {
    mount(node, function () {
      const p = palette();
      const chart = echarts.init(node);
      chart.setOption(Object.assign(base(p), {
        xAxis: axis(p, { type: "category", data: points.map(function (x) { return x.ts; }), boundaryGap: false }),
        yAxis: axis(p, { type: "value", min: 0, max: 1 }),
        series: [{
          type: "line", data: points.map(function (x) { return x.conviction; }),
          step: "end", symbol: "circle", symbolSize: 5,
          lineStyle: { color: p.brass, width: 1.6 }, itemStyle: { color: p.brass },
        }],
      }));
    });
  }

  /* Candlestick with the robo's own buys/sells marked. */
  function candles(node, data) {
    mount(node, function () {
      const p = palette();
      const chart = echarts.init(node);
      const dates = data.candles.map(function (c) { return c.date; });
      const ohlc = data.candles.map(function (c) { return [c.open, c.close, c.low, c.high]; });
      const marks = (data.trades || []).map(function (t) {
        const d = (t.date || "").slice(0, 10);
        const buy = t.side === "buy";
        return {
          coord: [d, t.fill_price],
          value: (buy ? "B" : "S"),
          itemStyle: { color: buy ? p.gain : p.loss },
          label: { color: "#fff", fontSize: 9 },
          symbol: "pin", symbolSize: 26,
        };
      });
      chart.setOption(Object.assign(base(p), {
        xAxis: axis(p, { type: "category", data: dates }),
        yAxis: axis(p, { type: "value", scale: true }),
        dataZoom: [
          { type: "inside" },
          { type: "slider", height: 18, bottom: 8, borderColor: p.hairline,
            backgroundColor: "transparent", fillerColor: "rgba(169,133,47,0.12)",
            handleStyle: { color: p.brass }, textStyle: { color: p.muted, fontSize: 10 } },
        ],
        series: [{
          type: "candlestick", data: ohlc,
          itemStyle: {
            color: p.gain, color0: p.loss,
            borderColor: p.gain, borderColor0: p.loss,
          },
          markPoint: { data: marks },
        }],
      }));
    });
  }

  return { line: line, conviction: conviction, candles: candles };
})();
