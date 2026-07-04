import { useState, useEffect, useRef, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

// ─────────────────────────────────────────────────────────────────────────
// DESIGN TOKENS
// A cold "trading-terminal" palette: near-black ink background, one
// signature electric-blue "flow" accent used for chrome + the live
// canvas, and the BUY/SELL/HOLD colors kept as-is because green/red/amber
// is a finance convention, not decoration.
// ─────────────────────────────────────────────────────────────────────────
const T = {
  bg:            "#04050a",
  surface:       "rgba(11,15,24,0.55)",
  surfaceSolid:  "#0b0f18",
  border:        "rgba(140,165,210,0.14)",
  borderStrong:  "rgba(112,180,255,0.4)",
  flow:          "#6fb6ff",
  flowDim:       "rgba(111,182,255,0.10)",
  textPrimary:   "#e9edf4",
  textMuted:     "#5c6577",
  textDim:       "#8b93a5",
  BUY:  { glow:"#10b981", bg:"rgba(16,185,129,0.10)",  border:"rgba(16,185,129,0.4)",  text:"#10b981" },
  SELL: { glow:"#ef4444", bg:"rgba(239,68,68,0.10)",   border:"rgba(239,68,68,0.4)",   text:"#ef4444" },
  HOLD: { glow:"#f59e0b", bg:"rgba(245,158,11,0.10)",  border:"rgba(245,158,11,0.4)",  text:"#f59e0b" },
  UNKNOWN: { glow:"#5c6577", bg:"rgba(92,101,119,0.10)", border:"rgba(92,101,119,0.3)", text:"#8b93a5" },
};

const QUICK_TICKERS = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "AMD"];
const API_BASE = "http://127.0.0.1:8000";
const API_KEY  = "innovation-ai-2024";

// ─── Helpers ─────────────────────────────────────────────────────────────
function parseRecommendation(text) {
  if (!text) return null;
  const signalMatch = text.match(/Final\s*Signal\s*:\s*(BUY|SELL|HOLD)/i);
  const confMatch   = text.match(/Confidence\s*:\s*(HIGH|MEDIUM|LOW)/i);
  const reasonMatch = text.match(/Reasoning\s*:([\s\S]*?)(?=Risk Factors|$)/i);
  const riskMatch   = text.match(/Risk Factors\s*:([\s\S]*?)(?=\n\n|\n[A-Z]|$)/i);
  const parseBullets = (raw) =>
    (raw||"").split("\n").map(l=>l.replace(/^[\s\-*•]+/,"").trim()).filter(Boolean);
  return {
    signal:     signalMatch ? signalMatch[1].toUpperCase() : "UNKNOWN",
    confidence: confMatch   ? confMatch[1].toUpperCase()   : "—",
    reasoning:  parseBullets(reasonMatch ? reasonMatch[1] : ""),
    risks:      parseBullets(riskMatch   ? riskMatch[1]   : ""),
    fullText:   text,
  };
}

// ─── Global styles ───────────────────────────────────────────────────────
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;1,9..144,400;1,9..144,500&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { background: #04050a; }
  body { background: #04050a; color: #e9edf4; font-family: 'Inter', sans-serif; overflow-x: hidden; }
  ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: #04050a; }
  ::-webkit-scrollbar-thumb { background: #1b2231; border-radius: 4px; }

  button, input { font-family: inherit; }
  button:focus-visible, input:focus-visible {
    outline: 2px solid #6fb6ff; outline-offset: 2px;
  }

  @keyframes fadeSlideIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
  @keyframes softPulse { 0%,100% { box-shadow: 0 0 0 0 rgba(111,182,255,0.35); } 50% { box-shadow: 0 0 0 6px rgba(111,182,255,0); } }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
  @keyframes dotLive { 0%,100%{opacity:1} 50%{opacity:0.4} }

  .fade-in { animation: fadeSlideIn 0.3s ease forwards; }
  .glass {
    background: rgba(11,15,24,0.55);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(140,165,210,0.14);
    border-radius: 12px;
  }
  .calling-dot {
    display: inline-block; width:6px; height:6px; border-radius:50%;
    background: #6fb6ff; animation: blink 1.2s ease-in-out infinite;
    margin-left:4px;
  }
  .calling-dot:nth-child(2){animation-delay:.2s}
  .calling-dot:nth-child(3){animation-delay:.4s}

  .chip {
    transition: border-color .15s ease, color .15s ease, background .15s ease;
  }
  .chip:hover { border-color: rgba(111,182,255,0.5) !important; color: #e9edf4 !important; }

  .drawer-backdrop { transition: opacity .25s ease; }
  .drawer-panel { transition: transform .3s cubic-bezier(.2,.8,.2,1); }

  @media (prefers-reduced-motion: reduce) {
    .fade-in, .calling-dot, .analyze-btn { animation: none !important; }
  }
`;

// ─────────────────────────────────────────────────────────────────────────
// FLUID CANVAS — the one signature element.
// Soft plasma blobs drift and lean away from the cursor (turbulence),
// layered under a faint particle mesh whose links brighten near the
// cursor — a quiet nod to the agent's own tool graph.
// ─────────────────────────────────────────────────────────────────────────
function FluidCanvas() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let width = 0, height = 0, dpr = 1;
    const mouse = { x: -9999, y: -9999 };

    const BLOB_COLORS = ["111,182,255", "128,120,255", "60,220,200", "170,140,255"];
    const blobs = Array.from({ length: 6 }).map((_, i) => ({
      ox: Math.random(), oy: Math.random(),
      r: 200 + Math.random() * 160,
      phase: Math.random() * Math.PI * 2,
      speed: 0.12 + Math.random() * 0.14,
      color: BLOB_COLORS[i % BLOB_COLORS.length],
    }));

    const NODE_COUNT = 46;
    const nodes = Array.from({ length: NODE_COUNT }).map(() => ({
      x: Math.random(), y: Math.random(),
      vx: (Math.random() - 0.5) * 0.00045,
      vy: (Math.random() - 0.5) * 0.00045,
    }));

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const onMove = (e) => { mouse.x = e.clientX; mouse.y = e.clientY; };
    const onLeave = () => { mouse.x = -9999; mouse.y = -9999; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseleave", onLeave);

    let t = 0;
    let raf = null;

    const frame = () => {
      t += 0.0045;
      ctx.clearRect(0, 0, width, height);

      // ---- plasma blobs ----
      ctx.globalCompositeOperation = "lighter";
      blobs.forEach((b) => {
        const bx = (0.5 + b.ox * 0.2 + Math.sin(t * b.speed + b.phase) * 0.3) * width;
        const by = (0.5 + b.oy * 0.2 + Math.cos(t * b.speed * 0.85 + b.phase * 1.4) * 0.3) * height;

        const dx = bx - mouse.x, dy = by - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const push = Math.max(0, 1 - dist / 460) * 34;
        const fx = bx + (dx / dist) * push;
        const fy = by + (dy / dist) * push;

        const grad = ctx.createRadialGradient(fx, fy, 0, fx, fy, b.r);
        grad.addColorStop(0, `rgba(${b.color},0.15)`);
        grad.addColorStop(1, `rgba(${b.color},0)`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(fx, fy, b.r, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.globalCompositeOperation = "source-over";

      // ---- particle mesh ----
      nodes.forEach((n) => {
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > 1) n.vx *= -1;
        if (n.y < 0 || n.y > 1) n.vy *= -1;
      });

      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        const ax = a.x * width, ay = a.y * height;
        const dxm = ax - mouse.x, dym = ay - mouse.y;
        const nearMouse = Math.sqrt(dxm * dxm + dym * dym) < 170;

        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          const bx = b.x * width, by = b.y * height;
          const dx = ax - bx, dy = ay - by;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < 130) {
            ctx.strokeStyle = nearMouse ? "rgba(140,200,255,0.22)" : "rgba(130,145,175,0.055)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.lineTo(bx, by);
            ctx.stroke();
          }
        }
        ctx.fillStyle = nearMouse ? "rgba(170,215,255,0.9)" : "rgba(150,160,180,0.35)";
        ctx.beginPath();
        ctx.arc(ax, ay, nearMouse ? 1.8 : 1.1, 0, Math.PI * 2);
        ctx.fill();
      }

      raf = requestAnimationFrame(frame);
    };

    if (reduceMotion) {
      frame(); // single static frame, no loop
    } else {
      raf = requestAnimationFrame(frame);
    }

    return () => {
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseleave", onLeave);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: "fixed", inset: 0, width: "100%", height: "100%", zIndex: 0, pointerEvents: "none" }}
    />
  );
}

// ─── Small UI primitives ──────────────────────────────────────────────────

function SignalBadge({ signal, large }) {
  const c = T[signal] || T.UNKNOWN;
  return (
    <span style={{
      display:"inline-flex", alignItems:"center", gap:6,
      background: c.bg, border:`1px solid ${c.border}`,
      color: c.text, borderRadius: 8,
      padding: large ? "6px 18px" : "3px 10px",
      fontSize: large ? 16 : 12,
      fontWeight: 700, letterSpacing: large ? 2 : 1,
      fontFamily:"'JetBrains Mono', monospace",
      boxShadow: `0 0 12px ${c.glow}33`,
    }}>
      {signal === "BUY" ? "▲" : signal === "SELL" ? "▼" : signal === "HOLD" ? "◆" : "?"}
      {" "}{signal}
    </span>
  );
}

function CachedBadge() {
  return (
    <span style={{
      display:"inline-flex", alignItems:"center", gap:5,
      background: T.flowDim, border:`1px solid ${T.borderStrong}`,
      color: T.flow, borderRadius: 8, padding:"3px 10px",
      fontSize: 11, fontWeight:600, letterSpacing:1,
      fontFamily:"'JetBrains Mono', monospace",
    }}>
      ⚡ CACHED
    </span>
  );
}

function Spinner() {
  return (
    <span style={{
      display:"inline-block", width:14, height:14,
      border:"2px solid rgba(111,182,255,0.25)",
      borderTopColor:"#6fb6ff",
      borderRadius:"50%", animation:"spin 0.7s linear infinite",
    }}/>
  );
}

function ToolCallCard({ event }) {
  return (
    <div className="glass fade-in" style={{
      display:"flex", alignItems:"center", gap:10,
      padding:"10px 16px", marginBottom:6,
      borderColor:"rgba(111,182,255,0.22)",
    }}>
      <Spinner />
      <span style={{fontWeight:600, color:T.flow, fontFamily:"'JetBrains Mono',monospace", fontSize:13}}>
        {event.tool}
      </span>
      <span style={{color:T.textMuted, fontSize:12, flex:1}}>
        {Object.entries(event.args||{}).map(([k,v])=>`${k}: ${v}`).join("  ·  ")}
      </span>
      <span style={{display:"flex", gap:2}}>
        <span className="calling-dot"/><span className="calling-dot"/><span className="calling-dot"/>
      </span>
    </div>
  );
}

function ToolResultCard({ event }) {
  const [open, setOpen] = useState(false);
  const r = event.result || {};
  const ok = r.success !== false;
  const c  = ok ? { dot:"#10b981", border:"rgba(16,185,129,0.2)", text:"#10b981" }
                : { dot:"#f59e0b", border:"rgba(245,158,11,0.2)",  text:"#f59e0b" };

  const shapData = ok && r.data?.top_shap_features
    ? r.data.top_shap_features.map(f=>({name:f.feature, value:parseFloat(f.mean_abs_shap.toFixed(4))}))
    : null;

  return (
    <div className="glass fade-in" style={{ padding:"10px 16px", marginBottom:6, borderColor: c.border }}>
      <div style={{display:"flex", alignItems:"center", gap:8}}>
        <span style={{ width:7, height:7, borderRadius:"50%", background:c.dot, boxShadow:`0 0 6px ${c.dot}` }}/>
        <span style={{fontWeight:600, color:c.text, fontFamily:"'JetBrains Mono',monospace", fontSize:13}}>
          {event.tool}
        </span>
        {!ok && <span style={{color:"#f59e0b", fontSize:12, marginLeft:4}}>{r.error}</span>}
        {ok && r.data && (
          <button onClick={()=>setOpen(o=>!o)} style={{
            marginLeft:"auto", background:"none", border:"none", cursor:"pointer",
            fontSize:11, color:T.textMuted,
          }}>
            {open ? "▲ hide" : "▼ details"}
          </button>
        )}
      </div>

      {open && ok && r.data && (
        <div style={{marginTop:10, animation:"fadeSlideIn .2s ease"}}>
          {shapData && (
            <div style={{marginBottom:10}}>
              <div style={{fontSize:11, color:T.textMuted, marginBottom:6, letterSpacing:1}}>SHAP FEATURE IMPORTANCE</div>
              <ResponsiveContainer width="100%" height={100}>
                <BarChart data={shapData} layout="vertical" margin={{left:0,right:8,top:0,bottom:0}}>
                  <XAxis type="number" tick={{fill:T.textMuted, fontSize:10}} axisLine={false} tickLine={false}/>
                  <YAxis type="category" dataKey="name" tick={{fill:T.textDim, fontSize:10, fontFamily:"'JetBrains Mono',monospace"}} width={90} axisLine={false} tickLine={false}/>
                  <Tooltip contentStyle={{background:"#0b0f18",border:"1px solid #1b2231",borderRadius:8,fontSize:11}}/>
                  <Bar dataKey="value" radius={[0,4,4,0]}>
                    {shapData.map((_,i)=>(
                      <Cell key={i} fill={`hsl(${205+i*18},85%,68%)`}/>
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          <pre style={{
            fontSize:11, color:T.textDim,
            background:"rgba(0,0,0,0.3)", borderRadius:8, padding:10,
            overflowX:"auto", fontFamily:"'JetBrains Mono',monospace",
            whiteSpace:"pre-wrap", wordBreak:"break-word",
          }}>
            {JSON.stringify(r.data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

function RecommendationCard({ rec, ticker, cached }) {
  const [showFull, setShowFull] = useState(false);
  const c = T[rec.signal] || T.UNKNOWN;

  return (
    <div className="fade-in" style={{
      position:"relative", marginTop:20,
      background: c.bg, border:`1px solid ${c.border}`, borderRadius:16, padding:24,
      boxShadow:`0 0 40px ${c.glow}1f, inset 0 1px 0 ${c.border}`, overflow:"hidden",
    }}>
      <div style={{
        position:"absolute", top:-40, right:-40, width:160, height:160, borderRadius:"50%",
        background:`radial-gradient(circle, ${c.glow}20 0%, transparent 70%)`, pointerEvents:"none",
      }}/>

      <div style={{display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:20}}>
        <div>
          <div style={{display:"flex", alignItems:"center", gap:8, marginBottom:6}}>
            <span style={{fontSize:11, color:c.text, letterSpacing:2, fontWeight:600}}>
              FINAL SIGNAL · {ticker}
            </span>
            {cached && <CachedBadge/>}
          </div>
          <SignalBadge signal={rec.signal} large />
        </div>
        <div style={{textAlign:"right"}}>
          <div style={{fontSize:11, color:T.textMuted, letterSpacing:1}}>CONFIDENCE</div>
          <div style={{ fontWeight:700, fontSize:20, color:c.text, fontFamily:"'JetBrains Mono',monospace", marginTop:2 }}>
            {rec.confidence}
          </div>
        </div>
      </div>

      {rec.reasoning.length > 0 && (
        <div style={{marginBottom:16}}>
          <div style={{fontSize:11, color:T.textMuted, letterSpacing:1, marginBottom:8}}>REASONING</div>
          <ul style={{listStyle:"none", paddingLeft:0}}>
            {rec.reasoning.map((r,i)=>(
              <li key={i} style={{ display:"flex", gap:8, color:T.textPrimary, fontSize:13, lineHeight:1.6, marginBottom:6, paddingLeft:4 }}>
                <span style={{color:c.text, marginTop:2, flexShrink:0}}>›</span>
                {r}
              </li>
            ))}
          </ul>
        </div>
      )}

      {rec.risks.length > 0 && (
        <div style={{ background:"rgba(0,0,0,0.2)", borderRadius:10, padding:"12px 14px", marginBottom:14 }}>
          <div style={{fontSize:11, color:"#f59e0b", letterSpacing:1, marginBottom:8}}>⚠ RISK FACTORS</div>
          <ul style={{listStyle:"none", paddingLeft:0}}>
            {rec.risks.map((r,i)=>(
              <li key={i} style={{ display:"flex", gap:8, color:T.textDim, fontSize:12, lineHeight:1.6, marginBottom:4 }}>
                <span style={{color:"#f59e0b", flexShrink:0}}>›</span>{r}
              </li>
            ))}
          </ul>
        </div>
      )}

      <button onClick={()=>setShowFull(o=>!o)} style={{
        background:"none", border:`1px solid ${c.border}`, borderRadius:6,
        padding:"4px 12px", cursor:"pointer", fontSize:11, color:c.text, letterSpacing:1,
      }}>
        {showFull ? "▲ HIDE FULL ANALYSIS" : "▼ FULL ANALYSIS"}
      </button>

      {showFull && (
        <pre style={{
          marginTop:12, padding:14, fontSize:11, lineHeight:1.7,
          background:"rgba(0,0,0,0.3)", borderRadius:10,
          whiteSpace:"pre-wrap", wordBreak:"break-word", color:T.textDim,
          fontFamily:"'JetBrains Mono',monospace",
        }}>
          {rec.fullText}
        </pre>
      )}
    </div>
  );
}

// ─── History drawer content ───────────────────────────────────────────────
function HistoryPanel() {
  const [records, setRecords] = useState([]);
  const [filter, setFilter]   = useState("");
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded]   = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const url = filter
        ? `${API_BASE}/history?ticker=${filter.toUpperCase()}&limit=30`
        : `${API_BASE}/history?limit=30`;
      const res = await fetch(url, { headers:{"x-api-key":API_KEY} });
      const d = await res.json();
      setRecords(d.records || []);
    } catch(e) { console.error(e); }
    finally { setLoading(false); setLoaded(true); }
  }, [filter]);

  useEffect(() => { load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const dist = ["BUY","SELL","HOLD"].map(s=>({
    name:s, value: records.filter(r=>r.signal===s).length
  })).filter(d=>d.value>0);

  return (
    <div>
      <div style={{display:"flex", gap:10, marginBottom:20}}>
        <input
          placeholder="Filter by ticker…"
          value={filter}
          onChange={e=>setFilter(e.target.value.toUpperCase())}
          onKeyDown={e=>e.key==="Enter"&&load()}
          style={{
            flex:1,
            padding:"9px 14px", borderRadius:8, border:"1px solid rgba(140,165,210,0.22)",
            background:"rgba(6,9,15,0.8)", color:T.textPrimary,
            fontSize:13, fontFamily:"'JetBrains Mono',monospace", letterSpacing:1,
            outline:"none",
          }}
        />
        <button onClick={load} disabled={loading} style={{
          padding:"9px 18px", background:T.flow, color:"#04050a", border:"none",
          borderRadius:8, cursor:"pointer", fontWeight:700, fontSize:12,
          opacity: loading ? 0.6 : 1,
        }}>
          {loading ? "…" : "Load"}
        </button>
      </div>

      {dist.length > 0 && (
        <div className="glass" style={{padding:"14px 16px", marginBottom:16}}>
          <div style={{fontSize:10, color:T.textMuted, letterSpacing:1, marginBottom:10}}>SIGNAL DISTRIBUTION</div>
          <ResponsiveContainer width="100%" height={56}>
            <BarChart data={dist} margin={{left:0,right:0,top:0,bottom:0}}>
              <XAxis dataKey="name" tick={{fill:T.textMuted,fontSize:10}} axisLine={false} tickLine={false}/>
              <YAxis hide/>
              <Tooltip contentStyle={{background:"#0b0f18",border:"1px solid #1b2231",borderRadius:8,fontSize:11}}/>
              <Bar dataKey="value" radius={[4,4,0,0]}>
                {dist.map((d,i)=>(<Cell key={i} fill={(T[d.name]||T.UNKNOWN).glow}/>))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {loaded && records.length === 0 && !loading && (
        <div style={{
          textAlign:"center", padding:36, color:T.textMuted, fontSize:13,
          border:"1px dashed rgba(140,165,210,0.16)", borderRadius:12,
        }}>
          No records yet — run an analysis to begin
        </div>
      )}

      <div style={{display:"flex", flexDirection:"column", gap:6}}>
        {records.map((r,i)=>{
          const c = T[r.signal]||T.UNKNOWN;
          return (
            <div key={i} className="glass fade-in" style={{
              display:"flex", justifyContent:"space-between", alignItems:"center",
              padding:"11px 14px", borderColor:c.border,
              animation:`fadeSlideIn .25s ease ${i*0.03}s both`,
            }}>
              <div style={{display:"flex", alignItems:"center", gap:10}}>
                <span style={{ fontFamily:"'JetBrains Mono',monospace", fontWeight:700, fontSize:14, color:T.textPrimary, letterSpacing:1 }}>
                  {r.ticker}
                </span>
                <SignalBadge signal={r.signal}/>
              </div>
              <div style={{fontSize:10, color:T.textMuted, fontFamily:"'JetBrains Mono',monospace"}}>
                {r.timestamp?.slice(0,16).replace("T"," ")}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function HistoryIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3v5h5"/><path d="M3.05 13a9 9 0 1 0 .5-4.5L3 8"/><path d="M12 7v5l4 2"/>
    </svg>
  );
}
function CloseIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 6 6 18"/><path d="M6 6l12 12"/>
    </svg>
  );
}

// ─── Main App ──────────────────────────────────────────────────────────────
export default function App() {
  const [ticker,   setTicker]   = useState("NVDA");
  const [events,   setEvents]   = useState([]);
  const [rec,      setRec]      = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [signal,   setSignal]   = useState(null);
  const [cached,   setCached]   = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);

  useEffect(() => {
    const el = document.createElement("style");
    el.textContent = GLOBAL_CSS;
    document.head.appendChild(el);
    return () => document.head.removeChild(el);
  }, []);

  const analyze = useCallback(async (overrideTicker) => {
    const t = overrideTicker || ticker;
    if (!t || loading) return;
    setTicker(t);
    setLoading(true);
    setEvents([]);
    setRec(null);
    setSignal(null);
    setCached(false);

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type":"application/json", "x-api-key":API_KEY },
        body: JSON.stringify({ ticker: t, query:`Analyze ${t}` }),
      });
      if (!res.body) throw new Error("No stream");

      const reader  = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream:true });
        const chunks = buf.split("\n\n");
        buf = chunks.pop();

        for (const chunk of chunks) {
          if (!chunk.startsWith("data:")) continue;
          let ev;
          try { ev = JSON.parse(chunk.replace(/^data:\s*/, "")); }
          catch { continue; }

          if (ev.type === "tool_call")   setEvents(p=>[...p, ev]);
          if (ev.type === "tool_result") setEvents(p=>[...p, ev]);
          if (ev.type === "final") {
            setRec(parseRecommendation(ev.content));
            if (ev.cached) setCached(true);
          }
          if (ev.type === "done") {
            setSignal(ev.signal);
            if (ev.cached) setCached(true);
            setLoading(false);
          }
          if (ev.type === "error") { setEvents(p=>[...p,ev]); setLoading(false); }
        }
      }
    } catch(e) {
      setEvents(p=>[...p,{type:"error",message:e.message}]);
      setLoading(false);
    }
  }, [ticker, loading]);

  const toolPairs = [];
  let pending = null;
  for (const ev of events) {
    if (ev.type === "tool_call") { pending = { call:ev, result:null }; toolPairs.push(pending); }
    else if (ev.type === "tool_result" && pending && pending.result === null) { pending.result = ev; pending = null; }
    else if (ev.type === "error") toolPairs.push({ call:null, result:null, error:ev });
  }

  return (
    <div style={{ minHeight:"100vh", background:T.bg, position:"relative", overflowX:"hidden" }}>

      <FluidCanvas/>

      {/* faint screen texture — ties the whole thing to "a monitor" */}
      <div style={{
        position:"fixed", inset:0, pointerEvents:"none", zIndex:0, opacity:0.35, mixBlendMode:"overlay",
        backgroundImage:"repeating-linear-gradient(180deg, rgba(255,255,255,0.025) 0px, rgba(255,255,255,0.025) 1px, transparent 1px, transparent 3px)",
      }}/>
      <div style={{
        position:"fixed", inset:0, pointerEvents:"none", zIndex:0,
        background:"radial-gradient(120% 90% at 50% -10%, rgba(111,182,255,0.06), transparent 55%)",
      }}/>

      {/* ── Header ── */}
      <div style={{
        position:"relative", zIndex:2, display:"flex", alignItems:"center", justifyContent:"space-between",
        maxWidth:880, margin:"0 auto", padding:"28px 20px 0",
      }}>
        <div style={{display:"flex", alignItems:"center", gap:10}}>
          <span style={{
            fontSize:11, letterSpacing:3, color:T.flow, fontWeight:600,
            fontFamily:"'JetBrains Mono',monospace",
          }}>
            INNOVATION AI
          </span>
          <span style={{display:"flex", alignItems:"center", gap:5, fontSize:10, color:T.textMuted, fontFamily:"'JetBrains Mono',monospace"}}>
            <span style={{width:5,height:5,borderRadius:"50%",background:"#10b981",animation:"dotLive 1.6s ease-in-out infinite"}}/>
            LIVE
          </span>
        </div>

        <button
          onClick={()=>setHistoryOpen(true)}
          style={{
            display:"flex", alignItems:"center", gap:7,
            background:"rgba(11,15,24,0.6)", border:`1px solid ${T.border}`, color:T.textDim,
            borderRadius:8, padding:"7px 14px", fontSize:12, letterSpacing:0.5,
            cursor:"pointer", fontFamily:"'JetBrains Mono',monospace",
          }}
        >
          <HistoryIcon/> History
        </button>
      </div>

      {/* ── Main content ── */}
      <div style={{ position:"relative", zIndex:1, maxWidth:880, margin:"0 auto", padding:"48px 20px 80px" }}>

        <div style={{marginBottom:36}}>
          <h1 style={{
            fontFamily:"'Fraunces', serif", fontStyle:"italic", fontWeight:500,
            fontSize:44, color:T.textPrimary, letterSpacing:-0.5, lineHeight:1.1,
          }}>
            Market Intelligence.
          </h1>
          <p style={{marginTop:10, color:T.textMuted, fontSize:13, fontFamily:"'JetBrains Mono',monospace", letterSpacing:0.5}}>
            REACT · LANGGRAPH · FINBERT · XGBOOST · RAG
          </p>
        </div>

        {/* Quick tickers */}
        <div style={{display:"flex", flexWrap:"wrap", gap:8, marginBottom:14}}>
          {QUICK_TICKERS.map(sym => (
            <button
              key={sym}
              className="chip"
              onClick={()=>analyze(sym)}
              disabled={loading}
              style={{
                padding:"6px 14px", borderRadius:20, fontSize:12, fontWeight:600, letterSpacing:1,
                fontFamily:"'JetBrains Mono',monospace", cursor: loading ? "default" : "pointer",
                background: sym===ticker ? T.flowDim : "rgba(11,15,24,0.5)",
                border: `1px solid ${sym===ticker ? T.borderStrong : T.border}`,
                color: sym===ticker ? T.flow : T.textMuted,
              }}
            >
              {sym}
            </button>
          ))}
        </div>

        {/* Input row */}
        <div style={{display:"flex", gap:10, marginBottom:28}}>
          <div style={{position:"relative", flex:1, maxWidth:220}}>
            <input
              value={ticker}
              onChange={e=>setTicker(e.target.value.toUpperCase())}
              onKeyDown={e=>e.key==="Enter"&&analyze()}
              placeholder="NVDA"
              style={{
                width:"100%", padding:"12px 16px",
                fontSize:20, fontWeight:700, letterSpacing:3,
                fontFamily:"'JetBrains Mono',monospace",
                background:"rgba(6,9,15,0.8)", color:T.textPrimary,
                border:`1px solid ${T.border}`, borderRadius:10,
                outline:"none", transition:"border-color .2s",
              }}
              onFocus={e=>e.target.style.borderColor=T.flow}
              onBlur={e=>e.target.style.borderColor=T.border}
            />
          </div>

          <button
            className="analyze-btn"
            onClick={()=>analyze()}
            disabled={loading}
            style={{
              padding:"12px 28px", fontSize:14, fontWeight:600,
              background: loading ? "rgba(111,182,255,0.35)" : "linear-gradient(135deg,#4f9dff,#8fc6ff)",
              color:"#04050a", border:"none", borderRadius:10,
              cursor: loading ? "not-allowed" : "pointer",
              boxShadow: loading ? "none" : "0 0 20px rgba(111,182,255,0.35)",
              transition:"all .2s",
              animation: !loading ? "softPulse 2.4s infinite" : "none",
            }}
          >
            {loading ? (
              <span style={{display:"flex",alignItems:"center",gap:8}}><Spinner/> Analyzing</span>
            ) : "Analyze →"}
          </button>
        </div>

        {/* Thought chain */}
        {toolPairs.length > 0 && (
          <div style={{marginBottom:20}}>
            <div style={{ fontSize:11, color:T.textMuted, letterSpacing:2, fontWeight:600, marginBottom:10 }}>
              THOUGHT CHAIN
            </div>
            {toolPairs.map((pair, i) => (
              <div key={i}>
                {pair.error && (
                  <div className="glass fade-in" style={{
                    padding:"10px 16px", marginBottom:6, borderColor:"rgba(239,68,68,0.3)",
                    color:"#ef4444", fontSize:13,
                  }}>
                    ✕ {pair.error.message}
                  </div>
                )}
                {pair.call && <ToolCallCard event={pair.call}/>}
                {pair.result && <ToolResultCard event={pair.result}/>}
              </div>
            ))}
          </div>
        )}

        {loading && toolPairs.length === 0 && (
          <div style={{ textAlign:"center", padding:32, color:T.textMuted, fontSize:13 }}>
            <Spinner/>&nbsp; Initializing agent…
          </div>
        )}

        {rec && <RecommendationCard rec={rec} ticker={ticker} cached={cached}/>}

        {signal && !loading && (
          <div style={{
            marginTop:12, display:"flex", justifyContent:"space-between", alignItems:"center",
            fontSize:11, color:T.textMuted, fontFamily:"'JetBrains Mono',monospace",
          }}>
            <span style={{color:"#10b981"}}>✓ Saved to memory</span>
            <span style={{display:"flex", alignItems:"center", gap:10}}>
              {cached && <CachedBadge/>}
              signal: <strong style={{color:(T[signal]||T.UNKNOWN).text}}>{signal}</strong>
            </span>
          </div>
        )}

        {events.length===0 && !rec && !loading && (
          <div style={{
            border:"1px dashed rgba(140,165,210,0.16)", borderRadius:14,
            padding:"56px 32px", textAlign:"center", color:T.textMuted,
          }}>
            <div style={{fontSize:28, marginBottom:12, opacity:0.7}}>◈</div>
            <div style={{fontSize:14, marginBottom:4, color:T.textDim}}>
              Pick a ticker above, or type one and press Analyze
            </div>
            <div style={{fontSize:12}}>45 tickers covered</div>
          </div>
        )}
      </div>

      {/* ── History drawer ── */}
      {historyOpen && (
        <div
          className="drawer-backdrop"
          onClick={()=>setHistoryOpen(false)}
          style={{ position:"fixed", inset:0, background:"rgba(2,3,6,0.55)", zIndex:10 }}
        />
      )}
      <div
        className="drawer-panel"
        style={{
          position:"fixed", top:0, right:0, height:"100vh", width:"min(420px, 92vw)",
          background:"#070a12", borderLeft:`1px solid ${T.border}`, zIndex:11,
          transform: historyOpen ? "translateX(0)" : "translateX(100%)",
          boxShadow: historyOpen ? "-30px 0 60px rgba(0,0,0,0.5)" : "none",
          display:"flex", flexDirection:"column",
        }}
      >
        <div style={{
          display:"flex", alignItems:"center", justifyContent:"space-between",
          padding:"22px 22px 16px", borderBottom:`1px solid ${T.border}`,
        }}>
          <span style={{
            fontFamily:"'Fraunces', serif", fontStyle:"italic", fontWeight:500,
            fontSize:20, color:T.textPrimary,
          }}>
            History
          </span>
          <button onClick={()=>setHistoryOpen(false)} style={{
            background:"none", border:"none", color:T.textMuted, cursor:"pointer", padding:6,
          }}>
            <CloseIcon/>
          </button>
        </div>
        <div style={{ padding:"20px 22px", overflowY:"auto", flex:1 }}>
          {historyOpen && <HistoryPanel/>}
        </div>
      </div>
    </div>
  );
}
