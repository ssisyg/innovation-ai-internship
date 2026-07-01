import { useState, useEffect, useRef, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

// ─── Design tokens ────────────────────────────────────────────────────────────
const T = {
  bg:      "#050810",
  surface: "rgba(15, 21, 32, 0.7)",
  border:  "rgba(99, 102, 241, 0.15)",
  accent:  "#6366f1",
  accentDim:"rgba(99,102,241,0.12)",
  textPrimary: "#e2e8f0",
  textMuted:   "#64748b",
  textDim:     "#94a3b8",
  BUY:  { glow:"#10b981", bg:"rgba(16,185,129,0.1)",  border:"rgba(16,185,129,0.4)",  text:"#10b981" },
  SELL: { glow:"#ef4444", bg:"rgba(239,68,68,0.1)",   border:"rgba(239,68,68,0.4)",   text:"#ef4444" },
  HOLD: { glow:"#f59e0b", bg:"rgba(245,158,11,0.1)",  border:"rgba(245,158,11,0.4)",  text:"#f59e0b" },
  UNKNOWN: { glow:"#64748b", bg:"rgba(100,116,139,0.1)", border:"rgba(100,116,139,0.3)", text:"#64748b" },
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
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

// ─── Global styles injected once ─────────────────────────────────────────────
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #050810; color: #e2e8f0; font-family: 'Inter', sans-serif; overflow-x: hidden; }
  ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: #050810; }
  ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }

  @keyframes fadeSlideIn {
    from { opacity:0; transform:translateY(8px); }
    to   { opacity:1; transform:translateY(0); }
  }
  @keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(99,102,241,0.4); }
    70%  { box-shadow: 0 0 0 8px rgba(99,102,241,0); }
    100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
  }
  @keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

  .fade-in { animation: fadeSlideIn 0.3s ease forwards; }
  .glass {
    background: rgba(15,21,32,0.7);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px;
  }
  .calling-dot {
    display: inline-block; width:6px; height:6px; border-radius:50%;
    background: #6366f1; animation: blink 1.2s ease-in-out infinite;
    margin-left:4px;
  }
  .calling-dot:nth-child(2){animation-delay:.2s}
  .calling-dot:nth-child(3){animation-delay:.4s}
`;

// ─── Sub-components ───────────────────────────────────────────────────────────

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

function Spinner() {
  return (
    <span style={{
      display:"inline-block", width:14, height:14,
      border:"2px solid rgba(99,102,241,0.3)",
      borderTopColor:"#6366f1",
      borderRadius:"50%", animation:"spin 0.7s linear infinite",
    }}/>
  );
}

function ToolCallCard({ event }) {
  return (
    <div className="glass fade-in" style={{
      display:"flex", alignItems:"center", gap:10,
      padding:"10px 16px", marginBottom:6,
      borderColor:"rgba(99,102,241,0.2)",
    }}>
      <Spinner />
      <span style={{fontWeight:600, color:T.accent, fontFamily:"'JetBrains Mono',monospace", fontSize:13}}>
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

  // SHAP bar chart data
  const shapData = ok && r.data?.top_shap_features
    ? r.data.top_shap_features.map(f=>({name:f.feature, value:parseFloat(f.mean_abs_shap.toFixed(4))}))
    : null;

  return (
    <div className="glass fade-in" style={{
      padding:"10px 16px", marginBottom:6,
      borderColor: c.border,
    }}>
      <div style={{display:"flex", alignItems:"center", gap:8}}>
        <span style={{
          width:7, height:7, borderRadius:"50%", background:c.dot,
          boxShadow:`0 0 6px ${c.dot}`,
        }}/>
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
          {/* SHAP chart */}
          {shapData && (
            <div style={{marginBottom:10}}>
              <div style={{fontSize:11, color:T.textMuted, marginBottom:6, letterSpacing:1}}>SHAP FEATURE IMPORTANCE</div>
              <ResponsiveContainer width="100%" height={100}>
                <BarChart data={shapData} layout="vertical" margin={{left:0,right:8,top:0,bottom:0}}>
                  <XAxis type="number" tick={{fill:T.textMuted, fontSize:10}} axisLine={false} tickLine={false}/>
                  <YAxis type="category" dataKey="name" tick={{fill:T.textDim, fontSize:10, fontFamily:"'JetBrains Mono',monospace"}} width={90} axisLine={false} tickLine={false}/>
                  <Tooltip contentStyle={{background:"#0f1520",border:"1px solid #1e293b",borderRadius:8,fontSize:11}}/>
                  <Bar dataKey="value" radius={[0,4,4,0]}>
                    {shapData.map((_,i)=>(
                      <Cell key={i} fill={`hsl(${240+i*20},70%,65%)`}/>
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          {/* Raw data */}
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

function RecommendationCard({ rec, ticker }) {
  const [showFull, setShowFull] = useState(false);
  const c = T[rec.signal] || T.UNKNOWN;

  return (
    <div className="fade-in" style={{
      position:"relative", marginTop:20,
      background: c.bg,
      border:`1px solid ${c.border}`,
      borderRadius:16,
      padding:24,
      boxShadow:`0 0 40px ${c.glow}22, inset 0 1px 0 ${c.border}`,
      overflow:"hidden",
    }}>
      {/* Glow orb top-right */}
      <div style={{
        position:"absolute", top:-40, right:-40,
        width:160, height:160, borderRadius:"50%",
        background:`radial-gradient(circle, ${c.glow}20 0%, transparent 70%)`,
        pointerEvents:"none",
      }}/>

      {/* Header */}
      <div style={{display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:20}}>
        <div>
          <div style={{fontSize:11, color:c.text, letterSpacing:2, fontWeight:600, marginBottom:6}}>
            FINAL SIGNAL · {ticker}
          </div>
          <SignalBadge signal={rec.signal} large />
        </div>
        <div style={{textAlign:"right"}}>
          <div style={{fontSize:11, color:T.textMuted, letterSpacing:1}}>CONFIDENCE</div>
          <div style={{
            fontWeight:700, fontSize:20, color:c.text,
            fontFamily:"'JetBrains Mono',monospace", marginTop:2,
          }}>
            {rec.confidence}
          </div>
        </div>
      </div>

      {/* Reasoning */}
      {rec.reasoning.length > 0 && (
        <div style={{marginBottom:16}}>
          <div style={{fontSize:11, color:T.textMuted, letterSpacing:1, marginBottom:8}}>REASONING</div>
          <ul style={{listStyle:"none", paddingLeft:0}}>
            {rec.reasoning.map((r,i)=>(
              <li key={i} style={{
                display:"flex", gap:8, color:T.textPrimary, fontSize:13, lineHeight:1.6,
                marginBottom:6, paddingLeft:4,
              }}>
                <span style={{color:c.text, marginTop:2, flexShrink:0}}>›</span>
                {r}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Risks */}
      {rec.risks.length > 0 && (
        <div style={{
          background:"rgba(0,0,0,0.2)", borderRadius:10, padding:"12px 14px", marginBottom:14,
        }}>
          <div style={{fontSize:11, color:"#f59e0b", letterSpacing:1, marginBottom:8}}>⚠ RISK FACTORS</div>
          <ul style={{listStyle:"none", paddingLeft:0}}>
            {rec.risks.map((r,i)=>(
              <li key={i} style={{
                display:"flex", gap:8, color:T.textDim, fontSize:12, lineHeight:1.6, marginBottom:4,
              }}>
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

// ─── History Tab ──────────────────────────────────────────────────────────────
function HistoryTab() {
  const [records, setRecords] = useState([]);
  const [filter, setFilter]   = useState("");
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const url = filter
        ? `http://127.0.0.1:8000/history?ticker=${filter.toUpperCase()}&limit=30`
        : "http://127.0.0.1:8000/history?limit=30";
      const res = await fetch(url, { headers:{"x-api-key":"innovation-ai-2024"} });
      const d = await res.json();
      setRecords(d.records || []);
    } catch(e) { console.error(e); }
    finally { setLoading(false); }
  };

  // Signal distribution for mini chart
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
            flex:1, maxWidth:200,
            padding:"9px 14px", borderRadius:8, border:"1px solid rgba(99,102,241,0.25)",
            background:"rgba(15,21,32,0.8)", color:T.textPrimary,
            fontSize:14, fontFamily:"'JetBrains Mono',monospace", letterSpacing:1,
            outline:"none",
          }}
        />
        <button onClick={load} disabled={loading} style={{
          padding:"9px 20px", background:T.accent, color:"#fff", border:"none",
          borderRadius:8, cursor:"pointer", fontWeight:600, fontSize:13,
          opacity: loading ? 0.6 : 1,
        }}>
          {loading ? "Loading…" : "Load"}
        </button>
      </div>

      {dist.length > 0 && (
        <div className="glass" style={{padding:"16px 20px", marginBottom:16}}>
          <div style={{fontSize:11, color:T.textMuted, letterSpacing:1, marginBottom:10}}>SIGNAL DISTRIBUTION</div>
          <ResponsiveContainer width="100%" height={60}>
            <BarChart data={dist} margin={{left:0,right:0,top:0,bottom:0}}>
              <XAxis dataKey="name" tick={{fill:T.textMuted,fontSize:11}} axisLine={false} tickLine={false}/>
              <YAxis hide/>
              <Tooltip contentStyle={{background:"#0f1520",border:"1px solid #1e293b",borderRadius:8,fontSize:11}}/>
              <Bar dataKey="value" radius={[4,4,0,0]}>
                {dist.map((d,i)=>(
                  <Cell key={i} fill={(T[d.name]||T.UNKNOWN).glow}/>
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {records.length === 0 && !loading && (
        <div style={{
          textAlign:"center", padding:48, color:T.textMuted, fontSize:14,
          border:"1px dashed rgba(99,102,241,0.15)", borderRadius:12,
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
              padding:"12px 16px", borderColor:c.border,
              animation:`fadeSlideIn .25s ease ${i*0.03}s both`,
            }}>
              <div style={{display:"flex", alignItems:"center", gap:12}}>
                <span style={{
                  fontFamily:"'JetBrains Mono',monospace", fontWeight:700,
                  fontSize:15, color:T.textPrimary, letterSpacing:1,
                }}>
                  {r.ticker}
                </span>
                <SignalBadge signal={r.signal}/>
              </div>
              <div style={{fontSize:11, color:T.textMuted, fontFamily:"'JetBrains Mono',monospace"}}>
                {r.timestamp?.slice(0,16).replace("T"," ")}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [ticker,   setTicker]   = useState("NVDA");
  const [events,   setEvents]   = useState([]);
  const [rec,      setRec]      = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [signal,   setSignal]   = useState(null);
  const [tab,      setTab]      = useState("analyze");
  const [mouse,    setMouse]    = useState({ x: -9999, y: -9999 });

  // Inject global styles once
  useEffect(() => {
    const el = document.createElement("style");
    el.textContent = GLOBAL_CSS;
    document.head.appendChild(el);
    return () => document.head.removeChild(el);
  }, []);

  // Mouse tracking
  useEffect(() => {
    const onMove = e => setMouse({ x: e.clientX, y: e.clientY });
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  const analyze = useCallback(async () => {
    if (!ticker || loading) return;
    setLoading(true);
    setEvents([]);
    setRec(null);
    setSignal(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        headers: { "Content-Type":"application/json", "x-api-key":"innovation-ai-2024" },
        body: JSON.stringify({ ticker, query:`Analyze ${ticker}` }),
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
          if (ev.type === "final")       setRec(parseRecommendation(ev.content));
          if (ev.type === "done")        { setSignal(ev.signal); setLoading(false); }
          if (ev.type === "error")       { setEvents(p=>[...p,ev]); setLoading(false); }
        }
      }
    } catch(e) {
      setEvents(p=>[...p,{type:"error",message:e.message}]);
      setLoading(false);
    }
  }, [ticker, loading]);

  // Derived: group events into call+result pairs for display
  const toolPairs = [];
  let pending = null;
  for (const ev of events) {
    if (ev.type === "tool_call") { pending = { call:ev, result:null }; toolPairs.push(pending); }
    else if (ev.type === "tool_result" && pending && pending.result === null) { pending.result = ev; pending = null; }
    else if (ev.type === "error") toolPairs.push({ call:null, result:null, error:ev });
  }

  const tabStyle = (name) => ({
    padding:"8px 20px", border:"none", background:"none", cursor:"pointer",
    fontWeight: tab===name ? 600 : 400,
    color: tab===name ? T.accent : T.textMuted,
    borderBottom: tab===name ? `2px solid ${T.accent}` : "2px solid transparent",
    fontSize:14, letterSpacing:0.5, transition:"all .15s",
  });

  return (
    <div style={{ minHeight:"100vh", background:T.bg, position:"relative", overflowX:"hidden" }}>

      {/* ── Mouse spotlight ── */}
      <div style={{
        position:"fixed", inset:0, pointerEvents:"none", zIndex:0,
        background:`radial-gradient(700px at ${mouse.x}px ${mouse.y}px, rgba(99,102,241,0.06), transparent 70%)`,
        transition:"background 0.1s ease",
      }}/>

      {/* ── Ambient glows (static) ── */}
      <div style={{position:"fixed",top:"-20%",left:"-10%",width:600,height:600,borderRadius:"50%",
        background:"radial-gradient(circle, rgba(99,102,241,0.05) 0%, transparent 60%)",pointerEvents:"none",zIndex:0}}/>
      <div style={{position:"fixed",bottom:"-20%",right:"-10%",width:500,height:500,borderRadius:"50%",
        background:"radial-gradient(circle, rgba(16,185,129,0.04) 0%, transparent 60%)",pointerEvents:"none",zIndex:0}}/>

      {/* ── Main content ── */}
      <div style={{ position:"relative", zIndex:1, maxWidth:800, margin:"0 auto", padding:"40px 20px" }}>

        {/* Header */}
        <div style={{marginBottom:36}}>
          <div style={{
            fontSize:11, letterSpacing:3, color:T.accent, fontWeight:600,
            fontFamily:"'JetBrains Mono',monospace", marginBottom:8,
          }}>
            AI INVESTMENT AGENT
          </div>
          <h1 style={{
            fontSize:28, fontWeight:700, color:T.textPrimary, letterSpacing:-0.5, lineHeight:1.2,
            background:"linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%)",
            WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent",
          }}>
            Market Intelligence
          </h1>
          <p style={{marginTop:6, color:T.textMuted, fontSize:13}}>
            ReAct · LangGraph · FinBERT · XGBoost · RAG
          </p>
        </div>

        {/* Tabs */}
        <div style={{borderBottom:"1px solid rgba(99,102,241,0.12)", marginBottom:28, display:"flex"}}>
          <button style={tabStyle("analyze")} onClick={()=>setTab("analyze")}>Analyze</button>
          <button style={tabStyle("history")} onClick={()=>setTab("history")}>History</button>
        </div>

        {/* ── Analyze tab ── */}
        {tab === "analyze" && (
          <div>
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
                    background:"rgba(15,21,32,0.8)", color:T.textPrimary,
                    border:"1px solid rgba(99,102,241,0.25)", borderRadius:10,
                    outline:"none", transition:"border-color .2s",
                  }}
                  onFocus={e=>e.target.style.borderColor="#6366f1"}
                  onBlur={e=>e.target.style.borderColor="rgba(99,102,241,0.25)"}
                />
              </div>

              <button
                onClick={analyze}
                disabled={loading}
                style={{
                  padding:"12px 28px", fontSize:14, fontWeight:600,
                  background: loading ? "rgba(99,102,241,0.4)" : "linear-gradient(135deg,#6366f1,#818cf8)",
                  color:"#fff", border:"none", borderRadius:10,
                  cursor: loading ? "not-allowed" : "pointer",
                  boxShadow: loading ? "none" : "0 0 20px rgba(99,102,241,0.4)",
                  transition:"all .2s",
                  animation: !loading ? "pulse-ring 2s infinite" : "none",
                }}
              >
                {loading ? (
                  <span style={{display:"flex",alignItems:"center",gap:8}}>
                    <Spinner/> Analyzing
                  </span>
                ) : "Analyze →"}
              </button>
            </div>

            {/* Thought chain */}
            {toolPairs.length > 0 && (
              <div style={{marginBottom:20}}>
                <div style={{
                  fontSize:11, color:T.textMuted, letterSpacing:2, fontWeight:600,
                  marginBottom:10,
                }}>
                  THOUGHT CHAIN
                </div>
                {toolPairs.map((pair, i) => (
                  <div key={i}>
                    {pair.error && (
                      <div className="glass fade-in" style={{
                        padding:"10px 16px", marginBottom:6,
                        borderColor:"rgba(239,68,68,0.3)",
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

            {/* Loading state — still waiting for results */}
            {loading && toolPairs.length === 0 && (
              <div style={{
                textAlign:"center", padding:32, color:T.textMuted, fontSize:13,
              }}>
                <Spinner/>&nbsp; Initializing agent…
              </div>
            )}

            {/* Recommendation card */}
            {rec && <RecommendationCard rec={rec} ticker={ticker}/>}

            {/* Done footer */}
            {signal && !loading && (
              <div style={{
                marginTop:12, display:"flex", justifyContent:"space-between",
                fontSize:11, color:T.textMuted, fontFamily:"'JetBrains Mono',monospace",
              }}>
                <span style={{color:"#10b981"}}>✓ Saved to memory</span>
                <span>signal: <strong style={{color:(T[signal]||T.UNKNOWN).text}}>{signal}</strong></span>
              </div>
            )}

            {/* Empty state */}
            {events.length===0 && !rec && !loading && (
              <div style={{
                border:"1px dashed rgba(99,102,241,0.15)", borderRadius:14,
                padding:"56px 32px", textAlign:"center", color:T.textMuted,
              }}>
                <div style={{fontSize:32, marginBottom:12}}>📊</div>
                <div style={{fontSize:14, marginBottom:4, color:T.textDim}}>
                  Enter a ticker and press Analyze
                </div>
                <div style={{fontSize:12}}>
                  Supported: NVDA · AAPL · TSLA · MSFT · GOOGL · AMD · and 39 more
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── History tab ── */}
        {tab === "history" && <HistoryTab/>}

      </div>
    </div>
  );
}