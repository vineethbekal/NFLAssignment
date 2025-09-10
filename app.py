# app.py
import os, re, time, json, requests, gradio as gr
from rag_utils import RAG

SPORTS_KEY = os.environ["SPORTS_DATA_API_KEY"]
BASE = "https://api.sportsdata.io/v3/nfl/scores/json"
HEADERS = {"Ocp-Apim-Subscription-Key": SPORTS_KEY}  # documented header

# 1) Helper: fetch all players once & cache (fallback for name search)
PLAYERS_CACHE = "/tmp/players.json"

def load_players():
    if os.path.exists(PLAYERS_CACHE) and (time.time()-os.path.getmtime(PLAYERS_CACHE) < 86400):
        return json.load(open(PLAYERS_CACHE))
    # Pull active players (small enough for demo). You can also hit roster-by-team and combine.
    url = f"{BASE}/Players"
    players = requests.get(url, headers=HEADERS, timeout=30).json()
    json.dump(players, open(PLAYERS_CACHE, "w"))
    return players

PLAYERS = load_players()

def find_player_id(name: str):
    name_norm = re.sub(r"[^a-z ]", "", name.lower()).strip()
    best = None
    for p in PLAYERS:
        full = f"{p.get('FirstName','')} {p.get('LastName','')}".strip().lower()
        if name_norm in full:
            best = p
            break
    return (best or {}), (best or {}).get("PlayerID")

def fetch_player_news(player, pid, team_fallback=True, limit=10):
    items = []
    # Preferred: player-specific news if plan supports it
    if pid:
        url_pid = f"{BASE}/NewsByPlayerID/{pid}"
        resp = requests.get(url_pid, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            items = resp.json()
    # Fallback: team news
    if not items and team_fallback and player.get("Team"):
        url_team = f"{BASE}/NewsByTeam/{player['Team']}"
        resp = requests.get(url_team, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            items = resp.json()
    # Massage into (title, content, source, updated)
    docs = []
    for art in (items or [])[:limit]:
        text = f"{art.get('Title','')}\n\n{art.get('Content','')}"
        docs.append({
            "title": art.get("Title",""),
            "content": text,
            "source": art.get("Source",""),
            "updated": art.get("Updated","")
        })
    return docs

RAG_ENGINE = RAG()

def pipeline(player_name):
    player, pid = find_player_id(player_name)
    if not pid and not player:
        return f"Couldn’t find a player matching “{player_name}”. Try full name or a team hint."
    docs = fetch_player_news(player, pid)
    if not docs:
        return f"No recent articles found for {player.get('FirstName','')} {player.get('LastName','')}."
    summary, refs = RAG_ENGINE.summarize(docs)
    header = f"{player.get('FirstName','')} {player.get('LastName','')} ({player.get('Team','')}, {player.get('Position','')})"
    bullets = "\n".join([f"• {d['title']} — {d['source']} ({d['updated']})" for d in refs])
    return f"{header}\n\n{summary}\n\nSources:\n{bullets}"

with gr.Blocks() as demo:
    gr.Markdown("# NFL Chatbot — Player Updates")
    name = gr.Textbox(label="Player name", placeholder="e.g., Patrick Mahomes")
    out = gr.Textbox(label="Summary", lines=14)
    btn = gr.Button("Get latest")
    btn.click(fn=pipeline, inputs=name, outputs=out)

if __name__ == "__main__":
    demo.launch()
