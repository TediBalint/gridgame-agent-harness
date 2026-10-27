#!/usr/bin/env python3
"""A starter agent: plays a level of one of our games with the AI, through the
game server, and shows it live in its own window (viewer.html, in Chrome's or
Edge's app window; in a browser tab where neither is): the game on the left,
the conversation on the right. Make it better: what it tells the model
(system_prompt.txt), how it shows the board, what it remembers, how many moves
it asks for at once.

    python3 starter_agent.py                 (opens the window; closing it stops the agent)
    python3 starter_agent.py --browser       (the same in a browser tab: http://localhost:8765)
    python3 starter_agent.py --no-ui         (plays one run in the terminal)
    python3 starter_agent.py --list

In the window, a reply of the model can be changed: its THINK (the model then
chooses the moves for that thought) or its MOVES too (they're made as they are,
and the model goes on from the next board). Each change is a new run, saved on
the server at once. "Restart rollout" starts the level again. The system
prompt can be changed there too: it's saved in its file, and a run going on
starts the level again with it.

The server says how the agents talk to the model now (GET /api/agent/phase):
on Day 3 they chat (system_prompt.txt: THINK and MOVES lines); on Day 4 the
model calls a tool instead. In the morning it's "play" (system_prompt_play.txt):
the model gives moves, the agent makes them and answers with the board
(transduction: it acts move by move). In the afternoon it's "python"
(system_prompt_python.txt): the model writes a program, the agent runs it
here, in a process of its own (at most PROGRAM_SECONDS, in an empty folder),
where play(moves) and board() reach the game through the server, so every
move is saved too; the program's output goes back to the model (induction:
it finds the rules and writes what wins). A changed reply works as on Day 3:
a new thought has the model call the tool for it; a new MOVES (or program)
is made (or run) as it is.

THE FIRST RULE: every run is saved on the game server, finished or not. The
server keeps whatever goes through it: each /api/agent/play call saves the run
as it is (the run's name, its every step so far), each /api/agent/chat call
the chat in full. So the agent always plays through the server (never with the
AI directly), names each run, and ends a run it leaves ({"end": true}, with
the moves it really made); a run that stems from another one (after a change)
says so ("parent", "fork"). If the agent dies, the server still saves the run
2 minutes later.

How a step works: the server is stateless. /api/agent/play gets the run's
every step so far (one letter per step: n = nothing, U D L R = moves, S =
space; a step is 4 game ticks: the move, then 3 of waiting, as the agents move
at most once every 4 frames) and answers the board as it is after them: a
character per cell, score, time, and whether the game was won or ended. Nobody
tells the agent the rules: it finds them out by playing. /api/agent/chat is a
normal chat (system, user and assistant messages) that the server sends on to
the model, within your team's budget; with "stream": true the reply comes in
pieces where the server can (the local one), else whole.
"""

import argparse
import atexit
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_SERVER = "https://s4ydzm4mvg.execute-api.eu-west-2.amazonaws.com"
# The team's agent key: a zip downloaded on a team member's device has it here.
DEFAULT_KEY = "036b770d33ca806f5620f6593c43ee1ae070bcbf4ef87bd4804bb8aaee195c73"
# The player whose zip this is (their device): the runs are theirs.
DEFAULT_PLAYER = "d25759c16ac3ff70c67caa99511f354a04893f54d222cda20f306d58d37cd72d"
SERVER = os.environ.get("SERVER", DEFAULT_SERVER).rstrip("/")
KEY = os.environ.get("AGENT_KEY") or DEFAULT_KEY
PLAYER = os.environ.get("AGENT_PLAYER") or DEFAULT_PLAYER
# The model (an OpenRouter model's id, as "vendor/name"; AGENT_MODEL="" for the
# server's, openai/gpt-6-luna). Any other works too: it costs what it costs,
# from the team's budget.
MODEL = os.environ.get("AGENT_MODEL", "openai/gpt-6-luna")
HERE = Path(__file__).resolve().parent
MOVES = {"UP": "U", "DOWN": "D", "LEFT": "L", "RIGHT": "R", "SPACE": "S", "NOOP": "n"}
MOVES_PER_REPLY = 10   # the AI plans this many moves at a time
MAX_REPLIES = 200      # a run stops after this many replies
KEEP_TURNS = 6         # the last turns the AI sees again in full (older ones go into its memory)
# The AI's memory (in the first user message each turn): the notes it wrote, and a
# digest of the turns that dropped out of the window. Made here, so it costs nothing.
MAX_NOTES = 25         # the newest notes kept
EXPERIENCE_TURNS = 100 # the AI sees its full thinking of this many last turns (boards only for the last KEEP_TURNS)
DIGEST_LINES = EXPERIENCE_TURNS - KEEP_TURNS   # older turns, one per line; beyond that, quiet stretches are merged
MEMORY_RULES = ("\n\nMemory: you only see the last few turns in full. Whenever you learn something lasting "
                "(what a character does, what gives or loses points, what kills you, how to win), add a line "
                "NOTE: <the fact> to your reply, before MOVES. Your notes and a digest of earlier turns are "
                "shown at the start of the conversation: use them, and don't repeat what failed.")
# What the AI found out about a game: when a run ends, the model sums it up in
# rules/<game>.md, and every later run of the game is shown it at the start.
RULES_DIR = HERE / "rules"
RULES_ASK = ("You keep the rules file of a video game for an AI that plays it. Nobody tells it the rules: it "
             "finds them out by playing. Given the file so far (if any) and what happened in the latest run, "
             "write the updated file: what each character on the board is, what the moves do, what gives or "
             "loses points, what ends the game (winning and losing), a plan that works, and mistakes to avoid. "
             "Keep what earlier runs confirmed unless this run shows it wrong; mark guesses as guesses. Levels "
             "differ in layout, so write rules, not positions. At most 300 words of plain Markdown; reply with "
             "the file only.")
VIEWER_PORT = 8765
BUSY = (429, 503, 504)   # the server is busy for a moment: try again
# A THINK changed by hand: the model is shown it as decided, and chooses the moves for it.
THINK_GIVEN = ("Your THINK for this turn is decided already:\nTHINK: {think}\n"
               "Reply with only the MOVES line that carries out exactly this THINK.")
# Day 4: the system prompt of each way of playing, and the tools the model calls.
PROMPT_FILES = {None: "system_prompt.txt", "play": "system_prompt_play.txt", "python": "system_prompt_python.txt"}
TOOLS = {
    "play": {"type": "function", "function": {
        "name": "play", "description": "Make moves in the game; it answers with the board after them, the score "
                                       "and the time (in ticks).",
        "parameters": {"type": "object", "required": ["moves"], "properties": {"moves": {
            "type": "string", "description": "up to 10 moves, separated by spaces, of UP DOWN LEFT RIGHT SPACE NOOP"}}}}},
    "python": {"type": "function", "function": {
        "name": "python", "description": "Run a Python program in which play(moves) makes moves and board() looks; "
                                         "it answers with what the program printed and the board at its end.",
        "parameters": {"type": "object", "required": ["code"], "properties": {"code": {
            "type": "string", "description": "the program"}}}}},
}
TOOL_THINK_GIVEN = "Your thought for this turn is decided already (above). Now call {tool} to carry out exactly that."
NUDGE = "You called no tool. Go on by calling {tool}."
PROGRAM_SECONDS = 60     # a program runs at most this long
PROGRAM_OUTPUT = 4000    # the model reads this much of what a program printed (its end)
ASK = "\x1eASK "        # how a program asks the agent to play (a line of its output)
# What runs a program: play() and board() ask the agent (a line of output), which answers on its input.
RUNNER = r"""
import json, sys
def _ask(kind, moves=""):
    sys.stdout.write("\n\x1eASK " + json.dumps({"kind": kind, "moves": moves}) + "\n")
    sys.stdout.flush()
    line = sys.stdin.readline()
    if not line:
        raise SystemExit("the agent stopped the program")
    return json.loads(line)
def play(moves):
    '''Makes moves (a list, or a string separated by spaces, of UP DOWN LEFT RIGHT SPACE NOOP).'''
    return _ask("play", " ".join(map(str, moves)) if isinstance(moves, (list, tuple)) else str(moves))
def board():
    '''The game as it is: {board, score, time, won, ended}.'''
    return _ask("board")
with open(sys.argv[1], encoding="utf-8") as f:
    code = f.read()
exec(compile(code, "program.py", "exec"), {"__name__": "__main__", "play": play, "board": board})
"""


class AgentError(Exception):
    pass


class Stopped(Exception):
    """The run was stopped (ended on the server) while it was playing."""


def call(method, path, body=None):
    """The game server's answer to a request (JSON), retrying while it's busy."""
    for attempt in range(6):
        request = urllib.request.Request(
            f"{SERVER}/api/{path}", method=method, data=json.dumps(body).encode() if body is not None else None,
            headers={"X-Agent-Key": KEY, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=300) as reply:
                return json.loads(reply.read())
        except urllib.error.HTTPError as error:
            text = error.read().decode()
            if error.code not in BUSY or attempt == 5:
                raise AgentError(f"{method} /api/{path}: {error.code} {text}")
        except urllib.error.URLError as error:
            if attempt == 5:
                raise AgentError(f"{method} /api/{path}: {error.reason}")
        time.sleep(2 ** attempt)   # 1, 2, 4, 8, 16 seconds


def chat(messages, run, on_piece=lambda text: None, tools=None):
    """The model's reply (with what the team spent and may; on Day 4 its tool
    calls), its pieces to `on_piece` as they come (all at once where the
    server can't stream, or the model calls tools)."""
    request = urllib.request.Request(
        f"{SERVER}/api/agent/chat", method="POST",
        data=json.dumps({"messages": messages, "run": run, "stream": True,
                         **({"tools": tools} if tools else {}), **({"model": MODEL} if MODEL else {})}).encode(),
        headers={"X-Agent-Key": KEY, "Content-Type": "application/json"})
    for attempt in range(6):   # (retrying while the server is busy)
        try:
            reply = urllib.request.urlopen(request, timeout=300)
            break
        except urllib.error.HTTPError as error:
            if error.code not in BUSY or attempt == 5:
                raise AgentError(f"POST /api/agent/chat: {error.code} {error.read().decode()}")
        except urllib.error.URLError as error:
            if attempt == 5:
                raise AgentError(f"POST /api/agent/chat: {error.reason}")
        time.sleep(2 ** attempt)   # 1, 2, 4, 8, 16 seconds
    with reply:
        if "event-stream" not in (reply.headers.get("Content-Type") or ""):
            answer = json.loads(reply.read())
            on_piece(answer["reply"])
            return answer
        event = None
        for raw in reply:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:])
                if event == "delta":
                    on_piece(data["text"])
                elif event == "done":
                    return data
                elif event == "error":
                    raise AgentError(f"POST /api/agent/chat: {data.get('status')} {data.get('error')}")
    raise AgentError("POST /api/agent/chat: the reply broke off")


def describe(state, before=None):
    """The board as the model sees it (`before`: the score before its last
    moves, to show what they gained or lost)."""
    change = state["score"] - before if before is not None else 0
    gained = f" ({change:+} from your last moves)" if change else ""
    return f"Score {state['score']}{gained}, time {state['time']}.\n{state['board']}"


def moves_line(reply):
    """A reply's MOVES line, without "MOVES:" ("" if it has none)."""
    line = next((l for l in reply.splitlines() if l.strip().upper().startswith("MOVES")), "")
    return line.strip()[5:].lstrip(" :").strip()


def think_of(reply):
    """A reply's thought: all but its MOVES line, without "THINK:"."""
    lines = [l for l in reply.splitlines() if not l.strip().upper().startswith("MOVES")]
    return re.sub(r"^\s*THINK\s*:?\s*", "", "\n".join(lines).strip(), flags=re.I)


def moves_of(reply, most=MOVES_PER_REPLY, none="n"):
    """The moves in a reply (its MOVES line, else any move words), as step
    letters: at most `most` (None: all), `none` if there are none."""
    line = next((l for l in reply.splitlines() if l.strip().upper().startswith("MOVES")), reply)
    words = re.findall(r"UP|DOWN|LEFT|RIGHT|SPACE|NOOP", line.upper())[:most]
    return "".join(MOVES[w] for w in words) or none


def notes_of(reply):
    """The NOTE lines of a reply, without "NOTE:"."""
    return [re.sub(r"^\s*NOTE\s*:?\s*", "", l, flags=re.I).strip()
            for l in (reply or "").splitlines() if l.strip().upper().startswith("NOTE")]


def avatar_at(text):
    """Where the A is on a board ("row r, col c"; "" if it isn't on it)."""
    for r, row in enumerate((text or "").splitlines()[1:]):   # (the first line is the score)
        if "A" in row:
            return f"row {r}, col {row.index('A')}"
    return ""


def memory(steps, action=lambda s: s.get("moves", "")):
    """The notes of all `steps`, and a digest of those but the last KEEP_TURNS
    (`action`: what a step did, as text)."""
    notes = []
    for s in steps:
        for note in notes_of(s.get("reply")):
            if note and note.lower() not in (n.lower() for n in notes):
                notes.append(note)
    lines = []
    older = steps[:-KEEP_TURNS] if len(steps) > KEEP_TURNS else []
    for i, s in enumerate(older):
        after = steps[i + 1].get("score", s.get("score"))
        gained = (after or 0) - (s.get("score") or 0)
        think = re.sub(r"\s+", " ", re.sub(r"(?im)^\s*NOTE.*$", "", think_of(s.get("reply") or s.get("think") or "")))
        where = avatar_at(s.get("view") or s.get("observation"))
        did = re.sub(r"\s+", " ", action(s)).strip()
        lines.append((gained, f"turn {i + 1}: {('A at ' + where + ', ') if where else ''}score {s.get('score')}, "
                              f"time {s.get('time')}; {think.strip()[:600]} -> {did[:200] or 'nothing'}"
                              f"{f' (score {gained:+})' if gained else ''}"))
    if len(lines) > DIGEST_LINES:   # keep every score change and the newest lines; merge the rest
        head, tail = lines[:-DIGEST_LINES], lines[-DIGEST_LINES:]
        merged, quiet = [], []
        for i, (gained, line) in enumerate(head):
            if gained:
                if quiet:
                    merged.append((0, f"turns {quiet[0]}-{quiet[-1]}: no change in score"))
                    quiet = []
                merged.append((gained, line))
            else:
                quiet.append(i + 1)
        if quiet:
            merged.append((0, f"turns {quiet[0]}-{quiet[-1]}: no change in score"))
        lines = merged + tail
    out = ""
    if notes:
        out += "\n\nYour notes so far:\n" + "\n".join(f"- {n}" for n in notes[-MAX_NOTES:])
    if lines:
        out += "\n\nYour experience: earlier turns, what you thought and did (newest last):\n" + "\n".join(line for _, line in lines)
    return out


def rules_file(game):
    return RULES_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', str(game))}.md"


def with_memory(out, steps, action=lambda s: s.get("moves", ""), rules=""):
    """`out` with the game's rules (from earlier runs) and the memory put
    before its first user message (it changes every turn, so it stays out of
    the system prompt)."""
    text = memory(steps, action).strip()
    if rules:
        text = f"The rules of this game, as your earlier runs found them (check them as you play):\n{rules}" \
            + (f"\n\n{text}" if text else "")
    first = next((m for m in out if m["role"] == "user"), None)
    if text and first:
        first["content"] = f"{text}\n\n{first['content']}"
    return out


def ended_note(state):
    return ("\nThe game is over: you won." if state["won"] else "\nThe game is over: you lost.") if state["ended"] else ""


def tool_arguments(call_):
    """A tool call's arguments (a dict; {} where they aren't JSON)."""
    try:
        args = json.loads(call_.get("arguments") or "{}")
    except ValueError:
        return {}
    return args if isinstance(args, dict) else {}


def action_of(step):
    """What a Day 4 step's (first) call was given: its moves, or its program."""
    args = tool_arguments((step.get("calls") or [{}])[0])
    given = args.get("moves", args.get("code", ""))
    return " ".join(map(str, given)) if isinstance(given, list) else str(given)


class Rollout:
    """One run of a level: a line of steps (what the model was shown, what it
    replied, the moves it made), played by the AI on the server. It's
    registered on the server at once (its first /api/agent/play call), and
    ended there when it stops, with the moves it really made."""

    def __init__(self, game, level, system, on_change, keep=(), parent=None, fork=None):
        self.game, self.level, self.system, self.on_change = game, level, system, on_change
        self.steps = [dict(s) for s in keep]   # {observation, reply, moves, score, time; think: a THINK given}
        self.run = f"r-{secrets.token_hex(4)}"
        self.parent, self.fork = parent, fork
        self.status, self.live, self.error = "starting", "", ""
        self.state = None
        self.spent = self.budget = None
        self.played = None       # the moves the server has (the last play call's)
        self.stopping = self.closed = False
        self.lock = threading.Lock()    # what shows
        self.calls = threading.Lock()   # one play call at a time, so the last one sent is the one kept
        self.thread = threading.Thread(target=self.play, daemon=True)
        path = rules_file(game)
        self.rules = path.read_text(encoding="utf-8").strip() if path.exists() else ""

    def begin(self):
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return {"run": self.run, "parent": self.parent, "fork": self.fork, "status": self.status,
                    "live": self.live, "error": self.error, "steps": [dict(s) for s in self.steps],
                    "state": self.state, "spent": self.spent, "budget": self.budget}

    def actions(self, steps=None):
        return "".join(s["moves"] for s in (self.steps if steps is None else steps) if s.get("moves"))

    def stop(self):
        """Ends the run now, as it was last played: a reply that came after
        that has its moves left unmade."""
        self.stopping = True
        with self.lock:
            if self.status in ("starting", "playing"):
                self.status = "stopping"
        self.on_change()
        self.close()

    def close(self):
        """Ends the run on the server (once), with the moves it really made."""
        with self.calls:
            if self.closed:
                return
            self.closed = True
            try:
                call("POST", "agent/play", {
                    "run": self.run, "game": self.game, "level": self.level,
                    "actions": self.played if self.played is not None else self.actions(),
                    "end": True, **({} if self.played is not None else self.first_call())})
            except AgentError:
                pass   # (the server saves it anyway, once it's been left 2 minutes)

    def first_call(self):
        return {"player": PLAYER or None, "parent": self.parent, "fork": self.fork, **({"model": MODEL} if MODEL else {})}

    def play_call(self, actions, **extra):
        with self.calls:
            if self.closed:
                raise Stopped()
            state = call("POST", "agent/play", {"run": self.run, "game": self.game, "level": self.level,
                                                "actions": actions, **extra})
            self.played = actions
            return state

    def messages(self):
        out = [{"role": "system", "content": self.system + MEMORY_RULES}]
        for s in self.steps[-KEEP_TURNS:]:
            if s.get("think") and not s.get("reply"):   # a THINK given: the moves for it
                out.append({"role": "user", "content": f"{s['observation']}\n\n{THINK_GIVEN.format(think=s['think'])}"})
                continue
            out.append({"role": "user", "content": s["observation"]})
            if s.get("reply"):
                out.append({"role": "assistant", "content": s["reply"]})
        return with_memory(out, self.steps, rules=self.rules)

    def play(self):
        replies = 0
        try:
            state = self.play_call(self.actions(), **self.first_call())
            with self.lock:
                self.status = "playing"
            while not state["ended"] and replies < MAX_REPLIES:
                with self.lock:
                    self.state = state
                    if not self.steps or self.steps[-1].get("reply"):
                        self.steps.append({"observation": describe(state)})
                    self.steps[-1].update({"score": state["score"], "time": state["time"]})
                    self.live = ""
                    step = self.steps[-1]
                self.on_change()
                if not step.get("reply"):   # (a kept reply, changed by hand: its moves are made)
                    if self.stopping:
                        raise Stopped()
                    answer = chat(self.messages(), self.run, self.piece)
                    replies += 1
                    reply = answer["reply"]
                    if step.get("think"):   # the moves the model chose for the THINK given
                        reply = f"THINK: {step['think']}\nMOVES: {moves_line(reply) or reply.strip()}"
                    with self.lock:
                        step.update({"reply": reply, "moves": moves_of(reply)})
                        self.spent, self.budget, self.live = answer.get("spent"), answer.get("budget"), ""
                    self.on_change()
                state = self.play_call(self.actions())
                with self.lock:
                    self.state = state
                    self.steps.append({"observation": describe(state, self.steps[-1].get("score")),
                                       "score": state["score"], "time": state["time"]})
                self.on_change()
            with self.lock:
                self.state = state
                self.status = ("won" if state["won"] else "lost") if state["ended"] else "stopped"
        except Stopped:
            with self.lock:
                self.status = "stopped"
                last = self.steps[-1] if self.steps else {}
                if last.get("moves") and self.played is not None and len(self.actions()) > len(self.played):
                    last["unmade"] = True   # (stopped before its moves were made)
        except AgentError as error:
            with self.lock:
                self.status, self.error = "error", str(error)
        finally:
            self.write_rules()
            self.close()
            with self.lock:
                self.live = ""
            self.on_change()

    def piece(self, text):
        with self.lock:
            self.live += text
        self.on_change()

    def write_rules(self, action=lambda s: s.get("moves", "")):
        """Sums up what the run found out about the game, with the rules known
        before it, in the game's rules file (only its notes, added to the file,
        if the model can't be asked)."""
        if not any(s.get("reply") for s in self.steps):
            return
        state = self.state or {}
        last = "\n\n".join(think_of(s["reply"]) for s in self.steps[-KEEP_TURNS:] if s.get("reply"))
        told = (f"Game {self.game}, level {self.level}. This run {self.status} with score {state.get('score')} "
                f"after {state.get('time')} ticks.\n\n"
                + (f"The rules file so far:\n{self.rules}\n\n" if self.rules else "The rules file is empty so far.\n\n")
                + f"What happened in this run:{memory(self.steps, action) or ' (nothing older than its last turns)'}"
                + f"\n\nIts last turns, what it thought:\n{last}")
        try:
            rules = chat([{"role": "system", "content": RULES_ASK}, {"role": "user", "content": told}],
                         self.run)["reply"].strip()
        except (AgentError, KeyError, ValueError):
            notes = [n for s in self.steps for n in notes_of(s.get("reply"))]
            rules = self.rules + (f"\n\nNotes of run {self.run} ({self.status}):\n" if notes else "") \
                + "\n".join(f"- {n}" for n in notes)
        if rules.strip():
            RULES_DIR.mkdir(exist_ok=True)
            rules_file(self.game).write_text(rules.strip() + "\n", encoding="utf-8")
            print(f"The rules of {self.game}: {rules_file(self.game)}")


class ToolRollout(Rollout):
    """A run on Day 4: the model calls the phase's tool instead of answering
    in lines, "play" (moves, made at once) or "python" (a program that plays,
    run here). A step is one reply: what the model was shown (the board, or
    what its calls answered), its thought, its calls and their answers, and
    the moves they made."""

    def __init__(self, tool, *args, **kw):
        super().__init__(*args, **kw)
        self.tool = tool
        self.program = None   # the program running now (python)

    def stop(self):
        program = self.program
        if program:
            program.kill()
        super().stop()

    def messages(self):
        out = [{"role": "system", "content": self.system + MEMORY_RULES}]
        kept = self.steps[-KEEP_TURNS:]
        for i, s in enumerate(kept):
            if i == 0 or not kept[i - 1].get("calls"):   # (else its calls' answers carry it)
                out.append({"role": "user", "content": s["observation"]})
            if s.get("think") and "reply" not in s:   # a thought given: the call for it
                out += [{"role": "assistant", "content": s["think"]},
                        {"role": "user", "content": TOOL_THINK_GIVEN.format(tool=self.tool)}]
                break
            if "reply" not in s:
                break
            message = {"role": "assistant", "content": s["reply"]}
            if s.get("calls"):
                message["tool_calls"] = [{"id": c["id"], "type": "function",
                                          "function": {"name": c["name"], "arguments": c["arguments"]}}
                                         for c in s["calls"]]
            out.append(message)
            out += [{"role": "tool", "tool_call_id": c["id"], "content": r}
                    for c, r in zip(s.get("calls") or [], s.get("results") or [])]
        return with_memory(out, self.steps, action_of, self.rules)

    def play(self):
        replies = 0
        try:
            state = self.play_call(self.actions(), **self.first_call())
            with self.lock:
                self.status, self.state = "playing", state
                if not self.steps:
                    self.steps.append({"observation": describe(state)})
                self.steps[-1].update({"view": self.steps[-1].get("view") or describe(state),
                                       "score": state["score"], "time": state["time"]})
            self.on_change()
            while not state["ended"] and replies < MAX_REPLIES:
                step = self.steps[-1]
                if "reply" not in step:
                    if self.stopping:
                        raise Stopped()
                    answer = chat(self.messages(), self.run, self.piece, [TOOLS[self.tool]])
                    replies += 1
                    with self.lock:
                        step.update({"reply": step["think"] if step.get("think") else answer.get("reply") or "",
                                     "calls": answer.get("calls") or [], "results": []})
                        self.spent, self.budget, self.live = answer.get("spent"), answer.get("budget"), ""
                    self.on_change()
                for call_ in step.get("calls") or []:   # (a kept, changed one's too)
                    if self.stopping:
                        raise Stopped()
                    text, state = self.answer_call(step, call_, state)
                    if self.stopping:
                        raise Stopped()   # (a program stopped midway: no answer of it)
                    with self.lock:
                        step["results"] = [*step.get("results", []), text]
                        self.state = state
                    self.on_change()
                seen = "\n\n".join(step["results"]) if step.get("calls") else NUDGE.format(tool=self.tool)
                with self.lock:
                    self.steps.append({"observation": seen, "view": describe(state), "score": state["score"],
                                       "time": state["time"]})
                self.on_change()
            with self.lock:
                self.state = state
                self.status = ("won" if state["won"] else "lost") if state["ended"] else "stopped"
        except Stopped:
            with self.lock:
                self.status = "stopped"
                last = self.steps[-1] if self.steps else {}
                if last.get("calls") and len(last.get("results") or []) < len(last["calls"]):
                    last["unmade"] = True   # (stopped before its calls were all made)
        except AgentError as error:
            with self.lock:
                self.status, self.error = "error", str(error)
        finally:
            self.write_rules()
            self.close()
            with self.lock:
                self.live = ""
            self.on_change()

    def write_rules(self, action=action_of):
        super().write_rules(action)

    def answer_call(self, step, call_, state):
        """Makes a call of the model: (what it answers, the game after it)."""
        args = tool_arguments(call_)
        if call_["name"] != self.tool:
            return f"There's no tool {call_['name']} now: only {self.tool}.", state
        if self.tool == "play":
            given = args.get("moves", "")
            given = " ".join(map(str, given)) if isinstance(given, list) else str(given)
            step["moves"] = step.get("moves", "") + moves_of(f"MOVES: {given}")
            before = state["score"]
            state = self.play_call(self.actions())
            return describe(state, before) + ended_note(state), state
        return self.run_program(step, str(args.get("code") or ""), state)

    def run_program(self, step, code, state):
        """Runs a program of the model here, in a process of its own (at most
        PROGRAM_SECONDS, in an empty folder): play() and board() reach the
        game through the server. (What it printed, the game after it.)"""
        folder = Path(tempfile.mkdtemp(prefix="agent-program-"))
        (folder / "program.py").write_text(code, encoding="utf-8")
        program = subprocess.Popen(
            [sys.executable, "-I", "-c", RUNNER, str(folder / "program.py")], cwd=folder, text=True,
            encoding="utf-8", errors="replace", stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env={k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT")})
        self.program = program
        timed_out = threading.Event()
        timer = threading.Timer(PROGRAM_SECONDS, lambda: (timed_out.set(), program.kill()))
        timer.start()
        printed = []
        try:
            for line in program.stdout:
                if not line.startswith(ASK):
                    printed.append(line)
                    continue
                if printed and printed[-1] == "\n":
                    printed.pop()   # (the line break the program put before its question)
                request = json.loads(line[len(ASK):])
                if request["kind"] == "play":
                    made = moves_of(f"MOVES: {request['moves']}", most=None, none="")
                    if made:
                        step["moves"] = step.get("moves", "") + made
                        state = self.play_call(self.actions())
                        with self.lock:
                            self.state = state
                        self.on_change()
                program.stdin.write(json.dumps({k: state[k] for k in ("board", "score", "time", "won", "ended")}) + "\n")
                program.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            timer.cancel()
            if program.poll() is None:
                program.kill()
            program.wait()
            self.program = None
            shutil.rmtree(folder, ignore_errors=True)
        text = "".join(printed).strip("\n")[-PROGRAM_OUTPUT:]
        late = f"\nIt ran out of time ({PROGRAM_SECONDS} seconds) and was stopped." if timed_out.is_set() else ""
        return f"The program printed:\n{text or '(nothing)'}{late}\n\nThe game now: {describe(state)}{ended_note(state)}", state


class Agent:
    """What the window shows and does: the game and level chosen, and its
    rollout (None until "Start"). A restart or a change begins a new run;
    the old one is ended (saved) first. Every change shows at once; what
    takes the server a moment happens behind (see behind)."""

    def __init__(self, games, game, level, system, on_change=lambda: None, tool=None):
        self.games, self.game, self.level, self.system = games, game, level, system
        self.on_change = on_change
        self.tool = tool   # Day 4: the tool the model calls ("play", "python"); None: it chats
        self.rollout = None
        self.spent = self.budget = None
        self.order = threading.Lock()   # the runs end and begin in the order asked

    def behind(self, work):
        def run():
            with self.order:
                work()
        threading.Thread(target=run, daemon=True).start()

    def replace(self, rollout):
        """`rollout` in the current one's place (None: back to "Start"): the
        current one ends first, so the team's agents never count one too many."""
        old, self.rollout = self.rollout, rollout
        self.on_change()

        def work():
            if old:
                old.stop()
            if rollout:
                rollout.begin()
        self.behind(work)

    def new_rollout(self, keep=(), parent=None, fork=None):
        if self.tool:
            return ToolRollout(self.tool, self.game, self.level, self.system, self.on_change, keep, parent, fork)
        return Rollout(self.game, self.level, self.system, self.on_change, keep, parent, fork)

    def select(self, game, level):
        if (game, level) == (self.game, self.level):
            return
        self.game, self.level = game, level
        self.replace(None)

    def start(self):
        """Start, or "Restart rollout": the level from its start, as a new run
        (in the phase's way of playing: see check_tool)."""
        self.check_tool()
        old = self.rollout
        self.replace(self.new_rollout(parent=old.run if old else None, fork=0 if old else None))

    def stop(self):
        if self.rollout:
            self.behind(self.rollout.stop)

    def check_tool(self):
        """The phase's tool, if it changed since (Day 4 AM to PM), with its own system prompt."""
        try:
            tools = call("GET", "agent/phase")["tools"]
        except AgentError:
            return
        tool = tools[0] if tools else None
        if tool != self.tool:
            self.tool = tool
            self.system = (HERE / PROMPT_FILES[tool]).read_text(encoding="utf-8").strip()

    def set_system(self, text):
        """A new system prompt, kept in system_prompt.txt; a run started
        already begins the level again with it (a new run, from the old one)."""
        text = text.strip()
        if not text or text == self.system:
            return
        self.system = text
        (HERE / PROMPT_FILES[self.tool]).write_text(text + "\n", encoding="utf-8")
        if self.rollout:
            self.start()
        else:
            self.on_change()

    def edit(self, index, think, moves):
        """A reply of the model at step `index`, changed: with its MOVES
        changed too, they're made as they are (the model goes on from the
        next board); else the model chooses the moves for the new THINK.
        Either way it's a new run from that step (the old one ends)."""
        old = self.rollout
        if not old:
            return
        with old.lock:
            steps = [dict(s) for s in old.steps[:index + 1]]
        if index >= len(steps):
            return
        step = steps[index]
        if self.tool:
            return self.edit_call(old, steps, index, think.strip(), moves.strip())
        think, moves = think.strip(), moves.strip()
        new_moves = moves_of(f"MOVES: {moves}") if moves else ""
        changed_moves = bool(moves) and new_moves != step.get("moves")
        if not changed_moves and think == think_of(step.get("reply", "")):
            return   # (nothing changed)
        kept = {k: step[k] for k in ("observation", "score", "time") if k in step}
        if changed_moves:
            kept.update({"reply": f"THINK: {think}\nMOVES: {moves}", "moves": new_moves})
        else:
            kept["think"] = think
        steps[index] = {**kept, "edited": True}
        shared = len(old.actions(steps[:index]))
        self.replace(self.new_rollout(steps, parent=old.run, fork=shared))

    def edit_call(self, old, steps, index, think, given):
        """Day 4's edit: with the call's moves (or program) changed it's made
        (or run) as it is; else the model calls the tool for the new thought."""
        step = steps[index]
        if self.tool == "play":
            changed = bool(given) and moves_of(f"MOVES: {given}") != moves_of(f"MOVES: {action_of(step)}")
        else:
            changed = bool(given) and given != action_of(step).strip()
        if not changed and think == (step.get("reply") or "").strip():
            return   # (nothing changed)
        kept = {k: step[k] for k in ("observation", "view", "score", "time") if k in step}
        if changed:
            argument = "moves" if self.tool == "play" else "code"
            kept.update({"reply": think, "results": [], "calls": [{
                "id": f"call-{uuid.uuid4().hex[:8]}", "name": self.tool, "arguments": json.dumps({argument: given})}]})
        else:
            kept["think"] = think
        steps[index] = {**kept, "edited": True}
        shared = len(old.actions(steps[:index]))
        self.replace(self.new_rollout(steps, parent=old.run, fork=shared))

    @staticmethod
    def reply_parts(reply):
        """A reply's THINK and MOVES, as the window shows them to be changed."""
        return think_of(reply), moves_line(reply)

    def snapshot(self):
        rollout = self.rollout.snapshot() if self.rollout else None
        if rollout and rollout["spent"] is not None:
            self.spent, self.budget = rollout["spent"], rollout["budget"]
        layout = self.games.get(self.game, {}).get("levels", {}).get(str(self.level), "")
        return {**(rollout or {"steps": [], "live": "", "error": "", "status": "ready"}), "tool": self.tool,
                "prompt_file": PROMPT_FILES[self.tool],
                "game": self.game, "level": self.level, "system": self.system, "started": rollout is not None,
                "spent": self.spent, "budget": self.budget, "preview": f"Score 0, time 0.\n{layout.strip()}"}

    def close(self):
        """Ends the run playing now (before the agent quits)."""
        if self.rollout:
            self.rollout.stop()


# --- The window: viewer.html, from a small local server, in an app window (or a browser tab) ---

def serve(agent, port):
    version = [0]
    changed = threading.Condition()

    def bump():
        with changed:
            version[0] += 1
            changed.notify_all()
    agent.on_change = bump

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                body = (HERE / "viewer.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/games":
                self.send_json(agent.games)
            elif self.path == "/state":
                self.send_json(agent.snapshot())
            elif self.path == "/events":   # the state on every change, as server-sent events
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with changed:
                            changed.wait_for(lambda: version[0] != seen, timeout=15)
                            seen = version[0]
                        self.wfile.write(f"data: {json.dumps(agent.snapshot())}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/select":
                agent.select(body["game"], int(body["level"]))
            elif self.path == "/start":
                agent.start()
            elif self.path == "/stop":
                agent.stop()
            elif self.path == "/system":
                agent.set_system(body.get("text", ""))
            elif self.path == "/edit":
                agent.edit(int(body["step"]), body.get("think", ""), body.get("moves", ""))
            else:
                return self.send_error(404)
            self.send_json({"ok": True})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def app_browser():
    """A browser that shows a page as a window of its own (Chrome, Edge,
    Chromium or Brave: their app mode, without tabs or address bar), if any."""
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge",
                 "microsoft-edge-stable", "brave-browser", "msedge", "chrome"):
        if shutil.which(name):
            return shutil.which(name)
    places = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
              "/Applications/Chromium.app/Contents/MacOS/Chromium",
              "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"]
    for root in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):   # (Windows)
        if os.environ.get(root):
            places += [os.path.join(os.environ[root], r"Microsoft\Edge\Application\msedge.exe"),
                       os.path.join(os.environ[root], r"Google\Chrome\Application\chrome.exe")]
    return next((place for place in places if os.path.exists(place)), None)


def open_window(url, port):
    """The page in an app window, as a process of its own (its own profile,
    so it ends when the window is closed); None if no browser can."""
    browser = app_browser()
    if not browser:
        return None
    profile = Path(tempfile.gettempdir()) / f"agent-window-{port}"
    return subprocess.Popen([browser, f"--app={url}", f"--user-data-dir={profile}", "--window-size=1280,820",
                             "--no-first-run", "--no-default-browser-check"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def first_game(games, wanted):
    playable = {key: game for key, game in games.items() if game["levels"]}
    if not playable:
        sys.exit("Your agent may play no game now.")
    game = wanted or ("bait_vgfmri3" if "bait_vgfmri3" in playable else next(iter(playable)))
    return game, min(map(int, playable.get(game, {"levels": {"0": ""}})["levels"]))


def auto(games, wanted, level, system, tool, parallel):
    """Plays the levels of the games (`wanted`, else all it may play; `level`,
    else every level) by itself, in the terminal: `parallel` games at once,
    each game's levels one after another, so each starts with the rules the
    ones before it found (rules/<game>.md)."""
    queue = [wanted] if wanted else list(games)
    lock = threading.Lock()   # the queue, the results and the printing
    results = []

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                game = queue.pop(0)
            for lv in [level] if level is not None else sorted(map(int, games[game]["levels"])):
                rollout = (ToolRollout(tool, game, lv, system, lambda: None) if tool
                           else Rollout(game, lv, system, lambda: None))
                atexit.register(rollout.stop)   # (Ctrl+C: the run is saved as it is)
                rollout.begin()
                rollout.thread.join()
                atexit.unregister(rollout.stop)
                state = rollout.state or {}
                replies = sum(1 for s in rollout.steps if s.get("reply"))
                with lock:
                    results.append((game, lv, rollout.status, state.get("score"), state.get("time", 0), replies))
                    print(f"{game} level {lv}: {rollout.status.upper()}, score {state.get('score')}, "
                          f"{state.get('time', 0)} ticks, {replies} replies {rollout.error}".rstrip(), flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, min(parallel, len(queue))))]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        time.sleep(1)   # (not join(): Ctrl+C gets through)
    won = sum(1 for r in results if r[2] == "won")
    print(f"\nWon {won} of {len(results)} runs, total score {sum(r[3] or 0 for r in results)}.")
    for game in dict.fromkeys(r[0] for r in results):
        mine = [r for r in results if r[0] == game]
        print(f"  {game:<24} won {sum(1 for r in mine if r[2] == 'won')}/{len(mine)}, "
              f"score {sum(r[3] or 0 for r in mine)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game", nargs="?", help="a game's key (see --list; default: Bait [vgfmri3], "
                                                "or the first game it may play)")
    parser.add_argument("level", nargs="?", type=int, help="a level of it (default: its first)")
    parser.add_argument("--list", action="store_true", help="list the games your agent may play")
    parser.add_argument("--no-ui", action="store_true", help="play one run in the terminal, without a window")
    parser.add_argument("--auto", action="store_true", help="play every level of every game (or of the game "
                                                            "given) by itself, in the terminal")
    parser.add_argument("--all", action="store_true", help="play every level of every game (or of the game "
    "given) by itself, in the terminal")

    parser.add_argument("--parallel", type=int, default=4, help="with --auto: how many games are played at once")
    parser.add_argument("--browser", action="store_true", help="show it in a browser tab instead of its own window")
    parser.add_argument("--port", type=int, default=VIEWER_PORT, help="the page's port on this computer")
    args = parser.parse_args()
    if not KEY:
        sys.exit("Set AGENT_KEY to your team's agent key (a zip from the play page has it inside).")
    try:
        games = {k: g for k, g in call("GET", "agent/games").items() if g["levels"]}
    except AgentError as error:
        sys.exit(str(error))
    if args.list:
        for key, game in games.items():
            print(f"{key:<28} {game['name']:<28} levels {', '.join(sorted(game['levels'], key=int))}")
        return
    game, level = first_game(games, args.game)
    level = args.level if args.level is not None else level
    try:
        tools = call("GET", "agent/phase")["tools"]
    except AgentError:
        tools = []   # (a server from before Day 4: the agents chat)
    tool = tools[0] if tools else None
    system = (HERE / PROMPT_FILES[tool]).read_text(encoding="utf-8").strip()
    if args.auto:
        if args.game and args.game not in games:
            sys.exit(f"No game {args.game} to play (see --list).")
        auto(games, args.game, args.level, system, tool, args.parallel)
        return
    agent = Agent(games, game, level, system, tool=tool)
    if args.no_ui:
        rollout = agent.new_rollout()
        atexit.register(rollout.stop)
        rollout.begin()
        rollout.thread.join()
        for i, step in enumerate(rollout.steps):
            print(f"step {i}: {(step.get('reply') or '').strip()}")
        print(rollout.status.upper(), rollout.error, f"after {(rollout.state or {}).get('time', 0)} ticks")
        return

    def leave(*_):   # the run playing now is saved before the agent stops
        agent.close()
        os._exit(0)
    signal.signal(signal.SIGINT, leave)
    signal.signal(signal.SIGTERM, leave)
    atexit.register(agent.close)
    server = serve(agent, args.port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://localhost:{args.port}"
    shown = None if args.browser else open_window(url, args.port)
    if shown:
        opened = time.time()
        print("The agent's window is open: closing it (or Ctrl+C here) stops the agent; the run is saved.")
        shown.wait()
        if time.time() - opened > 3:
            leave()
        print(f"The window closed at once: the agent goes on in {url}")   # (a browser that hands it on)
    else:
        print(f"The agent: {url} (Ctrl+C stops it; the run is saved)")
    webbrowser.open(url)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
