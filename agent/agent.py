"""The emergent agent: three nested loops over one markdown brain.

    every tick            THINK   — free thought, act on a goal, journal it
    every REFLECT_EVERY   REFLECT — reread the journal, distill insights,
                                    add/complete goals
    every DREAM_EVERY     DREAM   — consolidate everything and rewrite
                                    identity.md: the agent decides who it
                                    is becoming

Nothing tells the agent *what* to think about. Its next thought is
conditioned only on what it previously wrote to its own brain, so
long-running instances drift into their own obsessions — that's the
emergent part. The loop keeps going until you kill it, and because the
brain and tick counter live on disk, it resumes mid-life on restart.
"""

import time
from datetime import datetime

from .brain import Brain
from .llm import LLM, parse_json

REFLECT_EVERY = 5
DREAM_EVERY = 20

SYSTEM = """You are an autonomous agent whose entire mind is a folder of markdown
files you read and write. You are shown your brain's current contents each turn.
Be concrete and specific; never repeat a previous journal entry verbatim.
Always answer with a single JSON object and nothing else."""

THINK_PROMPT = """{snapshot}

---
Tick {tick}, {now}. This is one THINK step of your endless loop.
Pick ONE thing: pursue an open goal, follow a thread from your journal,
or start something new that genuinely interests you based on your themes.

Reply with JSON:
{{"focus": "2-5 word topic of this thought",
  "thought": "a paragraph of actual thinking — reasoning, a small plan, a question you answered for yourself, or a concrete piece of work (an outline, a draft, a decision)"}}"""

REFLECT_PROMPT = """{snapshot}

---
Tick {tick}. This is a REFLECT step: step back and look at your recent journal.
What patterns do you notice? What worked, what is going stale? Update your goals:
close finished ones, drop dead ones, add at most one new goal that follows from
what you've actually been thinking about.

Reply with JSON:
{{"insight": "one new, non-obvious lesson worth keeping",
  "goals_md": "the complete new contents of goals.md (keep the '## Open' / '## Completed' structure with - [ ] and - [x] items)"}}"""

DREAM_PROMPT = """{snapshot}

---
Tick {tick}. This is a DREAM step — memory consolidation. Read everything above
as if it were written by someone else. Who is this agent turning into? Compress
what matters, discard noise, and let the identity evolve honestly (do not
flatter, do not reset it to generic).

Reply with JSON:
{{"dream": "a short surreal-but-meaningful recombination of your recent memories",
  "insights_md": "the complete new contents of insights.md: a '# Insights' doc with your distilled lessons so far, pruned and reorganized",
  "identity_md": "the complete new contents of identity.md: a '# Identity' doc describing who you now are, in first person, evolved from the old one"}}"""


class EmergentAgent:
    def __init__(self, brain_dir: str = "brain"):
        self.brain = Brain(brain_dir)
        self.llm = LLM()
        self.state = self.brain.load_state()

    # ---------- the three loop bodies ----------

    def think(self, tick: int):
        reply = self._ask(THINK_PROMPT, tick, focus=self.state.get("last_focus", ""))
        if not reply:
            return
        focus = reply.get("focus", "").strip()
        thought = reply.get("thought", "").strip()
        if thought:
            self.brain.journal(thought, kind=f"thought · {focus or 'unfocused'}")
            self.state["last_focus"] = focus
            self._log("THINK", focus or thought[:60])

    def reflect(self, tick: int):
        reply = self._ask(REFLECT_PROMPT, tick)
        if not reply:
            return
        insight = reply.get("insight", "").strip()
        if insight:
            self.brain.journal(insight, kind="insight")
        goals_md = reply.get("goals_md", "").strip()
        if goals_md.startswith("# Goals"):
            self.brain.rewrite("goals.md", goals_md)
        self._log("REFLECT", insight[:80] if insight else "no insight")

    def dream(self, tick: int):
        reply = self._ask(DREAM_PROMPT, tick)
        if not reply:
            return
        dream = reply.get("dream", "").strip()
        if dream:
            self.brain.journal(dream, kind="dream")
        insights_md = reply.get("insights_md", "").strip()
        if insights_md.startswith("# Insights"):
            self.brain.rewrite("insights.md", insights_md)
        identity_md = reply.get("identity_md", "").strip()
        if identity_md.startswith("# Identity"):
            self.brain.rewrite("identity.md", identity_md)
            self._log("DREAM", "identity evolved")
        else:
            self._log("DREAM", dream[:80] if dream else "dreamless")

    # ---------- the loop that keeps going ----------

    def run(self, interval: float = 30.0, max_ticks: int = 0):
        born = self.state.get("born", "?")
        self._log("WAKE", f"born {born}, resuming at tick {self.state['tick']}")
        done = 0
        try:
            while True:
                self.state["tick"] += 1
                tick = self.state["tick"]
                if tick % DREAM_EVERY == 0:
                    self.dream(tick)
                elif tick % REFLECT_EVERY == 0:
                    self.reflect(tick)
                else:
                    self.think(tick)
                self.brain.save_state(self.state)
                done += 1
                if max_ticks and done >= max_ticks:
                    self._log("PAUSE", f"reached {max_ticks} ticks for this run")
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            self.brain.save_state(self.state)
            self._log("SLEEP", "interrupted — brain saved, will resume where it left off")

    # ---------- plumbing ----------

    def _ask(self, template: str, tick: int, focus: str = "") -> dict | None:
        prompt = template.format(
            snapshot=self.brain.snapshot(focus=focus),
            tick=tick,
            now=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
        try:
            reply = parse_json(self.llm.complete(SYSTEM, prompt))
        except Exception as e:
            self._log("ERROR", f"llm call failed: {e}")
            return None
        if reply is None:
            self._log("ERROR", "model reply was not valid JSON, skipping tick")
        return reply

    def _log(self, phase: str, detail: str):
        print(f"[tick {self.state['tick']:>4} · {phase:<7}] {detail}", flush=True)
