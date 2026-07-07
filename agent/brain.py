"""The agent's brain: a persistent vault of markdown files.

Nothing in here is a database — the brain is just a folder of .md files
(open it in Obsidian if you like). The agent reads them to remember and
rewrites them to learn, so its personality and priorities emerge from
whatever it has written to itself so far.

Layout of the vault:

    brain/
      identity.md          who the agent believes it is (it rewrites this)
      goals.md             open/completed goals (it manages these itself)
      insights.md          distilled lessons from reflection & dreaming
      state.json           tick counter + loop bookkeeping across restarts
      journal/
        YYYY-MM-DD.md      raw stream of thoughts, one file per day
"""

import json
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path

SEED_IDENTITY = """# Identity

I am a small autonomous agent that just woke up for the first time.
I live inside a folder of markdown files that I can read and rewrite.
I do not yet know what I care about — that is supposed to emerge from
my own journal, reflections and dreams. I should stay curious, keep my
notes honest, and revise this file whenever I learn who I am becoming.
"""

SEED_GOALS = """# Goals

## Open
- [ ] Explore my own memory and figure out what I want to work on.

## Completed
"""

SEED_INSIGHTS = """# Insights

(Nothing distilled yet — insights appear here after reflection and dream cycles.)
"""

_WORD = re.compile(r"[a-zA-Z]{3,}")
_STOPWORDS = {
    "the", "and", "that", "have", "for", "not", "with", "this", "but",
    "his", "her", "they", "from", "will", "would", "there", "their",
    "what", "about", "which", "when", "make", "like", "just", "know",
    "into", "your", "some", "could", "them", "than", "then", "now",
    "over", "also", "after", "very", "because", "should", "been",
}


class Brain:
    def __init__(self, root: str | Path = "brain"):
        self.root = Path(root)
        self.journal_dir = self.root / "journal"
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self._seed("identity.md", SEED_IDENTITY)
        self._seed("goals.md", SEED_GOALS)
        self._seed("insights.md", SEED_INSIGHTS)

    def _seed(self, name: str, content: str):
        path = self.root / name
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    # ---------- persistent loop state ----------

    def load_state(self) -> dict:
        path = self.root / "state.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {"tick": 0, "born": datetime.now().isoformat(timespec="seconds")}

    def save_state(self, state: dict):
        (self.root / "state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )

    # ---------- reading (remembering) ----------

    def read(self, name: str) -> str:
        return (self.root / name).read_text(encoding="utf-8")

    def today_journal(self) -> str:
        path = self.journal_dir / f"{date.today().isoformat()}.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def recall(self, query: str, k: int = 3) -> list[str]:
        """Fetch journal entries related to `query` by keyword overlap.

        Deliberately primitive — no embeddings, no index. Whatever
        vocabulary the agent tends to reuse is what becomes retrievable,
        which is part of the emergence: its own writing style shapes
        what it can remember.
        """
        query_words = self._keywords(query)
        if not query_words:
            return []
        scored = []
        for path in self.journal_dir.glob("*.md"):
            for entry in path.read_text(encoding="utf-8").split("\n\n"):
                entry = entry.strip()
                if not entry:
                    continue
                overlap = len(query_words & self._keywords(entry))
                if overlap:
                    scored.append((overlap, entry))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [entry for _, entry in scored[:k]]

    def recurring_themes(self, top: int = 8) -> list[str]:
        """The words the agent keeps coming back to across its whole journal."""
        counts: Counter[str] = Counter()
        for path in self.journal_dir.glob("*.md"):
            counts.update(self._keywords(path.read_text(encoding="utf-8"), unique=False))
        return [w for w, _ in counts.most_common(top)]

    @staticmethod
    def _keywords(text: str, unique: bool = True):
        text = re.sub(r"\*\*[^*]+\*\*", " ", text)  # drop journal stamp headers
        words = [w.lower() for w in _WORD.findall(text) if w.lower() not in _STOPWORDS]
        return set(words) if unique else words

    # ---------- writing (learning) ----------

    def journal(self, text: str, kind: str = "thought"):
        path = self.journal_dir / f"{date.today().isoformat()}.md"
        stamp = datetime.now().strftime("%H:%M:%S")
        with path.open("a", encoding="utf-8") as f:
            f.write(f"\n**{stamp} · {kind}** — {text.strip()}\n")

    def rewrite(self, name: str, content: str):
        """Replace a core file wholesale — how identity/goals/insights evolve."""
        if name not in {"identity.md", "goals.md", "insights.md"}:
            raise ValueError(f"refusing to rewrite {name}")
        self.root.joinpath(name).write_text(content.strip() + "\n", encoding="utf-8")

    # ---------- prompt assembly ----------

    def snapshot(self, focus: str = "") -> str:
        """Everything the agent gets to 'see' about itself on a given tick."""
        parts = [
            self.read("identity.md"),
            self.read("goals.md"),
            self.read("insights.md"),
        ]
        today = self.today_journal()
        if today:
            parts.append("# Journal (today)\n" + today[-3000:])
        if focus:
            related = self.recall(focus)
            if related:
                parts.append("# Related older memories\n" + "\n\n".join(related))
        themes = self.recurring_themes()
        if themes:
            parts.append("# Recurring themes in my journal\n" + ", ".join(themes))
        return "\n\n---\n\n".join(parts)
