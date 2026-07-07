# Test

test

## Emergent agent (`run_agent.py`)

An autonomous agent whose entire mind is a folder of markdown files
(`brain/` — open it as an Obsidian vault), driven by loops that keep going:

| Loop | When | What it does |
|---|---|---|
| **THINK** | every tick | one free thought or piece of work on a goal, appended to today's journal |
| **REFLECT** | every 5th tick | rereads the journal, distills an insight, rewrites `goals.md` |
| **DREAM** | every 20th tick | consolidates memory, rewrites `insights.md` and evolves `identity.md` |

Nothing tells it what to think about — each thought is conditioned only on
what it previously wrote to its own brain, so its interests, goals and even
identity emerge over time. The tick counter and brain live on disk, so
Ctrl-C puts it to sleep and the next run resumes the same life.

### Run it

```bash
pip install -r requirements.txt
# put ANTHROPIC_API_KEY or OPENAI_API_KEY in .env
python run_agent.py                 # loop forever, one tick / 30s
python run_agent.py --interval 10   # think faster
python run_agent.py --ticks 6       # take 6 steps, then pause
python run_agent.py --brain twin    # a second agent with a separate brain
```

Watch `brain/journal/`, `brain/goals.md` and `brain/identity.md` change
while it runs.
