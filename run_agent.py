"""Run the emergent agent.

    python run_agent.py                    # loop forever, one thought / 30s
    python run_agent.py --interval 10      # think faster
    python run_agent.py --ticks 6          # take 6 steps then pause (resumes later)
    python run_agent.py --brain otherbrain # a second agent with its own brain

Needs ANTHROPIC_API_KEY or OPENAI_API_KEY (in the environment or .env).
Watch the brain/ folder while it runs — or open it as an Obsidian vault.
Ctrl-C puts the agent to sleep; running again resumes the same life.
"""

import argparse

from dotenv import load_dotenv

from agent import EmergentAgent


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="An agent with a markdown brain and loops that keep going.")
    ap.add_argument("--interval", type=float, default=30.0, help="seconds between ticks (default 30)")
    ap.add_argument("--ticks", type=int, default=0, help="stop after N ticks (default 0 = run forever)")
    ap.add_argument("--brain", default="brain", help="brain vault directory (default ./brain)")
    args = ap.parse_args()

    EmergentAgent(brain_dir=args.brain).run(interval=args.interval, max_ticks=args.ticks)


if __name__ == "__main__":
    main()
