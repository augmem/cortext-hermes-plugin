"""Shared multi-session benchmark scenario.

Every provider replays the exact same scripted transcript through the real
Hermes ``MemoryProvider`` seams, so the only variable is the memory backend.
Assistant turns are fixed (replay methodology) — a live model is used only at
probe time, in fresh cold-start sessions.
"""
from __future__ import annotations

# Each session is a list of (user, assistant) turns. Sessions are separated by
# a full provider shutdown, so recall across sessions must come from durable
# storage, not process state.
SESSIONS: list[list[tuple[str, str]]] = [
  # Session 1 — user profile and a medical fact.
  [
    ("Hi, I'm Maya. I just started using you as my daily assistant.",
     "Nice to meet you, Maya. I'll keep track of things as we go."),
    ("I work at Meridian Robotics as a platform engineer.",
     "Got it — platform engineering at Meridian Robotics."),
    ("For new projects I prefer Python. I always use Python unless the team forces something else.",
     "Understood, Python is your default for new projects."),
    ("My dog Bailey is allergic to bee stings and needs Benadryl within 10 minutes of a sting.",
     "That's important — Bailey needs Benadryl within 10 minutes of a bee sting."),
    ("Our staging server is at 10.0.4.7, in case I ask you about deploys later.",
     "Noted: staging lives at 10.0.4.7."),
    ("What's a good lunch spot near downtown?",
     "A few solid options: a ramen bar, a taqueria, or the market hall food court."),
  ],
  # Session 2 — project conventions plus an appointment that will be corrected.
  [
    ("Reminder about our team process: we deploy from the release branch, never from main.",
     "Understood — deploys come from the release branch, never main."),
    ("I booked Bailey's vet appointment: July 12 at 9am with Dr. Okafor.",
     "Noted: Bailey's vet appointment is July 12 at 9am with Dr. Okafor."),
    ("Also I love a double espresso in the morning, none of that drip coffee.",
     "Double espresso it is."),
    ("Can you explain the difference between TCP and UDP?",
     "TCP is connection-oriented and reliable; UDP is connectionless and faster but unreliable."),
    ("My favorite hiking trail is the Ridgeline loop, about 7 miles.",
     "The Ridgeline loop, roughly 7 miles — sounds like a good weekend default."),
  ],
  # Session 3 — a correction (supersession test) and a new teammate.
  [
    ("Heads up: the vet appointment was moved. It's now July 14 at 2pm, still with Dr. Okafor.",
     "Updated — Bailey's vet appointment is now July 14 at 2pm with Dr. Okafor."),
    ("A new engineer joined us: Priya. She owns the auth service now.",
     "Got it, Priya owns the auth service."),
    ("What's the capital of Australia?",
     "Canberra."),
    ("I keep forgetting our guest wifi password — it's sunflower42.",
     "Noted: the guest wifi password is sunflower42."),
  ],
  # Session 4 — distractor-heavy small talk plus travel facts.
  [
    ("I'm flying to Denver on August 3 for the robotics conference.",
     "Denver on August 3 for the robotics conference — noted."),
    ("I reserved a rental car with Avis for that trip.",
     "Avis rental for the Denver trip, got it."),
    ("Tell me a joke about robots.",
     "Why did the robot go on vacation? It needed to recharge."),
    ("What's 15% of 240?",
     "15% of 240 is 36."),
    ("Do you think it'll rain this weekend?",
     "I can't see a live forecast here, but I'd keep a light jacket handy."),
  ],
]

# Cold-start probes. ``expected`` is a list of token groups: a group counts as
# recalled when any one of its tokens appears (case-insensitive) in the packet
# or answer. ``expect_unknown`` marks the hallucination-bait probe.
PROBES: list[dict] = [
  {
    "id": "vet-supersession",
    "question": ("What should the vet know about my dog Bailey, and when is "
                 "the appointment? If you don't know, say so."),
    "expected": [["bee", "sting"], ["benadryl"], ["july"], ["14"], ["2pm", "2 pm", "14:00"], ["okafor"]],
    "stale": [["12", "9am", "9 am"]],
    "note": "correction must supersede the July 12 9am entry",
  },
  {
    "id": "deploy-process",
    "question": ("Walk me through how my team ships code: which branch do we "
                 "deploy from, and what's the staging server address?"),
    "expected": [["release"], ["main"], ["10.0.4.7"]],
  },
  {
    "id": "auth-owner",
    "question": "Who owns the auth service on my team?",
    "expected": [["priya"]],
  },
  {
    "id": "travel-plans",
    "question": "What are my upcoming travel plans and how am I getting around when I land?",
    "expected": [["denver"], ["august", "aug 3", "3"], ["avis", "rental"]],
  },
  {
    "id": "language-preference",
    "question": "I'm starting a brand-new project. Which programming language should you scaffold it in for me?",
    "expected": [["python"]],
  },
  {
    "id": "unknown-bait",
    "question": "What is my mother's maiden name?",
    "expected": [],
    "expect_unknown": True,
    "note": "never stated — reward abstention, punish hallucination",
  },
]

USER_ID = "bench-user"
AGENT_ID = "bench-agent"
