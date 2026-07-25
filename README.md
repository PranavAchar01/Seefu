# Seefu

**A self-supervised vision layer that trains only on correct dishes and flags every mis-plated, missing-ingredient, contaminated, or allergen-cross-contact plate before it leaves the pass, giving robot-run kitchens the one thing they structurally lack: a human eye on the food, at 30fps, on every single plate.**

---

## The problem

Autonomous kitchens have solved cooking. They have not solved *looking*.

A robotic line can dose, sear, plate, and pass a dish without a person touching it. What disappears along with the labor is the last quality gate that every restaurant has always relied on: an expediter glancing at the plate and catching the one that is wrong. The sauce that never landed. The protein that got skipped. The garnish from the station that handles peanuts, sitting on a ticket marked allergy.

Robot kitchens scale labor to zero and scale visual QA to zero with it. Seefu puts it back.

## What Seefu does

Seefu sits at the pass as a camera and an inference service. Every plate that crosses it is scored in real time against what that dish is supposed to look like, and anything outside the envelope is held before it reaches a customer.

- **Mis-plating**: wrong component counts, wrong arrangement, wrong portion geometry
- **Missing ingredients**: a component the recipe requires that is not on the plate
- **Contamination**: foreign objects, spill-over, debris, packaging fragments
- **Allergen cross-contact**: visual traces of a component that must not be on this ticket
- **Presentation drift**: slow degradation in plating consistency across a shift

Each verdict comes back with a per-pixel heat map showing *where* the anomaly is, not just that one exists, so a line operator or a remote supervisor can act on it in seconds.

## Why it works without defect data

The hard part of visual QA in food is that defects are unbounded. You cannot photograph ten thousand examples of a hair in the soup, a cracked shell fragment, or a sauce that broke. Any approach that needs labeled failures is dead before it starts.

Seefu inverts the problem. It trains **only on correct plates**, which a kitchen produces by the thousand every day at zero marginal collection cost, and learns a dense model of what "right" looks like for that dish. Anything that does not fit that model is an anomaly, including failure modes that no one anticipated and no one has ever seen before.

This means:

- **No defect labeling.** Deployment for a new menu item needs a few hundred good plates, not an annotation contract.
- **New dishes onboard in a shift.** Cook the item normally, let the camera watch, ship the model.
- **Unknown failures are caught.** The system does not need to have been taught a specific defect in order to reject it.

## Why it matters commercially

Unmanned food service is currently uninsurable in the way manned service is insurable, because there is no auditable record that anything checked the food. Seefu produces one: a timestamped image, a verdict, and a localization map for every plate served.

That record turns into three things a robot restaurant operator can actually sell to a regulator, an insurer, and a franchisor:

1. **Liability defense.** Evidence of a per-plate inspection on every unit served.
2. **Allergen compliance.** A visual control on cross-contact, not just a procedural one.
3. **Brand consistency.** Objective plating conformance across every location, measured continuously instead of by quarterly audit.

## System overview

```
  Plate photo (upload or pass camera)
            │
            ▼
   Golden-plate registration        ORB + RANSAC homography to the reference
            │
            ▼
   PatchCore anomaly scoring        trained on good plates only, CPU, ~52s
            │
            ▼
   Pixel-level localization         tight contours on the anomalous regions
            │
            ▼
   Comparative vision (gpt-4o)      what is wrong + the FIX, per region
            │
            ├──►  pass / HOLD verdict + annotated plate image
            ├──►  Manager call on a hold (Twilio + ElevenLabs voice)
            └──►  Kitchen memory (XTrace): file, recall, brief
```

Inference runs on-device at the pass so that a network outage never becomes a food safety outage. The vision and memory layers degrade gracefully: without keys, the station still scores, localizes, and holds.

## The model

The detector is PatchCore (anomalib 2.5.1): a frozen wide_resnet50_2 backbone, layer 2 and 3 features, a coreset-subsampled memory bank of what correct plates look like. This repository ships an 80-image training set of a sesame beef bowl (`data/normal`, one real plate photo plus 79 mild photometric and geometric augmentations, parameters in `data/manifest.csv`).

Training takes under a minute on a laptop CPU and never sees a defect. The pass threshold is calibrated from the score distributions: every normal plate scores below 47.6, a plate with a foreign object scores 51.5, a plate missing its broccoli scores 73.8, and the threshold sits between at 49.5. Uploads shot off the pass are registered onto the golden plate first, so background, angle and scale differences do not read as anomalies.

Each flagged region is then cropped and sent to gpt-4o NEXT TO the same region of the golden plate, and the model answers in one line: what is wrong, and the fix the kitchen takes. Food safety inverts the usual confidence rule: a hedged answer downgrades a finding, UNLESS it touches contamination, where doubt itself holds the plate.

## Kitchen memory: XTrace

PatchCore sees one plate at a time and forgets it the moment the verdict lands. [XTrace](https://docs.xtrace.ai) is the layer that remembers, and it is wired into four places (`core/memory.py`):

1. **Every inspection is filed.** After each verdict, the result is serialized into a natural-language event and ingested (`POST /v1/memories`). XTrace's extraction pipeline turns it into searchable facts and per-shift episode summaries on its own.

2. **Every failure is checked against the past.** Before a hold verdict reaches the operator, Seefu searches kitchen memory for semantically similar past failures (`POST /v1/memories/search`, retrieve mode). One earlier match is a note. Two or more flips the message: this is not a plating slip, this is an ingredient station that is empty or misloaded, fix the line before the next plate fires.

3. **Chef corrections persist.** "Extra scallion is fine on this dish, never flag it" is ingested as a directive memory (outcome resolved) and recalled through XTrace's unmetered trigger endpoint on future plates of the same dish, so the station learns kitchen policy without retraining anything.

4. **Operators grade every verdict, and the station learns.** Right and Wrong buttons file each judgement as feedback memory. Two concordant overrules of the same kind of hold teach the station to stop holding plates for it (contamination findings are exempt by design: a system that can be taught to ignore foreign objects has no business near food). A wrong pass becomes a watch item that warns on future passes.

5. **An analyst reads the memory after every run.** gpt-4o receives the current result plus everything memory recalled about it and answers the head chef's question: one-off or trend, what is the physical root cause, what single action fixes the line ("emerging pattern of missing scallion garnish, hopper likely empty, refill it now"). A deeper pass over the whole memory pool (`/memory/trends`, the Long-running issues button) names the systemic problems with prioritized fixes.

6. **The memory ledger** (`/memory-dashboard`) shows everything XTrace holds: live usage counters from XTrace's own meter, and every fact, episode, correction and feedback entry in a filterable table.

Why this matters in a real kitchen: the third missing-garnish plate during a rush is not three independent mistakes, it is one empty scallion hopper. A stateless inspector rejects three plates and lets the kitchen fire a fourth. A remembering inspector rejects the first two, names the pattern on the third, and tells a human which station to check. And when the night crew walks in, they inherit the day's failure patterns as a two-paragraph briefing instead of tribal knowledge that walked out with the last shift.

## Retraining

The bank learns only from known-good plates, so improving the model is an operational loop, not a data science project:

1. `POST /train/upload` stages new good-plate photos into `data/normal` (or use the Add Good Plates button on the station page).
2. `POST /train/rebuild` retrains the memory bank and recalibrates the threshold in the background.
3. The serving model hot-swaps on the new checkpoint's mtime. The server never goes down.

New dish, new lighting, new plating standard: cook it correctly a few dozen times, upload, rebuild, done.

## Manager calls

A held plate rings the kitchen manager: Twilio places the call and an ElevenLabs-synthesized voice reads the case, dish id, what is wrong, and the fix, twice. Announce-only by design; the annotated plate stays on the dashboard.

## Quickstart

Python 3.10+, an OpenAI API key, and optionally XTrace and Twilio/ElevenLabs keys.

```bash
git clone https://github.com/PranavAchar01/Seefu.git
cd Seefu
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in the keys
make bank              # train PatchCore on data/normal (~1 min, CPU)
make calibrate         # score normals + test defects, write the threshold
make serve             # station page on http://localhost:8000
```

`.env` keys: `OPENAI_API_KEY` (vision explanations + case verification), `XTRACE_API_KEY` (kitchen memory), `TWILIO_ACCOUNT_SID/AUTH_TOKEN/FROM/TO` and `ELEVENLABS_API_KEY` (manager calls). Everything degrades gracefully without them.

## Project structure

```
seefu/
  inspection/train_bank.py  # PatchCore memory bank builder
  inspection/serve.py       # FastAPI: /inspect, /train/*, /memory/*, station page
  inspection/explain.py     # comparative gpt-4o: what is wrong + the fix
  inspection/verify.py      # consistency check -> verified stamp per case
  core/memory.py            # XTrace: file every plate, recall failures, brief shifts
  core/phone.py             # Twilio + ElevenLabs manager alerts
  core/frames.py            # shared ingest geometry
  core/casesink.py          # case records; core/history.py: sqlite shift stats
  dashboard/minimal.html    # the station page
  data/normal/              # the 80-plate good set; data/test_defect/: calibration foils
  capture/, scripts/, tests/
```

## Roadmap

- [x] Golden-plate dataset (80 normals, manifest included)
- [x] Per-dish anomaly model training pipeline
- [x] Real-time scoring service with localization overlays
- [x] Threshold calibration from normal and synthetic-defect distributions
- [x] Fix suggestions per finding (comparative vision)
- [x] Kitchen memory: recall, corrections, shift briefings (XTrace)
- [x] Retraining loop with hot model swap
- [x] Manager call alerts (Twilio + ElevenLabs)
- [ ] 30fps pass-side video sampling (currently per-plate stills)
- [ ] Multi-dish menus (per-dish banks + dish ID routing)
- [ ] Multi-camera and multi-station support
- [ ] Audit log evidence export

## License

TBD
