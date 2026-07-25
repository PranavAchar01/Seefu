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
  Overhead camera at the pass
            │
            ▼
   Plate detection + dish ID
            │
            ▼
   Per-dish anomaly scoring   ──►  heat map + score
            │
            ▼
   Threshold + hold decision  ──►  pass / hold / alert
            │
            ▼
   Audit log (image, verdict, map, ticket ID)
```

Inference runs on-device at the pass so that a network outage never becomes a food safety outage. The audit log syncs when connectivity returns.

## Status

Early development. The initial target is a single-camera pass-side deployment on a fixed menu, with the reference model, capture tooling, and scoring service in this repository.

## Roadmap

- [ ] Capture tooling for golden-plate dataset collection
- [ ] Per-dish anomaly model training pipeline
- [ ] Real-time scoring service with heat map output
- [ ] Threshold calibration UI for per-dish sensitivity
- [ ] Audit log and evidence export
- [ ] Multi-camera and multi-station support
- [ ] Edge deployment package

## License

TBD
