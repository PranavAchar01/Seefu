PY := .venv/bin/python

capture:
	$(PY) capture/capture.py

bank:
	$(PY) inspection/train_bank.py

serve:
	$(PY) inspection/serve.py

seed:
	$(PY) scripts/seed_history.py

tunnel:
	$(PY) scripts/start_tunnel.py

call:
	curl -s -X POST localhost:8000/phone/call

verify-voice:
	$(PY) scripts/verify_elevenlabs.py

calibrate:
	$(PY) inspection/serve.py --calibrate

.PHONY: capture bank serve seed tunnel call verify-voice calibrate
