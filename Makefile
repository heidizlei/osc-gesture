# Per-machine settings go in local.mk (gitignored), e.g.
#   NOTARY_PROFILE = lancelot
#   PYINSTALLER = /opt/miniconda3/envs/pipe/bin/pyinstaller
# Any of them can also be given for one run: make notarize NOTARY_PROFILE=...
-include local.mk

PYINSTALLER ?= .venv/bin/pyinstaller
CODESIGN_IDENTITY ?= Developer ID Application
NOTARY_PROFILE ?= osc-gesture

SPEC := osc-gesture.spec
APP := dist/OSC Gesture.app
WATCH_PATHS := main.py classes osc-gesture.spec requirements.txt hand_landmarker.task ui.html

.PHONY: build sign check notarize release watch clean

build:
	$(PYINSTALLER) --noconfirm $(SPEC)

# Developer ID signing and notarization, for sharing the app; see "Packaging"
# in README.md.
sign:
	CODESIGN_IDENTITY="$(CODESIGN_IDENTITY)" packaging/sign.sh "$(APP)"

# Launches the built app to make sure it starts. Runs after signing so the
# hardened runtime is in effect, and before notarizing so a broken build never
# gets uploaded.
check:
	packaging/smoke_test.sh "$(APP)"

notarize:
	NOTARY_PROFILE="$(NOTARY_PROFILE)" packaging/notarize.sh "$(APP)"

# Code change -> shareable zip, in one command.
release: build sign check notarize

# -l sets fswatch's own batching window (seconds); the inner `read -t` loop
# then drains any further events for another quiet 1s before rebuilding, so a
# burst of saves collapses into a single rebuild instead of one per file.
watch:
	@echo "Watching $(WATCH_PATHS) for changes (Ctrl+C to stop)..."
	@$(MAKE) build
	@fswatch -o -l 1 $(WATCH_PATHS) | while read -r _; do \
		while read -r -t 1 _; do :; done; \
		echo "Change detected, rebuilding..."; \
		$(MAKE) build; \
	done

clean:
	rm -rf build dist
