PYINSTALLER := /opt/miniconda3/envs/pipe/bin/pyinstaller
SPEC := osc-gesture.spec
WATCH_PATHS := main.py classes osc-gesture.spec requirements.txt hand_landmarker.task

.PHONY: build watch clean

build:
	$(PYINSTALLER) --noconfirm $(SPEC)

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
