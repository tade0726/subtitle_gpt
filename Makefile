format:
	black src/

run:
	uv run src/subtitle_gpt/main_ell.py

ell-dev:
	ell-studio --storage ./logdir
