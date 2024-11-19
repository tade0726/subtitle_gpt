format:
	black src/

run:
	uv run src/subtitle_gpt/main.py

ell-dev:
	ell-studio --storage ./logdir
