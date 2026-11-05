# Reaction Optimization Lab

A classroom game for comparing experiment strategies. Teams search for the best operating conditions of a black-box flow reactor with a limited number of experiments, using one factor at a time, a designed experiment or their own approach. The instructor can then run Bayesian optimization live on the same problem and reveal the true response surface.

The reactor is a series reaction A → P → S in a cooled plug-flow tube. **The results are for teaching, not for real process decisions.**

## Pages

| Page | For | What it does |
|---|---|---|
| Experiment | Teams | Run single experiments, download runs, state a final recommendation |
| Design | Teams | Create a full factorial, Box-Behnken or central composite design for chosen ranges, and run it |
| Analyse | Teams | Look at runs, fit a response surface model, inspect effects and slices, test the predicted optimum |
| Leaderboard | Everyone | Best measured margin per team |
| Instructor | Instructor | Live Bayesian optimization, the reveal, budget and factor settings, reset |

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Fill in the instructor password and the reactor constants.
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a public repository on github.com.
2. Create an app at [share.streamlit.io](https://share.streamlit.io), with `app.py` as the main file.
3. Under **Advanced settings**, choose Python 3.13 and paste the content of your local `secrets.toml` into **Secrets**.

The model structure is public, but the constants are not in this repository. Never commit `.streamlit/secrets.toml`.

## Before and during class

- Open the app a few minutes before class, because Community Cloud puts idle apps to sleep.
- The game state is held in memory and in a temporary file, and may be lost if the app restarts. Teams should download their runs. They can restore them on the Experiment page.
- Use short team codes, not names. There is no login for teams, and anyone with the link can see the Leaderboard.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use made-up constants. One extra test checks the real constants, and runs only if a local `secrets.toml` exists.
