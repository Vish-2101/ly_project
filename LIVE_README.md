# Live dashboard

Add to your project root (next to `env/`, `api.py`, ...):
- `live_app.py`
- `templates/model.html`

Required model artifacts in the project root:
- `ppo_dynamic_pricing_improved.zip`, `ppo_vecnormalize.pkl`
- `sac_dynamic_pricing.zip`,           `sac_vecnormalize.pkl`
- `a2c_dynamic_pricing.zip`,           `a2c_vecnormalize.pkl`

Run:
    pip install -r requirements.txt
    python live_app.py
Open http://localhost:5000/ppo , /sac , /a2c

The default data source is `deterministic_pricing_dataset.csv`. It is an Excel
workbook with a `.csv` filename, and is read correctly by the application.
Options (env vars): TICK_SECONDS=1.0  PORT=5000  DATA_PATH=deterministic_pricing_dataset.csv
If an artifact is missing, that model's prediction endpoint returns an error
instead of serving fabricated or heuristic predictions.
Use the Product dropdown on each page to select one product. The stream stays
on that product until you choose another, and the chart includes a moving
average price line. The selected model/product position and its last date and
price are persisted in `.live_state.json`, so restarting the app resumes from
the previous dataset record. Historical predictions are stored in
`.live_history.json`; use the chart history slider to browse earlier trends,
then press `Live` to follow new ticks again.

Use **Add new item** on a model page to persist a product in `.live_state.json`.
The form stores the company and product type plus CP (cost price), SP (selling
price), MP (market price), and starting price. CP, MP, and starting price feed
the live pricing environment; the starting price is the initial reference
price. Products can be removed with **Remove**; at least one product must
remain.
