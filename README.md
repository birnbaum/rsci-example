# rSCI: Reconciled Software Carbon Intensity

Example to reproduce the exemplary LLM example comparing SCI, oSCI, and rSCI in the paper "Reconciling Bottom-Up Metrics with Top-Down Reporting for Cloud Carbon Accounting".

## Data

The bundled workload is a minute-aggregated excerpt of Microsoft's [Azure LLM inference trace](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md) (Stojkovic et al., *DynamoLLM*, HPCA 2025), licensed under [CC BY 4.0](AZURE-DATA-LICENSE.txt).
Obtain hourly consumption-based carbon intensity with lifecycle emission factors from [Electricity Maps](https://portal.electricitymaps.com/) for zone `US-CAL-CISO`, from `2024-05-10T07:00:00Z` (inclusive) to `2024-05-13T07:00:00Z` (exclusive).
Save it as `data/carbon_intensity.csv` with columns `datetime,US-CAL-CISO`, exactly 72 hourly rows, timestamps with UTC offsets, and values in gCO2eq/kWh.

## Run

Run these commands from this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce_example.py
```

Use `--carbon-intensity PATH` for another input location and `--output-dir PATH` to change the output directory.
The figure (`sec5_example.pdf`), totals (`metrics.json`), and hourly allocations (`hourly_allocations.csv`) are written to `output/`.

Code is [MIT licensed](LICENSE).
