# LTReady

LTReady is a research tool for assessing the **low-temperature district
heating (LTDH) readiness** of Danish multi-family buildings. It combines Danish Energy
Performance Certificate (EPC) data, building registry (BBR) data, hourly
heat meter data, and weather data to estimate, for each building or group of
buildings ("component"):

* the nominal (EPC-derived) radiator design parameters and supply/return
temperatures,
* the data-driven, "actual" predicted peak space heating heat loss, and
* the minimum supply and return temperatures the building's existing
radiator system can support — i.e. how compatible it is with low- or
medium-temperature district heating.

## Table of Contents

* [Project Structure](#project-structure)
* [Workflow Overview](#workflow-overview)
* [Requirements](#requirements)
* [Dependencies and Licenses](#dependencies-and-licenses)
* [Required Input Data](#required-input-data)
* [Usage](#usage)
* [Data Sources / API Documentation](#data-sources--api-documentation)
* [Citation](#citation)
* [License](#license)

## Project Structure

```
Public\\\_Tool/
├── A. RadModel/        # Part A: EPC data and nominal radiator design values
│   ├── A1\\\_get\\\_EPC\\\_Identifier.py
│   ├── A2\\\_get\\\_EPC\\\_XML.py
│   ├── A3\\\_get\\\_design\\\_values.py
│   ├── A4\\\_get\\\_heatloss.py
│   ├── A5\\\_get\\\_DHTemperatures.py
│   ├── A6\\\_get\\\_BBRfromXML.py
│   ├── A7\\\_get\\\_heatCalcfromXML.py
│   └── A8\\\_get\\\_aggXML.py
├── B. SHModel/         # Part B: data-driven predicted space heating model
│   ├── B1\\\_get\\\_processedHM.py
│   ├── B2\\\_get\\\_disaggregated.py
│   ├── B3\\\_get\\\_weather\\\_data.py
│   ├── B4\\\_get\\\_addProcessHM.py
│   ├── B5\\\_get\\\_timeseries.py
│   ├── B6\\\_get\\\_actualUAs.py
│   ├── B7\\\_get\\\_actualHeat.py
│   ├── B8\\\_get\\\_energy.py
│   └── B9\\\_get\\\_aggHM.py
├── C. Mapping/         # Part C: minimum supply/return temperatures and visualization
│   ├── C1\\\_get\\\_actualDHTemperatures.py
│   └── C2\\\_get\\\_df\\\_visuals.py
└── run\\\_LTReady.py      # Main entry point - run this script
```

## Workflow Overview

`run\\\_LTReady.py` is the single entry point and runs three workflows in sequence:

1. **Part A - EPC data and radiator design values**: retrieves EPC records
via the EMOWeb API, calculates historical design heat losses and nominal
radiator parameters (K-value, design mass flow, design supply/return
temperatures), and aggregates these onto user-defined "components".
2. **Part B - Predicted peak space heating consumption**: processes raw
hourly heat meter data, disaggregates space heating from domestic hot
water, retrieves weather data, trains a grey-box space heating model per
meter, and predicts peak heat loss and annual energy consumption under
specified outdoor design conditions.
3. **Part C - Minimum supply/return temperatures**: combines the outputs of
Parts A and B to estimate the minimum supply and return temperatures each
component's radiator system can support, flags outliers, and produces the
final dataset and an interactive map.

See the docstrings in `run\\\_LTReady.py` and each module for details on inputs,
outputs and parameters.

## Requirements

* Python 3.10+ (developed and tested with Python 3.12)
* A DMI Open Data API key (free, see [Data Sources](#data-sources--api-documentation))
* An EMOWeb account (for retrieving EPC data, see [Data Sources](#data-sources--api-documentation))

Install the required Python packages with:

```bash
pip install pandas numpy requests scipy statsmodels plotly pyarrow
```

## Dependencies and Licenses

The table below lists the third-party packages used by LTReady and their
licenses. LTReady does not redistribute any of these packages; they are
listed here for copyright/license awareness when using or redistributing
this tool. Always check the package's own repository/PyPI page for the
current license terms.

|Package|Used for|License|
|-|-|-|
|[pandas](https://github.com/pandas-dev/pandas)|Data wrangling and tabular data structures|[BSD 3-Clause](https://github.com/pandas-dev/pandas/blob/main/LICENSE)|
|[numpy](https://github.com/numpy/numpy)|Numerical computing|[BSD 3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt)|
|[requests](https://github.com/psf/requests)|HTTP requests to the EMOWeb and DMI APIs|[Apache License 2.0](https://github.com/psf/requests/blob/main/LICENSE)|
|[scipy](https://github.com/scipy/scipy)|Optimization (model fitting)|[BSD 3-Clause](https://github.com/scipy/scipy/blob/main/LICENSE.txt)|
|[statsmodels](https://github.com/statsmodels/statsmodels)|OLS regression (cross-year deviation check, model fitting statistics)|[BSD 3-Clause](https://github.com/statsmodels/statsmodels/blob/main/LICENSE.txt)|
|[plotly](https://github.com/plotly/plotly.py)|Interactive map visualization|[MIT License](https://github.com/plotly/plotly.py/blob/master/LICENSE.txt)|
|[pyarrow](https://github.com/apache/arrow)|Reading/writing parquet files|[Apache License 2.0](https://github.com/apache/arrow/blob/main/LICENSE.txt)|

The following Python standard library modules are also used and require no
separate installation: `pickle`, `os`, `ast`, `copy`, `itertools`,
`datetime`, `typing`, `xml.etree.ElementTree`.

## Required Input Data

LTReady requires the following inputs, prepared by the user:

1. **Master ID file** (`master\\\_ids\\\_{CASE\\\_NAME}.pkl`): a pickled `pandas.DataFrame` with
one row per property, containing at least:

   * `BFEnummer` - BFE number (Danish property identifier), or `NaN`
   * `husnummer\\\_ID` - address unit ID, or `NaN`
   * `address` - property address (used to merge meter data, optional)
   * `lat`, `lon` - property coordinates (used for the map, optional)
   * `vejnavn`, `husnummer` - street name / house number (fallback display
fields if no EPC is found, optional)
2. **Component ID file** (`component\\\_ids\\\_{CASE\\\_NAME}.pkl`): a pickled `pandas.DataFrame`
defining how buildings/meters are grouped into "components" (e.g. all
buildings sharing a single district heating substation), containing:

   * `component\\\_id` - user-defined ID for each component
   * `bfe` - BFE number(s) belonging to this component
   * `epc` - EPC Serial(s) belonging to this component (optional)
   * `serial` - meter serial number(s) belonging to this component
   * `n\\\_serial` - number of meter serials in this component
3. **Raw hourly heat meter data** (one parquet file per year, e.g. `hourly\\\_meter\\\_raw\\\_{CASE\\\_NAME}\\\_{YEAR1}`): hourly heat
meter readings containing at least `timestamp`, `serial`, `energy`, `volume` , `heat\\\_power`,
`volume\\\_flow`, `inlet\\\_flow\\\_temperature`, `return\\\_flow\\\_temperature`, and
any identifier columns listed in `meter\\\_id\\\_list`.
4. **EMOWeb credentials** (`EMAIL`, `PASSWORD`) and a **DMI Open Data API
key** (`API\\\_KEY\\\_DMI`), set in `run\\\_LTReady.py`.
5. **DMI weather station IDs** for outdoor temperature, wind direction, wind
speed and solar radiation (`STATION\\\_TEMPERATURE`,
`STATION\\\_WIND\\\_DIRECTION`, `STATION\\\_WIND\\\_SPEED`,
`STATION\\\_SOLAR\\\_RADIATION`), set in `run\\\_LTReady.py`.
6. **Heating degree-days** (`HD\\\_year1`, `HD\\\_year2`) for the relevant year(s)
and location.

Full details of the expected file formats are documented in the "Required
input data" section at the top of `run\\\_LTReady.py`.

## Usage

1. Install the required packages (see [Requirements](#requirements)).
2. Place your raw hourly meter data files in `data/raw/`.
3. Make sure the file and variable names of the Master ID, Component ID and heat meter data files match as specified.
4. Open `run\\\_LTReady.py` and fill in the **USER-SPECIFIC INPUT** section:
`CASE\\\_NAME`, `YEAR1`/`YEAR2`, `EMAIL`/`PASSWORD`, `API\\\_KEY\\\_DMI`, the DMI
station IDs, `path\\\_comp\\\_id`, `path\\\_master\\\_id`, `meter\\\_id\\\_list`, the
outdoor design conditions (`Tout`, `Ws`, `Tin`, `Radn`), heating
degree-days, and the heat exchanger temperature offset (`temp\\\_HEX`).
5. Run the script from the project root:

```bash
python run\\\_LTReady.py
```

5. Intermediate results are written to `data/processed/` and final results
to `data/final/`. The final per-component datasets are
`final\\\_df\\\_mnorm\\\_<CASE\\\_NAME>.pkl` and `final\\\_df\\\_deltaT\\\_<CASE\\\_NAME>.pkl`,
and an interactive map is saved as `<CASE\\\_NAME>\\\_map\\\_plot.html`.

If only one year of meter data is available, set `YEAR2 = None` and
`HD\\\_year2 = None`; the workflow will automatically skip all year-2-specific
and cross-year processing steps.

## Data Sources / API Documentation

LTReady retrieves data from two external APIs. Familiarize yourself with
their documentation before running the tool:

* **Energy Performance Certificate (EPC) data** is retrieved from the Danish
Energy Agency's EMOWeb service. For background on the EPC data and how to
obtain access (an EMOWeb account/login), see the Danish Energy Agency's
page on energy labelling data:
[https://ens.dk/analyser-og-statistik/energimaerkningsdata](https://ens.dk/analyser-og-statistik/energimaerkningsdata)
* **Weather data** (outdoor temperature, wind direction, wind speed, solar
radiation) is retrieved from the Danish Meteorological Institute (DMI)
Open Data API. For API documentation, available datasets, parameter
definitions and how to obtain a free API key, see:
[https://www.dmi.dk/friedata/dokumentation/apis](https://www.dmi.dk/friedata/dokumentation/apis)

## Citation

If you use LTReady in your work, please cite:

> \\\[Author Name(s)], "\\\[Title of the associated paper/report/repository]",
> \\\[Journal/Conference/Repository], \\\[Year]. \\\[DOI or URL]

```bibtex
@misc{ltready,
  author = {\\\[Author Name(s)]},
  title  = {LTReady: A tool for assessing low-temperature district heating readiness of multi-family buildings},
  year   = {\\\[Year]},
  url    = {\\\[Repository URL]},
  note   = {\\\[DOI, if available]}
}
```

*(Replace the bracketed fields above with your name, the relevant
publication or repository details, and a DOI if one is assigned, e.g. via*
[*Zenodo*](https://zenodo.org)*.)*

## License

This project is licensed under the BSD 3-Clause Clear License - see the

\[LICENSE](LICENSE) file for details.



Copyright © 2026 Technical University of Denmark.

