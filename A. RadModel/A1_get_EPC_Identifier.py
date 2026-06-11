# -*- coding: utf-8 -*-
"""
A1_get_EPC_Identifier.py

Retrieves the EPC (Energy Performance Certificate) Serial Identifier(s) for each
property by querying the EMOWeb API using the property's BFE number ("BFEnummer")
and/or address unit ID ("husnummer_ID").

For some old EPCs the BFE number is not registered, so the address unit ID is
used as a fallback to identify the EPC Serial Identifier.

Output: a pickle file containing a list of dictionaries, one per EPC entry found
for each property, including basic location and validity information.
"""

import copy
from datetime import datetime
import pickle

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _make_empty_prep_record(uid_key: str, uid_val) -> dict:
    """
    Build a placeholder record (all fields set to None) used when no EPC
    information could be found for a given identifier.

    uid_key: str
        Name of the identifier column (e.g. "BFEnummer" or "husnummer_ID").
    uid_val:
        Value of the identifier for the current property.
    """
    return {
        uid_key: uid_val,
        "streetName": None,
        "house_number": None,
        "PropertyNumber": None,
        "lat": None,
        "lon": None,
        "EPCSerial": None,
        "LabelStatus": None,
        "ValidFrom": None,
        "ValidTo": None,
    }


def _normalize_val(x):
    """Convert pandas/NumPy missing values (NaN) to None for stable dict keys."""
    if pd.isna(x):
        return None
    return x


# ---------------------------------------------------------------------------
# API fetcher
# ---------------------------------------------------------------------------
def get_EnergyLabelData(UID, search_by: str, EMAIL: str, PASSWORD: str) -> list:
    """
    Request energy label data for a single identifier from the EMOWeb API.

    UID:
        The identifier value to search by (BFE number or address unit ID).
    search_by: str
        Which identifier type UID represents. One of:
            "BFEnummer"    -> endpoint SearchEnergyLabelBFE/{UID}
            "husnummer_ID" -> endpoint SearchEnergyLabelUID/{UID}/AdgAdr,TL,BA
            "bygning_ID"   -> endpoint SearchEnergyLabelUID/{UID}/Bygning,TL,BA
    EMAIL: str
        EMOWeb account email used for HTTP basic authentication.
    PASSWORD: str
        EMOWeb account password used for HTTP basic authentication.

    Returns
    -------
    list
        A list containing one dict: {search_by: UID, "EPC_Identifier": <list or None>}
    """

    # ---- Configuration / session ----
    BASE_URL = "https://emoweb.dk/EMOData/EMOData.svc/"
    AUTH = HTTPBasicAuth(EMAIL, PASSWORD)
    SESSION = requests.Session()

    # decide endpoint and date format used for ValidFrom/ValidTo parsing
    if search_by == "BFEnummer":
        endpoint = "SearchEnergyLabelBFE/{}"
        parse_fmt = "%d-%m-%Y"
    elif search_by == "husnummer_ID":
        endpoint = "SearchEnergyLabelUID/{}/AdgAdr,TL,BA"
        parse_fmt = "%Y-%m-%d"
    elif search_by == "bygning_ID":
        endpoint = "SearchEnergyLabelUID/{}/Bygning,TL,BA"
        parse_fmt = "%Y-%m-%d"
    else:
        raise ValueError("Invalid search_by parameter")

    url = BASE_URL + endpoint.format(UID)
    try:
        resp = SESSION.get(url, auth=AUTH, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        # If the request fails for any reason, return EPC_Identifier = None
        return [{search_by: UID, "EPC_Identifier": None}]

    # pick items container and EPC id key depending on search type
    if search_by in ("husnummer_ID", "bygning_ID"):
        items = data.get("SearchResults", [])
        epc_id_key = "EntityIdentifier"
    else:  # BFEnummer
        items = data.get("EnergyLabels", [])
        epc_id_key = "EnergyLabelSerialIdentifier"

    epc_list = []
    for item in items:
        try:
            valid_from_raw = item.get("ValidFrom")
            valid_to_raw = item.get("ValidTo")

            valid_from = (
                datetime.strptime(valid_from_raw, parse_fmt)
                if valid_from_raw
                else None
            )
            valid_to = (
                datetime.strptime(valid_to_raw, parse_fmt) if valid_to_raw else None
            )

            epc_list.append(
                {
                    "EPCSerial": item.get(epc_id_key),
                    "streetName": item.get("StreetName"),
                    "house_number": item.get("HouseNumber"),
                    "PropertyNumber": item.get("PropertyNumber"),
                    "lat": item.get("Wgs84Latitude"),
                    "lon": item.get("Wgs84Longitude"),
                    "LabelStatus": item.get("LabelStatus"),
                    "ValidFrom": valid_from,
                    "ValidTo": valid_to,
                }
            )
        except Exception:
            # skip items that cannot be parsed
            continue

    return [{search_by: UID, "EPC_Identifier": epc_list}]


# ---------------------------------------------------------------------------
# Reshape API results into flat records
# ---------------------------------------------------------------------------
def prep_df(data: list, UID: str) -> list:
    """
    Convert the result of get_EnergyLabelData() into a flat list of per-EPC
    dictionaries.

    data: list
        Output of get_EnergyLabelData() - a list containing one dict.
    UID: str
        Name of the identifier column used in `data` (e.g. "BFEnummer").

    Returns
    -------
    list of dict
        One dict per EPC entry found, with a fallback (all-None) record if
        nothing was found for the given identifier.
    """
    result = []
    for subset in data:
        uid_val = subset.get(UID)
        if uid_val is None:
            continue

        epc_entries = subset.get("EPC_Identifier")
        if not epc_entries:
            result.append(_make_empty_prep_record(UID, uid_val))
            continue

        for entry in epc_entries:
            result.append(
                {
                    UID: uid_val,
                    "streetName": entry.get("streetName"),
                    "house_number": entry.get("house_number"),
                    "PropertyNumber": entry.get("PropertyNumber"),
                    "lat": entry.get("lat"),
                    "lon": entry.get("lon"),
                    "EPCSerial": entry.get("EPCSerial"),
                    "LabelStatus": entry.get("LabelStatus"),
                    "ValidFrom": entry.get("ValidFrom"),
                    "ValidTo": entry.get("ValidTo"),
                }
            )

    return result


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------
def retrieve_EPC(stamdata: pd.DataFrame, EMAIL: str, PASSWORD: str, path_epc_id: str) -> None:
    """
    Fetch EPC identifiers for all properties in `stamdata` and save the result
    to `path_epc_id`.

    EPC data is fetched once per unique (BFEnummer, husnummer_ID) pair and then
    mapped back to all rows sharing that pair, so repeated lookups are avoided.

    stamdata: pd.DataFrame
        Master ID dataframe. Must contain the columns "BFEnummer" and
        "husnummer_ID" (one row per property).
    EMAIL: str
        EMOWeb account email used for HTTP basic authentication.
    PASSWORD: str
        EMOWeb account password used for HTTP basic authentication.
    path_epc_id: str
        File path (.pkl) where the resulting list of EPC identifier records
        is saved.

    Returns
    -------
    None
        The result is written to `path_epc_id` as a pickle file.
    """
    # Build list of unique pairs (normalize NaN -> None so keys are stable)
    pairs_df = stamdata[["BFEnummer", "husnummer_ID"]].copy()
    pairs_df["BFEnummer"] = pairs_df["BFEnummer"].apply(_normalize_val)
    pairs_df["husnummer_ID"] = pairs_df["husnummer_ID"].apply(_normalize_val)
    unique_pairs = pairs_df.drop_duplicates().to_records(index=False)

    pair_cache = {}

    for bfe_val, husnr_val in unique_pairs:
        # Look up EPCs via husnummer_ID
        if husnr_val is not None:
            try:
                res_hus = get_EnergyLabelData(husnr_val, "husnummer_ID", EMAIL, PASSWORD)
                epc_husnr_valid = prep_df(res_hus, UID="husnummer_ID")
            except Exception:
                epc_husnr_valid = []
        else:
            epc_husnr_valid = [
                {
                    "BFEnummer": bfe_val,
                    "husnummer_ID": husnr_val,
                    "lat": None,
                    "lon": None,
                    "streetName": None,
                    "house_number": None,
                    "PropertyNumber": None,
                    "EPCSerial": None,
                    "LabelStatus": None,
                    "ValidFrom": None,
                    "ValidTo": None,
                }
            ]

        # Look up EPCs via BFEnummer
        if bfe_val is not None:
            try:
                res_bfe = get_EnergyLabelData(bfe_val, "BFEnummer", EMAIL, PASSWORD)
                epc_bfe_valid = prep_df(res_bfe, UID="BFEnummer")
            except Exception:
                epc_bfe_valid = []
        else:
            epc_bfe_valid = [
                {
                    "BFEnummer": bfe_val,
                    "husnummer_ID": husnr_val,
                    "lat": None,
                    "lon": None,
                    "streetName": None,
                    "house_number": None,
                    "PropertyNumber": None,
                    "EPCSerial": None,
                    "LabelStatus": None,
                    "ValidFrom": None,
                    "ValidTo": None,
                }
            ]

        # Enrich and combine results from both lookups
        for r in epc_bfe_valid:
            r["husnummer_ID"] = husnr_val
        for r in epc_husnr_valid:
            r["BFEnummer"] = bfe_val

        combined = epc_bfe_valid + epc_husnr_valid

        # Deduplicate by EPCSerial
        seen = set()
        unique_records = []
        for r in combined:
            serial = r.get("EPCSerial")
            if serial not in (None, "") and serial not in seen:
                unique_records.append(r)
                seen.add(serial)

        pair_cache[(bfe_val, husnr_val)] = unique_records

    # Map cached results back to the original rows
    EPC_IDs_all = []
    for row in stamdata.itertuples(index=False):
        bfe_key = _normalize_val(getattr(row, "BFEnummer", None))
        husnr_key = _normalize_val(getattr(row, "husnummer_ID", None))
        cached = pair_cache.get((bfe_key, husnr_key), [])

        row_records = [copy.deepcopy(r) for r in cached]

        if not row_records:
            row_records = [
                {
                    "BFEnummer": bfe_key,
                    "husnummer_ID": husnr_key,
                    "lat": None,
                    "lon": None,
                    "streetName": getattr(row, "vejnavn", None),
                    "house_number": getattr(row, "husnummer", None),
                    "PropertyNumber": None,
                    "EPCSerial": None,
                    "LabelStatus": None,
                    "ValidFrom": None,
                    "ValidTo": None,
                }
            ]

        EPC_IDs_all.extend(row_records)

    with open(path_epc_id, "wb") as f:
        pickle.dump(EPC_IDs_all, f)
