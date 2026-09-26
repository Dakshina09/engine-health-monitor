"""Privacy policy and terms and conditions pages.

Every statement here describes what this code actually does. If you add
analytics, sign-in, forms or third-party scripts, update these pages first.
"""
from __future__ import annotations

import streamlit as st

from site_config import (CONTACT_URL, EFFECTIVE_DATE, GOVERNING_LAW, OWNER,
                         SITE_NAME, SITE_URL)

_SITE = SITE_URL or "this website"


def _header(title: str) -> None:
    st.title(title)
    st.caption(f"Effective {EFFECTIVE_DATE}")


def privacy_policy() -> None:
    _header("Privacy policy")
    st.markdown(f"""
{SITE_NAME} ("the site", {_SITE}) is run by {OWNER}. This page explains what information the site
handles when you use it.

### What the site does not collect

- There are no user accounts, sign-up forms or newsletters.
- The site does not set advertising or tracking cookies and does not use third-party analytics.
- Streamlit's built-in usage statistics are switched off in the site configuration.

### What is processed when you use the site

- **Your requests.** When you open a page or change a filter, your browser sends a request to the
  server so it can return the chart or table you asked for. The engine data shown is the public
  NASA C-MAPSS simulated dataset, not data about you.
- **Data you send to the API.** If you call the `/predict` endpoint, the sensor readings you submit
  are used to compute a prediction in memory and are not saved to a database or file.
- **Server logs.** Like most web servers, the server and the hosting provider may record standard
  technical logs such as IP address, time of request, requested page and browser type. These are
  used only to keep the site running and secure.
- **Session cookie.** Streamlit may use a small functional cookie or local browser storage to keep
  your session working. It is not used to identify or track you.

### Sharing

Information is not sold or shared with anyone, except the hosting provider as needed to serve the
site, or where required by law.

### Your choices

You can use the site without giving any personal information. If you want any log data connected
to you deleted, contact the site owner using the link below.

### Changes

If this policy changes, the updated version will be posted here with a new effective date.

### Contact

Questions about this policy: [{CONTACT_URL}]({CONTACT_URL})
""")


def terms_and_conditions() -> None:
    _header("Terms and conditions")
    st.markdown(f"""
By using {SITE_NAME} ("the site", {_SITE}), run by {OWNER}, you agree to these terms.

### 1. What the site is

The site is a demonstration of predictive maintenance methods. It predicts remaining useful life,
failure risk and inspection timing for **simulated** turbofan engines from the NASA C-MAPSS dataset.
No real aircraft or engines are monitored.

### 2. Not for operational use

Predictions, risk levels and inspection recommendations are produced by a statistical model and can
be wrong. They must not be used to make airworthiness, maintenance, safety or financial decisions
about any real equipment. Always follow the manufacturer's maintenance programme and the rules of
the relevant aviation authority.

### 3. No warranty

The site and its outputs are provided "as is", without any warranty of accuracy, availability or
fitness for a particular purpose.

### 4. Limitation of liability

To the extent permitted by law, {OWNER} is not liable for any loss or damage arising from use of,
or reliance on, the site or its outputs.

### 5. Acceptable use

Do not attempt to disrupt the site or its API, overload it with automated requests, probe it for
vulnerabilities, or use it for anything unlawful.

### 6. Data and credits

Engine data comes from the NASA Prognostics Center of Excellence C-MAPSS dataset (A. Saxena,
K. Goebel, D. Simon and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure
Simulation", PHM08). The site's code, models and design belong to {OWNER} unless stated otherwise.

### 7. Changes

These terms may be updated. The current version will always be on this page with its effective date.

### 8. Governing law

These terms are governed by the laws of {GOVERNING_LAW}.

### 9. Contact

[{CONTACT_URL}]({CONTACT_URL})
""")
