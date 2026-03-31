import os
import streamlit as st

st.set_page_config(
    page_title="Cryogenic Tools Suite",
    page_icon="🧊",
    layout="wide",
)

BASE_DIR = os.path.dirname(__file__)
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .hero {
        padding: 1.4rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #eef4ff 0%, #f9fbff 100%);
        border: 1px solid rgba(49, 91, 161, 0.12);
        margin-bottom: 1.25rem;
    }
    .hero h1 {
        margin: 0 0 0.25rem 0;
        font-size: 2.4rem;
    }
    .hero p {
        margin: 0;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .card {
        background: white;
        border: 1px solid rgba(49, 91, 161, 0.12);
        border-radius: 16px;
        padding: 1.1rem 1.1rem 0.9rem 1.1rem;
        box-shadow: 0 1px 8px rgba(20, 30, 60, 0.04);
        min-height: 200px;
    }
    .card h3 {
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .card p {
        line-height: 1.55;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH)

st.markdown(
    """
    <div class="hero">
        <h1>Cryogenic Tools Suite</h1>
        <p>
            This app bundles two tools for fast, physically transparent estimates in cryogenic experiments.
            The laser-heating module estimates the local temperature rise produced by a fiber-delivered optical
            pulse train. The dilution-refrigerator module estimates effective thermal noise, effective temperature,
            coherent drive power, and photon flux along a staged attenuator chain.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
        <div class="card">
            <h3>Laser Heating Calculator</h3>
            <p>
                Estimate emitted power, delivered power at the fiber end, absorbed power at the target,
                thermal time constant, steady-state temperature rise, and the burst duration required to
                reach a target temperature.
            </p>
            <p>
                This module is designed to connect trigger settings, pulse parameters, optical losses,
                overlap, and thermal anchoring to a predicted temperature trace.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="card">
            <h3>Dilution Refrigerator Noise Explorer</h3>
            <p>
                Explore how staged attenuation redistributes thermal noise along a microwave line,
                and inspect effective photon number, effective temperature, noise reduction,
                coherent drive power, and photon flux.
            </p>
            <p>
                This module is useful for understanding attenuation placement, thermalization,
                and the effective bath seen by a device.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.info("Use the page selector in the sidebar to open either module.")