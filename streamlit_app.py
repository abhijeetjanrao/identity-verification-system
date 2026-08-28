import mimetypes
import os

import requests
import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Identity Verification Demo",
    page_icon="🧑‍🤝‍🧑",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(90deg, #4f46e5, #2563eb);
        color: white;
        border: none;
        box-shadow: 0 6px 18px rgba(37,99,235,0.25);
    }

    .stButton > button:hover {
        opacity: 0.95;
        transform: translateY(-1px);
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0f172a, #111827);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 0.9rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }

    .st-emotion-cache-1kyxq5g {
        border-radius: 14px;
    }

    .stAlert {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BACKEND API
# ============================================================

API_URL = os.getenv(
    "FACE_API_URL",
    "https://identity-verification-system-production.up.railway.app/predict",
)

HISTORY_URL = os.getenv(
    "FACE_HISTORY_URL",
    "https://identity-verification-system-production.up.railway.app/history",
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        margin-bottom: 1rem;
    ">
        <h1 style="color: white; margin-bottom: 0.3rem;">
            Identity Verification Demo
        </h1>

        <p style="color: #cbd5e1; margin: 0;">
            AI-powered identity verification for attendance,
            access control, and recognition workflows.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Upload a face image to verify a known identity "
    "through the deployed AI system."
)


# ============================================================
# IMAGE UPLOAD SECTION
# ============================================================

col1, col2 = st.columns([1.2, 0.8])


with col1:

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file)

            st.image(
                image,
                caption="Uploaded image",
                use_container_width=True,
            )

        except Exception as exc:

            st.error(f"Could not open the image: {exc}")
            uploaded_file = None


# ============================================================
# HOW IT WORKS
# ============================================================

with col2:

    st.subheader("How it works")

    st.markdown(
        "- Detects a face in the uploaded image"
    )

    st.markdown(
        "- Extracts feature embeddings"
    )

    st.markdown(
        "- Matches the result against the trained identity database"
    )

    st.markdown(
        "- Returns a confidence-based verification result"
    )

    st.markdown("---")

    st.info(
        "Use case: attendance verification, "
        "access control demo, or identity recognition"
    )


# ============================================================
# VERIFY BUTTON
# ============================================================

if uploaded_file is not None:

    if st.button(
        "Verify Identity",
        use_container_width=True,
    ):

        with st.spinner("Analyzing image..."):

            # ------------------------------------------------
            # Determine content type
            # ------------------------------------------------

            content_type, _ = mimetypes.guess_type(
                uploaded_file.name
            )

            if not content_type:
                content_type = "application/octet-stream"


            # ------------------------------------------------
            # Prepare multipart file
            # ------------------------------------------------

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    content_type,
                )
            }


            try:

                # ====================================================
                # CALL RAILWAY FASTAPI BACKEND
                # ====================================================

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=120,
                )


                # ------------------------------------------------
                # Check HTTP response
                # ------------------------------------------------

                response.raise_for_status()


                # ------------------------------------------------
                # Convert JSON response
                # ------------------------------------------------

                result = response.json()


                # ====================================================
                # SUCCESS
                # ====================================================

                st.success(
                    "Verification completed successfully!"
                )

                st.subheader("Prediction Result")


                # ------------------------------------------------
                # Get result values
                # ------------------------------------------------

                status = result.get(
                    "status",
                    "unknown",
                )

                name = result.get(
                    "name",
                    "Unknown",
                )

                confidence = result.get(
                    "confidence",
                    0,
                )

                profession = result.get(
                    "profession",
                    "Unknown",
                )

                description = result.get(
                    "description",
                    "",
                )


                # ====================================================
                # MATCHED
                # ====================================================

                if status == "matched":

                    st.success(
                        f"Identity Matched: {name}"
                    )

                else:

                    st.warning(
                        "Identity could not be verified."
                    )


                # ====================================================
                # METRICS
                # ====================================================

                metric1, metric2 = st.columns(2)


                with metric1:

                    if status == "matched":

                        st.metric(
                            "Verified Identity",
                            name,
                        )

                    else:

                        st.metric(
                            "Verified Identity",
                            "Unknown",
                        )


                with metric2:

                    st.metric(
                        "Confidence",
                        f"{float(confidence):.3f}",
                    )


                # ====================================================
                # ADDITIONAL INFORMATION
                # ====================================================

                if profession:

                    st.write(
                        f"**Profession:** {profession}"
                    )


                if description:

                    st.write(
                        f"**Description:** {description}"
                    )


                # ====================================================
                # DETAILED JSON RESPONSE
                # ====================================================

                st.markdown("#### Detailed Response")

                st.json(result)


                # ====================================================
                # HISTORY
                # ====================================================

                try:

                    history_response = requests.get(
                        HISTORY_URL,
                        timeout=30,
                    )

                    history_response.raise_for_status()

                    history = history_response.json()


                    if history.get("records"):

                        st.subheader(
                            "Recent Activity"
                        )

                        st.json(
                            history["records"][-3:]
                        )

                except requests.exceptions.RequestException:

                    st.caption(
                        "History endpoint unavailable."
                    )


            # ====================================================
            # REQUEST ERROR
            # ====================================================

            except requests.exceptions.Timeout:

                st.error(
                    "The API request timed out. "
                    "The AI model may still be processing the image. "
                    "Please try again."
                )


            except requests.exceptions.ConnectionError as exc:

                st.error(
                    f"Could not connect to the backend API.\n\n"
                    f"API URL: {API_URL}\n\n"
                    f"Error: {exc}"
                )


            except requests.exceptions.HTTPError as exc:

                st.error(
                    f"The backend returned an HTTP error.\n\n"
                    f"Status code: {response.status_code}\n\n"
                    f"Response: {response.text}"
                )


            except requests.exceptions.RequestException as exc:

                st.error(
                    f"Could not reach the API.\n\n"
                    f"API URL: {API_URL}\n\n"
                    f"Error: {exc}"
                )


            except ValueError as exc:

                st.error(
                    f"The backend returned an invalid JSON response.\n\n"
                    f"Error: {exc}"
                )
