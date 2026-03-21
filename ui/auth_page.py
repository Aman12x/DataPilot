"""
ui/auth_page.py — Sign-in and sign-up UI for DataPilot.

Renders only. All auth logic lives in auth/store.py.

On successful auth, writes to st.session_state:
    user: dict  — {user_id, username, email, created_at}
"""

from __future__ import annotations

import streamlit as st

from auth.store import create_user, verify_user


def render_auth_page() -> None:
    """
    Render the login / sign-up page. Sets st.session_state.user on success
    and calls st.rerun() to transition into the main app.
    """
    st.title("🧭 DataPilot")
    st.caption("AI Product Data Scientist")
    st.markdown("---")

    tab_signin, tab_signup = st.tabs(["Sign in", "Create account"])

    # ── Sign in ───────────────────────────────────────────────────────────────
    with tab_signin:
        st.markdown("#### Welcome back")
        with st.form("signin_form"):
            login    = st.text_input("Username or email")
            password = st.text_input("Password", type="password")
            submit   = st.form_submit_button("Sign in", type="primary", use_container_width=True)

        if submit:
            if not login or not password:
                st.error("Please enter your username/email and password.")
            else:
                user = verify_user(login, password)
                if user is None:
                    st.error("Invalid credentials. Please try again.")
                else:
                    st.session_state.user = user.to_dict()
                    st.success(f"Welcome back, {user.username}!")
                    st.rerun()

    # ── Sign up ───────────────────────────────────────────────────────────────
    with tab_signup:
        st.markdown("#### Create your account")
        with st.form("signup_form"):
            new_username = st.text_input("Username", help="Letters, numbers, underscores. Min 3 chars.")
            new_email    = st.text_input("Email")
            new_password = st.text_input("Password", type="password", help="Minimum 8 characters.")
            confirm_pwd  = st.text_input("Confirm password", type="password")
            submit_up    = st.form_submit_button("Create account", type="primary", use_container_width=True)

        if submit_up:
            errors = _validate_signup(new_username, new_email, new_password, confirm_pwd)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                result = create_user(new_username, new_email, new_password)
                if isinstance(result, str):
                    st.error(result)
                else:
                    st.session_state.user = result.to_dict()
                    st.success(f"Account created! Welcome, {result.username}.")
                    st.rerun()


def _validate_signup(
    username: str,
    email: str,
    password: str,
    confirm: str,
) -> list[str]:
    errors: list[str] = []
    if len(username.strip()) < 3:
        errors.append("Username must be at least 3 characters.")
    if not username.replace("_", "").replace("-", "").isalnum():
        errors.append("Username can only contain letters, numbers, hyphens, and underscores.")
    if "@" not in email or "." not in email.split("@")[-1]:
        errors.append("Please enter a valid email address.")
    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    if password != confirm:
        errors.append("Passwords do not match.")
    return errors
