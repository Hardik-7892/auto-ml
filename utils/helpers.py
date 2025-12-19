def clear_session_state():
    """Reset all session state variables"""
    keys = list(st.session_state.keys())
    for key in keys:
        del st.session_state[key]
