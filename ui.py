import streamlit as st

def use_global_style():
    st.markdown(
        """
        <style>
        body {
            font-family: 'Comfortaa', sans-serif;
            background-color: white;
            color: black;
        }

        /* Navbar */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            border-bottom: 1px solid #ddd;
        }

        .navbar-left img {
            width: 50px;
            height: 50px;
        }

        .navbar-center a {
            margin: 0 1rem;
            text-decoration: none;
            font-weight: 600;
            color: black;
        }

        .navbar-center a.active {
            color: #e63946;
        }

        .navbar-right {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .btn-login {
            font-weight: 600;
            color: black;
            text-decoration: none;
        }

        .btn-primary {
            background-color: #e63946;
            color: white !important;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-weight: 600;
            text-decoration: none;
        }

        .btn-primary:hover {
            background-color: #c92c3c;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def header(active="Home"):
    st.markdown(
        f"""
        <div class="navbar">
            <!-- Left -->
            <div class="navbar-left">
                <img src="assets/logo.png2"/>
            </div>

            <!-- Center -->
            <div class="navbar-center">
                <a href="/" class="{'active' if active=='Home' else ''}">Home</a>
                <a href="/EV_Finder" class="{'active' if active=='EV Finder' else ''}">EV Finder</a>
                <a href="/Tools" class="{'active' if active=='Tools' else ''}">Tools</a>
                <a href="/Resources" class="{'active' if active=='Resources' else ''}">Resources</a>
                <a href="/Subscription" class="{'active' if active=='Subscription' else ''}">Subscription</a>
            </div>

            <!-- Right -->
            <div class="navbar-right">
                <a href="/Subscription" class="btn-login">Login</a>
                <a href="/Subscription" class="btn-primary">Try for Free</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def footer():
    st.markdown(
        """
        <div style="margin-top:3rem; padding:2rem; background:#111; color:#fff; text-align:center;">
            <p>© 2025 TruLine Betting</p>
            <p>
                <a href="mailto:contact@trulinebetting.com">Contact</a> |
                <a href="https://discord.com" target="_self">Discord</a> |
                <a href="https://youtube.com" target="_self">YouTube</a> |
                <a href="https://tiktok.com" target="_self">TikTok</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
