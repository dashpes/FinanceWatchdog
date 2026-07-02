"""Archie's dashboard — a LAN web GUI over the robo advisor's data.

FastAPI JSON API + server-rendered shells + vanilla-JS hydration. Read-only
against the SQLite store (PRAGMA query_only), with a small token-gated control
surface (pause/kill/blocklist/settings) that writes through the same modules the
CLI uses (robo.control, robo.blocklist, robo.tunables) — never the DB.
"""
