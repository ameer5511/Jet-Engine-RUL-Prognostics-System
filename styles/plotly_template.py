"""Plotly 'jerps' template for Jet Engine RUL Prognostics System.

Applied to every figure: transparent background, monospace text, hairline gridlines,
horizontal legend, tight margins, hidden mode bar.
"""

from __future__ import annotations

import plotly.graph_objects as go

from tokens import (
    ACCENT,
    BORDER_HAIRLINE,
    BG_PAGE,
    BG_PANEL,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    FONT_MONO,
)


def jerps_template() -> go.layout.Template:
    """Return the custom 'jerps' Plotly template."""
    return go.layout.Template(
        layout=go.Layout(
            # Backgrounds
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # Fonts
            font=dict(family=FONT_MONO, size=11, color=TEXT_SECONDARY),
            title=dict(font=dict(family=FONT_MONO, size=12, color=TEXT_PRIMARY)),
            # Axes
            xaxis=dict(
                showgrid=True,
                gridcolor=BORDER_HAIRLINE,
                zeroline=False,
                showline=False,
                tickfont=dict(family=FONT_MONO, size=10, color=TEXT_MUTED),
                ticklabelmode="period",
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=BORDER_HAIRLINE,
                zeroline=False,
                showline=False,
                tickfont=dict(family=FONT_MONO, size=10, color=TEXT_MUTED),
                ticklabelmode="period",
            ),
            # Legend: horizontal, top-left, small
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(family=FONT_MONO, size=10, color=TEXT_MUTED),
                bgcolor="rgba(0,0,0,0)",
                bordercolor=BORDER_HAIRLINE,
                borderwidth=1,
                itemsizing="constant",
                itemwidth=30,
                traceorder="normal",
            ),
            # Hover
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor=BG_PANEL,
                font=dict(family=FONT_MONO, size=11, color=TEXT_PRIMARY),
                bordercolor=BORDER_HAIRLINE,
            ),
            # Mode bar hidden
            modebar=dict(orientation="v", bgcolor="rgba(0,0,0,0)"),
            # Margins tight
            margin=dict(l=8, r=8, t=24, b=8),
        )
    )


def apply_jerps(fig: go.Figure) -> go.Figure:
    """Apply the jerps template to an existing figure in-place."""
    fig.update_layout(template=jerps_template())
    fig.update_layout(margin=dict(l=8, r=8, t=24, b=8))
    fig.update_layout(showlegend=True)
    # Hide mode bar via config instead of layout
    return fig
