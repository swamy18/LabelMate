#code for progress tracking and data visualization.

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List

def display_progress_metrics(stats: Dict):
    """
    Display key labeling metrics in a clean dashboard format.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Progress", 
            value=f"{stats['progress']:.1f}%",
            delta=f"{stats['labeled']}/{stats['total']} items"
        )
    
    with col2:
        st.metric(
            label="🤖 AI Generated", 
            value=stats['ai_generated'],
            delta=f"{stats['ai_generated']/max(stats['total'],1)*100:.1f}% of total"
        )
    
    with col3:
        st.metric(
            label="✏️ Human Edits", 
            value=stats['human_changed'],
            delta=f"{stats['human_changed']/max(stats['ai_generated'],1)*100:.1f}% of AI labels"
        )
    
    with col4:
        st.metric(
            label="🎯 AI Accuracy", 
            value=f"{stats['accuracy']:.1f}%",
            delta="Higher is better"
        )

def animated_progress_bar(current: int, total: int, label: str = "Labeling Progress"):
    """
    Enhanced progress bar with animation and status.
    """
    progress = current / max(total, 1)
    
    # Create progress bar
    st.progress(progress, text=f"{label}: {current}/{total} ({progress*100:.1f}%)")
    
    # Add status indicator
    if progress == 1.0:
        st.success("✅ All items labeled!")
    elif progress > 0.8:
        st.info("🏃 Almost done!")
    elif progress > 0.5:
        st.info("⚡ Good progress!")
    elif progress > 0.2:
        st.warning("📝 Keep going!")
    else:
        st.warning("🚀 Just getting started!")

def label_distribution_chart(df: pd.DataFrame, label_column: str, title: str = "Label Distribution"):
    """
    Interactive bar chart showing label distribution with live updates.
    """
    if label_column not in df.columns or df.empty:
        st.info("📈 Chart will appear as you add labels...")
        return
    
    # Filter out empty labels for cleaner visualization
    labeled_df = df[df[label_column] != ""]
    
    if labeled_df.empty:
        st.info("📊 Start labeling to see distribution...")
        return
    
    # Calculate distribution
    counts = labeled_df[label_column].value_counts().reset_index()
    counts.columns = ["Label", "Count"]
    counts["Percentage"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
    
    # Create interactive bar chart
    fig = px.bar(
        counts, 
        x="Label", 
        y="Count",
        title=title,
        color="Count",
        color_continuous_scale="viridis",
        hover_data=["Percentage"],
        text="Count"
    )
    
    # Customize layout
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Labels",
        yaxis_title="Count",
        showlegend=False,
        height=400,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed statistics
    with st.expander("📈 Detailed Statistics"):
        st.dataframe(counts, use_container_width=True)

def ai_vs_human_comparison(df: pd.DataFrame):
    """
    Chart comparing AI suggestions vs human corrections.
    """
    if "human_changed" not in df.columns:
        return
    
    ai_accepted = (~df["human_changed"]).sum()
    human_changed = df["human_changed"].sum()
    
    # Create pie chart
    labels = ["AI Accepted", "Human Modified"]
    values = [ai_accepted, human_changed]
    colors = ["#2E8B57", "#FF6347"]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=0.4,
        marker_colors=colors,
        textinfo="label+percent+value",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        title="AI vs Human Labeling",
        height=300,
        showlegend=True,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def export_analytics_summary(df: pd.DataFrame, stats: Dict) -> str:
    """
    Generate and export analytics summary.
    """
    summary = {
        "Dataset Statistics": stats,
        "Label Distribution": df["label"].value_counts().to_dict() if "label" in df.columns else {},
        "AI Performance": {
            "Total AI Suggestions": stats.get("ai_generated", 0),
            "Human Corrections": stats.get("human_changed", 0),
            "AI Accuracy Rate": f"{stats.get('accuracy', 0):.1f}%"
        }
    }
    
    import json
    from pathlib import Path
    
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    
    summary_path = exports_dir / "labeling_analytics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    return str(summary_path)
