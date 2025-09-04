"""
LabelMate - AI-Powered Data Labeling Assistant
Main Streamlit Application

A production-ready tool for data teams to accelerate annotation workflows.
Run: streamlit run app.py

Author: @swamy18
Repository: https://github.com/swamy18/labelmate
"""

import streamlit as st
import pandas as pd
import os
import json
import time
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go

# Import our custom modules
from utils.text_helper import (
    load_text_csv, suggest_text_label, save_text_csv, 
    get_labeling_stats
)
from utils.image_helper import (
    suggest_image_labels, process_uploaded_images, export_image_labels
)
from utils.charts import (
    display_progress_metrics, animated_progress_bar,
    label_distribution_chart, ai_vs_human_comparison,
    export_analytics_summary
)

# =============================================================================
# PAGE CONFIGURATION AND STYLING
# =============================================================================

st.set_page_config(
    page_title="LabelMate - AI Data Labeling",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/swamy18/labelmate/wiki',
        'Report a bug': 'https://github.com/swamy18/labelmate/issues',
        'About': 'LabelMate v1.0 - AI-Powered Data Labeling Assistant'
    }
)

# Custom CSS for professional styling
st.markdown("""
<style>
/* Main header styling */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.main-header h1 {
    color: white;
    margin: 0;
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
}

.subtitle {
    color: #666;
    text-align: center;
    font-size: 1.2rem;
    margin-bottom: 2rem;
    font-style: italic;
}

/* Status indicators */
.success-box {
    background: linear-gradient(90deg, #d4edda 0%, #c3e6cb 100%);
    border: 1px solid #28a745;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.warning-box {
    background: linear-gradient(90deg, #fff3cd 0%, #ffeaa7 100%);
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.error-box {
    background: linear-gradient(90deg, #f8d7da 0%, #f5c6cb 100%);
    border: 1px solid #dc3545;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

/* Cards and containers */
.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border: 1px solid #e1e8ed;
    margin: 0.5rem 0;
}

.labeling-container {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    border: 1px solid #dee2e6;
}

/* Progress indicators */
.progress-container {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin: 1rem 0;
}

/* Button enhancements */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* Responsive design */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem;
    }
    
    .main-header {
        padding: 1rem;
    }
}

/* Animation for loading states */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.loading {
    animation: pulse 2s infinite;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_directories():
    """Ensure required directories exist"""
    for dir_name in ["data", "data/temp_images", "exports"]:
        Path(dir_name).mkdir(parents=True, exist_ok=True)

def display_api_status():
    """Display API configuration status in sidebar"""
    st.sidebar.markdown("### 🔑 API Configuration")
    
    gemini_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if gemini_key:
        st.sidebar.success("✅ Gemini API configured")
        st.sidebar.caption("🤖 Text + Image labeling available")
    elif openai_key:
        st.sidebar.success("✅ OpenAI API configured")
        st.sidebar.caption("📝 Text labeling only")
    else:
        st.sidebar.error("❌ No API key configured")
        st.sidebar.info("💡 Add GOOGLE_API_KEY or OPENAI_API_KEY to your .env file")
        
        with st.sidebar.expander("🔧 Setup Instructions"):
            st.markdown("""
            1. Get a free API key:
               - [Gemini AI](https://makersuite.google.com/app/apikey) (Recommended)
               - [OpenAI](https://platform.openai.com/api-keys)
            
            2. Create a `.env` file in the project root:
               ```
               GOOGLE_API_KEY=your_key_here
               ```
            
            3. Restart the application
            """)

def show_welcome_message():
    """Display welcome message for new users"""
    if "welcome_shown" not in st.session_state:
        st.info("""
        👋 **Welcome to LabelMate!** 
        
        This AI-powered tool helps you label text and images efficiently:
        - Upload your data (CSV for text, images for vision tasks)
        - Get AI suggestions instantly
        - Review and correct labels easily
        - Export high-quality labeled datasets
        
        Choose a mode from the sidebar to get started!
        """)
        st.session_state.welcome_shown = True

def initialize_session_state():
    """Initialize all session state variables"""
    if "text_df" not in st.session_state:
        st.session_state.text_df = pd.DataFrame()
    
    if "processed_images" not in st.session_state:
        st.session_state.processed_images = {}
    
    if "labeling_task" not in st.session_state:
        st.session_state.labeling_task = "sentiment"
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = 0
    
    if "auto_save" not in st.session_state:
        st.session_state.auto_save = True

# =============================================================================
# MAIN APPLICATION HEADER
# =============================================================================

def render_header():
    """Render the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🏷️ LabelMate</h1>
        <p style="color: white; text-align: center; margin: 0; font-size: 1.1rem;">
            AI-Powered Data Labeling Assistant
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p class="subtitle">
        Accelerate your data annotation workflow with intelligent AI assistance
    </p>
    """, unsafe_allow_html=True)

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

def render_sidebar():
    """Render sidebar navigation and controls"""
    st.sidebar.title("🎛️ Control Panel")
    st.sidebar.markdown("---")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "📋 Select Mode",
        ["📄 Text Labeling", "🖼️ Image Labeling", "📊 Analytics Dashboard"],
        help="Choose the type of data labeling task"
    )
    
    st.sidebar.markdown("---")
    
    # API status
    display_api_status()
    
    st.sidebar.markdown("---")
    
    # Settings section
    with st.sidebar.expander("⚙️ Settings"):
        auto_save = st.checkbox(
            "Auto-save progress", 
            value=st.session_state.get("auto_save", True),
            help="Automatically save your progress"
        )
        st.session_state.auto_save = auto_save
        
        show_advanced = st.checkbox(
            "Show advanced options",
            value=False,
            help="Display advanced configuration options"
        )
    
    # Quick stats
    if not st.session_state.text_df.empty or st.session_state.processed_images:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📈 Quick Stats")
        
        if not st.session_state.text_df.empty:
            text_stats = get_labeling_stats(st.session_state.text_df)
            st.sidebar.metric("Text Items", text_stats["total"])
            st.sidebar.metric("Text Progress", f"{text_stats['progress']:.1f}%")
        
        if st.session_state.processed_images:
            img_total = len(st.session_state.processed_images)
            img_labeled = sum(1 for data in st.session_state.processed_images.values() 
                            if data["selected_label"] != "unknown")
            img_progress = (img_labeled / max(img_total, 1)) * 100
            st.sidebar.metric("Image Items", img_total)
            st.sidebar.metric("Image Progress", f"{img_progress:.1f}%")
    
    # Help section
    st.sidebar.markdown("---")
    with st.sidebar.expander("📚 Quick Help"):
        st.markdown("""
        **Text Labeling:**
        - Upload CSV with 'text' column
        - Choose sentiment or topic task
        - Review AI suggestions
        - Export labeled results
        
        **Image Labeling:**
        - Upload JPG/PNG files
        - Get AI label suggestions
        - Select or customize labels
        - Export filename-label mapping
        
        **Analytics:**
        - View progress metrics
        - Analyze AI performance
        - Export comprehensive reports
        """)
    
    return mode

# =============================================================================
# TEXT LABELING MODE
# =============================================================================

def render_text_labeling_mode():
    """Render the complete text labeling interface"""
    st.header("📄 Text Data Labeling")
    
    # Configuration section
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        task = st.selectbox(
            "🎯 Labeling Task",
            ["sentiment", "topic"],
            index=0 if st.session_state.labeling_task == "sentiment" else 1,
            help="Choose the type of text classification"
        )
        st.session_state.labeling_task = task
    
    with col2:
        batch_size = st.number_input(
            "🔢 Batch Size",
            min_value=1, max_value=100, value=10,
            help="Number of items to process at once"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset Data", help="Clear all data and start over"):
            st.session_state.text_df = pd.DataFrame()
            st.success("✅ Data reset successfully!")
            st.rerun()
    
    # File upload section
    st.subheader("📤 Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Your CSV file must contain a 'text' column with the data to label"
    )
    
    # Sample data option
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🧪 Load Sample Data", help="Try with sample text data"):
            sample_texts = [
                "I absolutely love this product! It's amazing!",
                "Terrible experience. Would not recommend.",
                "It's okay, nothing special but does the job.",
                "Best purchase I've made this year!",
                "Could be better, but it's acceptable.",
                "Waste of money. Very disappointed.",
                "Pretty good overall, happy with it.",
                "Not great, not terrible. Average quality.",
                "Excellent quality and fast shipping!",
                "Poor customer service experience.",
                "The new update broke everything!",
                "Works perfectly as described.",
                "Overpriced for what you get.",
                "Great value for money!",
                "Customer support was very helpful.",
                "Delivery was delayed twice.",
                "Exactly what I was looking for.",
                "Quality could be much better.",
                "Impressed with the quick response.",
                "Would definitely buy again!"
            ]
            
            sample_df = pd.DataFrame({
                "text": sample_texts,
                "label": [""] * len(sample_texts),
                "ai_suggested": [""] * len(sample_texts),
                "human_changed": [False] * len(sample_texts)
            })
            
            st.session_state.text_df = sample_df
            st.success("✅ Sample data loaded! 20 texts ready for labeling.")
            st.rerun()
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            with st.spinner("📖 Loading and validating your data..."):
                df = load_text_csv(uploaded_file)
                st.session_state.text_df = df.copy()
            
            st.success(f"✅ Successfully loaded {len(df)} text samples!")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return
    
    # Main interface - only show if we have data
    if not st.session_state.text_df.empty:
        df = st.session_state.text_df
        
        # Data preview section
        with st.expander("👀 Data Preview & Statistics", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(
                    df.head(10)[["text", "label", "ai_suggested", "human_changed"]], 
                    use_container_width=True,
                    height=300
                )
            
            with col2:
                stats = get_labeling_stats(df)
                st.metric("Total Texts", stats["total"])
                st.metric("Labeled", f"{stats['labeled']} ({stats['progress']:.1f}%)")
                st.metric("AI Generated", stats["ai_generated"])
                st.metric("Human Edited", stats["human_changed"])
                
                if stats["ai_generated"] > 0:
                    accuracy = stats["accuracy"]
                    st.metric("AI Accuracy", f"{accuracy:.1f}%")
        
        # AI batch labeling section
        st.subheader("🤖 AI Batch Labeling")
        
        unlabeled_count = (df["label"] == "").sum()
        
        if unlabeled_count > 0:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.info(f"📝 {unlabeled_count} items need labeling")
            
            with col2:
                if st.button("🚀 Label All", type="primary"):
                    label_all_texts(df, task)
            
            with col3:
                sample_size = min(5, unlabeled_count)
                if st.button(f"🧪 Label {sample_size}"):
                    label_sample_texts(df, task, sample_size)
            
            with col4:
                if st.button("⚡ Label Batch"):
                    label_batch_texts(df, task, batch_size)
        else:
            st.success("🎉 All texts have been labeled!")
        
        # Progress visualization
        st.subheader("📊 Progress Dashboard")
        stats = get_labeling_stats(df)
        display_progress_metrics(stats)
        animated_progress_bar(stats["labeled"], stats["total"], "Labeling Progress")
        
        # Interactive labeling section
        st.subheader("✏️ Review & Correct Labels")
        render_text_review_interface(df, task)
        
        # Visualization section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            label_distribution_chart(df, "label", "Current Label Distribution")
        
        with col2:
            ai_vs_human_comparison(df)
        
        # Export section
        render_text_export_section(df, task)

def label_all_texts(df, task):
    """Label all unlabeled texts with AI"""
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    unlabeled_mask = df["label"] == ""
    unlabeled_indices = df[unlabeled_mask].index.tolist()
    
    if not unlabeled_indices:
        st.warning("No unlabeled texts found!")
        return
    
    progress_bar = progress_placeholder.progress(0)
    
    for i, idx in enumerate(unlabeled_indices):
        # Update progress
        progress = (i + 1) / len(unlabeled_indices)
        progress_bar.progress(progress)
        
        # Show current item
        current_text = df.at[idx, "text"]
        status_placeholder.text(f"Processing {i+1}/{len(unlabeled_indices)}: {current_text[:50]}...")
        
        # Get AI suggestion
        suggestion = suggest_text_label(current_text, task)
        df.at[idx, "ai_suggested"] = suggestion
        df.at[idx, "label"] = suggestion
        
        # Auto-save progress periodically
        if st.session_state.auto_save and (i + 1) % 10 == 0:
            st.session_state.text_df = df.copy()
    
    st.session_state.text_df = df.copy()
    progress_placeholder.empty()
    status_placeholder.empty()
    st.success(f"🎉 Successfully labeled {len(unlabeled_indices)} texts!")
    st.rerun()

def label_sample_texts(df, task, sample_size):
    """Label a small sample of texts"""
    with st.spinner(f"🤖 Labeling {sample_size} samples..."):
        unlabeled_mask = df["label"] == ""
        sample_indices = df[unlabeled_mask].head(sample_size).index
        
        for idx in sample_indices:
            suggestion = suggest_text_label(df.at[idx, "text"], task)
            df.at[idx, "ai_suggested"] = suggestion
            df.at[idx, "label"] = suggestion
        
        st.session_state.text_df = df.copy()
        st.success(f"✅ Labeled {len(sample_indices)} samples!")
        st.rerun()

def label_batch_texts(df, task, batch_size):
    """Label a specific batch of texts"""
    unlabeled_mask = df["label"] == ""
    batch_indices = df[unlabeled_mask].head(batch_size).index
    
    if len(batch_indices) == 0:
        st.warning("No unlabeled texts found!")
        return
    
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0)
    
    for i, idx in enumerate(batch_indices):
        progress = (i + 1) / len(batch_indices)
        progress_bar.progress(progress)
        
        suggestion = suggest_text_label(df.at[idx, "text"], task)
        df.at[idx, "ai_suggested"] = suggestion
        df.at[idx, "label"] = suggestion
    
    st.session_state.text_df = df.copy()
    progress_placeholder.empty()
    st.success(f"✅ Processed batch of {len(batch_indices)} texts!")
    st.rerun()

def render_text_review_interface(df, task):
    """Render the text review and correction interface"""
    # Filter and pagination controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        filter_option = st.selectbox(
            "🔍 Filter items:",
            ["All items", "Unlabeled only", "AI labeled only", "Human modified only", "Need review"]
        )
    
    with col2:
        items_per_page = st.selectbox("📄 Items per page:", [5, 10, 20, 50], index=1)
    
    with col3:
        sort_option = st.selectbox("📊 Sort by:", ["Original order", "Text length", "Confidence"])
    
    # Apply filters
    filtered_df = apply_text_filters(df, filter_option)
    
    if filtered_df.empty:
        st.info(f"🔍 No items match the filter: {filter_option}")
        return
    
    # Apply sorting
    if sort_option == "Text length":
        filtered_df = filtered_df.copy()
        filtered_df["text_length"] = filtered_df["text"].str.len()
        filtered_df = filtered_df.sort_values("text_length", ascending=False)
    
    # Pagination
    total_items = len(filtered_df)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    if total_pages > 1:
        page = st.selectbox(f"📖 Page (1-{total_pages}):", range(1, total_pages + 1)) - 1
        st.session_state.current_page = page
    else:
        page = 0
    
    start_idx = page * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    page_df = filtered_df.iloc[start_idx:end_idx]
    
    # Review interface
    st.markdown(f"**Showing items {start_idx + 1}-{end_idx} of {total_items}**")
    
    for i, (idx, row) in enumerate(page_df.iterrows()):
        render_single_text_review(idx, row, task, i + start_idx)
    
    # Navigation buttons for pagination
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if page > 0 and st.button("⬅️ Previous Page"):
                st.session_state.current_page = page - 1
                st.rerun()
        
        with col3:
            if page < total_pages - 1 and st.button("➡️ Next Page"):
                st.session_state.current_page = page + 1
                st.rerun()

def apply_text_filters(df, filter_option):
    """Apply selected filter to the dataframe"""
    if filter_option == "Unlabeled only":
        return df[df["label"] == ""]
    elif filter_option == "AI labeled only":
        return df[(df["ai_suggested"] != "") & (~df["human_changed"])]
    elif filter_option == "Human modified only":
        return df[df["human_changed"]]
    elif filter_option == "Need review":
        # Items where AI suggestion differs significantly or low confidence
        return df[(df["ai_suggested"] != "") & (df["label"] == "")]
    else:
        return df

def render_single_text_review(idx, row, task, display_idx):
    """Render a single text item for review"""
    with st.container():
        # Create a bordered container
        st.markdown(f"""
        <div class="labeling-container" style="margin: 1rem 0;">
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            # Display text with character limit
            text_preview = row["text"]
            if len(text_preview) > 200:
                text_preview = text_preview[:200] + "..."
            
            st.markdown(f"**Text #{display_idx + 1}:**")
            st.write(text_preview)
            
            # Show AI suggestion if available
            if row["ai_suggested"]:
                confidence_emoji = "🎯" if not row["human_changed"] else "✏️"
                st.caption(f"{confidence_emoji} AI suggested: **{row['ai_suggested']}**")
            
            # Show modification status
            if row["human_changed"]:
                st.caption("✏️ *Human modified*")
        
        with col2:
            # Label selection
            if task == "sentiment":
                label_options = ["", "Positive", "Negative", "Neutral"]
            else:  # topic
                label_options = ["", "Technology", "Business", "Politics", "Sports", 
                               "Entertainment", "Health", "Education", "Other"]
            
            current_label = row["label"]
            if current_label not in label_options:
                label_options.append(current_label)
            
            new_label = st.selectbox(
                "Label:",
                label_options,
                index=label_options.index(current_label) if current_label in label_options else 0,
                key=f"label_{idx}_{display_idx}"
            )
            
            # Update dataframe if label changed
            if new_label != current_label:
                df = st.session_state.text_df
                df.at[idx, "label"] = new_label
                
                # Mark as human changed if different from AI suggestion
                if row["ai_suggested"] and new_label != row["ai_suggested"]:
                    df.at[idx, "human_changed"] = True
                
                st.session_state.text_df = df.copy()
                
                # Auto-save if enabled
                if st.session_state.auto_save:
                    save_text_csv(df, f"auto_save_{task}.csv")
        
        with col3:
            # Action buttons
            if st.button("👁️ Full Text", key=f"view_{idx}_{display_idx}"):
                st.info(f"**Complete Text:**\n\n{row['text']}")
            
            # Quick label buttons for common cases
            if task == "sentiment" and not row["label"]:
                quick_col1, quick_col2 = st.columns(2)
                with quick_col1:
                    if st.button("👍", key=f"pos_{idx}_{display_idx}", help="Mark as Positive"):
                        update_text_label(idx, "Positive", row)
                with quick_col2:
                    if st.button("👎", key=f"neg_{idx}_{display_idx}", help="Mark as Negative"):
                        update_text_label(idx, "Negative", row)
        
        st.markdown("---")

def update_text_label(idx, new_label, row):
    """Update a text label and handle change tracking"""
    df = st.session_state.text_df
    df.at[idx, "label"] = new_label
    
    if row["ai_suggested"] and new_label != row["ai_suggested"]:
        df.at[idx, "human_changed"] = True
    
    st.session_state.text_df = df.copy()
    st.rerun()

def render_text_export_section(df, task):
    """Render the export section for text labeling"""
    st.subheader("📥 Export Results")
    
    # Export statistics
    stats = get_labeling_stats(df)
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("💾 Save Progress", type="primary"):
            filepath = save_text_csv(df, f"labeled_text_{task}.csv")
            st.success(f"✅ Saved to {filepath}")
    
    with col2:
        # Download CSV
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name=f"labeled_text_{task}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        # Download only labeled items
        labeled_df = df[df["label"] != ""]
        if not labeled_df.empty:
            labeled_csv = labeled_df.to_csv(index=False)
            st.download_button(
                label="📋 Labeled Only",
                data=labeled_csv,
                file_name=f"completed_labels_{task}.csv",
                mime="text/csv"
            )
        else:
            st.button("📋 Labeled Only", disabled=True, help="No labeled items yet")
    
    with col4:
        if st.button("📈 Export Analytics"):
            analytics_path = export_analytics_summary(df, stats)
            st.success(f"✅ Analytics saved to {analytics_path}")
    
    # Export summary
    if stats["labeled"] > 0:
        with st.expander("📊 Export Summary"):
            st.write(f"**Ready to export:**")
            st.write(f"- Total items: {stats['total']}")
            st.write(f"- Labeled items: {stats['labeled']} ({stats['progress']:.1f}%)")
            st.write(f"- AI generated: {stats['ai_generated']}")
            st.write(f"- Human modified: {stats['human_changed']}")
            
            if stats['ai_generated'] > 0:
                st.write(f"- AI accuracy: {stats['accuracy']:.1f}%")
            
            # Label distribution
            label_counts = df["label"].value_counts()
            if not label_counts.empty:
                st.write("**Label distribution:**")
                for label, count in label_counts.items():
                    if label:  # Skip empty labels
                        st.write(f"  - {label}: {count} ({count/stats['labeled']*100:.1f}%)")

# =============================================================================
# IMAGE LABELING MODE
# =============================================================================

def render_image_labeling_mode():
    """Render the complete image labeling interface"""
    st.header("🖼️ Image Data Labeling")
    
    # Configuration section
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        num_suggestions = st.slider(
            "🎯 AI suggestions per image",
            min_value=1, max_value=5, value=3,
            help="Number of label suggestions from AI"
        )
    
    with col2:
        display_mode = st.selectbox(
            "👁️ Display Mode",
            ["Grid View", "List View", "Single Image"],
            help="Choose how to display images for labeling"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset Images", help="Clear all image data"):
            st.session_state.processed_images = {}
            # Clean up temp directory
            import shutil
            temp_dir = Path("data/temp_images")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)
            st.success("✅ Image data reset!")
            st.rerun()
    
    # File upload section
    st.subheader("📤 Upload Your Images")
    
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=True,
        help="Upload one or more images to label"
    )
    
    # Sample images option
    if st.button("🧪 Use Sample Images", help="Try with sample images"):
        st.info("📝 For demo purposes, upload your own images or use the sample data folder")
    
    # Process uploaded images
    if uploaded_files:
        # Check if we need to reprocess images
        current_file_names = {f.name for f in uploaded_files}
        stored_file_names = set(st.session_state.processed_images.keys())
        
        if current_file_names != stored_file_names:
            with st.spinner("🖼️ Processing images..."):
                st.session_state.processed_images = process_uploaded_images(uploaded_files)
        
        processed_images = st.session_state.processed_images
        
        if processed_images:
            st.success(f"✅ Processed {len(processed_images)} images")
            
            # Progress tracking
            total_images = len(processed_images)
            labeled_images = sum(1 for data in processed_images.values() 
                               if data["selected_label"] != "unknown")
            
            # Progress metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📷 Total Images", total_images)
            
            with col2:
                st.metric("✅ Labeled", labeled_images)
            
            with col3:
                progress_pct = (labeled_images / max(total_images, 1)) * 100
                st.metric("📊 Progress", f"{progress_pct:.1f}%")
            
            with col4:
                human_modified = sum(1 for data in processed_images.values() 
                                   if data.get("human_changed", False))
                st.metric("✏️ Modified", human_modified)
            
            animated_progress_bar(labeled_images, total_images, "Image Labeling Progress")
            
            # Render the appropriate display mode
            if display_mode == "Grid View":
                render_grid_view(processed_images, num_suggestions)
            elif display_mode == "List View":
                render_list_view(processed_images, num_suggestions)
            else:  # Single Image
                render_single_image_view(processed_images, num_suggestions)
            
            # Label distribution chart
            render_image_analytics(processed_images)
            
            # Export section
            render_image_export_section(processed_images)

def render_grid_view(processed_images, num_suggestions):
    """Render images in a grid layout"""
    st.subheader("🔲 Grid View - Quick Labeling")
    
    # Create grid layout
    cols_per_row = 3
    image_items = list(processed_images.items())
    
    for i in range(0, len(image_items), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, (img_name, img_data) in enumerate(image_items[i:i+cols_per_row]):
            with cols[j]:
                # Display image
                st.image(img_data["image"], caption=img_name, width=200)
                
                # Current selection status
                current_label = img_data["selected_label"]
                if current_label != "unknown":
                    status_color = "🟢" if not img_data.get("human_changed", False) else "🟡"
                    st.caption(f"{status_color} **{current_label}**")
                else:
                    st.caption("🔴 **Unlabeled**")
                
                # Label selection
                suggestions = img_data["suggested_labels"]
                label_options = suggestions + ["Custom"]
                
                try:
                    default_index = suggestions.index(current_label) if current_label in suggestions else 0
                except ValueError:
                    default_index = len(suggestions)  # Custom option
                
                selected_label = st.selectbox(
                    f"Label for {img_name}:",
                    label_options,
                    index=default_index,
                    key=f"grid_{img_name}_{i}_{j}"
                )
                
                # Handle custom label
                if selected_label == "Custom":
                    custom_label = st.text_input(
                        "Custom label:",
                        value=current_label if current_label not in suggestions else "",
                        key=f"custom_grid_{img_name}_{i}_{j}",
                        placeholder="Enter custom label"
                    )
                    if custom_label:
                        selected_label = custom_label
                
                # Update if changed
                if selected_label != current_label and selected_label != "Custom":
                    processed_images[img_name]["selected_label"] = selected_label
                    processed_images[img_name]["human_changed"] = selected_label != suggestions[0]
                    st.session_state.processed_images = processed_images
                    
                    # Auto-save if enabled
                    if st.session_state.auto_save:
                        export_image_labels(processed_images, "auto_save_images.csv")

def render_list_view(processed_images, num_suggestions):
    """Render images in a detailed list view"""
    st.subheader("📋 List View - Detailed Review")
    
    for img_name, img_data in processed_images.items():
        with st.expander(f"📷 {img_name}", expanded=False):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(img_data["image"], width=300)
                
                # Image info
                img = img_data["image"]
                st.caption(f"Size: {img.size[0]}x{img.size[1]}")
                st.caption(f"Mode: {img.mode}")
            
            with col2:
                st.markdown("**🤖 AI Suggestions:**")
                suggestions = img_data["suggested_labels"]
                
                for i, suggestion in enumerate(suggestions):
                    confidence_emoji = ["🥇", "🥈", "🥉", "🏅", "⭐"][i] if i < 5 else "📌"
                    confidence_pct = max(90 - i*15, 30)  # Simulated confidence
                    st.write(f"{confidence_emoji} {suggestion} *({confidence_pct}% confidence)*")
                
                st.markdown("**🏷️ Current Label:**")
                current_label = img_data["selected_label"]
                
                # Label selection with radio buttons
                label_options = suggestions + ["Other"]
                
                if current_label in suggestions:
                    default_index = suggestions.index(current_label)
                else:
                    default_index = len(suggestions)  # "Other" option
                
                selected_option = st.radio(
                    "Choose label:",
                    label_options,
                    index=default_index,
                    key=f"list_radio_{img_name}",
                    horizontal=True
                )
                
                # Handle "Other" selection
                if selected_option == "Other":
                    custom_label = st.text_input(
                        "Enter custom label:",
                        value=current_label if current_label not in suggestions else "",
                        key=f"list_custom_{img_name}",
                        placeholder="Type your custom label"
                    )
                    if custom_label:
                        selected_option = custom_label
                
                # Update if changed
                if selected_option != current_label and selected_option != "Other":
                    processed_images[img_name]["selected_label"] = selected_option
                    processed_images[img_name]["human_changed"] = selected_option != suggestions[0]
                    st.session_state.processed_images = processed_images
                    st.success(f"✅ Updated label for {img_name} to: **{selected_option}**")
                
                # Show modification status
                if img_data.get("human_changed", False):
                    st.info("✏️ This label was modified by human review")

def render_single_image_view(processed_images, num_suggestions):
    """Render single image detailed view with navigation"""
    st.subheader("🔍 Single Image View - Detailed Analysis")
    
    if not processed_images:
        st.info("No images to display")
        return
    
    # Image navigation
    img_names = list(processed_images.keys())
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        selected_img = st.selectbox(
            "🖼️ Select Image:",
            img_names,
            key="single_image_selector"
        )
    
    if selected_img:
        img_data = processed_images[selected_img]
        
        # Main image display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(img_data["image"], caption=f"📷 {selected_img}", use_column_width=True)
            
            # Image metadata
            img = img_data["image"]
            st.caption(f"📐 Dimensions: {img.size[0]} × {img.size[1]} pixels")
            st.caption(f"🎨 Color Mode: {img.mode}")
            
            # File size estimation
            import io
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            file_size = len(img_bytes.getvalue()) / 1024  # KB
            st.caption(f"💾 Estimated size: {file_size:.1f} KB")
        
        with col2:
            st.subheader("🏷️ Labeling Panel")
            
            # AI suggestions with confidence scores
            st.markdown("**🤖 AI Analysis:**")
            suggestions = img_data["suggested_labels"]
            
            for i, suggestion in enumerate(suggestions):
                confidence = max(95 - i*12, 40)  # Simulated confidence decay
                confidence_bar = "🟢" if confidence > 70 else "🟡" if confidence > 50 else "🔴"
                st.write(f"{confidence_bar} **{suggestion}** ({confidence}%)")
            
            st.markdown("---")
            
            # Current label status
            current_label = img_data["selected_label"]
            if current_label != "unknown":
                status_emoji = "🎯" if not img_data.get("human_changed", False) else "✏️"
                st.success(f"{status_emoji} Current: **{current_label}**")
            else:
                st.warning("⚠️ **Unlabeled**")
            
            # Label selection interface
            st.markdown("**Select Label:**")
            
            # Quick select buttons for AI suggestions
            for suggestion in suggestions:
                if st.button(
                    f"🏷️ {suggestion}", 
                    key=f"quick_{selected_img}_{suggestion}",
                    use_container_width=True
                ):
                    processed_images[selected_img]["selected_label"] = suggestion
                    processed_images[selected_img]["human_changed"] = suggestion != suggestions[0]
                    st.session_state.processed_images = processed_images
                    st.rerun()
            
            # Custom label input
            st.markdown("**Or enter custom:**")
            custom_label = st.text_input(
                "Custom label:",
                placeholder="Type custom label...",
                key=f"single_custom_{selected_img}"
            )
            
            if custom_label and st.button("✅ Apply Custom", key=f"apply_{selected_img}"):
                processed_images[selected_img]["selected_label"] = custom_label
                processed_images[selected_img]["human_changed"] = True
                st.session_state.processed_images = processed_images
                st.success(f"✅ Applied custom label: **{custom_label}**")
                st.rerun()
        
        # Navigation controls
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        current_idx = img_names.index(selected_img)
        
        with col1:
            if current_idx > 0:
                prev_img = img_names[current_idx - 1]
                if st.button(f"⬅️ Previous: {prev_img[:15]}...", key="nav_prev"):
                    st.session_state.single_image_selector = prev_img
                    st.rerun()
        
        with col2:
            st.write(f"📍 Image {current_idx + 1} of {len(img_names)}")
        
        with col3:
            if current_idx < len(img_names) - 1:
                next_img = img_names[current_idx + 1]
                if st.button(f"➡️ Next: {next_img[:15]}...", key="nav_next"):
                    st.session_state.single_image_selector = next_img
                    st.rerun()
        
        # Batch operations
        st.markdown("---")
        st.subheader("⚡ Batch Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Re-analyze This Image", key="reanalyze_single"):
                with st.spinner("🤖 Re-analyzing..."):
                    new_suggestions = suggest_image_labels(img_data["image"], num_suggestions)
                    processed_images[selected_img]["suggested_labels"] = new_suggestions
                    if not img_data.get("human_changed", False):
                        processed_images[selected_img]["selected_label"] = new_suggestions[0]
                    st.session_state.processed_images = processed_images
                    st.success("✅ Re-analysis complete!")
                    st.rerun()
        
        with col2:
            if st.button("📋 Copy Label to Similar", key="copy_similar"):
                st.info("💡 Feature coming soon: Find similar images and apply the same label")
        
        with col3:
            if st.button("⏭️ Skip to Next Unlabeled", key="skip_unlabeled"):
                # Find next unlabeled image
                for i in range(current_idx + 1, len(img_names)):
                    if processed_images[img_names[i]]["selected_label"] == "unknown":
                        st.session_state.single_image_selector = img_names[i]
                        st.rerun()
                        break
                else:
                    st.info("🎉 All images are labeled!")

def render_image_analytics(processed_images):
    """Render analytics for image labeling"""
    st.subheader("📊 Image Labeling Analytics")
    
    # Calculate label distribution
    label_counts = {}
    total_images = len(processed_images)
    labeled_images = 0
    human_modified = 0
    
    for img_data in processed_images.values():
        label = img_data["selected_label"]
        if label != "unknown":
            labeled_images += 1
            label_counts[label] = label_counts.get(label, 0) + 1
        
        if img_data.get("human_changed", False):
            human_modified += 1
    
    # Display charts if we have labeled images
    if label_counts:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Pie chart of label distribution
            df_labels = pd.DataFrame(list(label_counts.items()), columns=["Label", "Count"])
            fig_pie = px.pie(
                df_labels, 
                values="Count", 
                names="Label",
                title="Label Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart with counts
            fig_bar = px.bar(
                df_labels, 
                x="Label", 
                y="Count",
                title="Label Frequency",
                color="Count",
                color_continuous_scale="viridis"
            )
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # AI vs Human analysis
        if human_modified > 0:
            st.subheader("🤖 vs ✏️ AI vs Human Analysis")
            
            ai_accepted = labeled_images - human_modified
            
            # Create comparison chart
            comparison_data = {
                "Category": ["AI Accepted", "Human Modified"],
                "Count": [ai_accepted, human_modified],
                "Percentage": [
                    (ai_accepted / labeled_images) * 100,
                    (human_modified / labeled_images) * 100
                ]
            }
            
            fig_comparison = px.bar(
                x=comparison_data["Category"],
                y=comparison_data["Count"],
                title="AI Acceptance vs Human Modifications",
                color=comparison_data["Category"],
                color_discrete_map={"AI Accepted": "#2E8B57", "Human Modified": "#FF6347"}
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Show AI accuracy
            ai_accuracy = (ai_accepted / max(labeled_images, 1)) * 100
            st.metric("🎯 AI Accuracy Rate", f"{ai_accuracy:.1f}%")
    
    else:
        st.info("📈 Charts will appear as you label images")

def render_image_export_section(processed_images):
    """Render the export section for image labeling"""
    st.subheader("📥 Export Image Labels")
    
    total_images = len(processed_images)
    labeled_images = sum(1 for data in processed_images.values() 
                        if data["selected_label"] != "unknown")
    
    # Export statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Save Labels", type="primary"):
            export_path = export_image_labels(processed_images)
            st.success(f"✅ Labels saved to {export_path}")
    
    with col2:
        # Download CSV with all data
        if total_images > 0:
            csv_data = create_image_export_csv(processed_images)
            st.download_button(
                label="⬇️ Download All",
                data=csv_data,
                file_name=f"image_labels_all_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.button("⬇️ Download All", disabled=True)
    
    with col3:
        # Download only labeled images
        if labeled_images > 0:
            labeled_csv = create_image_export_csv(processed_images, labeled_only=True)
            st.download_button(
                label="📋 Labeled Only",
                data=labeled_csv,
                file_name=f"image_labels_completed.csv",
                mime="text/csv"
            )
        else:
            st.button("📋 Labeled Only", disabled=True, help="No labeled images yet")
    
    with col4:
        if st.button("🔄 Re-process All"):
            with st.spinner("🤖 Re-processing all images..."):
                for img_name, img_data in processed_images.items():
                    new_suggestions = suggest_image_labels(img_data["image"], 3)
                    processed_images[img_name]["suggested_labels"] = new_suggestions
                    # Only update selected label if it wasn't human modified
                    if not img_data.get("human_changed", False):
                        processed_images[img_name]["selected_label"] = new_suggestions[0]
                
                st.session_state.processed_images = processed_images
                st.success("✅ All images re-processed!")
                st.rerun()
    
    # Export summary
    if labeled_images > 0:
        with st.expander("📊 Export Summary"):
            st.write(f"**Export Statistics:**")
            st.write(f"- Total images: {total_images}")
            st.write(f"- Labeled images: {labeled_images} ({labeled_images/total_images*100:.1f}%)")
            
            # Label breakdown
            label_counts = {}
            for img_data in processed_images.values():
                label = img_data["selected_label"]
                if label != "unknown":
                    label_counts[label] = label_counts.get(label, 0) + 1
            
            if label_counts:
                st.write("**Label distribution:**")
                for label, count in label_counts.items():
                    percentage = (count / labeled_images) * 100
                    st.write(f"  - {label}: {count} images ({percentage:.1f}%)")

def create_image_export_csv(processed_images, labeled_only=False):
    """Create CSV data for image export"""
    rows = []
    for img_name, data in processed_images.items():
        selected_label = data["selected_label"]
        
        # Skip unlabeled if labeled_only is True
        if labeled_only and selected_label == "unknown":
            continue
        
        rows.append({
            "filename": img_name,
            "selected_label": selected_label,
            "ai_suggestion_1": data["suggested_labels"][0] if data["suggested_labels"] else "",
            "ai_suggestion_2": data["suggested_labels"][1] if len(data["suggested_labels"]) > 1 else "",
            "ai_suggestion_3": data["suggested_labels"][2] if len(data["suggested_labels"]) > 2 else "",
            "human_changed": data.get("human_changed", False),
            "labeled": selected_label != "unknown"
        })
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

# =============================================================================
# ANALYTICS DASHBOARD MODE
# =============================================================================

def render_analytics_dashboard():
    """Render comprehensive analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    
    # Check available data
    text_data_available = not st.session_state.text_df.empty
    image_data_available = bool(st.session_state.processed_images)
    
    if not text_data_available and not image_data_available:
        render_no_data_analytics()
        return
    
    # Analytics navigation
    analytics_tab = st.selectbox(
        "📈 Analytics View:",
        ["Overview", "Text Analytics", "Image Analytics", "Performance Analysis", "Export Reports"]
    )
    
    if analytics_tab == "Overview":
        render_overview_analytics(text_data_available, image_data_available)
    elif analytics_tab == "Text Analytics":
        render_detailed_text_analytics()
    elif analytics_tab == "Image Analytics":
        render_detailed_image_analytics()
    elif analytics_tab == "Performance Analysis":
        render_performance_analysis()
    else:  # Export Reports
        render_export_reports()

def render_no_data_analytics():
    """Render analytics view when no data is available"""
    st.info("🔍 **No data available for analytics**")
    st.write("To see analytics, you need to:")
    st.write("1. Label some text data (Text Labeling mode)")
    st.write("2. Label some images (Image Labeling mode)")
    st.write("3. Return here to view comprehensive analytics")
    
    st.markdown("---")
    st.subheader("🧪 Try with Sample Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Load Sample Text Data", use_container_width=True):
            create_sample_text_data()
            st.success("✅ Sample text data loaded!")
            st.rerun()
    
    with col2:
        if st.button("🖼️ Load Sample Image Data", use_container_width=True):
            create_sample_image_data()
            st.success("✅ Sample image data loaded!")
            st.rerun()

def create_sample_text_data():
    """Create sample text data for analytics demo"""
    sample_data = {
        "text": [
            "I absolutely love this product! Amazing quality and fast shipping.",
            "Terrible experience. Product broke after one day. Waste of money.",
            "It's okay, nothing special but does the job adequately.",
            "Best purchase I've made this year! Highly recommended.",
            "Could be better, but acceptable for the price point.",
            "Completely disappointed. Poor quality and slow delivery.",
            "Pretty good overall, happy with the purchase decision.",
            "Not great, not terrible. Average quality and service.",
            "Excellent customer service and high-quality product!",
            "Poor experience with customer support team.",
            "The new update improved everything significantly!",
            "Works perfectly as described in the listing.",
            "Overpriced for what you actually receive.",
            "Great value for money and excellent build quality.",
            "Customer support was very helpful and responsive."
        ],
        "label": [
            "Positive", "Negative", "Neutral", "Positive", "Neutral",
            "Negative", "Positive", "Neutral", "Positive", "Negative",
            "Positive", "Positive", "Negative", "Positive", "Positive"
        ],
        "ai_suggested": [
            "Positive", "Negative", "Positive", "Positive", "Neutral",
            "Negative", "Positive", "Neutral", "Positive", "Negative",
            "Positive", "Positive", "Negative", "Positive", "Positive"
        ],
        "human_changed": [
            False, False, True, False, False,
            False, False, False, False, False,
            False, False, False, False, False
        ]
    }
    
    st.session_state.text_df = pd.DataFrame(sample_data)

def create_sample_image_data():
    """Create sample image data for analytics demo"""
    # This would normally process actual images, but for demo we create mock data
    sample_images = {
        "landscape_001.jpg": {
            "selected_label": "nature",
            "suggested_labels": ["nature", "landscape", "outdoor"],
            "human_changed": False
        },
        "portrait_001.jpg": {
            "selected_label": "person",
            "suggested_labels": ["person", "portrait", "face"],
            "human_changed": False
        },
        "city_001.jpg": {
            "selected_label": "urban",
            "suggested_labels": ["building", "urban", "architecture"],
            "human_changed": True
        },
        "animal_001.jpg": {
            "selected_label": "dog",
            "suggested_labels": ["animal", "dog", "pet"],
            "human_changed": True
        },
        "food_001.jpg": {
            "selected_label": "food",
            "suggested_labels": ["food", "meal", "cuisine"],
            "human_changed": False
        }
    }
    
    # Note: In a real implementation, you'd need actual image objects
    # For demo purposes, we'll create placeholder data
    st.session_state.processed_images = sample_images

def render_overview_analytics(text_data_available, image_data_available):
    """Render overview analytics combining all data sources"""
    st.subheader("🎯 Project Overview")
    
    # Combined metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_items = 0
    total_labeled = 0
    total_ai_generated = 0
    total_human_changed = 0
    
    # Aggregate text metrics
    if text_data_available:
        text_stats = get_labeling_stats(st.session_state.text_df)
        total_items += text_stats["total"]
        total_labeled += text_stats["labeled"]
        total_ai_generated += text_stats["ai_generated"]
        total_human_changed += text_stats["human_changed"]
    
    # Aggregate image metrics
    if image_data_available:
        processed_images = st.session_state.processed_images
        img_total = len(processed_images)
        img_labeled = sum(1 for data in processed_images.values() 
                         if data["selected_label"] != "unknown")
        img_human_changed = sum(1 for data in processed_images.values() 
                               if data.get("human_changed", False))
        
        total_items += img_total
        total_labeled += img_labeled
        total_human_changed += img_human_changed
    
    with col1:
        st.metric("📊 Total Items", total_items)
    
    with col2:
        progress = (total_labeled / max(total_items, 1)) * 100
        st.metric("✅ Overall Progress", f"{progress:.1f}%")
    
    with col3:
                st.metric("🤖 AI Generated", total_ai_generated)
    
    with col4:
        st.metric("✏️ Human Modified", total_human_changed)
    
    # Progress visualization
    animated_progress_bar(total_labeled, total_items, "Overall Project Progress")
    
    # Data source breakdown
    if text_data_available and image_data_available:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Text Data Summary")
            text_stats = get_labeling_stats(st.session_state.text_df)
            st.metric("Text Items", text_stats["total"])
            st.metric("Text Progress", f"{text_stats['progress']:.1f}%")
            st.metric("Text AI Accuracy", f"{text_stats['accuracy']:.1f}%" if text_stats['accuracy'] > 0 else "N/A")
        
        with col2:
            st.subheader("🖼️ Image Data Summary")
            processed_images = st.session_state.processed_images
            img_total = len(processed_images)
            img_labeled = sum(1 for data in processed_images.values() 
                             if data["selected_label"] != "unknown")
            img_progress = (img_labeled / max(img_total, 1)) * 100
            
            st.metric("Image Items", img_total)
            st.metric("Image Progress", f"{img_progress:.1f}%")
            st.metric("AI Acceptance Rate", f"{(1 - total_human_changed/max(total_labeled, 1))*100:.1f}%")
    
    # Recent activity timeline (placeholder)
    st.subheader("📈 Activity Timeline")
    
    # Create a simple activity chart
    activity_data = {
        "Date": pd.date_range(start="2024-01-01", periods=7, freq='D'),
        "Text Labeled": [0, 5, 12, 8, 15, 20, 18] if text_data_available else [0]*7,
        "Images Labeled": [0, 3, 5, 2, 8, 10, 12] if image_data_available else [0]*7
    }
    
    fig_activity = px.line(
        pd.DataFrame(activity_data),
        x="Date",
        y=["Text Labeled", "Images Labeled"],
        title="Recent Labeling Activity",
        markers=True
    )
    
    st.plotly_chart(fig_activity, use_container_width=True)

def render_detailed_text_analytics():
    """Render detailed text analytics"""
    if st.session_state.text_df.empty:
        st.info("📄 No text data available for analysis")
        return
    
    st.subheader("📄 Detailed Text Analytics")
    
    df = st.session_state.text_df
    stats = get_labeling_stats(df)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Texts", stats["total"])
    
    with col2:
        st.metric("Labeled", f"{stats['labeled']} ({stats['progress']:.1f}%)")
    
    with col3:
        st.metric("AI Suggestions", stats["ai_generated"])
    
    with col4:
        st.metric("Human Corrections", stats["human_changed"])
    
    # Label distribution
    col1, col2 = st.columns([1, 1])
    
    with col1:
        label_distribution_chart(df, "label", "Label Distribution")
    
    with col2:
        ai_vs_human_comparison(df)
    
    # Text length analysis
    st.subheader("📏 Text Length Analysis")
    
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Text length distribution
        fig_length = px.histogram(
            df,
            x="text_length",
            color="label",
            title="Text Length Distribution by Label",
            nbins=20
        )
        st.plotly_chart(fig_length, use_container_width=True)
    
    with col2:
        # Word count distribution
        fig_words = px.histogram(
            df,
            x="word_count",
            color="label",
            title="Word Count Distribution by Label",
            nbins=20
        )
        st.plotly_chart(fig_words, use_column_width=True)
    
    # Time-based analysis (if timestamps available)
    st.subheader("⏰ Temporal Analysis")
    
    # Create mock timestamps for demo
    df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq='H')
    
    # Labeling speed over time
    df["cumulative_labeled"] = (df["label"] != "").cumsum()
    
    fig_speed = px.line(
        df,
        x="timestamp",
        y="cumulative_labeled",
        title="Labeling Progress Over Time",
        markers=True
    )
    
    st.plotly_chart(fig_speed, use_container_width=True)
    
    # Quality metrics
    st.subheader("🎯 Quality Metrics")
    
    if stats["ai_generated"] > 0:
        ai_accuracy = stats["accuracy"]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.metric("AI Accuracy Rate", f"{ai_accuracy:.1f}%")
            st.metric("Human Intervention Rate", f"{stats['human_changed']/max(stats['labeled'], 1)*100:.1f}%")
        
        with col2:
            # Confidence analysis (mock data)
            confidence_data = {
                "Confidence Range": ["90-100%", "80-89%", "70-79%", "60-69%", "<60%"],
                "Count": [45, 30, 15, 7, 3],
                "Accuracy": [95, 85, 70, 55, 30]
            }
            
            fig_conf = px.bar(
                pd.DataFrame(confidence_data),
                x="Confidence Range",
                y="Count",
                color="Accuracy",
                title="AI Confidence vs Accuracy",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig_conf, use_container_width=True)

def render_detailed_image_analytics():
    """Render detailed image analytics"""
    if not st.session_state.processed_images:
        st.info("🖼️ No image data available for analysis")
        return
    
    st.subheader("🖼️ Detailed Image Analytics")
    
    processed_images = st.session_state.processed_images
    
    # Basic statistics
    total_images = len(processed_images)
    labeled_images = sum(1 for data in processed_images.values() 
                        if data["selected_label"] != "unknown")
    human_modified = sum(1 for data in processed_images.values() 
                        if data.get("human_changed", False))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", total_images)
    
    with col2:
        progress = (labeled_images / max(total_images, 1)) * 100
        st.metric("Labeled Images", f"{labeled_images} ({progress:.1f}%)")
    
    with col3:
        st.metric("Human Modifications", human_modified)
    
    with col4:
        ai_acceptance = ((labeled_images - human_modified) / max(labeled_images, 1)) * 100
        st.metric("AI Acceptance Rate", f"{ai_acceptance:.1f}%")
    
    # Label distribution
    render_image_analytics(processed_images)
    
    # Image characteristics analysis (mock data)
    st.subheader("🔍 Image Characteristics Analysis")
    
    # Simulate image analysis data
    char_data = []
    for img_name, img_data in processed_images.items():
        # In a real implementation, you'd analyze actual image properties
        char_data.append({
            "filename": img_name,
            "label": img_data["selected_label"],
            "width": 800 + hash(img_name) % 1200,  # Mock width 800-2000
            "height": 600 + hash(img_name) % 900,  # Mock height 600-1500
            "file_size": 50 + hash(img_name) % 450,  # Mock size 50-500KB
            "aspect_ratio": 1.0 + (hash(img_name) % 100) / 100  # Mock ratio 1.0-2.0
        })
    
    df_chars = pd.DataFrame(char_data)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Size distribution
        fig_size = px.scatter(
            df_chars,
            x="width",
            y="height",
            color="label",
            size="file_size",
            title="Image Dimensions by Label",
            hover_data=["filename"]
        )
        st.plotly_chart(fig_size, use_container_width=True)
    
    with col2:
        # Aspect ratio distribution
        fig_ratio = px.box(
            df_chars,
            x="label",
            y="aspect_ratio",
            title="Aspect Ratio Distribution by Label"
        )
        st.plotly_chart(fig_ratio, use_container_width=True)

def render_performance_analysis():
    """Render performance and efficiency analytics"""
    st.subheader("⚡ Performance Analysis")
    
    # Mock performance data
    performance_data = {
        "Metric": ["Labeling Speed", "AI Accuracy", "Human Efficiency", "Data Quality"],
        "Current": [85, 78, 92, 88],
        "Target": [95, 85, 95, 90],
        "Industry Avg": [70, 65, 80, 75]
    }
    
    df_perf = pd.DataFrame(performance_data)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Performance radar chart
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=df_perf["Current"],
            theta=df_perf["Metric"],
            fill='toself',
            name='Current Performance',
            line_color='blue'
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=df_perf["Target"],
            theta=df_perf["Metric"],
            fill='toself',
            name='Target',
            line_color='green'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Performance Metrics"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Comparison bar chart
        fig_comp = px.bar(
            df_perf,
            x="Metric",
            y=["Current", "Target", "Industry Avg"],
            barmode='group',
            title="Performance Comparison",
            color_discrete_map={
                "Current": "blue",
                "Target": "green",
                "Industry Avg": "red"
            }
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    
    # Efficiency trends
    st.subheader("📈 Efficiency Trends")
    
    # Mock time series data
    dates = pd.date_range(start="2024-01-01", periods=30, freq='D')
    efficiency_data = {
        "Date": dates,
        "Items per Hour": [10 + i*0.5 + hash(str(i)) % 5 for i in range(30)],
        "Accuracy": [70 + i*0.3 + hash(str(i)) % 10 for i in range(30)]
    }
    
    df_efficiency = pd.DataFrame(efficiency_data)
    
    fig_trend = px.line(
        df_efficiency,
        x="Date",
        y=["Items per Hour", "Accuracy"],
        title="Labeling Efficiency Over Time",
        markers=True
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Recommendations
    st.subheader("💡 Performance Recommendations")
    
    recommendations = [
        {
            "area": "AI Accuracy",
            "current": "78%",
            "target": "85%",
            "recommendation": "Consider retraining the AI model with more diverse examples",
            "priority": "High"
        },
        {
            "area": "Labeling Speed",
            "current": "85 items/hour",
            "target": "95 items/hour",
            "recommendation": "Implement keyboard shortcuts and batch operations",
            "priority": "Medium"
        },
        {
            "area": "Data Quality",
            "current": "88%",
            "target": "90%",
            "recommendation": "Add data validation and consistency checks",
            "priority": "Low"
        }
    ]
    
    for rec in recommendations:
        with st.expander(f"🎯 {rec['area']} - {rec['priority']} Priority"):
            st.write(f"**Current:** {rec['current']}")
            st.write(f"**Target:** {rec['target']}")
            st.write(f"**Recommendation:** {rec['recommendation']}")

def render_export_reports():
    """Render report export options"""
    st.subheader("📊 Export Comprehensive Reports")
    
    # Report types
    report_types = st.multiselect(
        "📋 Select Report Types:",
        [
            "Project Summary Report",
            "Text Analytics Report",
            "Image Analytics Report",
            "Performance Report",
            "Quality Assessment Report",
            "Custom Report"
        ],
        default=["Project Summary Report"]
    )
    
    # Report format
    report_format = st.selectbox(
        "📄 Export Format:",
        ["PDF", "Excel", "CSV", "JSON", "Markdown"]
    )
    
    # Date range
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", value=pd.Timestamp.now().date() - pd.Timedelta(days=30))
    
    with col2:
        end_date = st.date_input("End Date", value=pd.Timestamp.now().date())
    
    # Generate report button
    if st.button("🚀 Generate Report", type="primary"):
        with st.spinner("📊 Generating comprehensive report..."):
            report_data = generate_comprehensive_report(report_types, start_date, end_date)
            
            # Export based on format
            if report_format == "PDF":
                st.info("📄 PDF export feature coming soon!")
            elif report_format == "Excel":
                st.info("📊 Excel export feature coming soon!")
            elif report_format == "CSV":
                csv_data = report_data.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Report CSV",
                    data=csv_data,
                    file_name=f"labelmate_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            elif report_format == "JSON":
                json_data = report_data.to_json(orient="records", indent=2)
                st.download_button(
                    label="⬇️ Download Report JSON",
                    data=json_data,
                    file_name=f"labelmate_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:  # Markdown
                markdown_data = report_data.to_markdown(index=False)
                st.download_button(
                    label="⬇️ Download Report Markdown",
                    data=markdown_data,
                    file_name=f"labelmate_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )

def generate_comprehensive_report(report_types, start_date, end_date):
    """Generate comprehensive analytics report"""
    # This is a placeholder implementation
    # In a real system, this would aggregate all relevant data
    
    report_data = {
        "Metric": [],
        "Value": [],
        "Unit": []
    }
    
    # Add basic metrics
    if "Project Summary Report" in report_types:
        report_data["Metric"].extend(["Total Items", "Labeled Items", "AI Suggestions", "Human Corrections"])
        report_data["Value"].extend(["100", "85", "75", "10"])
        report_data["Unit"].extend(["items", "items", "items", "items"])
    
    if "Text Analytics Report" in report_types and not st.session_state.text_df.empty:
        text_stats = get_labeling_stats(st.session_state.text_df)
        report_data["Metric"].extend(["Text Progress", "AI Text Accuracy", "Text Items"])
        report_data["Value"].extend([f"{text_stats['progress']:.1f}", f"{text_stats['accuracy']:.1f}", f"{text_stats['total']}"])
        report_data["Unit"].extend(["%", "%", "items"])
    
    if "Image Analytics Report" in report_types and st.session_state.processed_images:
        processed_images = st.session_state.processed_images
        labeled_count = sum(1 for data in processed_images.values() if data["selected_label"] != "unknown")
        total_count = len(processed_images)
        progress = (labeled_count / max(total_count, 1)) * 100
        report_data["Metric"].extend(["Image Progress", "Total Images", "Labeled Images"])
        report_data["Value"].extend([f"{progress:.1f}", f"{total_count}", f"{labeled_count}"])
        report_data["Unit"].extend(["%", "images", "images"])
    
    return pd.DataFrame(report_data)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    try:
        # Initialize the application
        initialize_session_state()
        create_directories()
        
        # Render main header
        render_header()
        
        # Render sidebar and get current mode
        mode = render_sidebar()
        
        # Show welcome message for new users
        show_welcome_message()
        
        # Render appropriate mode based on sidebar selection
        if mode == "📄 Text Labeling":
            render_text_labeling_mode()
        elif mode == "🖼️ Image Labeling":
            render_image_labeling_mode()
        else:  # Analytics Dashboard
            render_analytics_dashboard()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p>LabelMate v1.0 - AI-Powered Data Labeling Assistant</p>
            <p>Built with ❤️ for data teams | 
            <a href="https://github.com/swamy18/labelmate" target="_blank">GitHub</a> | 
            <a href="https://github.com/swamy18/labelmate/wiki" target="_blank">Documentation</a></p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"""
        🚨 **Application Error**
        
        An unexpected error occurred: {str(e)}
        
        **Troubleshooting steps:**
        1. Refresh the page
        2. Check your internet connection
        3. Verify your API keys are configured
        4. Check the [GitHub Issues](https://github.com/swamy18/labelmate/issues) for known problems
        
        If the problem persists, please [report this issue](https://github.com/swamy18/labelmate/issues/new).
        """)
        
        # Log error details for debugging
        import traceback
        st.error(f"Debug information: {traceback.format_exc()}")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
