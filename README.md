# 🏷️ LabelMate - AI-Powered Data Labeling Assistant

**By [@swamy18](https://github.com/swamy18)**

A production-ready mini-product that accelerates data annotation workflows for ML teams using AI assistance.

## 🚀 What LabelMate Does

LabelMate is a comprehensive data labeling tool that combines AI automation with human oversight to create high-quality labeled datasets efficiently:

### 📄 **Text Labeling**
- **Sentiment Analysis**: Automatically classify text as Positive, Negative, or Neutral
- **Topic Classification**: Categorize text into Technology, Business, Politics, Sports, etc.
- **Batch Processing**: Label hundreds of texts with one click
- **Human Review**: Easy interface to correct AI mistakes
- **Change Tracking**: Monitor which labels were AI-generated vs human-corrected

### 🖼️ **Image Labeling**  
- **Multi-Label Suggestions**: AI provides top-3 label suggestions per image
- **Flexible Views**: Grid, List, or Single-image detailed views
- **Custom Labels**: Add your own labels beyond AI suggestions
- **Batch Operations**: Re-process all images with updated AI models

### 📊 **Analytics Dashboard**
- **Real-time Progress**: Track labeling completion rates
- **AI Performance**: Monitor accuracy and human correction rates  
- **Data Quality**: Insights into label distribution and consistency
- **Export Reports**: Comprehensive analytics in JSON and Markdown formats

## ⚙️ Quick Setup (< 5 minutes)

### 1. Clone and Install
```bash
git clone https://github.com/swamy18/labelmate.git
cd labelmate
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_gemini_key_here" > .env
# OR for OpenAI
echo "OPENAI_API_KEY=your_openai_key_here" > .env
```

**Get Free API Keys:**
- [Google Gemini](https://makersuite.google.com/app/apikey) (Recommended - handles both text and images)
- [OpenAI](https://platform.openai.com/api-keys) (Text only, requires paid account)

### 3. Launch Application
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

## 🎯 Demo Workflow

### Text Labeling Demo:
1. **Upload Data**: Use the provided `data/sample_texts.csv` or upload your own CSV with a 'text' column
2. **AI Processing**: Click "Auto-Label All" to get AI sentiment predictions
3. **Human Review**: Review and correct any mislabeled items using the dropdown selectors
4. **Export Results**: Download the labeled CSV with tracking of AI vs human labels

### Image Labeling Demo:
1. **Upload Images**: Drop 5-10 sample images (JPG/PNG format)
2. **AI Suggestions**: System provides 3 label suggestions per image using Gemini Vision
3. **Label Selection**: Choose the best label or add custom ones
4. **Export Labels**: Download CSV mapping filenames to selected labels

### Analytics Demo:
1. **Progress Tracking**: Real-time charts showing completion rates
2. **Performance Metrics**: AI accuracy rates and human correction statistics  
3. **Data Insights**: Label distribution analysis and quality metrics
4. **Report Export**: Generate comprehensive analytics reports

## 📁 Project Structure

```
labelmate/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies  
├── .env                   # API keys (create this)
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── utils/                # Helper modules
│   ├── __init__.py
│   ├── text_helper.py    # Text labeling logic
│   ├── image_helper.py   # Image labeling logic  
│   └── charts.py         # Analytics and visualization
├── data/                 # Sample data and temporary files
│   ├── sample_texts.csv  # Demo text dataset
│   └── temp_images/      # Temporary image storage
└── exports/              # Output folder (auto-generated)
    ├── labeled_text.csv
    ├── labeled_images.csv
    └── analytics_report.json
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit (Python-based web interface)
- **AI Backend**: Google Gemini 1.5 Flash (text + vision) or OpenAI GPT-3.5
- **Data Processing**: Pandas for CSV manipulation
- **Visualization**: Plotly for interactive charts and progress tracking
- **Image Processing**: PIL (Pillow) for image handling
- **State Management**: Streamlit session state for real-time updates

## 🔧 Advanced Configuration

### Multiple AI Providers
The system automatically detects available API keys:
1. **Gemini** (preferred): Handles both text and image labeling
2. **OpenAI**: Text labeling only (requires separate vision solution)

### Custom Labeling Tasks
Modify `utils/text_helper.py` to add new classification tasks:
```python
CUSTOM_PROMPT = """
Your custom classification prompt here.
Return one of: Option1, Option2, Option3
"""
```

### Batch Processing Settings
Adjust batch sizes in the UI:
- **Text**: Process 1-100 items at once
- **Images**: Upload and process multiple files simultaneously

## 📈 Production Deployment

### Local Production Mode
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.headless", "true"]
```

### Cloud Deployment
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Use provided Procfile
- **AWS/GCP**: Container-based deployment

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black app.py utils/
```

### Performance Optimization:
- **Text**: Process in batches of 10-20 for optimal speed
- **Images**: Resize large images before upload
- **Memory**: Clear session state periodically for large datasets

## 📊 Analytics & Metrics

LabelMate tracks comprehensive metrics:

- **Progress Tracking**: Real-time completion percentages
- **AI Performance**: Accuracy rates and confidence scores  
- **Human Oversight**: Correction rates and consistency metrics
- **Data Quality**: Label distribution and anomaly detection
- **Export Analytics**: JSON reports for further analysis

## 🔒 Data Privacy & Security

- **Local Processing**: All data stays on your machine by default
- **API Calls**: Only text/image content sent to AI providers (no personal data)
- **No Storage**: No permanent data storage in cloud services
- **Session Based**: Data cleared when application restarts

## 📚 Use Cases

### ML/AI Teams
- **Dataset Creation**: Rapidly label training data for supervised learning
- **Data Augmentation**: Generate labels for synthetic or scraped data
- **Quality Assurance**: Validate existing labels with AI assistance

### Research Organizations  
- **Survey Analysis**: Classify open-ended survey responses
- **Content Analysis**: Categorize research papers, articles, or documents
- **Image Annotation**: Label research images for computer vision projects

### Business Applications
- **Customer Feedback**: Analyze sentiment in reviews and support tickets
- **Content Moderation**: Classify user-generated content for policy compliance
- **Market Research**: Tag and categorize social media posts and comments

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit** team for the amazing web framework
- **Google** for the powerful Gemini API
- **OpenAI** for GPT models and vision capabilities
- **Plotly** for interactive visualization components
- **PIL/Pillow** for robust image processing

## 🌟 Star History

If you find LabelMate useful, please give it a star on GitHub! ⭐

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/swamy18/labelmate/issues)
- **Discussions**: [GitHub Discussions](https://github.com/swamy18/labelmate/discussions)  
- **Email**: Contact via GitHub profile
- **Documentation**: [Wiki](https://github.com/swamy18/labelmate/wiki)

---

**Made by [@swamy18](https://github.com/swamy18)**

*LabelMate - Accelerating AI development through intelligent data labeling*
"""
