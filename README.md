# 📧 AI-Powered Email Agent

A local, privacy-focused email automation tool that uses AI to generate personalized emails for multiple recipients. Built with Qwen2.5 0.5B Instruct language model and Gmail API integration.

## 🌟 Features

- **🤖 AI-Generated Emails**: Uses Qwen2.5 0.5B Instruct LLM to write unique, personalized emails for each recipient
- **⚡ Ultra-Fast**: Generates emails in 1-3 seconds on GPU, 5-10 seconds on CPU
- **🔒 Completely Local**: All AI processing runs on your machine - no external API calls, no data sent to cloud
- **💾 Low Requirements**: Works on modest hardware with only ~1GB model download
- **🎯 Context-Aware**: Understands complex email context and generates natural, professional emails
- **✅ Preview & Confirm**: Review all emails before sending - manual approval required
- **📊 Email Validation**: Automatic validation of email addresses before sending
- **🔐 OAuth2 Security**: Secure Gmail integration using official Google OAuth2
- **📈 Detailed Reports**: Comprehensive sending reports with success/failure tracking
- **🚀 FP16 Optimization**: Efficient half-precision inference for both GPU and CPU

## 🏗️ Architecture

### **AI/ML Components:**
- **Language Model**: Qwen2.5 0.5B Instruct (HuggingFace Transformers)
- **Inference**: FP16 precision with `torch.cuda.amp.autocast`
- **Framework**: PyTorch for model loading and inference
- **Tokenization**: Qwen chat template for proper instruction formatting

### **Web/API Integration:**
- **Gmail API**: Google API Python Client for email sending
- **OAuth2 Authentication**: Google OAuth2 flow for secure access
- **HTTP**: Gmail API uses RESTful endpoints

### **Backend:**
- **Language**: Python 3.8+
- **Data Processing**: Pandas for recipient management, Tabulate for report formatting
- **Validation**: Regex-based email validation

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: ~1GB for model download
- **GPU**: Optional (CUDA-enabled GPU for faster generation)
- **Google Account**: For Gmail API access

## 🚀 Installation

### 1. Clone or Download

```bash
git clone <your-repo-url>
cd mail_agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Gmail API Credentials

**Important**: You need to create your own Google Cloud project and credentials.

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable **Gmail API** for your project
4. Go to **APIs & Services** → **OAuth consent screen**
   - Choose "External" (if not an organization account)
   - Fill in required fields (app name, user support email, developer email)
   - Add yourself as a **Test User** in the "Test users" section
5. Go to **Credentials** → **Create Credentials** → **OAuth 2.0 Client ID**
   - Choose "Desktop app" as application type
   - Download the credentials JSON file
6. **Save the downloaded file as `credentials.json`** in the project folder

**Security Note**: Never commit `credentials.json` or `token.json` to version control!

## 📖 Usage

### Quick Start

1. **Open the Notebook**: Launch `mail_agent_v1.ipynb` in Jupyter or VS Code

2. **Run Setup Cells**: Execute cells 1-11 in order to:
   - Install dependencies (if not already installed)
   - Import libraries
   - Load the Qwen2.5 model (downloads ~1GB on first run)

3. **Configure Recipients** (Section A):
```python
recipients = [
    {'name': 'John Doe', 'email': 'john@example.com'},
    {'name': 'Jane Smith', 'email': 'jane@example.com'},
]

email_context = """
Your detailed email context here...
Be as specific as possible - the AI will use all details!
"""
```

4. **Generate Emails** (Section B): Run the cell to see AI-generated emails

5. **Review & Send** (Section C): 
   - Review each email carefully
   - Change `send_confirmation = "yes"` to approve sending
   - Run the cell to send

### Example Context

```python
email_context = """
I'm a recent Computer Science graduate with strong experience in 
machine learning and full-stack development. Currently seeking 
entry-level or internship positions.

Key qualifications:
- 3 years Python experience with PyTorch, TensorFlow, scikit-learn
- Built 5+ ML projects including NLP chatbot and image classifier
- Full-stack skills: React, Node.js, PostgreSQL, REST APIs
- Strong problem-solving with 300+ LeetCode problems solved
- Bachelor's in CS from State University (GPA: 3.8/4.0)

Available to start immediately. GitHub: https://github.com/harshad-k-135 
Portfolio: https://portfolioharshad.vercel.app/
Looking for opportunities in AI/ML, backend, or data science roles.
"""
```

## 🎓 Key Concepts & Learning Outcomes

### **AI/ML Concepts**

1. **Language Model Inference**
   - Loading pre-trained models from HuggingFace
   - Using `AutoTokenizer` and `AutoModelForCausalLM`
   - Understanding model parameters (0.5B = 500 million parameters)

2. **FP16 Precision**
   - Half-precision floating point for efficient inference
   - Using `torch_dtype=torch.float16`
   - Automatic Mixed Precision (AMP) with `torch.cuda.amp.autocast`

3. **Prompt Engineering**
   - Structured prompts with clear instructions
   - Chat templates for instruct-tuned models
   - Context injection and formatting

4. **Generation Parameters**
   - `temperature`: Controls randomness (0.7 = balanced)
   - `top_p`: Nucleus sampling (0.9 = diverse but coherent)
   - `repetition_penalty`: Prevents repetitive text (1.1)
   - `max_new_tokens`: Output length limit

5. **Model Optimization**
   - Device mapping for GPU/CPU
   - Memory-efficient loading techniques
   - Inference optimization with `torch.no_grad()`

### **Web Development & API Integration**

1. **OAuth2 Authentication**
   - Understanding OAuth2 flow
   - Client credentials and tokens
   - Token refresh mechanisms
   - Secure credential storage

2. **RESTful APIs**
   - Gmail API service construction
   - HTTP request/response handling
   - Error handling with `HttpError`

3. **Email Protocols**
   - MIME multipart messages
   - Base64 encoding for email transmission
   - Email header construction

4. **Data Validation**
   - Regex patterns for email validation
   - Input sanitization
   - Error handling and user feedback

### **Software Engineering Practices**

1. **Object-Oriented Programming**
   - Classes for modularity (`EmailAgent`, `QwenEmailGenerator`)
   - Encapsulation of functionality
   - State management

2. **Error Handling**
   - Try-except blocks for robust code
   - Graceful fallbacks (template mode)
   - User-friendly error messages

3. **Code Organization**
   - Separation of concerns
   - Reusable functions
   - Clear documentation

4. **User Experience**
   - Interactive Jupyter notebook interface
   - Progress indicators
   - Confirmation workflows
   - Comprehensive reporting

## 🔧 Configuration Options

### Model Settings

```python
# In QwenEmailGenerator.generate()
qwen_generator.generate(
    prompt=prompt,
    max_length=1024,      # Max tokens to generate
    temperature=0.7       # Creativity (0.0-1.0)
)
```

### Email Generation

```python
# Disable AI, use template mode
agent.toggle_llm(False)

# Re-enable AI mode
agent.toggle_llm(True)
```

## 📊 Performance Benchmarks

| Hardware | Generation Time (per email) |
|----------|----------------------------|
| GPU (CUDA) with FP16 | 1-3 seconds ⚡ |
| CPU with FP16 | 5-10 seconds |
| GPU (First run) | +5 seconds (model loading) |

**Comparison with larger models:**
- 10x faster than Phi-3 Mini (3.8B parameters)
- 20x faster than Mistral 7B (7B parameters)

## 🛡️ Privacy & Security

- **Local Processing**: All AI inference runs locally on your machine
- **No External APIs**: No third-party AI services used
- **OAuth2**: Industry-standard secure authentication
- **Token Storage**: Credentials stored locally in `token.json`
- **Manual Approval**: All emails require explicit user confirmation before sending

**Security Best Practices:**
- Never share `credentials.json` or `token.json`
- Add both files to `.gitignore`
- Use test users in OAuth consent screen during development
- Review all generated emails before sending

## 📁 Project Structure

```
mail_agent/
├── mail_agent_v1.ipynb     # Main notebook
├── requirements.txt         # Python dependencies
├── credentials.json         # Gmail API credentials (YOU MUST CREATE)
├── token.json              # OAuth2 token (auto-generated)
└── README.md               # This file
```

## 🐛 Troubleshooting

### Gmail API Issues

**Error: `credentials.json not found`**
- Solution: Follow Gmail API setup instructions above

**Error: `access_denied` or `Error 403`**
- Solution: Add yourself as a test user in OAuth consent screen

**Token expired errors**
- Solution: Delete `token.json` and re-authenticate

### Model Loading Issues

**Out of memory errors**
- Solution: Close other applications, try CPU mode, or reduce batch size

**Model download fails**
- Solution: Check internet connection, try again (downloads resume)

### Email Generation Issues

**Generic/poor quality emails**
- Solution: Provide more detailed context, increase temperature slightly

**Parsing errors (no subject/body)**
- Solution: Falls back to template mode automatically

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Support for other LLM models
- HTML email formatting
- Attachment support
- Email scheduling
- Additional email providers (Outlook, etc.)

## 📜 License

This project is provided as-is for educational and personal use.

## ⚠️ Disclaimer

- Use responsibly and in compliance with Gmail's terms of service
- Test thoroughly before sending to real recipients
- Always respect recipients' privacy and anti-spam regulations
- The AI-generated content should be reviewed before sending

## 🙏 Acknowledgments

- **Qwen2.5**: Alibaba Cloud for the excellent small language model
- **HuggingFace**: For the transformers library and model hosting
- **Google**: For Gmail API and comprehensive documentation
- **PyTorch**: For the deep learning framework

---

**Built with ❤️ for learning AI, automation, and practical software development**
