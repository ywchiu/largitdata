[繁體中文](README.md) | **English**

# LargitData Academy: Free Python Web Scraping, AI, and ChatGPT Tutorials with Code

**LargitData Academy (大數學堂)** is a free online course platform run by [LargitData](https://www.largitdata.com/). Each lesson is a short video paired with a runnable code example, teaching you how to use Python for web scraping, data analysis, deep learning, and generative AI. This repository contains the example code (Jupyter notebooks) for every lesson on the [LargitData Academy course site](https://www.largitdata.com/courses/). Every notebook matches a free lesson, and you can download it or open it in Google Colab.

- **Course site**: <https://www.largitdata.com/courses/>
- **YouTube channel**: <https://www.youtube.com/@Largitdata>
- **Facebook page**: <https://www.facebook.com/largitdata/>
- **Lessons**: example code for more than 100 lessons, with new lessons added regularly
- **Language and format**: lessons taught in Traditional Chinese; Python 3; Jupyter notebooks (`.ipynb`)

## How to use the examples

1. Find a topic on the [course site](https://www.largitdata.com/courses/) and watch the video.
2. Open the notebook for lesson number `N` at `code/Course_N.ipynb`, or click the **Colab** link in the tables below to run it in your browser.
3. To run the notebooks locally:

   ```bash
   git clone https://github.com/ywchiu/largitdata.git
   cd largitdata
   pip install jupyter
   jupyter notebook code/
   ```

   Install the packages each notebook imports at the top (for example `requests`, `beautifulsoup4`, `pandas`, `selenium`, or `openai`). For lessons that need an API key, use your own key and keep it out of version control.

## Lesson index

Lessons are grouped by the categories on the [LargitData Academy site](https://www.largitdata.com/courses/), newest first within each category. Lesson titles are English translations; the videos and course pages are in Traditional Chinese. Click a lesson title to watch the video, or a notebook to read the code.

- [Vibe Coding](#vibe-coding)
- [Artificial Intelligence](#artificial-intelligence)
- [ChatGPT](#chatgpt)
- [Financial Data Scraping](#financial-data-scraping)
- [Web Scraping in Practice](#web-scraping-in-practice)
- [Selenium Tutorials](#selenium-tutorials)
- [RPA (Robotic Process Automation)](#rpa-robotic-process-automation)
- [Deep Learning](#deep-learning)
- [Algorithmic Trading](#algorithmic-trading)
- [Open Jarvis](#open-jarvis)
- [Other Topics](#other-topics)

### Vibe Coding

Building software with AI coding assistants such as Claude Code and Codex. ([Category page](https://www.largitdata.com/course_list/23))

| # | Lesson | Notebook | Tools |
|---|---|---|---|
| 259 | [Getting the Most out of Fable 5 by Mixing Models](https://www.largitdata.com/course/259/) | [Course_259.md](code/Course_259.md) | Claude, Fable5, DeepSeek, OpenRouter |

### Artificial Intelligence

Hands-on generative AI: Gemini, DeepSeek, Ollama, Whisper, AI agents, and Computer Use. ([Category page](https://www.largitdata.com/course_list/22))

| # | Lesson | Notebook | Tools |
|---|---|---|---|
| 254 | [Beating reCAPTCHA with AI](https://www.largitdata.com/course/254/) | [Course_254.ipynb](code/Course_254.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_254.ipynb) | ReCAPTCHA, Gemini, Selenium |
| 251 | [Deploying DeepSeek: From the 1.5B Distilled Model to the Full 671B Model](https://www.largitdata.com/course/251/) | [Course_251.ipynb](code/Course_251.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_251.ipynb) | LLM, Ollama, DeepSeek, vLLM |
| 250 | [AI Agent Tutorial: Building a Stock Analyst AI for Beginners](https://www.largitdata.com/course/250/) | [Course_250.ipynb](code/Course_250.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_250.ipynb) | AIAgent, OpenAI |
| 249 | [Real-Time Translation in About 100 Lines of Code with Gemini 2 Flash](https://www.largitdata.com/course/249/) | [Course_249.py](code/Course_249.py) | Gemini2Flash |
| 247 | [Letting AI Control My Computer: Testing Claude's Computer Use](https://www.largitdata.com/course/247/) | [Course_247.ipynb](code/Course_247.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_247.ipynb) | Anthropic, ComputerUse, AIAssistant |
| 246 | [Building a NotebookLM-Style Podcast Generator with the OpenAI API](https://www.largitdata.com/course/246/) | [Course_246.ipynb](code/Course_246.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_246.ipynb) | ChatGPT, GPT4o-mini, NotebookLM, OpenAI |
| 243 | [Multimodal AI in Practice: Real-Time Speech Translation and Synthesis with Gemini and ElevenLabs](https://www.largitdata.com/course/243/) | [Course_243.ipynb](code/Course_243.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_243.ipynb) | Gemini, ElevenLabs |
| 241 | [Building a Real-Time Subtitle Transcriber with Whisper](https://www.largitdata.com/course/241/) | [Course_241.ipynb](code/Course_241.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_241.ipynb) | whisper |
| 240 | [Generating Articles and Describing Images with Local LLMs in Ollama](https://www.largitdata.com/course/240/) | [Course_240.ipynb](code/Course_240.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_240.ipynb) | Ollama, llm, breeze7b, llava |
| 238 | [The Tech Behind the Hong Kong Deepfake Scam: Making Deepfake Videos with Free Colab and Basic Python](https://www.largitdata.com/course/238/) | [Course_238.ipynb](code/Course_238.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_238.ipynb) | DeepFake, Roop |
| 233 | [EasyOCR vs. PaddleOCR: Which Is the Best Image-to-Text Tool?](https://www.largitdata.com/course/233/) | [Course_233.ipynb](code/Course_233.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_233.ipynb) | OCR, EasyOCR, PaddleOCR, AIMochi |
| 230 | [Creating Music from Text with AudioCraft](https://www.largitdata.com/course/230/) | [Course_230.ipynb](code/Course_230.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_230.ipynb) | meta, audiocraft, musicgen, audiogen |

### ChatGPT

ChatGPT and the OpenAI API, Llama 2 fine-tuning, RAG, PDF translation, and voice chat. ([Category page](https://www.largitdata.com/course_list/21))

| # | Lesson | Notebook | Tools |
|---|---|---|---|
| 242 | [Translating PDFs on Google Drive with LlamaParse and ChatGPT](https://www.largitdata.com/course/242/) | [Course_242.ipynb](code/Course_242.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_242.ipynb) | ChatGPT, LlamaParse, LlamaIndex, RAG |
| 237 | [Building a Financial News Aggregator with Python Web Scraping and GPTs](https://www.largitdata.com/course/237/) | [Course_237.ipynb](code/Course_237.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_237.ipynb) | moneydj, proxy, brightdata, BrightDataProxy |
| 231 | [Fine-Tuning ChatGPT into a Customer Service Assistant](https://www.largitdata.com/course/231/) | [Course_231.ipynb](code/Course_231.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_231.ipynb) | chatgpt, finetuning |
| 229 | [Fine-Tuning Llama 2 on Your Own Dataset in Colab](https://www.largitdata.com/course/229/) | [Course_229.ipynb](code/Course_229.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_229.ipynb) | Llama2, LLaMAEfficientTuning, FineTune, ChatGPT |
| 228 | [Building Your Own ChatGPT with Meta's Open-Source Llama 2](https://www.largitdata.com/course/228/) | [Course_228.ipynb](code/Course_228.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_228.ipynb) | chatgpt, llama2, huggingface, transformers |
| 226 | [Building a General-Purpose Scraper with ChatGPT to Track Airfares](https://www.largitdata.com/course/226/) | [Course_226.ipynb](code/Course_226.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_226.ipynb) | chatgpt, langchain, selenium |
| 225 | [Web Scraping x Midjourney x ChatGPT: Generating Eye-Catching News Covers Automatically (2/2)](https://www.largitdata.com/course/225/) | [Course_225.ipynb](code/Course_225.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_225.ipynb) | chatgpt, discord, midjourney |
| 224 | [Web Scraping x Midjourney x ChatGPT: Generating Eye-Catching News Covers Automatically (1/2)](https://www.largitdata.com/course/224/) | [Course_224.ipynb](code/Course_224.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_224.ipynb) | MidJourney, ChatGPT, AIPRM, OpenAIAPI |
| 223 | [Understanding Foreign Analysts' Views on TSMC Earnings Calls with ChatGPT](https://www.largitdata.com/course/223/) | [Course_223.ipynb](code/Course_223.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_223.ipynb) | chatgpt, llama_index, langchain, ChatPDF |
| 222 | [Translating PDF Documents Quickly with ChatGPT](https://www.largitdata.com/course/222/) | [Course_222.ipynb](code/Course_222.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_222.ipynb) | chatgpt |
| 221 | [Whisper or CapCut? Choosing the Best Subtitle Tool for Your Videos](https://www.largitdata.com/course/221/) | [Course_221.ipynb](code/Course_221.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_221.ipynb) | chatgpt, whisper, pydub, ytdlp |
| 220 | [Summarizing YouTube Videos with the Whisper and ChatGPT APIs](https://www.largitdata.com/course/220/) | [Course_220.ipynb](code/Course_220.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_220.ipynb) | chatgpt, whisper, pydub, yt-dlp |
| 219 | [Become a Faceless YouTuber with AI](https://www.largitdata.com/course/219/) | [Course_219.ipynb](code/Course_219.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_219.ipynb) | ChatGPT, Midjourney |
| 218 | [Improving ChatGPT's Answers with a Python Web Scraper](https://www.largitdata.com/course/218/) | [Course_218.ipynb](code/Course_218.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_218.ipynb) | chatgpt, revChatGPT, selenium |
| 217 | [Talking to ChatGPT with Your Voice](https://www.largitdata.com/course/217/) | [Course_217.ipynb](code/Course_217.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_217.ipynb) | chatgpt, revChatGPT |

### Financial Data Scraping

Scraping TWSE, TPEx, Goodinfo, Yahoo Finance, TDCC, and other financial data with Python. ([Category page](https://www.largitdata.com/course_list/18))

| # | Lesson | Notebook | Tools |
|---|---|---|---|
| 253 | [Using AI to Solve CAPTCHAs and Scrape TWSE Daily Trading Reports](https://www.largitdata.com/course/253/) | [Course_253.ipynb](code/Course_253.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_253.ipynb) | OpenAI |
| 248 | [Scraping TDCC Shareholding Distribution Tables with Python](https://www.largitdata.com/course/248/) | [Course_248.ipynb](code/Course_248.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_248.ipynb) | Requests, BeautifulSoup, Pandas |
| 244 | [Scraping Goodinfo with Python and Spotting Promising Stocks with GPT-4o](https://www.largitdata.com/course/244/) | [Course_244.ipynb](code/Course_244.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_244.ipynb) | GoodInfo, GPT-4o, ChatGPT |
| 213 | [Scraping Real-Time Taiwan Index Futures Quotes from Yahoo with Python](https://www.largitdata.com/course/213/) | [Course_213.ipynb](code/Course_213.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_213.ipynb) | Requests, Pandas |
| 146 | [Getting Past reCAPTCHA with 2Captcha to Scrape TPEx Broker Branch Trading Data](https://www.largitdata.com/course/146/) | [Course_146.ipynb](code/Course_146.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_146.ipynb) | 2Captcha, reCAPTCHA, Requests |
| 145 | [Scraping the Latest Hong Kong Stock Exchange Trading Data with Python](https://www.largitdata.com/course/145/) | [Course_145.ipynb](code/Course_145.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_145.ipynb) | Requests, Pandas |
| 143 | [Scraping Real-Time Quotes from the New Yahoo Stock Site with Python](https://www.largitdata.com/course/143/) | [Course_143.ipynb](code/Course_143.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_143.ipynb) | Requests, Pandas |
| 134 | [Scraping All Listed Company Stock Codes with Regular Expressions](https://www.largitdata.com/course/134/) | [Course_134.ipynb](code/Course_134.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_134.ipynb) | TEJ |
| 132 | [Scraping Goodinfo Taiwan Stock Data with Python](https://www.largitdata.com/course/132/) | [Course_132.ipynb](code/Course_132.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_132.ipynb) | Goodinfo |
| 129 | [Scraping and Analyzing Gold Prices with Pandas](https://www.largitdata.com/course/129/) | [Course_129.ipynb](code/Course_129.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_129.ipynb) | PythonCrawler, Pandas |

### Web Scraping in Practice

E-commerce price tracking, shopping festival deals, and getting past anti-bot measures (Cloudflare, CAPTCHAs, encrypted APIs). ([Category page](https://www.largitdata.com/course_list/8))

| # | Lesson | Notebook | Tools |
|---|---|---|---|
| 245 | [Getting Past Cloudflare's Anti-Bot Protection](https://www.largitdata.com/course/245/) | [Course_245.ipynb](code/Course_245.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_245.ipynb) | Puppeteer, pyppeteer, pyppeteerstealth, Cloudflare |
| 236 | [Grabbing Double 11 Red Envelopes with PyAutoGUI](https://www.largitdata.com/course/236/) | [Course_236.ipynb](code/Course_236.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_236.ipynb) | AndroidStudio, PyAutoGUI |
| 216 | [Scraping World Cup Betting Odds from Taiwan Sports Lottery with Python](https://www.largitdata.com/course/216/) | [Course_216.ipynb](code/Course_216.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_216.ipynb) | Requests, Pandas |
| 215 | [Skipping Double 11 Shopping? Auto Check-In for Shopee Coins with a Python Scraper](https://www.largitdata.com/course/215/) | [Course_215.ipynb](code/Course_215.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_215.ipynb) | Selenium |
| 214 | [The Pound Plunges! Comparing Global Prices to Find Bargains with a Python Scraper](https://www.largitdata.com/course/214/) | [Course_214.ipynb](code/Course_214.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_214.ipynb) | Requests, BeautifulSoup, Pandas |
| 212 | [Building an Automatic Translation System by Scraping Youdao Translate with Python](https://www.largitdata.com/course/212/) | [Course_212.ipynb](code/Course_212.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_212.ipynb) | Playwright, RPA |
| 151 | [Scraping momo Shop Double 11 Deals with Playwright](https://www.largitdata.com/course/151/) | [Course_151.ipynb](code/Course_151.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_151.ipynb) | nocode, lowcode, Playwright, RPA |
| 150 | [Writing Browser Automation without Code Using Playwright (Low-Code / No-Code)](https://www.largitdata.com/course/150/) | [Course_150.ipynb](code/Course_150.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_150.ipynb) | nocode, lowcode, Playwright, RPA |
| 148 | [Scraping PChome Product Prices with Pyppeteer](https://www.largitdata.com/course/148/) | [Course_148.ipynb](code/Course_148.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_148.ipynb) | Pyppeteer, Puppeteer |
| 144 | [Analyzing NetEase Cloud Music's Personality Color Quiz with Python](https://www.largitdata.com/course/144/) | [Course_144.ipynb](code/Course_144.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_144.ipynb) | Pandas |
| 142 | [Decoding Taiwan Real Estate Price Data Automatically with Python Flask](https://www.largitdata.com/course/142/) | [Course_142.ipynb](code/Course_142.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_142.ipynb) | Flask |
| 141 | [Cracking the Encrypted Strings in Taiwan's Real Estate Price API with Chrome DevTools](https://www.largitdata.com/course/141/) | [Course_141.ipynb](code/Course_141.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_141.ipynb) | Chrome DevTools, Requests |
| 136 | [Scraping Shopee Flash Sale Discounts for the Double 11 Shopping Festival](https://www.largitdata.com/course/136/) | [Course_136.ipynb](code/Course_136.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_136.ipynb) | Selenium |
| 135 | [Scraping and Comparing iPhone 12 Carrier Plans with Pandas](https://www.largitdata.com/course/135/) | [Course_135.ipynb](code/Course_135.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_135.ipynb) | iPhone12 |
| 133 | [Collecting Free Proxy IPs for Python Web Scraping](https://www.largitdata.com/course/133/) | [Course_133.ipynb](code/Course_133.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_133.ipynb) | Proxy, ipify |
| 131 | [Analyzing Employee Salaries at Taiwan Listed Companies with Pandas](https://www.largitdata.com/course/131/) | [Course_131.ipynb](code/Course_131.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_131.ipynb) | Requests, Pandas |
| 122 | [Scraping Weibo Public Opinion on COVID-19](https://www.largitdata.com/course/122/) | [Course_122.ipynb](code/Course_122.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_122.ipynb) | 2019-nCoV, weibo |
| 121 | [Scraping momo Shop Listings for the Double 12 Shopping Festival](https://www.largitdata.com/course/121/) | [Course_121.ipynb](code/Course_121.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_121.ipynb) | momo |
| 120 | [Scraping Taobao Product Listings for the Double 11 Shopping Festival](https://www.largitdata.com/course/120/) | [Course_120.ipynb](code/Course_120.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_120.ipynb) | Requests, Pandas |
| 109 | [Scraping and Organizing Taiwan's 2018 Referendum Results with Python](https://www.largitdata.com/course/109/) | [Course_109.ipynb](code/Course_109.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_109.ipynb) | Requests, Selenium, Pandas |
| 108 | [Finding the Biggest Double 11 Discounts with a Python Scraper (2018 Edition)](https://www.largitdata.com/course/108/) | [Course_108.ipynb](code/Course_108.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_108.ipynb) | Requests, BeautifulSoup, Pandas |
| 100 | [Getting Past TWSE Rate Limits to Scrape the Latest Trading Data](https://www.largitdata.com/course/100/) | [Course_100.ipynb](code/Course_100.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_100.ipynb) | Crawler, rate, limiting |
| 98 | [Scraping Tmall Double 11 Deals Fast](https://www.largitdata.com/course/98/) | [Course_98.ipynb](code/Course_98.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_98.ipynb) | Requests, BeautifulSoup4, Pandas |
| 97 | [Cracking the Taiwan High Speed Rail CAPTCHA (2): Removing Curves with Regression](https://www.largitdata.com/course/97/) | [Course_97.ipynb](code/Course_97.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_97.ipynb) | sklearn, linear, model |
| 96 | [Cracking the Taiwan High Speed Rail CAPTCHA (1): Removing Image Noise](https://www.largitdata.com/course/96/) | [Course_96.ipynb](code/Course_96.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_96.ipynb) | fastNlMeansDenoisingColored |
| 95 | [Capturing CAPTCHA Images with Selenium](https://www.largitdata.com/course/95/) | [Course_95.ipynb](code/Course_95.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_95.ipynb) | Requests, selenium |
| 94 | [Cracking CAPTCHAs with Machine Learning (4): Saving and Loading the Trained Model](https://www.largitdata.com/course/94/) | [Course_94.ipynb](code/Course_94.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_94.ipynb) | pickle |
| 93 | [Cracking CAPTCHAs with Machine Learning (3): Recognizing Digits with a Neural Network](https://www.largitdata.com/course/93/) | [Course_93.ipynb](code/Course_93.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_93.ipynb) | scikit-learn, MLPClassfier |
| 92 | [Cracking CAPTCHAs with Machine Learning (2): Segmenting Each Digit](https://www.largitdata.com/course/92/) | [Course_92.ipynb](code/Course_92.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_92.ipynb) | OpenCV3, findContours |
| 90 | [Finding the Best Time to Buy Bitcoin with Python Pandas](https://www.largitdata.com/course/90/) | [Course_90.ipynb](code/Course_90.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_90.ipynb) | Pandas |
| 89 | [Getting Past Shopee's Restrictions to Scrape Product Listings](https://www.largitdata.com/course/89/) | [Course_89.ipynb](code/Course_89.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_89.ipynb) | XHR, POST |
| 86 | [Exporting slides.com Web Slides to Images with Selenium](https://www.largitdata.com/course/86/) | [Course_86.ipynb](code/Course_86.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_86.ipynb) | Selenium, slides.com, HTML, pdf |
| 85 | [Plotting Recent Japanese Yen Exchange Rate Trends with Pandas](https://www.largitdata.com/course/85/) | [Course_85.ipynb](code/Course_85.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_85.ipynb) | Pandas, read_csv, csv |
| 84 | [Getting the Latest Exchange Rates by Email in Real Time](https://www.largitdata.com/course/84/) | [Course_84.ipynb](code/Course_84.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_84.ipynb) | smtplib, GMAIL |
| 83 | [Scheduling a Job to Store Exchange Rates in a Database Automatically](https://www.largitdata.com/course/83/) | [Course_83.ipynb](code/Course_83.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_83.ipynb) | SQLite, Pandas |
| 82 | [Saving Bank of Taiwan Exchange Rates to a Database with Pandas](https://www.largitdata.com/course/82/) | [Course_82.ipynb](code/Course_82.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_82.ipynb) | Excel, Pandas |
| 81 | [Writing a Python Scraper for Bank of Taiwan Exchange Rates](https://www.largitdata.com/course/81/) | [Course_81.ipynb](code/Course_81.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_81.ipynb) | Pandas |
| 80 | [Scraping Double 11 (1111) Shopping Festival Deals at Top Speed](https://www.largitdata.com/course/80/) | [Course_80.ipynb](code/Course_80.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_80.ipynb) | Pyhton |

### Selenium Tutorials

Selenium from opening a browser and locating elements to automatic login. ([Category page](https://www.largitdata.com/course_list/15))

| # | Lesson | Notebook | Tools |
|---|---|---|---|
| 147 | [Logging In to momo Shop Automatically with Cookies in Selenium](https://www.largitdata.com/course/147/) | [Course_147.ipynb](code/Course_147.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_147.ipynb) | PS5, Cookie, Selenium |
| 137 | [Pre-Ordering a PS5 Automatically with Selenium](https://www.largitdata.com/course/137/) | [Course_137.ipynb](code/Course_137.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_137.ipynb) | PS5, Selenium |
| 107 | [Setting an Implicit Wait in Selenium](https://www.largitdata.com/course/107/) | [Course_107.ipynb](code/Course_107.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_107.ipynb) | Selenium, NoSuchElementException, implicit_wait |
| 106 | [Writing a Web Scraper with Selenium](https://www.largitdata.com/course/106/) | [Course_106.ipynb](code/Course_106.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_106.ipynb) | Selenium, page_source, BeautifulSoup |
| 105 | [Interacting with Web Page Elements in Selenium](https://www.largitdata.com/course/105/) | [Course_105.ipynb](code/Course_105.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_105.ipynb) | Selenium, send_keys |
| 104 | [Locating Elements with Selenium](https://www.largitdata.com/course/104/) | [Course_104.ipynb](code/Course_104.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_104.ipynb) | Selenium, find_element_by_id, find_element_by_class_name, find_element_by_name |
| 103 | [Opening Chrome with Selenium](https://www.largitdata.com/course/103/) | [Course_103.ipynb](code/Course_103.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_103.ipynb) | Selenium |

### RPA (Robotic Process Automation)

Automating workflows with PyAutoGUI, Selenium, and LINE Notify. ([Category page](https://www.largitdata.com/course_list/17))

| # | Lesson | Notebook | Tools |
|---|---|---|---|
| 119 | [Sending the Latest Comic Chapter through LINE](https://www.largitdata.com/course/119/) | [Course_119.ipynb](code/Course_119.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_119.ipynb) | SQLite, LineNotify, Selenium, RPA |
| 118 | [Getting Instant Notifications with LINE Notify](https://www.largitdata.com/course/118/) | [Course_118.ipynb](code/Course_118.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_118.ipynb) | LineNotify, RPA |
| 117 | [Merging Images into a PDF with img2pdf](https://www.largitdata.com/course/117/) | [Course_117.ipynb](code/Course_117.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_117.ipynb) | img2pdf, RPA |
| 116 | [Downloading Comics Automatically with Selenium (1)](https://www.largitdata.com/course/116/) | [Course_116.ipynb](code/Course_116.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_116.ipynb) | Selenium |
| 115 | [Getting Past reCAPTCHA with PyAutoGUI to Download TPEx Broker Trading Reports](https://www.largitdata.com/course/115/) | [Course_115.ipynb](code/Course_115.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_115.ipynb) | PyAutoGUI, reCAPTCHA |
| 114 | [Building a Python Keyboard and Mouse Macro Tool with PyAutoGUI](https://www.largitdata.com/course/114/) | [Course_114.ipynb](code/Course_114.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_114.ipynb) | RPA, PyAutoGUI |

### Deep Learning

Computer vision projects: CNN face recognition, YOLO mask detection, and DeepFakes. ([Category page](https://www.largitdata.com/course_list/16))

| # | Lesson | Notebook | Tools |
|---|---|---|---|
| 130 | [Installing and Using YOLOv4 on Google Colab](https://www.largitdata.com/course/130/) | [Course_130.ipynb](code/Course_130.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_130.ipynb) | DeepLearning, GoogleColab, YOLOv4 |
| 128 | [Building a Real-Time Face Mask Detector with YOLO (3): Real-Time Detection](https://www.largitdata.com/course/128/) | [Course_128.ipynb](code/Course_128.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_128.ipynb) | DeepLearning, YOLO, COVID19 |
| 127 | [Building a Real-Time Face Mask Detector with YOLO (2): Training the Mask Detection Model](https://www.largitdata.com/course/127/) | [Course_127.ipynb](code/Course_127.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_127.ipynb) | DeepLearning, YOLO, COVID19 |
| 126 | [Building a Real-Time Face Mask Detector with YOLO (1): Introduction to YOLO](https://www.largitdata.com/course/126/) | [Course_126.ipynb](code/Course_126.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_126.ipynb) | DeepLearning, YOLO, COVID19 |
| 125 | [Swapping Faces in Videos with DeepFakes (3)](https://www.largitdata.com/course/125/) | [Course_125.ipynb](code/Course_125.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_125.ipynb) | DeepFakes, DeepFaceLab, DeepLearning |
| 112 | [Building a Deep Learning Model to Tell Three Taiwanese Actors Apart (3)](https://www.largitdata.com/course/112/) | [Course_112.ipynb](code/Course_112.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_112.ipynb) | OpenCV |
| 111 | [Building a Deep Learning Model to Tell Three Taiwanese Actors Apart (2)](https://www.largitdata.com/course/111/) | [Course_111.ipynb](code/Course_111.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_111.ipynb) | OpenCV, PIL |
| 110 | [Building a Deep Learning Model to Tell Three Taiwanese Actors Apart (1)](https://www.largitdata.com/course/110/) | [Course_110.ipynb](code/Course_110.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_110.ipynb) | Requests, BeautifulSoup, PIL |

### Algorithmic Trading

Historical Bitcoin prices, TA-Lib indicators, and strategy backtesting with Backtesting.py. ([Category page](https://www.largitdata.com/course_list/19))

| # | Lesson | Notebook | Tools |
|---|---|---|---|
| 140 | [Backtesting Trading Strategies with Backtesting.py](https://www.largitdata.com/course/140/) | [Course_140.ipynb](code/Course_140.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_140.ipynb) | BTC, Backtesting |
| 139 | [Building Bitcoin Technical Indicators with TA-Lib](https://www.largitdata.com/course/139/) | [Course_139.ipynb](code/Course_139.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_139.ipynb) | BTC, TALib |
| 138 | [Getting Historical Bitcoin Prices through an API](https://www.largitdata.com/course/138/) | [Course_138.ipynb](code/Course_138.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_138.ipynb) | BTC |

### Open Jarvis

Speech recognition, speech synthesis, and chatbots. ([Category page](https://www.largitdata.com/course_list/14))

| # | Lesson | Notebook | Tools |
|---|---|---|---|
| 102 | [Building a Real-Time Translator in Python](https://www.largitdata.com/course/102/) | [Course_102.ipynb](code/Course_102.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_102.ipynb) | py-googletrans, Google |
| 101 | [Letting a Chatbot Answer Expert Questions with Wikipedia](https://www.largitdata.com/course/101/) | [Course_101.ipynb](code/Course_101.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_101.ipynb) | Wikipedia |
| 99 | [Building a Real Voice Chatbot in Under 30 Lines of Python](https://www.largitdata.com/course/99/) | [Course_99.ipynb](code/Course_99.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_99.ipynb) | SpeechRecognition, gTTS |
| 88 | [Making Your Computer Talk with Python](https://www.largitdata.com/course/88/) | [Course_88.ipynb](code/Course_88.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_88.ipynb) | gTTS, pygame |
| 87 | [Transcribing Speech to Text Automatically with Python](https://www.largitdata.com/course/87/) | [Course_87.ipynb](code/Course_87.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_87.ipynb) | SpeechRecognition |

### Other Topics

Small data science projects: Wordle, machine learning in Excel, and box office analysis. ([Category page](https://www.largitdata.com/course_list/12))

| # | Lesson | Notebook | Tools |
|---|---|---|---|
| 235 | [Making Your Own Memes with ROOP Face Swap](https://www.largitdata.com/course/235/) | [Course_235.ipynb](code/Course_235.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_235.ipynb) | meme, roop |
| 234 | [Machine Learning in Excel with Python](https://www.largitdata.com/course/234/) | [Course_234.ipynb](code/Course_234.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_234.ipynb) | Excel |
| 152 | [Using Data Science to Find the Best First Word in Wordle](https://www.largitdata.com/course/152/) | [Course_152.ipynb](code/Course_152.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_152.ipynb) | wordle, nltk, pandas |
| 113 | [Scraping Box Office Data for Avengers: Endgame](https://www.largitdata.com/course/113/) | [Course_113.ipynb](code/Course_113.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_113.ipynb) | Requests, Pandas |

### Other examples

| Notebook | Description |
|---|---|
| [Course_232.ipynb](code/Course_232.ipynb) | Scraping chart data from MacroMicro |
| [20230311_Whisper_and_ChatGPT.ipynb](code/20230311_Whisper_and_ChatGPT.ipynb) | Speech-to-text with Whisper and summarization with ChatGPT |

## FAQ

**Are the LargitData Academy lessons free?**
Yes. Every video is free to watch on the [LargitData Academy site](https://www.largitdata.com/courses/) and the [YouTube channel](https://www.youtube.com/@Largitdata), and all example code is public in this repository.

**How do I find the code for a lesson?**
By lesson number: the code for `https://www.largitdata.com/course/N/` is `code/Course_N.ipynb`.

**What background do I need?**
Basic Python syntax is enough. Start with Selenium Tutorials or Web Scraping in Practice, then move on to financial data scraping, deep learning, and AI.

**What if an older example no longer works?**
Website redesigns and package updates often break older scrapers. Look for a newer lesson on the same topic (financial scraping and CAPTCHA solving each have several versions), or adjust the CSS selectors and package versions based on the error message.

## Repository layout

```
code/      Lesson code (Course_N.ipynb matches https://www.largitdata.com/course/N/)
data/      Sample data used in lessons
config/    YOLO face mask detection config
archive/   Code from past talks and workshops (archive/speeches/)
```

### Talks and workshops

| Date | Topic | Code |
|---|---|---|
| 2017-04-06 | Data science in Python: Google Trends and stock prices | [20170406Speech](archive/speeches/20170406Speech/) |
| 2017-07-21 | Scraping 591 rental listings and visualizing with Tableau | [20170721Speech](archive/speeches/20170721Speech/) |
| 2017-07-27 | Rental listing data analysis | [20170727Speech](archive/speeches/20170727Speech/) |
| 2017-08-10 | Text mining news articles | [20170810Speech](archive/speeches/20170810Speech/) |
| 2017-08-22 | Housing price scraping and credit risk machine learning | [20170822Speech](archive/speeches/20170822Speech/) |
| 2018-03-12 | Big data in finance: practice and applications (slides) | [20180312Speech](archive/speeches/20180312Speech/) |
| 2018-09-22 | Data cleaning with Pandas | [104class](archive/speeches/104class/) |
| 2019-03-26 | Facebook scraping | [20190326](archive/speeches/20190326/) |
| 2019-09-17 | WeChat official account scraping | [20190917Speech](archive/speeches/20190917Speech/) |
| 2020-03-04 | Deep Learning 101 and DeepFaceLab | [20200304Speech](archive/speeches/20200304Speech/) |

## LargitData products

- [InfoMiner social listening platform](https://www.largitdata.com/infominer/)
- [InfoLite web data extraction Chrome extension](https://chromewebstore.google.com/detail/infolite/ipjbadabbpedegielkhgpiekdlmfpgal)
