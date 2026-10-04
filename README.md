**繁體中文** | [English](README.en.md)

# 大數學堂 LargitData｜Python 網路爬蟲、AI 人工智慧與 ChatGPT 免費教學範例程式碼

**大數學堂**是[大數軟體 LargitData](https://www.largitdata.com/) 經營的免費線上課程平台，以「一支影片、一份範例程式」的方式，教你用 Python 完成網路爬蟲、資料分析、深度學習與生成式 AI 應用。本 repo 收錄[大數學堂課程頁面](https://www.largitdata.com/courses/)上所有影片的範例程式碼（Jupyter Notebook），每份程式都對應一堂可以免費觀看的課程，可直接下載或在 Google Colab 開啟執行。

- **課程網站**：<https://www.largitdata.com/courses/>
- **YouTube 頻道**：<https://www.youtube.com/@Largitdata>
- **Facebook 粉絲頁**：<https://www.facebook.com/largitdata/>
- **課程數量**：本 repo 收錄 100 多堂課程的範例程式，網站持續更新中
- **語言與格式**：繁體中文教學，Python 3，Jupyter Notebook（`.ipynb`）

## 如何使用範例程式

1. 在[課程網站](https://www.largitdata.com/courses/)找到想學的主題，觀看影片。
2. 依課程編號 `N` 開啟對應的範例程式 `code/Course_N.ipynb`，或點下方表格的 **Colab** 連結直接在瀏覽器執行。
3. 想在本機執行：

   ```bash
   git clone https://github.com/ywchiu/largitdata.git
   cd largitdata
   pip install jupyter
   jupyter notebook code/
   ```

   各課程需要的套件（如 `requests`、`beautifulsoup4`、`pandas`、`selenium`、`openai`）請依 notebook 開頭的 `import` 自行安裝。需要 API 金鑰的課程，請改成自己的金鑰，並避免提交到版本控制。

## 課程目錄

課程依[大數學堂網站](https://www.largitdata.com/courses/)的分類整理，每個分類內由新到舊排列。點課程名稱看教學影片，點範例程式看原始碼。

- [Vibe Coding](#vibe-coding)
- [AI 人工智慧](#ai-人工智慧)
- [ChatGPT](#chatgpt)
- [財經爬蟲](#財經爬蟲)
- [網路爬蟲實戰](#網路爬蟲實戰)
- [Selenium 爬蟲教程](#selenium-爬蟲教程)
- [RPA 流程機器人](#rpa-流程機器人)
- [深度學習](#深度學習)
- [程式交易](#程式交易)
- [Open Jarvis](#open-jarvis)
- [其他專題](#其他專題)

### Vibe Coding

用 Claude Code、Codex 等 AI 程式助理開發軟體的實戰流程。（[網站分類](https://www.largitdata.com/course_list/23)）

| # | 課程 | 範例程式 | 關鍵技術 |
|---|---|---|---|
| 259 | [如何透過混用模型將 Fable 5 的效益發揮最大?!](https://www.largitdata.com/course/259/) | [Course_259.md](code/Course_259.md) | Claude、Fable5、DeepSeek、OpenRouter |

### AI 人工智慧

Gemini、DeepSeek、Ollama、Whisper、AI Agent、Computer Use 等生成式 AI 應用實作。（[網站分類](https://www.largitdata.com/course_list/22)）

| # | 課程 | 範例程式 | 關鍵技術 |
|---|---|---|---|
| 254 | [如何用 AI 打敗reCAPTCHA驗證碼？！](https://www.largitdata.com/course/254/) | [Course_254.ipynb](code/Course_254.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_254.ipynb) | 網路爬蟲、ReCAPTCHA、Gemini、Selenium |
| 251 | [DeepSeek 部署全攻略 –從 1.5B 蒸餾模型到 671B 滿血模型](https://www.largitdata.com/course/251/) | [Course_251.ipynb](code/Course_251.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_251.ipynb) | LLM、Ollama、DeepSeek、vLLM |
| 250 | [AI Agent 實戰教學：新手也能輕鬆打造股票AI分析師！](https://www.largitdata.com/course/250/) | [Course_250.ipynb](code/Course_250.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_250.ipynb) | AI教學、股票分析、AIAgent、Python程式 |
| 249 | [只要100多行程式碼？！ Gemini 2 Flash 顛覆你對即時翻譯的想像](https://www.largitdata.com/course/249/) | [Course_249.py](code/Course_249.py) | 多模態模型、AI教學、Python程式設計、Gemini2Flash |
| 247 | [AI直接操控我的電腦？！Computer Use功能實測大揭密](https://www.largitdata.com/course/247/) | [Course_247.ipynb](code/Course_247.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_247.ipynb) | Anthropic、ComputerUse、AIAssistant |
| 246 | [如何用OpenAI API 快速搭建一個類似 NotebookLM 的 Podcast 功能 ?](https://www.largitdata.com/course/246/) | [Course_246.ipynb](code/Course_246.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_246.ipynb) | ChatGPT、GPT4o-mini、NotebookLM、OpenAI |
| 243 | [多模態AI應用實戰:輕鬆用Gemini 與 ElevenLabs 實現即時語音翻譯與合成](https://www.largitdata.com/course/243/) | [Course_243.ipynb](code/Course_243.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_243.ipynb) | Gemini、ElevenLabs、即時翻譯、語音複製 |
| 241 | [運用 Whisper 輕鬆打造即時字幕轉錄神器！😎](https://www.largitdata.com/course/241/) | [Course_241.ipynb](code/Course_241.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_241.ipynb) | whisper、語音轉文字、即時字幕轉錄 |
| 240 | [使用 Ollama 調用本地語言模型生成文章並且辨識圖片內容](https://www.largitdata.com/course/240/) | [Course_240.ipynb](code/Course_240.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_240.ipynb) | 大型語言模型、Ollama、llm、breeze7b |
| 238 | [探索香港Deepfake詐騙案背後的科技：如何只憑免費Colab與基本Python知識製作深度偽造影片?](https://www.largitdata.com/course/238/) | [Course_238.ipynb](code/Course_238.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_238.ipynb) | DeepFake、Roop、深度偽造、兩億港幣 |
| 233 | [EasyOCR v.s. PaddleOCR 誰才是圖片轉文字(OCR)的最佳神器?!](https://www.largitdata.com/course/233/) | [Course_233.ipynb](code/Course_233.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_233.ipynb) | OCR、EasyOCR、PaddleOCR、AIMochi |
| 230 | [你也能成為編曲大師！探索如何運用 AudioCraft 以文字創造音樂](https://www.largitdata.com/course/230/) | [Course_230.ipynb](code/Course_230.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_230.ipynb) | meta、audiocraft、musicgen、audiogen |

### ChatGPT

ChatGPT / OpenAI API、Llama 2 微調、RAG、PDF 翻譯與語音對話。（[網站分類](https://www.largitdata.com/course_list/21)）

| # | 課程 | 範例程式 | 關鍵技術 |
|---|---|---|---|
| 242 | [使用Llama Parse和 ChatGPT 翻譯 Google Drive 上的PDF文件](https://www.largitdata.com/course/242/) | [Course_242.ipynb](code/Course_242.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_242.ipynb) | PDF論文翻譯、ChatGPT、LlamaParse、LlamaIndex |
| 237 | [如何結合Python網路爬蟲和GPTs打造你自己的財經新聞聚合應用程式！](https://www.largitdata.com/course/237/) | [Course_237.ipynb](code/Course_237.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_237.ipynb) | python網路爬蟲、財經新聞、工商時報、moneydj |
| 231 | [運用微調之力！如何將 ChatGPT 訓練成公司的客服助理](https://www.largitdata.com/course/231/) | [Course_231.ipynb](code/Course_231.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_231.ipynb) | 大型語言模型、微調技術、chatgpt、finetuning |
| 229 | [個人化Llama2 ！如何在Colab中運用自己的資料集微調 Llama2 模型](https://www.largitdata.com/course/229/) | [Course_229.ipynb](code/Course_229.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_229.ipynb) | 大型語言模型、Llama2、FineTune、監督式微調 |
| 228 | [如何利用Meta開源的Llama2模型，打造屬於自己的ChatGPT](https://www.largitdata.com/course/228/) | [Course_228.ipynb](code/Course_228.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_228.ipynb) | chatgpt、llama2、huggingface、transformers |
| 226 | [利用 ChatGPT 打造萬用網路爬蟲追蹤最新機票價格](https://www.largitdata.com/course/226/) | [Course_226.ipynb](code/Course_226.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_226.ipynb) | python網路爬蟲、chatgpt、langchain、selenium |
| 225 | [網路爬蟲 X MidJourney X ChatGPT 自動化產生吸睛新聞封面 (2/2)](https://www.largitdata.com/course/225/) | [Course_225.ipynb](code/Course_225.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_225.ipynb) | chatgpt、discord、midjourney、自動化 |
| 224 | [網路爬蟲 X MidJourney X ChatGPT 自動化產生吸睛新聞封面 (1/2)](https://www.largitdata.com/course/224/) | [Course_224.ipynb](code/Course_224.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_224.ipynb) | Python網路爬蟲、MidJourney、ChatGPT、AIPRM |
| 223 | [用ChatGPT輕鬆掌握外資對台積電法說會的看法](https://www.largitdata.com/course/223/) | [Course_223.ipynb](code/Course_223.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_223.ipynb) | chatgpt、llama_index、langchain、ChatPDF |
| 222 | [如何使用ChatGPT 快速翻譯 PDF 文件?](https://www.largitdata.com/course/222/) | [Course_222.ipynb](code/Course_222.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_222.ipynb) | chatgpt、翻譯、pdf文件、AI語言模型 |
| 221 | [Whisper還是剪映？選擇最佳字幕創建工具讓你的影片更專業！](https://www.largitdata.com/course/221/) | [Course_221.ipynb](code/Course_221.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_221.ipynb) | chatgpt、whisper、pydub、ytdlp |
| 220 | [如何使用Whisper API 與 ChatGPT API 快速摘要YouTube 影片?](https://www.largitdata.com/course/220/) | [Course_220.ipynb](code/Course_220.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_220.ipynb) | chatgpt、whisper、pydub、yt-dlp |
| 219 | [完全不露臉？ 沒問題！ AI人工智慧讓你輕鬆當上 YouTuber](https://www.largitdata.com/course/219/) | [Course_219.ipynb](code/Course_219.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_219.ipynb) | ChatGPT、Midjourney、AI 影片 |
| 218 | [如何使用Python 網路爬蟲強化ChatGPT 的問答能力?](https://www.largitdata.com/course/218/) | [Course_218.ipynb](code/Course_218.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_218.ipynb) | chatgpt、revChatGPT、python網路爬蟲、網路爬蟲 |
| 217 | [用說的也會通！如何用語音與ChatGPT 對話](https://www.largitdata.com/course/217/) | [Course_217.ipynb](code/Course_217.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_217.ipynb) | chatgpt、revChatGPT、語音識別、語音合成 |

### 財經爬蟲

用 Python 抓取證交所、櫃買中心、Goodinfo、Yahoo 股市、集保等財經資料。（[網站分類](https://www.largitdata.com/course_list/18)）

| # | 課程 | 範例程式 | 關鍵技術 |
|---|---|---|---|
| 253 | [運用 AI 之力突破驗證碼：解鎖證交所買賣日報表網路爬蟲技術](https://www.largitdata.com/course/253/) | [Course_253.ipynb](code/Course_253.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_253.ipynb) | Python爬蟲、AI教學、驗證碼破解、大型語言模型 |
| 248 | [使用 Python 網路爬蟲輕鬆爬取集保戶股權分散表](https://www.largitdata.com/course/248/) | [Course_248.ipynb](code/Course_248.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_248.ipynb) | Python爬蟲、網路爬蟲教學、集保資訊、股權分散表 |
| 244 | [手把手帶你用Python網路爬蟲抓取Goodinfo，再結合GPT-4o快速分析潛力股!](https://www.largitdata.com/course/244/) | [Course_244.ipynb](code/Course_244.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_244.ipynb) | Python網路爬蟲、GoodInfo、GPT-4o、財報分析 |
| 213 | [如何使用Python 網路爬蟲抓取Yahoo 台指期的即時行情?](https://www.largitdata.com/course/213/) | [Course_213.ipynb](code/Course_213.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_213.ipynb) | python網路爬蟲、財經爬蟲、即時行情、交易機器人 |
| 146 | [怎麼繞過驗證碼? 利用 2Captcha 驗證碼識別服務突破  reCAPTCHA 驗證碼，抓取證券櫃買中心的券商買賣證券日報表上分點交易資訊](https://www.largitdata.com/course/146/) | [Course_146.ipynb](code/Course_146.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_146.ipynb) | 驗證碼識別服務、怎麼繞過驗證碼、驗證碼怎麼識別、Python網路爬蟲 |
| 145 | [如何透過Python 網路爬蟲爬取香港交易所最新成交資訊?](https://www.largitdata.com/course/145/) | [Course_145.ipynb](code/Course_145.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_145.ipynb) | Python網路爬蟲、交易機器人、香港交易所 |
| 143 | [如何使用Python 網路爬蟲抓取新版Yahoo 股市上的即時行情?](https://www.largitdata.com/course/143/) | [Course_143.ipynb](code/Course_143.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_143.ipynb) | Python網路爬蟲、財經爬蟲、即時行情、交易機器人 |
| 134 | [如何使用正規表達法快速抓取所有上市公司代號?](https://www.largitdata.com/course/134/) | [Course_134.ipynb](code/Course_134.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_134.ipynb) | Python網路爬蟲、正規表達法、TEJ |
| 132 | [如何透過Python 網路爬蟲抓取Goodinfo 台灣股市資訊網?](https://www.largitdata.com/course/132/) | [Course_132.ipynb](code/Course_132.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_132.ipynb) | Goodinfo、Python網路爬蟲、財經爬蟲 |
| 129 | [如何透過Pandas 快速抓取並分析黃金價格?](https://www.largitdata.com/course/129/) | [Course_129.ipynb](code/Course_129.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_129.ipynb) | Python網路爬蟲、PythonCrawler、黃金價格、Pandas |

### 網路爬蟲實戰

電商比價、購物節特價、反爬蟲（Cloudflare、驗證碼、加密字串）破解實戰。（[網站分類](https://www.largitdata.com/course_list/8)）

| # | 課程 | 範例程式 | 關鍵技術 |
|---|---|---|---|
| 245 | [如何破解Cloudflare 的反爬蟲機制](https://www.largitdata.com/course/245/) | [Course_245.ipynb](code/Course_245.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_245.ipynb) | Python網路爬蟲、Puppeteer、pyppeteer、pyppeteerstealth |
| 236 | [如何使用 PyAutoGUI 搶雙 11 百萬紅包](https://www.largitdata.com/course/236/) | [Course_236.ipynb](code/Course_236.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_236.ipynb) | Python網路爬蟲、AndroidStudio、雙11紅包、PyAutoGUI |
| 216 | [如何用Python網路爬蟲抓取台灣運彩上的世界杯足球賠率?](https://www.largitdata.com/course/216/) | [Course_216.ipynb](code/Course_216.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_216.ipynb) | python網路爬蟲、台灣運彩、世界杯足球、世足賠率 |
| 215 | [1111 不購物?! 來用Python網路爬蟲每天簽到領蝦幣](https://www.largitdata.com/course/215/) | [Course_215.ipynb](code/Course_215.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_215.ipynb) | 1111購物狂歡節、雙11、Python網路爬蟲、Selenium |
| 214 | [英鎊暴跌! 如何利用Python 網路爬蟲進行全球商品比價、撿便宜](https://www.largitdata.com/course/214/) | [Course_214.ipynb](code/Course_214.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_214.ipynb) | python網路爬蟲、比價爬蟲、英鎊暴跌、貨幣競貶 |
| 212 | [如何利用Python網路爬蟲爬取有道翻譯打造自動化翻譯系統](https://www.largitdata.com/course/212/) | [Course_212.ipynb](code/Course_212.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_212.ipynb) | 自動翻譯軟體、有道翻譯、Python網路爬蟲、Playwright |
| 151 | [如何使用工具 Playwright爬取 MOMO 購物網 1111 特價資訊](https://www.largitdata.com/course/151/) | [Course_151.ipynb](code/Course_151.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_151.ipynb) | 1111購物狂歡節、雙11、nocode、lowcode |
| 150 | [如何不寫任何一行程式碼透過低代碼Low-Code / No-Code 工具 Playwright撰寫網頁自動化瀏覽程式](https://www.largitdata.com/course/150/) | [Course_150.ipynb](code/Course_150.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_150.ipynb) | nocode、lowcode、Python網路爬蟲、Playwright |
| 148 | [如何使用 Pyppeteer抓取 PCHOME 商品價格資訊?](https://www.largitdata.com/course/148/) | [Course_148.ipynb](code/Course_148.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_148.ipynb) | PCHOME爬蟲、Pyppeteer、Puppeteer、Python網路爬蟲 |
| 144 | [如何利用Python快速分析網易雲性格主導色心理測驗?](https://www.largitdata.com/course/144/) | [Course_144.ipynb](code/Course_144.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_144.ipynb) | Python網路爬蟲、網易雲、性格主導色、心理測驗 |
| 142 | [如何利用Python Flask自動轉換實價登錄網站加密字串?](https://www.largitdata.com/course/142/) | [Course_142.ipynb](code/Course_142.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_142.ipynb) | Python網路爬蟲、實價登錄資訊、Flask |
| 141 | [如何透過開發人員工具破解實價登錄網新版API中的加密字串?](https://www.largitdata.com/course/141/) | [Course_141.ipynb](code/Course_141.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_141.ipynb) | Python網路爬蟲、實價登錄資訊、Chrome開發人員工具 |
| 136 | [如何在1111購物狂歡節快速爬取蝦皮限時特賣的商品折扣資訊?](https://www.largitdata.com/course/136/) | [Course_136.ipynb](code/Course_136.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_136.ipynb) | 1111購物狂歡節、蝦皮API、蝦皮特賣商品折扣、Selenium |
| 135 | [如何使用Pandas 快速抓取並分析iPhone 12 購機方案?](https://www.largitdata.com/course/135/) | [Course_135.ipynb](code/Course_135.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_135.ipynb) | Python網路爬蟲、5G購機方案、iPhone12 |
| 133 | [如何快速蒐集免費IP作為Python 網路爬蟲跳板Proxy?](https://www.largitdata.com/course/133/) | [Course_133.ipynb](code/Course_133.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_133.ipynb) | Python網路爬蟲、Proxy、ipify、跳板 |
| 131 | [如何使用Pandas快速分析上市櫃公司員工的薪資水平?](https://www.largitdata.com/course/131/) | [Course_131.ipynb](code/Course_131.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_131.ipynb) | 網路爬蟲、上市櫃公司薪資水平、平均值與中位數 |
| 122 | [如何撰寫網路爬蟲快速爬取微博上所有關於新冠肺炎的輿情?](https://www.largitdata.com/course/122/) | [Course_122.ipynb](code/Course_122.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_122.ipynb) | Python網路爬蟲、武漢肺炎、2019-nCoV、微博 |
| 121 | [如何在1212購物狂歡節快速爬取momo購物網上的商品資訊?](https://www.largitdata.com/course/121/) | [Course_121.ipynb](code/Course_121.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_121.ipynb) | Python網路爬蟲、1212購物狂歡節、momo |
| 120 | [如何在1111購物狂歡節 快速爬取淘寶上的商品資訊?](https://www.largitdata.com/course/120/) | [Course_120.ipynb](code/Course_120.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_120.ipynb) | Python網路爬蟲、1111購物狂歡節、淘寶、不信你可以下來看看 |
| 109 | [如何透過 Python 網路爬蟲 抓取並整理 2018 公投選舉資料?](https://www.largitdata.com/course/109/) | [Course_109.ipynb](code/Course_109.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_109.ipynb) | 選舉、公投、中選會、投票統計資料 |
| 108 | [如何透過 Python 網路爬蟲快速找出1111購物狂歡節折扣最多的商品? (2018年版)](https://www.largitdata.com/course/108/) | [Course_108.ipynb](code/Course_108.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_108.ipynb) | 購物狂歡節、精打細算、購買清單、數據做決策 |
| 100 | [如何突破證交所的限制，穩穩抓取最新成交資訊?](https://www.largitdata.com/course/100/) | [Course_100.ipynb](code/Course_100.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_100.ipynb) | 證交所、爬蟲、Crawler、網頁伺服器 |
| 98 | [如何快速爬取天貓TMALL 雙11 特價商品資訊?](https://www.largitdata.com/course/98/) | [Course_98.ipynb](code/Course_98.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_98.ipynb) | 雙11、購物狂歡、天貓TMALL、網路爬蟲 |
| 97 | [如何破解高鐵驗證碼 (2) - 使用迴歸方法去除多餘弧線?](https://www.largitdata.com/course/97/) | [Course_97.ipynb](code/Course_97.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_97.ipynb) | 去除弧線、高鐵驗證碼、二項式迴歸公式、sklearn |
| 96 | [如何破解高鐵驗證碼 (1) - 去除圖片噪音點?](https://www.largitdata.com/course/96/) | [Course_96.ipynb](code/Course_96.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_96.ipynb) | 高鐵、驗證碼、破解、噪音點 |
| 95 | [如何使用Selenium 抓取驗證碼?](https://www.largitdata.com/course/95/) | [Course_95.ipynb](code/Course_95.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_95.ipynb) | Requests、擷取驗證碼圖片、selenium、存下頁面快照 |
| 94 | [如何使用機器學習方法破解驗證碼 (4) ? – 如何存取訓練模型](https://www.largitdata.com/course/94/) | [Course_94.ipynb](code/Course_94.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_94.ipynb) | 訓練模型、pickle、檔、系統 |
| 93 | [如何使用機器學習方法破解驗證碼 (3) ? – 使用類神經網路自動辨認驗證碼](https://www.largitdata.com/course/93/) | [Course_93.ipynb](code/Course_93.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_93.ipynb) | 驗證碼、切成、數字、scikit-learn |
| 92 | [如何使用機器學習方法破解驗證碼 (2) ? – 切割出驗證碼中的各個數字](https://www.largitdata.com/course/92/) | [Course_92.ipynb](code/Course_92.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_92.ipynb) | OpenCV3、爬蟲、經濟部、公司基本資料 |
| 90 | [如何使用Python Pandas 分析比特幣最佳買點?](https://www.largitdata.com/course/90/) | [Course_90.ipynb](code/Course_90.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_90.ipynb) | 比特幣、以太幣、虛擬貨幣、投資浪潮 |
| 89 | [如何突破蝦皮拍賣的重重限制以順利抓取拍賣商品資訊?](https://www.largitdata.com/course/89/) | [Course_89.ipynb](code/Course_89.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_89.ipynb) | 蝦皮拍賣、爬蟲實戰、抓取方法、XHR |
| 86 | [如何使用Selenium 自動將slides.com 的網頁投影片輸出成圖檔?](https://www.largitdata.com/course/86/) | [Course_86.ipynb](code/Course_86.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_86.ipynb) | 爬蟲、網路爬蟲、Selenium、slides.com |
| 85 | [如何使用Pandas 快速繪製日幣近期的匯率走勢?](https://www.largitdata.com/course/85/) | [Course_85.ipynb](code/Course_85.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_85.ipynb) | 資料分析、圖表、Pandas、處理 |
| 84 | [如何透過EMAIL即時獲取最新匯率資訊?](https://www.largitdata.com/course/84/) | [Course_84.ipynb](code/Course_84.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_84.ipynb) | 匯率資訊、自動排程、爬蟲、EMAIL通知 |
| 83 | [如何設定工作排程自動將牌告匯率存進資料庫之中?](https://www.largitdata.com/course/83/) | [Course_83.ipynb](code/Course_83.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_83.ipynb) | 爬蟲、定期執行、爬取工作、自動化 |
| 82 | [如何使用Pandas 函式將台灣銀行的牌告匯率存進資料庫中?](https://www.largitdata.com/course/82/) | [Course_82.ipynb](code/Course_82.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_82.ipynb) | 牌告匯率、Excel、管理、新增 |
| 81 | [如何撰寫Python爬蟲抓取台灣銀行的牌告匯率?](https://www.largitdata.com/course/81/) | [Course_81.ipynb](code/Course_81.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_81.ipynb) | 買進最低價位的日圓、爬蟲、Pandas、台灣銀行 |
| 80 | [如何極速擷取1111購物狂歡節的特價商品資訊?](https://www.largitdata.com/course/80/) | [Course_80.ipynb](code/Course_80.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_80.ipynb) | 購物狂歡、網路爬蟲、Pyhton、Python網路爬蟲 |

### Selenium 爬蟲教程

從開啟瀏覽器、元素定位到自動登入的 Selenium 完整教學。（[網站分類](https://www.largitdata.com/course_list/15)）

| # | 課程 | 範例程式 | 關鍵技術 |
|---|---|---|---|
| 147 | [如何利用Cookie 資訊 自動登入 momo 購物網的使用者帳戶中?](https://www.largitdata.com/course/147/) | [Course_147.ipynb](code/Course_147.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_147.ipynb) | Python購物小幫手、PS5、PS5預購、Cookie |
| 137 | [如何使用 Selenium  自動預購PS5?](https://www.largitdata.com/course/137/) | [Course_137.ipynb](code/Course_137.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_137.ipynb) | Python購物小幫手、PS5、PS5預購、Selenium |
| 107 | [如何設定 Selenium 中的隱含等待(Implicit Wait)?](https://www.largitdata.com/course/107/) | [Course_107.ipynb](code/Course_107.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_107.ipynb) | Selenium、抓取資料、implicit_wait、頁面載入 |
| 106 | [如何使用 Selenium 撰寫網路爬蟲?](https://www.largitdata.com/course/106/) | [Course_106.ipynb](code/Course_106.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_106.ipynb) | Selenium、自動化流程、爬取頁面內容、page_source |
| 105 | [如何使用 Selenium 操作網頁元素?](https://www.largitdata.com/course/105/) | [Course_105.ipynb](code/Course_105.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_105.ipynb) | Selenium、網頁元素、點擊、按鈕 |
| 104 | [如何使用 Selenium 查找元素定位?](https://www.largitdata.com/course/104/) | [Course_104.ipynb](code/Course_104.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_104.ipynb) | Selenium、瀏覽器、元素定位、操作 |
| 103 | [如何使用 Selenium 開啟 Chrome 瀏覽器?](https://www.largitdata.com/course/103/) | [Course_103.ipynb](code/Course_103.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_103.ipynb) | Selenium、基礎教程、擬人化的操作、爬蟲開發者 |

### RPA 流程機器人

用 PyAutoGUI、Selenium、Line Notify 打造自動化流程。（[網站分類](https://www.largitdata.com/course_list/17)）

| # | 課程 | 範例程式 | 關鍵技術 |
|---|---|---|---|
| 119 | [如何透過 Line 發送最新一集的漫畫?](https://www.largitdata.com/course/119/) | [Course_119.ipynb](code/Course_119.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_119.ipynb) | SQLite、LineNotify、Selenium、RPA |
| 118 | [如何使用 Line Notify 取得第一手通知?](https://www.largitdata.com/course/118/) | [Course_118.ipynb](code/Course_118.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_118.ipynb) | LineNotify、RPA、Python自動化 |
| 117 | [如何使用 img2pdf  將圖檔合併成 pdf 檔 ?](https://www.largitdata.com/course/117/) | [Course_117.ipynb](code/Course_117.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_117.ipynb) | img2pdf、RPA、Python自動化 |
| 116 | [如何使用 Selenium  自動下載漫畫 (1)?](https://www.largitdata.com/course/116/) | [Course_116.ipynb](code/Course_116.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_116.ipynb) | Selenium、Python爬蟲 |
| 115 | [如何使用 PyAutoGUI 突破 reCAPTCHA 順利下載櫃買中心券商買賣證券日報表?](https://www.largitdata.com/course/115/) | [Course_115.ipynb](code/Course_115.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_115.ipynb) | PyAutoGUI、reCAPTCHA、券商買賣證券日報表 |
| 114 | [如何用PyAutoGUI 建立Python 版的按鍵精靈?](https://www.largitdata.com/course/114/) | [Course_114.ipynb](code/Course_114.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_114.ipynb) | RPA、流程機器人、自動化程序、PyAutoGUI |

### 深度學習

CNN 人臉辨識、YOLO 口罩檢測、DeepFakes 等電腦視覺專案。（[網站分類](https://www.largitdata.com/course_list/16)）

| # | 課程 | 範例程式 | 關鍵技術 |
|---|---|---|---|
| 130 | [如何在Google Colab上安裝與使用 YOLOv4 ?](https://www.largitdata.com/course/130/) | [Course_130.ipynb](code/Course_130.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_130.ipynb) | DeepLearning、GoogleColab、YOLOv4 |
| 128 | [如何使用 YOLO 製作即時口罩檢測系統(三) – 建立即時口罩檢測系統](https://www.largitdata.com/course/128/) | [Course_128.ipynb](code/Course_128.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_128.ipynb) | DeepLearning、YOLO、COVID19、新冠肺炎 |
| 127 | [如何使用 YOLO 製作即時口罩檢測系統(二) – 建立口罩檢測模型?](https://www.largitdata.com/course/127/) | [Course_127.ipynb](code/Course_127.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_127.ipynb) | DeepLearning、YOLO、COVID19、新冠肺炎 |
| 126 | [如何使用 YOLO 製作即時口罩檢測系統(一) - YOLO簡介?](https://www.largitdata.com/course/126/) | [Course_126.ipynb](code/Course_126.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_126.ipynb) | DeepLearning、YOLO、COVID19、新冠肺炎 |
| 125 | [如何使用 DeepFakes 技術移花接木影片人物的臉(三)?](https://www.largitdata.com/course/125/) | [Course_125.ipynb](code/Course_125.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_125.ipynb) | DeepFakes、DeepFaceLab、DeepLearning、深度偽造 |
| 112 | [如何建構深度學習模型分辨誰是屈中恆、宋少卿、鈕承澤 (3)?](https://www.largitdata.com/course/112/) | [Course_112.ipynb](code/Course_112.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_112.ipynb) | 鈕承澤、卷積神經網路、OpenCV、Python網路爬蟲 |
| 111 | [如何建構深度學習模型分辨誰是屈中恆、宋少卿、鈕承澤 (2)?](https://www.largitdata.com/course/111/) | [Course_111.ipynb](code/Course_111.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_111.ipynb) | 鈕承澤、卷積神經網路、Python網路爬蟲、深度學習 |
| 110 | [如何建構深度學習模型分辨誰是屈中恆、宋少卿、鈕承澤 (1)?](https://www.largitdata.com/course/110/) | [Course_110.ipynb](code/Course_110.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_110.ipynb) | 鈕承澤、屈中恆、宋少卿、驗證碼 |

### 程式交易

比特幣歷史報價、TA-Lib 技術指標與 Backtesting.py 策略回測。（[網站分類](https://www.largitdata.com/course_list/19)）

| # | 課程 | 範例程式 | 關鍵技術 |
|---|---|---|---|
| 140 | [如何使用 Backtesting.py回測交易策略?](https://www.largitdata.com/course/140/) | [Course_140.ipynb](code/Course_140.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_140.ipynb) | 程式交易、比特幣、BTC、Backtesting |
| 139 | [如何使用TA-Lib快速建立比特幣技術分析指標?](https://www.largitdata.com/course/139/) | [Course_139.ipynb](code/Course_139.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_139.ipynb) | 程式交易、比特幣、BTC、TALib |
| 138 | [如何透過API獲取比特幣歷史報價數據?](https://www.largitdata.com/course/138/) | [Course_138.ipynb](code/Course_138.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_138.ipynb) | 程式交易、比特幣、BTC、API串接 |

### Open Jarvis

語音辨識、語音合成與對話機器人。（[網站分類](https://www.largitdata.com/course_list/14)）

| # | 課程 | 範例程式 | 關鍵技術 |
|---|---|---|---|
| 102 | [如何使用Python寫一個翻譯蒟蒻?](https://www.largitdata.com/course/102/) | [Course_102.ipynb](code/Course_102.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_102.ipynb) | 小叮噹、翻譯蒟蒻、py-googletrans、Google |
| 101 | [如何讓對話機器人利用 Wikipedia 回答專業知識?](https://www.largitdata.com/course/101/) | [Course_101.ipynb](code/Course_101.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_101.ipynb) | Wikipedia、對話機器人、網路爬蟲 |
| 99 | [如何用不到30行Python程式碼寫出「真‧對話機器人」?](https://www.largitdata.com/course/99/) | [Course_99.ipynb](code/Course_99.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_99.ipynb) | 對話機器人、語音辨識、gTTS |
| 88 | [如何用Python 讓電腦說話?](https://www.largitdata.com/course/88/) | [Course_88.ipynb](code/Course_88.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_88.ipynb) | gTTS、pygame、語音合成 |
| 87 | [如何讓Python 自動將語音轉譯成文字?](https://www.largitdata.com/course/87/) | [Course_87.ipynb](code/Course_87.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_87.ipynb) | Open、Jarvis、Project、電腦自動 |

### 其他專題

資料科學小專題：Wordle、Excel + Python 機器學習、票房分析。（[網站分類](https://www.largitdata.com/course_list/12)）

| # | 課程 | 範例程式 | 關鍵技術 |
|---|---|---|---|
| 235 | [ROOP 換臉中文教學：製作自己的迷因圖](https://www.largitdata.com/course/235/) | [Course_235.ipynb](code/Course_235.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_235.ipynb) | 迷因圖、meme、黑人問號、roop |
| 234 | [完美結合！  Excel 中也可以用 Python 做機器學習？](https://www.largitdata.com/course/234/) | [Course_234.ipynb](code/Course_234.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_234.ipynb) | Excel、機器學習、資料分析、資料科學 |
| 152 | [運用數據科學分析Wordle 該從哪個字開始猜？](https://www.largitdata.com/course/152/) | [Course_152.ipynb](code/Course_152.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_152.ipynb) | wordle、nltk、pandas、資料科學 |
| 113 | [如何抓取電影 「復仇者聯盟4-終局之戰」的票房數據?](https://www.largitdata.com/course/113/) | [Course_113.ipynb](code/Course_113.ipynb) · [Colab](https://colab.research.google.com/github/ywchiu/largitdata/blob/master/code/Course_113.ipynb) | 票房數據、復仇者聯盟4、網路爬蟲 |

### 其他範例程式

| 範例程式 | 說明 |
|---|---|
| [Course_232.ipynb](code/Course_232.ipynb) | 抓取財經 M 平方（MacroMicro）圖表資料 |
| [20230311_Whisper_and_ChatGPT.ipynb](code/20230311_Whisper_and_ChatGPT.ipynb) | Whisper 語音轉文字搭配 ChatGPT 摘要示範 |

## 常見問題

**大數學堂的課程要收費嗎？**
不用。所有影片都可以在[大數學堂網站](https://www.largitdata.com/courses/)和 [YouTube 頻道](https://www.youtube.com/@Largitdata)免費觀看，範例程式也都公開在這個 repo。

**範例程式和課程影片怎麼對應？**
用課程編號對應：網站上 `https://www.largitdata.com/course/N/` 的範例程式就是 `code/Course_N.ipynb`。

**適合什麼程度的人學習？**
具備 Python 基礎語法即可。建議從「Selenium 爬蟲教程」或「網路爬蟲實戰」入門，再進階到財經爬蟲、深度學習與 AI 應用。

**舊範例跑不起來怎麼辦？**
網站改版或套件更新常會讓舊爬蟲失效。可以參考同主題的較新課程（例如財經爬蟲、驗證碼破解都有多個版本），或依錯誤訊息調整 CSS 選擇器與套件版本。

## 目錄結構

```
code/      課程範例程式（Course_N.ipynb 對應 https://www.largitdata.com/course/N/）
data/      課程使用的範例資料
config/    YOLO 口罩檢測模型設定檔
archive/   歷年演講與工作坊的範例程式（archive/speeches/）
```

### 演講與工作坊範例

| 日期 | 主題 | 範例 |
|---|---|---|
| 2017-04-06 | 用 Python 做資料科學：Google Trends 與股價分析 | [20170406Speech](archive/speeches/20170406Speech/) |
| 2017-07-21 | 591 租屋網爬蟲與 Tableau 視覺化 | [20170721Speech](archive/speeches/20170721Speech/) |
| 2017-07-27 | 租屋資料分析 | [20170727Speech](archive/speeches/20170727Speech/) |
| 2017-08-10 | 新聞文字探勘 | [20170810Speech](archive/speeches/20170810Speech/) |
| 2017-08-22 | 房價爬蟲與信用風險機器學習 | [20170822Speech](archive/speeches/20170822Speech/) |
| 2018-03-12 | 金融大數據實務與應用（投影片） | [20180312Speech](archive/speeches/20180312Speech/) |
| 2018-09-22 | Pandas 資料清理 | [104class](archive/speeches/104class/) |
| 2019-03-26 | Facebook 爬蟲 | [20190326](archive/speeches/20190326/) |
| 2019-09-17 | 微信公眾號爬蟲 | [20190917Speech](archive/speeches/20190917Speech/) |
| 2020-03-04 | Deep Learning 101 與 DeepFaceLab | [20200304Speech](archive/speeches/20200304Speech/) |

## 大數軟體相關產品

- [InfoMiner 輿情分析平台](https://www.largitdata.com/infominer/)
- [InfoLite 網頁資料擷取 Chrome 擴充功能](https://chromewebstore.google.com/detail/infolite/ipjbadabbpedegielkhgpiekdlmfpgal)

## 其他線上課程

- [人人都爱数据科学家！Python 数据科学精华实战课程](https://edu.hellobi.com/course/159)
- [手把手教你用 Python 实践深度学习](https://edu.hellobi.com/course/278)
