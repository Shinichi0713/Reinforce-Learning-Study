
近年頻繁な自分で考えるエージェンティックトレンドについて。
"これっていつからのトレンドなん？"と疑問に感じたので調べてみました。
まだ、このエージェンティックに乗っかって、どんな技術開発されているのかも気になっているので合わせてサーベイです。

![1784369093928](image/16_agentic_movement/1784369093928.png)

## そもそもエージェントとは

結論から言うと、**「エージェンティックAIの動作」について、唯一の厳密な公式定義があるわけではありませんが、複数のベンダー・リサーチ機関がほぼ共通した特徴で定義しており、それらをまとめると「自律性・目標志向・行動」の3要素が中核になっています。**


### 1. 代表的な定義の例

いくつかの代表的な定義を挙げると、以下のようになります。

- **AWS**  
  「エージェンティックAIは、事前に設定された目標を達成するために独立して動作できる自律型AIシステム。従来のAIはプロンプトやステップバイステップの指示を必要とするが、エージェンティックAIはプロアクティブに動き、人間の継続的な監視なしに複雑なタスクを実行できる。」[AWS](https://aws.amazon.com/jp/what-is/agentic-ai)

- **HPE**  
  「エージェンティックAIは、人間の介入なしに自律的に動作し、意思決定を行ってタスクを実行できるAIシステムのクラスを指す。」[HPE](https://www.hpe.com/jp/ja/what-is/agentic-ai.html)

- **IBM**  
  「エージェンティックAIは、限られた監督のもとで特定の目標を達成できるAIシステム。AIエージェント（人間の意思決定を模倣するMLモデル）から構成され、マルチエージェントシステムでは各エージェントがサブタスクを担当し、AIオーケストレーションで協調する。」[IBM](https://www.ibm.com/think/topics/agentic-ai)

- **BAP Software**  
  「エージェント型AI（エージェントベースAI／自律型AIエージェント）は、目標を自律的に実行できるAIシステム。自律性・目標志向・行動の3つを中核とし、Sense → Model → Plan → Act → Learn のループを自ら回す。」[BAP Software](https://bap-software.net/knowledge/what-is-agentic-ai)

これらを総合すると、「エージェンティックAIの動作」はおおむね次のように定義されています。


### 2. 共通する「動作」の定義・特徴

多くの解説では、エージェンティックAIの動作は以下の3点で特徴づけられています。

__(1) 自律性（Autonomy）__
- 人間の細かい指示を待たずに、**自ら**環境を観察し、状況をモデル化し、計画を立て、行動し、結果から学習するループ（Sense → Model → Plan → Act → Learn）を回す。[BAP Software](https://bap-software.net/knowledge/what-is-agentic-ai)
- 従来のAIが「入力待ちの反応型」であるのに対し、エージェンティックAIは「目標主導の積極型」と説明されることが多いです。[AWS](https://aws.amazon.com/jp/what-is/agentic-ai)

__(2) 目標志向（Goal-driven）__
- 事前に定義された**明確な目標**（例：在庫削減、顧客対応の完了、スケジュール最適化など）を受け取り、その達成に向けて行動を適応させる。[BAP Software](https://bap-software.net/knowledge/what-is-agentic-ai)
- 単に「テキストを生成する」のではなく、「目標を達成するために何をすべきか」を自分で判断する点が重要です。[HPE](https://www.hpe.com/jp/ja/what-is/agentic-ai.html)

__(3) 行動（Action）__
- API呼び出し、メール送信、システム設定変更、ワークフローの起動など、**外部のデジタル環境に直接影響を与える行動**を複数ステップで実行できる。[BAP Software](https://bap-software.net/knowledge/what-is-agentic-ai)
- 単なる「回答生成」ではなく、「実際にタスクを完遂する」ことが求められます。[UiPath](https://www.uipath.com/ja/ai/agentic-ai)


## エージェントトレンド

エージェンティックAI（Agentic AI）やエージェンティックLLMの流れは、**2023年春ごろから**本格的に注目され始め、**2024〜2025年にかけて**「次の大きなトレンド」として語られるようになった、というのがおおまかなタイムラインです。


### 1. 背景：AIエージェントという概念自体はかなり前から存在

- 「エージェント（agent）」という概念は、AI・システム理論の分野で古くから使われてきました。  
- これは「環境を観測し、意思決定し、行動するソフトウェア」という意味で、自律的なシステムの基本単位として研究されてきました。[BAP Software](https://bap-software.net/knowledge/what-is-agentic-ai)

この段階では「エージェント」は主に学術・研究の文脈で使われており、一般向けのトレンドというよりは基礎概念でした。


### 2. LLMベースの「自律エージェント」が一気に注目された時期：2023年春ごろ

- ChatGPT（2022年11月公開）やGPT-4（2023年3月公開）の登場で、LLMの能力が大きく向上しました。
- これを受けて、**2023年春ごろ**に以下のようなオープンソースプロジェクトが相次いで登場し、「LLMをループさせて自律的にタスクをこなすエージェント」というアイデアが一気に広まりました。
  - **AutoGPT**：GPT-4を自律的に動かし、目標達成のために自分で計画・実行する実験的アプリ
  - **BabyAGI**：タスク管理ループ（実行→新タスク生成→優先順位付け）をLLMで回す仕組み
  - **AgentGPT**：Web上で動く自律AIエージェント
- これらのプロジェクトは「Agentic AI」「autonomous LLM agents」という言葉とともに、**2023年春〜夏にかけて**コミュニティで大きな話題になりました。[Medium](https://medium.com/@roseserene/agentic-ai-autogpt-babyagi-and-autonomous-llm-agents-substance-or-hype-8fa5a14ee265)

この時期が、「LLMを単なるチャットボットではなく、**目標達成のために自律的に行動するエージェント**として使う」という流れが明確になった起点と言えます。

### 3. 「エージェンティックAI」というトレンド語としての本格化：2024〜2025年

- 2024年後半〜2025年にかけて、大手企業やリサーチ会社が「エージェンティックAI」「AIエージェント」を**次の大きな波**として位置づけ始めました。
  - **Microsoft**：Ignite 2024で「エージェンティックワールド」のビジョンを表明
  - **Google**：Gemini 2.0を「エージェント時代に向けた次世代モデル」と位置づけ
  - **NVIDIA**：CEOジェンスン・フアン氏が「2025年はAIエージェントの年」と発言
  - **OpenAI**：2025年1月にブラウザ操作エージェント「Operator」を発表[ONEDER](https://oneder.hakuhodody-one.co.jp/blog/ai-agent-2025-issues)
- 同時に、**Gartner**や**Salesforce**なども「AIエージェント／エージェンティックAI」を2025年の重要トレンドとして取り上げています。[Salesforce](https://www.salesforce.com/jp/news/stories/ai-agents-trends-2025)
- 日本語圏でも、2024〜2025年に「エージェンティックAIが次の大きなトレンドかもしれない」といった解説記事が増え、2025年を「AIエージェント元年」と位置づける声が多数出てきました。[note](https://note.com/shimap_sampo/n/n52e5ae518b31)[ONEDER](https://oneder.hakuhodody-one.co.jp/blog/ai-agent-2025-issues)


## 技術トレンド

### 1. 現在の主なトレンド

__(1) マルチエージェントシステムの本格化__

さほどニュースではありませんが。

- **Salesforce**は2025年のAIエージェント動向レポートで、**マルチエージェント・オーケストレーション**を中核トレンドとして位置づけています。

> 「AIエージェントは、特定のタスクを自律的に遂行するよう設計された、先見性のあるアプリケーションです。AIエージェントは大規模言語モデル（LLM）を活用し、リクエストやプロンプト、または自動トリガーの背景情報を分析・理解し、次のステップを自律的に判断して実行します。」  
> 「Agentforceは、Salesforceプラットフォームに新たに追加されたレイヤーで、企業が自律型AIエージェントを構築・展開することを可能にします。」  
> — Salesforce「AIエージェントの未来：2025年の注目予測とトレンド」[Salesforce](https://www.salesforce.com/jp/news/stories/ai-agents-trends-2025)

- **Gartner**も、2026年の戦略的テクノロジートレンドとして**マルチエージェント・システム**を選定しており、単一エージェントから「専門化した複数エージェントの協調」へ移行していると指摘しています。[Gartner](https://www.gartner.com/en/newsroom/press-releases/2025-08-26-gartner-predicts-40-percent-of-enterprise-apps-will-feature-task-specific-ai-agents-by-2026-up-from-less-than-5-percent-in-2025)

__(2) 小規模言語モデル（SLM）の台頭__

- 2025年9月時点の技術解説では、**小規模言語モデル（SLM）** がエージェンティックAIシステムでより効率的・経済的であることが示されています。

> 「最新の研究では、**小規模言語モデル（SLM）がエージェンティックAIシステムにおいてより効率的で経済的**であることが実証されています。特化タスクの反復実行において、大規模モデルと同等の性能を発揮しながらコストを大幅に削減できることが明らかになりました。」  
> — 「エージェンティックAI最前線：2025年9月の重要動向と企業への影響」[Zenn](https://zenn.dev/ino_h/articles/agentic-ai-trends-2025-09)

__(3) エンタープライズ領域への本格浸透__

- **ITCross**の2025年トレンド解説では、エージェンティックAIの「エンタープライズ領域への本格浸透」が主要トレンドとして挙げられています。[ITCross](https://www.itcross.jp/media/266)
- **Market.us**のレポートでは、2024年時点で**Ready-To-Deploy Agents（すぐに使えるエージェント）** が市場の58.5%を占め、**Multi Agent（マルチエージェント）** が66.4%を占めていると報告されています。[Market.us](https://market.us/report/agentic-ai-market)

### 2. 主要企業・プラットフォームの動向

__Microsoft：オープンエージェンティックウェブ__

- **Microsoft Build 2025**では、「**オープンエージェンティックウェブ**」のビジョンが発表され、Azure AI Foundry Agent Serviceの一般提供開始など、エージェント基盤の整備が進んでいます。[Microsoft Build 2025](https://blogs.microsoft.com/blog/2025/05/19/microsoft-build-2025-the-age-of-ai-agents-and-building-the-open-agentic-web)

__Google：Gemini 2.0 と Project Mariner__

- Googleは**Gemini 2.0 Flash**および**Gemini 2.0 Flash Thinking Mode**をリリースし、エージェンティックAI分野での競争力を強化しています。[Zenn](https://zenn.dev/ino_h/articles/agentic-ai-trends-2025-09)
- 特に**Project Mariner**は、オンライン買い物やウェブブラウジングを代行するエージェンティック機能として注目されています。

__OpenAI：Operator の発表__

- OpenAIは2025年1月にAIエージェント「**Operator**」をリリース予定と発表し、「2025年はエージェンティックシステムが主流になる年」という予測を体現するものとされています。[Zenn](https://zenn.dev/ino_h/articles/agentic-ai-trends-2025-09)

__Salesforce：Agentforce 3 とエージェント統合__

- Salesforceは**Agentforce 3**を発表し、AIエージェントを企業プラットフォームに統合する基盤を強化しています。

> 「Agentforce 3は、AIエージェント、ビジネスプロセスの自動化、Salesforce Cloud全体でのインテリジェントなコラボレーションを統合する、完全なアーキテクチャ上の飛躍です。」  
> — 「Agentforce 3: How Salesforce Is Revolutionizing AI Agent Integration in 2025」[Generative.ai](https://www.getgenerative.ai/agentforce-3-how-salesforce-revolutionizing-ai-agents-integration)

### 3. 市場規模・成長見通し

合わせて調べたという内容ですが。
立派な成長市場です。
今は仮想環境でのエージェントですが、しばらくすると物理環境に対するブームも本格化するでしょう。

__市場規模の急拡大__

- **Market.us**のレポートによると、エージェンティックAI市場は2024年に**52億ドル**、2034年には**約1,966億ドル**に達し、**CAGR 43.8%**で成長すると予測されています。[Market.us](https://market.us/report/agentic-ai-market)
- **Precedence Research**も、2025年の市場規模を**75.5億ドル**、2034年には**1,990.5億ドル**（CAGR 43.84%）と見込んでいます。[Precedence Research](https://www.precedenceresearch.com/agentic-ai-market)

__エンタープライズアプリへの組み込み__

- **Gartner**は、2026年までに**40%のエンタープライズアプリケーションがタスク特化型AIエージェントを組み込む**と予測しています。[Gartner](https://www.gartner.com/en/newsroom/press-releases/2025-08-26-gartner-predicts-40-percent-of-enterprise-apps-will-feature-task-specific-ai-agents-by-2026-up-from-less-than-5-percent-in-2025)

### 4. 今後の展望と課題

__(1) 自律エージェントの「実装元年」としての2026年__

- 新規事業・社内起業の事例集である**IntraStar Wiki**は、2026年をエージェンティックAIの「**実装元年**」と位置づけています。

> 「『AIエージェント元年』と呼ばれた2025年を経て、2026年は『実装元年』と位置付けられる。PoC（概念実証）から本格運用への移行が加速しており、先行する企業と停滞する企業の差が、新規事業の成果に直結し始めている。」  
> — 「エージェンティックAI×新規事業の最前線——2026年の企業イノベーション最前線」[IntraStar Wiki](https://intrastar.wiki/articles/agentic-ai-corporate-innovation-2026)

__(2) 人材・組織への影響__

- **Mercer**のレポートでは、エージェンティックAI時代の人材マネジメントについて、AIが定型業務を担う一方で、**創造的思考・リーダーシップ・レジリエンス**といった人間固有のスキルの価値が高まると指摘しています。[Mercer](https://www.mercer.com/ja-jp/insights/people-strategy/hr-transformation/hr-management-in-the-era-of-agentic-ai)

__(3) 課題：成功率・コスト・ガバナンス__

- **Gartner**は、**40%以上のエージェンティックAIプロジェクトが2027年までに中止される**と予測し、ビジネス価値の不明確さやコスト増大を理由に挙げています。[Gravity.global](https://www.gravity.global/en/blog/salesforce-and-gartner-cast-doubt-on-ai-agents)
- **Salesforce**のCRMベンチマークでは、複雑なマルチターン対話におけるAIエージェントの成功率が**35%**にとどまると報告されており、現状の限界も明らかになっています。[Gravity.global](https://www.gravity.global/en/blog/salesforce-and-gartner-cast-doubt-on-ai-agents)

## 妄想

ここまでの情報を踏まえ、「エージェンティックLLM」という技術トレンドを**客観視かつ抽象化して総括**し、そのうえで**今後どうすべきか**を激しく妄想してみます。

### 1. 客観視：何が起きているのか

__(1) 技術の進化段階としての位置づけ__

- これまでのAIは、主に「**入力に応じて出力を返す反応型**」でした（生成AIも含め、プロンプト依存）。
- エージェンティックAIは、**目標を与えられたら、自ら計画・実行・適応する自律型システム**へと進化した段階です。[AWS](https://aws.amazon.com/jp/what-is/agentic-ai)
- つまり、**「AIが人間の作業を“代行”する」**フェーズから、**「AIが人間と協調して業務を“推進”する」**フェーズへ移行しつつあります。

__(2) 市場・社会へのインパクト__

- 市場規模は2024年52億ドルから2034年約1,966億ドルへ、**約40倍**に拡大すると予測されています。[Market.us](https://market.us/report/agentic-ai-market)
- Gartnerは、2026年までに**40%のエンタープライズアプリがタスク特化型AIエージェントを組み込む**と予測し、[Gartner](https://www.gartner.com/en/newsroom/press-releases/2025-08-26-gartner-predicts-40-percent-of-enterprise-apps-will-feature-task-specific-ai-agents-by-2026-up-from-less-than-5-percent-in-2025)  
  企業システムの「標準装備」になりつつあります。
- 一方で、Gartnerは**40%以上のエージェンティックAIプロジェクトが2027年までに中止される**とも予測しており、[Gravity.global](https://www.gravity.global/en/blog/salesforce-and-gartner-cast-doubt-on-ai-agents)  
  **「実用化の壁」と「過剰期待」の両方が顕在化**しています。

__(3) 技術的な抽象化__

- エージェンティックLLMは、**「LLMを“頭脳”とする自律エージェント」** と捉えられます。
- その中核は、  
  - **自律性**（Sense → Model → Plan → Act → Learn のループ）  
  - **目標志向**（KPI・ゴールに基づく行動）  
  - **行動**（API・ツール・外部システムへの作用）  
  の3要素です。[BAP Software](https://bap-software.net/knowledge/what-is-agentic-ai)
- つまり、**「LLM＋ツール連携＋ループ制御」** というアーキテクチャが、**“仕事を進める主体”として機能し始めている**、という抽象化ができます。

### 2. 抽象化：このトレンドの本質は何か

__(1) 「人間の拡張」から「人間とAIの協働」へ__

- 生成AIは「**人間の補助**」としての位置づけが強かったのに対し、  
  エージェンティックAIは「**人間と並ぶ“デジタルワーカー”**」としての性格が強いです。[UiPath](https://www.uipath.com/ja/ai/agentic-ai)
- Salesforceは、AIエージェントを「**調整された労働力（Coordinated Workforce）** 」の一部と位置づけ、[IntraStar Wiki](https://intrastar.wiki/articles/agentic-ai-corporate-innovation-2026)  
  **人間とAIエージェントが役割分担して協働するチーム構造**が次のステージとしています。

__(2) 「単一モデル」から「マルチエージェント・オーケストレーション」へ__

- 2025年以降のトレンドは、**単一の巨大モデル**ではなく、  
  **専門化した複数エージェントの協調（マルチエージェント・システム）** が主流になりつつあります。[ITCross](https://www.itcross.jp/media/266)
- Microsoftの「オープンエージェンティックウェブ」やSalesforceのAgentforce 3は、**エージェント間の連携・オーケストレーション基盤**を提供しています。[Microsoft Build 2025](https://blogs.microsoft.com/blog/2025/05/19/microsoft-build-2025-the-age-of-ai-agents-and-building-the-open-agentic-web)[Generative.ai](https://www.getgenerative.ai/agentforce-3-how-salesforce-revolutionizing-ai-agents-integration)

__(3) 「汎用モデル」から「SLM＋RAG＋ツール」の組み合わせへ__

- Zennの解説では、**小規模言語モデル（SLM）** がエージェンティックAIシステムで効率的・経済的であることが示されています。[Zenn](https://zenn.dev/ino_h/articles/agentic-ai-trends-2025-09)
- つまり、**「巨大モデルで何でもやる」から、「SLM＋RAG＋外部ツール連携」でタスク特化のエージェントを構成する**方向にシフトしています。

### 3. 今後どうすべきか：実務的な示唆

__(1) 企業・組織としてのスタンス__

1. **「AIを“使う”」から「AIと“働く”」への意識転換**  
   - エージェンティックAIは、単なるツールではなく、**チームメンバー**として扱う必要があります。  
   - Mercerは、AIが定型業務を担う中で、**創造的思考・リーダーシップ・レジリエンス**といった人間固有のスキルの価値が高まると指摘しています。[Mercer](https://www.mercer.com/ja-jp/insights/people-strategy/hr-transformation/hr-management-in-the-era-of-agentic-ai)

2. **PoC地獄を避け、「実装元年」に向けた本格運用へ**  
   - IntraStar Wikiは2026年を「実装元年」と位置づけ、PoCから本格運用への移行が加速していると述べています。[IntraStar Wiki](https://intrastar.wiki/articles/agentic-ai-corporate-innovation-2026)  
   - Gartnerが指摘するように、**ビジネス価値が不明確なPoCは中止されるリスクが高い**ため、  
     **「どの業務を誰と分担するか」を明確にしたうえで導入**する必要があります。

__(2) 技術・アーキテクチャの方向性__

1. **マルチエージェント・オーケストレーションを前提に設計**  
   - 単一エージェントで何でもやろうとせず、  
     **「調査エージェント」「分析エージェント」「実行エージェント」など、役割分担した複数エージェントをオーケストレーターで統合**する設計が現実的です。[ITCross](https://www.itcross.jp/media/266)

2. **SLM＋RAG＋ツール連携でコストと性能を最適化**  
   - 巨大モデルに依存しすぎず、**SLM＋RAG＋外部API連携**でタスク特化のエージェントを構築する方が、  
     コスト・レイテンシ・セキュリティの観点で有利になるケースが増えています。[Zenn](https://zenn.dev/ino_h/articles/agentic-ai-trends-2025-09)

3. **ガバナンス・監査・説明可能性の確保**  
   - Deloitteは、エージェントの行動を追跡するための**Model Context Protocol（MCP）**や**ゼロトラスト認証**の重要性を指摘しています。[Deloitte](https://www.deloitte.com/us/en/insights/topics/technology-management/tech-trends/2026/agentic-ai-strategy.html)  
   - 「何をやったか」「なぜそう判断したか」を**人間が監査できる仕組み**が必須です。

__(3) 個人としてのキャリア・スキル__

1. **「AIに指示する側」ではなく「AIと協働する側」のスキル**  
   - プロンプトエンジニアリングだけでなく、**エージェント設計・オーケストレーション・評価指標（KPI）設計**といったスキルが重要になります。
2. **AIには代替しにくい領域の強化**  
   - MercerやWEFのレポートが示すように、**創造的思考・倫理的判断・変化への心理的耐性**など、  
     人間固有のスキルの価値が高まっています。[Mercer](https://www.mercer.com/ja-jp/insights/people-strategy/hr-transformation/hr-management-in-the-era-of-agentic-ai)

### 4. まとめ：今後どうすべきか

- **技術的には**、  
  - 「LLM＋ツール連携＋ループ制御」という**エージェントアーキテクチャ**を前提に、  
  - **マルチエージェント・オーケストレーション**と**SLM＋RAG＋ツール連携**で、  
    コスト・性能・安全性をバランスさせる設計が主流になります。
- **組織的には**、  
  - AIエージェントを「**チームメンバー**」として位置づけ、  
  - **人間とAIの役割分担・協働プロセス**を再設計することが不可欠です。
- **個人的には**、  
  - AIに指示するだけでなく、**AIと協働して成果を出すスキル**と、  
  - AIには代替しにくい**創造性・倫理観・レジリエンス**を磨くことが、長期的な価値につながります。

このトレンドは、単なる「AIの進化」ではなく、**仕事の進め方・組織のあり方・人間の役割そのものの再定義**を迫る大きな転換点だと捉えるのが、最も客観的かつ本質的な見方だと言えるでしょう。

![1784369069391](image/16_agentic_movement/1784369069391.png)