# aiDAPTIV+ Client & OpenWebUI 安裝與使用手冊

## 📋 目錄

- [Chap.0 安裝包檔案介紹](#chap0-檔案介紹)
- [Chap.1 首次安裝指南](#chap1-首次安裝指南)
- [Chap.2 啟動 OpenWebUi](#chap2-啟動OpenWebUi)
- [Chap.3 Demo 執行流程](#chap3-demo-執行流程)
- [Chap.4 補充操作](#chap4-補充操作)
- [Chap.5 卸載流程](#chap5-卸載流程)
- [Chap.6 附錄](#chap6-附錄)
- [版本修改記錄](#版本修改記錄)

---

## Chap.0 檔案介紹

### 📦 安裝包內容

安裝包包含以下檔案與目錄：

- **`AgentBuilderClientInstaller_0.1.1.exe`**：主程式安裝檔
- **`/package/embedding_model`**：Embedding Model，產生 vector db 時所使用，**目前不開放更換**
- **`/package/inference_modell`**： LLM Model，預設為 Meta-Llama-3.1-8B-Instruct-Q4_K_M
- **`/maestro/aiDAPTIV_vNXWI_3_01J00`**：Phison masetro 相關執行檔案
- **`/maestro/kv_cache`**：KV cache bin（適用於 Meta-Llama-3.1-8B-Instruct-Q4_K_M 模型）
- **`/maestro/openwebui`**：安裝 openwebui 時所需要的相關套件

---

## Chap.1 首次安裝指南 
<mark>Note: 若已安裝過可跳過</mark>

### 🔧 Middleware 環境設定

#### 前置需求

在安裝主程式前，請先完成以下環境設定：

1. **參考文件**：請詳閱 `aiDAPTIV_vNXWI_3_01J00.pdf` 文件
2. **執行章節**：
   - Prerequisite 環境設定
   - MSVC Redistributable upgrade
3. **SSD 分割區設定**：將 AI SSD 的分割區設為 `R:\`

### 💻 Agentbuilder Client 安裝

#### 步驟 1：安裝 Agentbuilder Client Service

Agentbuilder Client Service 將會把三個 windows service 進行安裝，並啟動

1. 雙擊執行 `AgentBuilderClientInstaller_0.1.1.exe`
2. 依照安裝精靈指示，安裝至使用者自定的位置後選擇**Next**
   ![chap1_installation_kit_page1](./images/chap1_installation_kit_page1.png)
3. 進行初始設定，設定完後選擇**Next**
   - NATS server IP: 是設定 AgentClient Server 的 IP
   - Select inference model : 設定初始啟動 masetro 的 LLM Model
   - Select maestro package : 設定 masetro 的版本
   - Prompt language: 設定 KM prompt language
     ![chap1_installation_kit_setup_page](./images/chap1_installation_kit_setup_page.png)
4. 選擇 AI SSD 的分割區安裝 aiDAPTIV 並進行**Next**。
   <mark>(必須要選擇 AI SSD 的分割區)</mark>
   ![chap1_installation_kit_aidaptiv_cache](./images/chap1_installation_kit_aidaptiv_cache.png)
   ![chap1_installation_kit_installing](./images/chap1_installation_kit_installing.png)
5. 因為會設定到環境變數，建議重新開機

#### 步驟 2：確認 Agentbuilder Client Service 安裝完成

安裝完成並重新啟動後，您將看到：

- Task Manager, Services 中會出現
  - ada_service
  - AgentBuilderClient
  - KMClient
  - MaestroMcpServer
  - llamacpp
  ![chap1_agnetbuilder_services](./images/chap1_agnetbuilder_services.png)
  
- 使用桌面的 Tail AgentBuilder Logs.bat，可以看到目前 service 的執行狀況
  (AgentbuilderClient, KMClient, MaestroMcpServer, llamacpp logs)

  ![chap1_agnetbuilder_losgs](./images/chap1_agnetbuilder_losgs.png)
  ![chap1_agnetbuilder_losgs_content](./images/chap1_agnetbuilder_losgs_content.png)

---

## Chap.2 啟動 OpenWebUI和確認Client services

#### 🚀 啟動 OpenWebUi

1. 點選桌面上的 `Phison_aiDAPTIV_OpenWebUI` 執行檔捷徑

   ![chap2_start_openwebui](./images/chap2_start_openwebui.png)

2. 系統將自動開啟兩個命令提示字元視窗：
   1. Backend Process
   2. Frontend Process
      ![chap2_start_openwebui_BF](./images/chap2_start_openwebui_BF.png)

### 🚀 啟動 LlamaCpp

1. 重啟電腦後，必須要先等待 Agentbuilder Server 進行 kv cache publish，並等待 LlamaCpp 啟動完成
   ![chap2_start_llamacpp_check_aidaptiv](./images/chap2_start_llamacpp_check_aidaptiv.png)
   ![chap2_start_llamacpp_check_aidaptiv_2](./images/chap2_start_llamacpp_check_aidaptiv_2.png)
   <mark>(每次 server publish 後皆會觸發 LlamaCpp service Restart)</mark>

### ✅ 確認服務狀態

請依序確認以下項目：

1. 確認所有服務皆正常運行中
2. 開啟 Chrome 瀏覽器
3. 在網址列輸入 `http://localhost:5173/`
4. 成功載入 OpenWebUI 主畫面
   ![chap2_openWebUI_home_page](./images/chap2_openWebUI_home_page.png)

> Note: 第一次登入使用帳號密碼
> 賬號:`phison@phison.com`, 密碼:`phison`

## Chap.3 Demo 執行流程

### 📚 Part 1：RAG Collection Information

#### 進入知識工作區

1. 點選 OpenWebUI 左側選單中的 **Workspace**
2. 點選上方標籤頁中的 **"Knowledge"**
   ![chap3_knowledge_collections_v2](./images/chap3_knowledge_collections_v2.png)

3. 點選Phison_Collections即可看到該agent內的原始文檔
![chap3_knowledge_original_files](./images/chap3_knowledge_original_files.png)



> ⚠️ **提醒**：此處的知識庫僅供示範展示使用，實際 RAG 邏輯已內建於系統中。

### 🤖 Part 2：執行 Inference + RAG

#### 提問方式

1. **Normal Chat:** 直接於對話框輸入問題(不會進行collection retrieve)
2. **Agent Chat:** 
   (1) 於對話框內打 <# hashtag 符號> 並選擇Phison_Collection Agent
   (2) 在hashtag後輸入相關問題
   ![chap3_inference_hashtag](./images/chap3_inference_hashtag.png)

#### 驗證執行流程

請確認以下指標：

1. **首字元生成時間（TTFT）介於 2 ～ 8 秒**
2. **於response下方會顯示RAG reference文件**
   ![chap3_inference_llamacpp_check_new](./images/chap3_inference_llamacpp_check_new.png)

---

## Chap.4 補充操作

### 🔄 單獨重啟 Phison LlamaCpp Server

#### 重啟指令

如需單獨重啟 LlamaCpp Server，請參考：

- **檔案位置**：在桌面有一個`Start Llama Service.bat`

  ![chap4_startllamacpp_bat](./images/chap4_startllamacpp_bat.png)

- **內容說明**：該檔案包含重啟 LlamaCpp Server 的完整指令
- **執行流程**：修改相對應的設定，執行 bat 即可

```bat
set "EXE=D:\AgentBuilderClient\MaestroMcpServer\maestro\llama-server.exe"
set "MODEL=D:\AgentBuilderClient\downloads\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
set "CACHE=D:\."
set "LOG=D:\AgentBuilderClient\logs\maestro_llama.log"
```

### 🗑️ 刪除舊的對話視窗

#### 操作步驟

1. 在 OpenWebUI 左側聊天室列表中選擇要刪除的對話框
2. 點選對話框右側的 **"..."** 選單按鈕
3. 選擇 **"Delete"** 選項進行刪除
   ![chap4_delete_chat_history](./images/chap4_delete_chat_history.png)

---

## Chap.5 卸載流程

### 🗑️ 卸載步驟

#### 標準卸載流程

1. 開啟 **Windows 設定**
2. 點選 **應用程式**
3. 在應用程式列表中搜尋 **"Agentbuilder Client"**
4. 點選 **解除安裝** 按鈕
5. 依照卸載精靈指示完成卸載
   ![images/chap5_uninstall_1.png](./images/chap5_uninstall_1.png)

#### 注意事項

- 卸載前請先停止所有服務
- 卸載後會自動移除桌面捷徑
- 安裝目錄中的檔案會被自動清理

---

## Chap.6 附錄

### ⚙️ 更改 AgentBuilder Client Server IP

當 Server 的 IP 位置有更改的話，可以到安裝路徑底下的**AgentBuilderClient\scripts**資料夾中有**set_nats_ip_restart.bat**以及**reset_services.ps1**，並使用系統管理權限進行以下步驟

![images/chap6_scripts.png](./images/chap6_scripts.png)

1. 執行**set_nats_ip_restart.bat**

```
 D:\AgentBuilderClient\scripts> .\set_nats_ip_and_restart.bat <新的ip位置> 4222
```

![images/chap6_set_nets_ip.png](./images/chap6_set_nets_ip.png)

2. 執行**reset_services.ps1**

```
powershell -NoProfile -ExecutionPolicy Bypass -File .\reset_services.ps1
```

![images/chap6_reset_services.png](./images/chap6_reset_services.png)

3. 重啟電腦

### 🎯 Model Download Path

當 AgentBuilder Server Publish Model 更新後
其下載的路徑會在**AgentBuilderClient\downloads\models** 底下

![images/chap6_model_download_path.png](./images/chap6_model_download_path.png)

### ⚠️ LlamaCpp 因空間不足無法啟動

當 LlamaCpp 的 Log 出現空間不足的 Error Message 時
![images/chap6_llamacpp_space_issue.png](./images/chap6_llamacpp_space_issue.png)

1. 將 R:\底下的 Kv cache 刪除
   ![images/chap6_unexception_kvcache.png](./images/chap6_unexception_kvcache.png)

2. 重新進行**publish kv cache update**，便可以快速的重新啟動 LlamaCpp


### 🎓 Legacy (Without aiDAPTIV Solution) Setup

設置 Legacy 並啟動 OpenWebUi，需要進行以下的三個事項

1. 啟動 KM，雙擊**Start_KM.bat**
2. 啟動 LlamaCpp，雙擊**Start_Llama_Server.bat**
3. 啟動 OpenWebUi，雙擊**Start_Openwebui.bat**

更換 KM Collection Data

1. 確認 KM Service 以在運行中
2. 將 Server 製作好的 Kv Cache data 中，將以下檔案複製到相對應的內容:
   1. Source files
   ```
   C:\Users\phison\AppData\Local\openwebui_phison\open-webui\backend\data\aiDAPTIV_RAG\Phison_Collection
   ```
   2. Collection chucks.json
   ```
   KM_agent_builder_client\test_data
   ```
   3. Collection source files
   ```
   KM_agent_builder_client\test_data\Phison_Collection
   ```
3. 初始化 Vector DB
   1. 打開 cmd，並執行以下 command line
   ```bat
   curl.exe -X POST "http://localhost:13142/create_db" -H "Content-Type: application/json" -d '{ \"json_path\": \"D:/AgentBuilderClient/KMClient/KM_agent_builder_client/test_data/chunks.json\", \"collection_name\": \"Phison_Collection\" }'
   ```
   ![images/chap6_call_KM_Init.png](./images/chap6_call_KM_Init.png)

---

### 🖥️ System Configuration
**HP EliteBook 8 G1i 16 inch Notebook Next Gen AI PC**
- CPU: Intel(R) Core(TM) Ultra 7 256V (Lunar Lake)
- GPU: Intel(R) Arc(TM) 140V GPU
- DRAM: LPDDR5X 8533 MHz 16GB
- SSD slot: 1

---

## 版本修改記錄

### [1.0.0] - 2025-09-09

#### 🎯 初始版本

- 建立基礎安裝與使用手冊
- 包含 Chap.0 至 Chap.6 完整內容
- 提供詳細的安裝、啟動、使用指南

---

_最後更新：2025/09/09_
