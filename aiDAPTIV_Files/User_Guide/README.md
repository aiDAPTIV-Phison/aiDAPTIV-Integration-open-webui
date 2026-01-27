# Openwebui User Manual

### 📦 Part 1: Installation

Before starting Open-WebUI, make sure you have already started the LLM model and embedding model through middleware.

1. Please download [Open-webui_installation.zip](https://phison-software-bucket.s3.ap-northeast-1.amazonaws.com/aiDAPTIV_Application/open-webui/Open-webui_installation.zip)
2. Unzip `Open-webui_installation.zip`, you will get `open-webui.zip` and `setup_and_run.bat` in the same directory.
![1](./images/1.png)
3. Double-click `setup_and_run.bat` to launch Open-WebUI.
4. The first time you launch Open-WebUI, it will unzip `open-webui.zip`, install Python, and set up the environment. Then you can use Open-WebUI in your browser.

### ⚙️ Part 2: Setting Parameters for LLM, Embedding Model, Knowledge Management, and Open-WebUI
1. If you have already set it up, the settings will be stored in config.txt. You can directly use it. The batch file will verify that the settings still work.
![alt text](./images/image-7.png)
1. Setting the LLM model endpoint:
   You can see the model information in the "models" endpoint, and provide the endpoint with version as the LLM URL.
   ![alt text](./images/image-8.png)
   ![alt text](./images/image-9.png)
2. Setting the Embedding model endpoint for Knowledge Management:
   You can see the model information in the "models" endpoint, and provide the endpoint with version as the Embedding URL.
   ![alt text](./images/image-1.png)
   ![alt text](./images/image-4.png)
3. Setting tokenizer model for Knowledge Management:
   ![alt text](./images/image.png)
   ![alt text](./images/image-2.png)

### 🎯 Part 3: Getting Started
1. When Open-WebUI is ready, you will see the log like the following image.
![alt text](./images/image-5.png)
2. You can get started by opening it in your browser.
![alt text](./images/image-6.png)
3. Upload files to Knowledge. We have prepared some sample questions in the open-webui folder. Please wait until the LLM finishes processing.
![3](./images/3.png)


---

### 🤖 Part 4: Running Inference + RAG

#### Query Methods

1. **Normal Chat:** Directly enter your question in the chat box (will not perform collection retrieval)
2. **Agent Chat:**
   (1) Type <# hashtag symbol> in the chat box and click the collection you create.
   (2) Enter your question after the hashtag

   ![chap3_inference_hashtag](./images/chap3_inference_hashtag.png)

#### Verifying Execution Flow

Please confirm the following indicators:

1. **Time To First Token (TTFT) is between 2 ~ 8 seconds**
2. **RAG reference documents are displayed below the response**

   ![chap3_inference_llamacpp_check_new](./images/chap3_inference_llamacpp_check_new.png)

---
