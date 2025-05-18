# 專案總覽

本專案目標是把 **CNN 網路的每一層拆解成獨立的 Pod / Service**，由一個「Layer Controller」微服務在 Kubernetes 叢集裡**動態建立、初始化、訓練、儲存與載入**這些層。如此可讓研究人員：

* **彈性地插拔或重排層**（實驗不同網路結構時無須重新打包整個模型）
* **獨立縮放或部署特定層**（例如把計算量大的卷積層排到 GPU 節點）
* **跨語言或跨框架混合組裝**（日後可替換成異質層，如 TensorRT、ONNX Runtime 等）

---

## 1. 架構一覽

![image](https://github.com/xleshing/K8S/blob/main/CNN_MNIST/architecture.png)

---

## 2. 關鍵元件與檔案

| 元件                        | 位置 / 映像                                              | 角色                                                            | 重點 API / 方法                                                             |
| ------------------------- | ---------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Layer Controller**      | `layer_controller.py` → `ycair/cnn_layer_controller` | orchestration - 接收命令後呼叫 K8s API 動態建立/刪除 Pod 與 Service、串接正反向傳遞 | `create_layers()` 建立所有層 ； `forward()` 逐層呼叫子服務前傳 ； `backward()` 反傳梯度     |
| **Layer Service (單層微服務)** | `layer_service.py` → `ycair/cnn_layer_service`       | 真正的 PyTorch 層；讀取環境變數 `LAYER_TYPE / LAYER_CONFIG` 動態建構模型       | `/initialize` 依型別建層 ； `/forward` 回傳輸出 ； `/backward` 接收上層梯度、更新權重並回傳下游梯度  |
| **Trainer (客戶端)**         | `main.py`                                            | 下載 MNIST、計算展平大小、組裝層清單、驅動訓練/測試                                 | 產生 `layers` 陣列與 `requests.post(.../create_layers)`                      |
| **K8s 佈署**                | `layer_controller.yaml`                              | 部署/Service/PVC                                                | 單副本 Deployment 與公開 5000 port                                            |
| **RBAC**                  | `layercontroller-rbac.yaml`                          | 允許 Controller 動態管理 Pod/Service                                | ClusterRole `pods, services get/list/create/delete`                     |
| **Dockerfile**            | (兩份；分別對應 Controller / Service)                       | 打包基底映像                                                        | 內容未顯示但在叢集使用 `ycair/…`                                                   |

---

## 3. 流程細節

### 3.1 建構與部署

1. **build & push**

   ```bash
   docker build -t ycair/cnn_layer_controller .
   docker build -t ycair/cnn_layer_service .
   ```
2. **RBAC + Deployment**

   ```bash
   kubectl apply -f layercontroller-rbac.yaml
   kubectl apply -f layer_controller.yaml
   ```

### 3.2 動態建立層

* `main.py` 將 `layers` 陣列 POST 至 `/create_layers`。
  Controller 逐層呼叫 `create_pod()` 建 Pod，並用 label `app: layer-N` 為 Service 綁定 。
* 每個 Pod 透過環境變數攜帶 **層型別與設定**，Service 固定對外曝露 5000 埠。

### 3.3 訓練循環

1. **/initialize**：Controller 依序轉呼叫各層 Service `/initialize`，建構 PyTorch 模型 。
2. **前向**：`main.py` 取一批 MNIST 影像 → POST `/forward` → Controller 逐層轉發；最後回傳 logits 。
3. **計算損失**：Trainer 在本地計算 `CrossEntropyLoss`、呼叫 `/backward` 把最終梯度推回 Controller。
4. **反向**：Controller 從最後一層開始，向前逐層 POST `/backward`，各層用 Adam 更新權重並回傳下一層所需梯度 。
5. **持久化**：

   * `/save_layers` 把 JSON 結構寫到 PVC&#x20;
   * `/save_model` / `/load_model` 讓整張網路能分散儲存或熱重載權重&#x20;

---

## 4. 文字化元件說明

### Layer Controller

* **REST API**

  * `POST /create_layers`: 傳入 `[{"type": "...", "config": {...}}, …]`，自動重試最多 3 次確保 Pod 啟動成功。
  * `POST /forward`: 逐層前向；若任何層失敗即回傳 500。
  * `POST /backward`: 包含 `learning_rate` 與 `output_grad`；完成後回傳 200。
  * 其他：`/initialize`、`/save_model`、`/load_model`、`/save_layers`。

* **Kubernetes 控制**
  透過 `client.CoreV1Api` 實作 CRUD，並用 `wait_for_pod_ready()` 與 `wait_for_deletion()` 同步狀態 。

### Layer Service

* **模型定義**

  * `ConvLayer`：`Conv2d → ReLU → MaxPool`。
  * `FcLayer`：`Linear → ReLU/Identity`。
* **梯度回傳**：接收 `output_grad` 用 `output_data.backward(gradient=output_grad)`，算完後 `optimizer.step()` 更新本層並把 `input_grad` 吐回 。

### Trainer (main.py)

* **資料管線**：`torchvision.datasets.MNIST` + Normalize → DataLoader。
* **動態展平大小**：`compute_flatten_size()` 依 conv/pool 超參數計算 FC in\_features 。
* **訓練迴圈**：epochs × batches → 前傳/計損失/反傳。

---

## 5. 延伸與最佳化方向

| 主題               | 說明                                                                 |
| ---------------- | ------------------------------------------------------------------ |
| **Auto-scaling** | 透過 HPA 或自訂指標（迴圈秒數、GPU/CPU 負載）來縮放指定卷積層 Pod。                         |
| **複合層**          | 可新增 BatchNorm、Dropout 等 layer\_type，在 Layer Service 中擴寫 `if…elif`。 |
| **服務網格**         | 將層間流量接入 Istio，以觀察延遲、重試與熔斷。                                         |
| **資料並行**         | 在 Controller 端一次同批 broadcast 給多個通道，於外層再做聚合，以支援多卡訓練。                |

---

