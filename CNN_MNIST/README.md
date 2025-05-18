## Architecture Diagram

```mermaid
flowchart TD
    %% ---------- Client ----------
    subgraph "Client"
        A["Researcher main.py"] -->|REST| B["Layer Controller"];
    end

    %% ---------- Cluster ----------
    subgraph "Kubernetes Cluster"
        %% Controller → K8s API
        B -->|calls&nbsp;K8s&nbsp;API| API["API Server"];

        %% Controller 動態產生 Layer Pods
        B -.|creates| L0["Layer Pod 0"];
        B -.|creates| LN["Layer Pod N"];

        %% ===== Forward pass =====
        B -->|/forward| L0;
        L0 -->|/forward| Lmid["⋯"];
        Lmid -->|/forward| LN;
        LN -->|output| B;

        %% ===== Back-prop =====
        B <--|/backward| LN;
        LN <--|/backward| Lmid;
        Lmid <--|/backward| L0;
        L0 <--|grad| B;

        %% ===== Persistence =====
        B -->|save&nbsp;/&nbsp;load| PVC["Persistent Volume"];
    end

    %% ---------- Style ----------
    classDef ctl fill:#ffd,stroke:#333;
    class B,L0,Lmid,LN ctl;
