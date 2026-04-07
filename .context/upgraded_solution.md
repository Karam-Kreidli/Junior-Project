# Upgraded Final Solution

## Stage 1:

**1. Input Layer: The Context Harvester**
In a standard pipeline, the background is considered "noise" and is cropped away. In this upgraded Stage 1, the background is **"signal."** When a candidate object is detected, the **Context Harvester** generates four concentric, center-aligned crops at a fixed resolution (typically **$256 \times 256$**):
• **ROI (1x):** Focuses strictly on the morphological features (fins, eyes, scales).
• **Context (3x):** Captures the "social context" (e.g., are there other fish of the same species nearby?).
• **Context (5x):** Captures the "micro-habitat" (e.g., is the fish hovering over a coral reef or a sandy patch?).
• **Full Frame:** Captures the "macro-environment" (e.g., light depth, water turbidity, and overall scene structure).

**2. Feature Extraction: The ViT Backbone**
Instead of a standard CNN, we utilize a **Vision Transformer (ViT)**—specifically **DINOv2**—as the backbone.
• **Patching**: Each of the 4 streams is broken into $16 \times 16$ patches.
• **Saliency-Guided Masking**: Before high-level feature extraction, a lightweight saliency module (like DeepLIFT or Grad-CAM) generates heatmaps to highlight the most biologically relevant pixels (e.g., scale patterns, fin shapes) . This ensures the model ignores "marine snow" and focuses on the fish's unique "fingerprint".
• **The [CLS] Token**: The ViT produces a Global Class Token ($g$) for the ROI (the "fish signature").
• **Patch Embeddings ($P$)**: The context streams produce a grid of Local Patch Embeddings (the "habitat clues")

**3. The MCEAM: Cross-Attention Fusion**
The "magic" happens in the **MCEAM**. It uses **Multi-Head Cross-Attention** to allow the fish features to "query" the environment.

**The Mathematical Logic**
The ROI serves as the **Query ($Q$)**, while the context patches serve as the **Keys (K)** and **Values ($V$)**

$F_{attn}^{(r)} = \sum_{j} \text{Softmax}\left(\frac{(W_q g) \cdot (W_k P_{r,j})^T}{\sqrt{d}}\right) (W_v P_{r,j})$
• $g$: [CLS] embedding from the ROI (the fish).

• $P_{r,j}$: $j$-th patch from context level $r$ (the environment).

• $W$: Learned weight matrices.

**4. Detection Head: CARAFE Refinement**
To solve the issue of small fish detection—a common failure point in underwater AI—the standard bilinear upsampling in the detection "neck" is replaced with **CARAFE (Content-Aware ReAssembly of Features)**.

• **Mechanism**: CARAFE uses a larger receptive field to aggregate semantic information based on the content of the image rather than fixed pixels.

• **Benefit**: This significantly improves the **mAP** and recall for small objects or fish that are partially occluded by reef structures in the Fujairah ecosystem.

**5. Supervision: The Taxonomic Anchor**
To enforce biological accuracy, the training process utilizes a **Weighted Hierarchical Loss** within the HSLM.
• **The Strategy**: The model is penalized more heavily for high-level "Family" errors than for "Species" errors early in training.
• **Goal**: This establishes a strong **"Taxonomic Anchor,"** ensuring the model achieves a low **Hierarchical Distance (HD)** and stays scientifically accurate even when the species-level details are murky.

**Attention Behaviors**
The MCEAM specifically learns three types of "marine expert" behaviors:
1. **Separated Attention:** Effectively separates the fish's silhouette from the murky water.
2. **Complementary Attention:** Notices if the substrate (sand vs. rock) matches the biological preference of the fish.
3. **Clustered Attention:** Recognizes schooling patterns, which is a major hint for identifying species like *Yellowfin Tuna*.

| **Component** | **Standard Option 3** | **MATANet-Enhanced Stage 1** |
| --- | --- | --- |
| **Model** | Faster R-CNN (CNN) | **Multi-Stream ViT (DINOv2)** |
| **Input** | 1 Image | **4 Synchronized Crops** |
| **Logic** | Box Regression | **Cross-Attention Fusion** |
| **Data Output** | $[x, y, w, h]$ | **$[x, y, w, h] +$ Context-Aware Embedding ($z$)** |

![image.png](image.png)

![image.png](image%201.png)

![image.png](image%202.png)

![image.png](image%203.png)

![image.png](image%204.png)

![image.png](image%205.png)

---

---

The tracking stage transitions the system from identifying fish in individual frames to maintaining their biological identities over time. This is critical for accurate biodiversity counts, ensuring that a single fish swimming in and out of the reef is not counted multiple times.

### 1. The Core Engine: Dual-Threshold Association

The primary technical framework utilizes **ByteTrack** logic to handle the unique visibility challenges of underwater environments. Instead of discarding low-confidence boxes, it uses them to "fill in the gaps."

- **Phase 1 (High-Confidence):** The tracker first matches detections with high scores ($\geq 0.6$) to existing tracks using IoU (Intersection over Union) and motion prediction.
- **Phase 2 (Low-Confidence Rescue):** For detections that are blurry or partially hidden (scores $0.1–0.6$), the system "rescues" them by checking if they align with where the Kalman Filter predicted the fish should be. This prevents new IDs from being created unnecessarily during periods of heavy silt or bubbles.

### 2. The Kalman Filter: Motion Prediction with CMC

The system maintains a State Vector ($\mathbf{x}$) for every fish, representing its physical movement:

$\mathbf{x} = [u, v, a, h, \dot{u}, \dot{v}, \dot{a}, \dot{h}]^T$
• $(u, v)$: The center coordinates of the fish.
• $a$: Aspect ratio of the bounding box.
• $h$: Height of the bounding box.
• $(\dot{u}, \dot{v}, \dot{a}, \dot{h})$: The respective **velocities** (how fast those numbers are changing).

**The Benefit:** If a fish is temporarily obscured by a bubble or silt, the Kalman Filter "guesses" its position for a few frames. When the fish reappears, ByteTrack re-attaches the old ID to the new detection.

- **Camera Motion Compensation (CMC):** Before state prediction, the system incorporates a CMC module to correct for background drift. In coastal environments, water surge or camera vibrations can cause the background to "move." CMC aligns the frames so the Kalman Filter tracks only the fish's biological movement, maintaining a "sticky" lock even when the camera is unstable.

### 3. Leveraging DINOv2 with EMA for Re-ID

In this architecture, DINOv2 embeddings serve as the primary "fingerprint" to distinguish visually similar individuals in crowded schools.

- **EMA-Based Feature Bank:** Instead of matching a new detection against only the previous frame, the tracker utilizes an **Exponential Moving Average (EMA)** to maintain a running "Appearance Signature" for each fish.
- **The Logic:** As a fish turns from a side-view to a front-view, its appearance changes. The EMA smooths these transitions, creating a stable, high-dimensional representation that is more resistant to temporary biological distortion or light refraction.
- **Cosine Distance:** When motion is ambiguous—for example, if two fish of the same size cross paths—the tracker calculates the Cosine Similarity between the new detection and the track’s **smoothed EMA embedding**.

### 4. The Handoff: Generating Spatiotemporal Tracklets

The most important output of Stage 2 is the generation of **Tracklets** for the final classification phase.

- **Definition:** A Tracklet is a sequence of 16–30 consecutive crops of the same individual fish.
- **Context Inclusion:** Because Stage 1 used the MCEAM module, each frame in the tracklet is already "habitat-aware," containing the fused data of the fish and its surrounding environment.
- **Structure:** Track ID #105: [Frame_01 + Context, Frame_02 + Context, ... Frame_20 + Context].

| **Feature** | **Option 3 (Standard ByteTrack)** | **New Architecture (DINOv2 + ByteTrack)** |
| --- | --- | --- |
| **Association Signal** | Primarily Motion & Overlap (IoU). | Motion + **EMA-Smoothed Fingerprints**. |
| **Motion Stability** | Vulnerable to camera surge. | **CMC-Corrected** Motion Prediction. |
| **ID Persistence** | High risk of switches in schools. | High stability; distinguishes via texture. |
| **Handling Occlusion** | Relies on Kalman Filter "guessing." | Uses **EMA Re-ID** to find fish after reappearance. |
| **Validation Metric** | Basic MOTA. | **HOTA** (Detection + Association balance). |

---

### Merging the Data: The Hierarchy of Trust

The merge happens inside a process called **Cascaded Matching**. The system manages the opinions of "The Physicist" (Kalman Filter) and "The Biometrician" (DINOv2) using a strict hierarchy:

**Step 1: The Motion "Gate" (CMC-Corrected)**

Before looking at appearance, the system uses Path 1 to establish physical possibilities. The Kalman Filter, corrected by **Camera Motion Compensation**, predicts a likely search area. If a detection appears outside this area, the match is rejected immediately—fish cannot "teleport" across the screen.

**Step 2: Primary Match (Motion-First with Visual Veto)**

The system tentatively matches detections to tracks based on the highest IoU. Before finalizing, it performs a **Veto Check** using the DINOv2 Cosine Similarity. If the similarity is too low, the system realizes they are different fish (e.g., a predator crossing in front of prey) and breaks the match.

**Step 3: Secondary Match (EMA Appearance Rescue)**

For unmatched detections where motion was erratic or occluded (e.g., a fish swimming behind a rock for 20 frames), the system relies entirely on the **EMA Feature Bank**. By comparing the DINOv2 embeddings against the gallery of lost tracks, the system can recognize unique scale patterns and re-identify the fish even when the motion signal is lost.

### Hungarian Algorithm: The Global Matchmaker

The Hungarian Algorithm acts as the mathematical optimization engine that solves the merged cost matrix.

- **Global Minimum:** Instead of picking the "best-looking" match for the first fish it sees, it looks at every possible combination in the frame to find the mathematical best fit for all fish simultaneously.
- **Optimal Assignment:** It ensures that one detection is assigned to exactly one track, preventing the error of assigning multiple IDs to the same individual.

**How it Works (A Simple Example)**
Imagine you have two fish detections ($D_1, D_2$) and two existing tracks ($T_1, T_2$). The algorithm looks at a "Cost Matrix":

|  | **Track 1 (T1)** | **Track 2 (T2)** |
| --- | --- | --- |
| **Detection 1 ($D_1$)** | Cost: 10 (Very Similar) | Cost: 90 (Different) |
| **Detection 2 ($D_2$)** | Cost: 20 (Similar) | Cost: 50 (Somewhat Similar) |

If you just picked the best match for $D_1$, you'd pick $T_1$. But then $D_2$ would be forced to take $T_2$ (Cost 50). Total Cost = **60**.

The Hungarian Algorithm checks if another combination is better. In this case, $D_1 \rightarrow T_1$ and $D_2 \rightarrow T_2$ is indeed the best (Total 60). If $D_2$ had a cost of 5 with $T_1$, the algorithm might shift $D_1$ to $T_2$ to make the *entire* system more accurate.

### Validation: Higher Order Tracking Accuracy (HOTA)

To ensure scientific validity for marine research, the project utilizes **HOTA** as the primary evaluation metric. It perfectly balances **Detection Accuracy** (finding the fish) and **Association Accuracy** (keeping the same ID), providing a robust proof of performance for biodiversity assessments.

![image.png](image%206.png)

![image.png](image%207.png)

![image.png](image%208.png)

![image.png](image%209.png)

[https://youtu.be/L7niSuVq8js](https://youtu.be/L7niSuVq8js)

---

## Stage 3:

### 1. The Input: Feature-Rich Tracklets

Instead of just sending a sequence of "dumb" image crops, Stage 2 hands over a Spatiotemporal Tracklet. Each frame in this tracklet is already "pre-digested" because it carries the MCEAM (Context) embeddings from Stage 1.

- **Content:** Each frame contains the fish's pixels plus the mathematical "summary" of the surrounding habitat (coral, sand, or blue water).
- **Tracklet Size:** Typically 16–30 frames.

### 2. Spatial Refinement: Time-Distributed ViT (DINOv2)

Each individual frame in the tracklet is passed through the DINOv2 (Vision Transformer).

- **The "Global" Eye:** Unlike a CNN that looks at local pixels, the ViT uses Global Self-Attention. It looks at the relationship between every part of the fish simultaneously.
- **Feature Extraction:** It identifies fine-grained "biomarkers"—the exact shape of the dorsal fin, the spacing of spots, or the unique curve of the gill plates.
- **Output:** A sequence of high-dimensional "feature vectors" (digital fingerprints) that describe the fish's appearance in every frame.

### 3. Temporal Logic: The LSTM "Brain"

The sequence of vectors from the ViT is fed into the LSTM. This is where the model "watches the movie" of the fish swimming.

- **Behavioral Fingerprinting:** The LSTM identifies Kinematics (swimming patterns). Example: A Grouper has a "burst-and-hover" rhythm, while a Jack or Tuna has a "high-frequency constant beat."
- **The Benefit:** Even if two fish look identical in a still photo (Spatial), the LSTM can distinguish them by their "body language" (Temporal).
- **[ADDED] Temporal Majority Voting:** The system does not rely on a single frame for the final label. It performs a "consensus vote" across the entire tracklet. If the model is uncertain in one frame due to a bubble or blur, it uses the clear views from the rest of the 30-frame sequence to stabilize the identification.

### 4. The "Scientist": Hierarchical Separation-Induced Learning (HSLM)

The final summary vector from the LSTM doesn't just guess a name. It is processed by the HSLM, which forces the AI to follow the Taxonomic Tree. The HSLM contains three distinct classification "heads":

- **Family Head:** Must decide the broad group (e.g., *Carangidae*).
- **Genus Head:** Must decide the middle group (e.g., *Caranx*).
- **Species Head:** Must decide the final ID (e.g., *Caranx ignobilis*).
- **[ADDED] Taxonomic Guardrails (Masking):** To prevent scientifically impossible predictions, the system utilizes a **Consistency Mask** during inference. If the Family head predicts "Snapper," the Genus and Species heads are restricted to only choose from valid sub-categories. This eliminates "biological leaps" (like identifying a fish as a Shark family but a Snapper species) caused by pixel noise.
- **The Separation Advantage:** By training with a Hierarchical Loss, the model learns that certain features are "Family-level" (overall body shape) while others are "Species-level" (tiny color variations). If the model is 99% sure of the Family but only 60% sure of the Species, your report will reflect that scientific uncertainty, which is vital for a Biodiversity Count.

### **5. Performance Evaluation: Hierarchical Distance (HD)**

To reach professional-grade standards for marine research, the project adopts **Hierarchical Distance (HD)** as a primary success metric.

- **The Logic:** Unlike standard accuracy, HD penalizes the model based on how "biologically far" a mistake is. Confusing two species in the same Genus results in a very low penalty, while confusing a Shark with a Snapper results in a high penalty.
- **Scientific Value:** This proves that even when the model is uncertain, it remains "biologically intelligent," providing valid data for sustainable fisheries management.

### **6. Supporting Research & Statistics**

- **1.54 Hierarchical Distance (HD):** Research proves that hierarchical models capture the true biological structure of marine data, significantly reducing major taxonomic errors compared to "flat" classification models.
    - *Reference: Lee, D. H., et al. (2026). "MATANet: A Multi-Context Attention and Taxonomy-Aware Network...".*
- **Taxonomic Consistency:** Enforcing biological constraints through masking is the "Gold Standard" for modern electronic monitoring (EM) systems in fisheries.
    - *Reference: Thilakarathna, S. N., et al. (2026). "Towards Visual Re-Identification of Fish using Fine-Grained Classification...".*
- **Weighted Hierarchical Loss:** Studies show that establishing a strong "Family" anchor early in the learning process improves final species-level recall and precision.
    - *Reference: Yang, C., et al. (2024). "FishAI: Automated hierarchical marine fish image classification with vision transformer".*

| **Output Metric** | **Value Example** | **Why it matters** |
| --- | --- | --- |
| **Track ID** | #105 | Ensures this individual isn't double-counted. |
| **Species** | *Epinephelus coioides* | Final biodiversity identification. |
| **Taxonomic Path** | Serranidae $\rightarrow$ Epinephelus | Provides scientific validity. |
| **Confidence** | 94.2% | Allows you to filter out "uncertain" detections. |

| **Feature** | **Option 3 (CNN-LSTM)** | **New Architecture (ViT-LSTM + HSLM)** |
| --- | --- | --- |
| **Backbone (Spatial)** | Standard CNN (e.g., MobileNet/ResNet). | **Vision Transformer (DINOv2).** |
| **Attention Logic** | Local (convolutions look at small areas). | **Global** (Self-attention looks at the whole fish). |
| **Temporal Head** | Standard LSTM. | **LSTM** (enhanced with Contextual Embeddings). |
| **Decision Logic** | "Flat" Classification (guesses name). | **Hierarchical Classification** (Path-based logic). |
| **Error Control** | No biological "safeguards." | **Taxonomic Consistency** (HSLM-enforced). |
| **Accuracy Source** | Primarily the appearance of the fish. | **Appearance + Environment + Behavior + Biology.** |
| **Scientific Value** | Basic Identification. | **Audit-ready Biodiversity Data.** |
| **Hardware Need** | Low to Moderate. | **High** (Large memory/VRAM for ViT). |

![image.png](image%2010.png)

![image.png](image%2011.png)

![image.png](image%2012.png)

![image.png](image%2013.png)

![image.png](image%2014.png)

## Data Summary

| **Stage** | **Data Input Type** | **Key Requirements** | **Recommended Volume** | **Technical Notes** |
| --- | --- | --- | --- | --- |
| **1. Context-Aware Detection** | **Multi-Stream Frames** (Still images)  | 4 concentric crops (1x, 3x, 5x, Full); 256x256 resolution; Bounding box labels.  | **1,500 – 3,000 labeled frames** (strided, not consecutive). | Uses a **frozen DINOv2 ViT**; the MCEAM is the primary trained component.  |
| **2. Hybrid Temporal Tracking** | **Context-Aware Proposals** (Tensors)  | **Chronological video** (no skipping); stable frame rate; DINOv2 embeddings.  | **10 – 60s clips**; variety of schooling/occlusion scenarios. | **No training.** This is a configuration phase for Kalman Filter and Hungarian Algorithm.  |
| **3. Hierarchical Classification** | **Spatiotemporal Tracklets** (Video sequences)  | **16 – 30 consecutive frames**; labels must follow a Taxonomic Tree (Family/Genus/Species).  | **30 – 50 tracklets per species**; ~1,000 frames per species. | Trains the **LSTM** (motion) and **HSLM heads** (taxonomy) using Hierarchical Loss.  |
| **4. Reporting & Analytics** | **Verified Tracked IDs** (Structured data)  | Track IDs for deduplication; Species names + Confidence scores.  | N/A (Process all video output). | An **Analytics layer** that calculates MaxN and Diversity Indices ($H', D, S$). |