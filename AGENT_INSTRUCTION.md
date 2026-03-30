🚀 NExT-ST-Mamba: Research Codebase Blueprint

1. Project Overview & Objective

Model Name: NExT-ST-Mamba (NExT-QA Spatio-Temporal Hybrid Mamba-Transformer)

Target Task: Causal and Temporal Video Question Answering (VideoQA).

Target Dataset: NExT-QA.

Goal: Xây dựng một research codebase hoàn chỉnh, chuẩn bị cho việc submit paper. Hệ thống cần giải quyết bài toán lập luận nhân quả/thời gian trên video dài mà vẫn đảm bảo hiệu suất bộ nhớ.

Hardware Constraint: Tối ưu hóa để chạy trên 1 GPU RTX 3090/4090 (24GB VRAM) hoặc Colab T4. Yêu cầu sử dụng bfloat16, FlashAttention, và Mamba-2 kernels.

2. Core Architecture & Technical Specifications

Yêu cầu Agent implement kiến trúc dựa trên các kỹ thuật lõi sau:

2.1. Dual-Stream Language Model (BERT + SLM)

Question Encoder: Sử dụng BERT-base để encode câu hỏi.

Reasoning Decoder: Sử dụng SLM 1.5B (ví dụ: Qwen/Qwen2.5-1.5B). SLM này nhận feature đã được tinh cất từ video và sinh ra câu trả lời cuối cùng.

2.2. Vision & Temporal Processing (ActionChunk + Mamba)

ActionChunk Formulation: Dữ liệu đầu vào là các frame features được lấy từ file .h5. Cần ghép các features liên tiếp thành một ActionChunk để bảo toàn motion cues.

Hybrid Mamba-Transformer: Sử dụng kiến trúc Mamba làm xương sống (backbone) để xử lý chuỗi sequence video dài với độ phức tạp $O(L)$.

2.3. Relational & Causal Modeling (ST-Graph)

Spatio-Temporal Graph (ST-Graph): Áp dụng Graph Neural Network (GNN). Kết nối các node theo không gian và thời gian để nắm bắt sự thay đổi trạng thái và tương tác nhân quả.

2.4. Memory & Compute Optimization (ShaRP Pruning)

Adaptive Token Pruning (ShaRP): Triển khai module cắt tỉa token. Đo lường lưu lượng thông tin của từng visual token để loại bỏ token tĩnh (background), giảm tải KV Cache cho SLM phía sau.

2.5. Training Strategy & Alignment (GRPO)

RLHF / Alignment: KHÔNG dùng PPO. BẮT BUỘC dùng GRPO (Group Relative Policy Optimization) để tối ưu hóa trực tiếp policy, tiết kiệm 40-60% VRAM.

Reward Functions: Cần định nghĩa: Causal Reward (sinh đúng lý do) và Temporal Accuracy Reward (focus đúng timestamp).

Anti-Shortcut Mechanism: Kỹ thuật Frame Shuffling và Masking ngẫu nhiên các ActionChunk trong lúc pre-training.

3. Expected Directory Structure

Agent phải khởi tạo project theo cấu trúc sau. Lưu ý kỹ phần thư mục datasets/:

NExT-ST-Mamba/
├── configs/                  # Chứa file cấu hình yaml (Hydra config)
│   ├── model/next_st_mamba.yaml
│   ├── data/nextqa.yaml
│   └── train/grpo_training.yaml
├── datasets/                 # Thư mục chứa data gốc (Đưa vào .gitignore)
│   ├── nextqa/
│   │   ├── train.csv         # File chứa câu hỏi, lựa chọn và label
│   │   ├── val.csv
│   │   ├── map_vid_vidorID.json # Map giữa index và video ID thực tế
│   │   ├── glove_embed.npy   # (Optional)
│   │   └── vocab.pkl         # (Optional)
│   └── vid_feat/
│       ├── app_mot_train.h5  # File HDF5 chứa Appearance + Motion features (Train)
│       └── app_mot_val.h5    # File HDF5 chứa Appearance + Motion features (Val)
├── data/                     # Thư mục chứa SOURCE CODE xử lý data
│   ├── dataloader.py         # PyTorch Lightning DataModule
│   ├── nextqa_dataset.py     # Class xử lý NExT-QA dataset (sử dụng h5py để đọc file .h5)
│   └── transforms.py         # Logic tạo ActionChunk & Frame Shuffling/Masking
├── models/
│   ├── components/
│   │   ├── text_encoder.py   
│   │   ├── mamba_layer.py    
│   │   ├── st_graph.py       
│   │   └── sharp_pruner.py   
│   ├── decoder/
│   │   └── qwen_reasoner.py  
│   └── next_st_mamba.py      # Main model file
├── training/
│   ├── lightning_module.py   
│   ├── grpo_trainer.py       
│   └── reward_funcs.py       
├── scripts/
│   ├── train.py              
│   └── evaluate.py           
├── utils/
│   ├── metrics.py            
│   └── visualizer.py         
├── requirements.txt
└── README.md


4. Strict Coding Directives for AI Agent

Framework & Libraries: PyTorch >= 2.1, PyTorch Lightning, transformers, mamba-ssm, causal-conv1d, và đặc biệt là h5py (để đọc dữ liệu .h5).

Memory Efficiency: Khởi tạo model mặc định với torch.bfloat16. Dùng Gradient Checkpointing trong file next_st_mamba.py.

Type Hinting: Bắt buộc có Python Type Hinting và Docstring rõ ràng.

Modularity & Quality: KHÔNG được viết code "placeholder" (pass, TODO). Phải implement logic thực tế.

5. Master Execution Plan for Agent (Toàn bộ dự án)

Agent hãy đóng vai trò là một AI Research Engineer tự trị. Thực hiện tuần tự các Phase dưới đây.
Chỉ dẫn: Hoàn thành Phase 1 và 2 trong lần đầu tiên. Cuối mỗi phản hồi, hãy hỏi người dùng xem có muốn tiếp tục Phase tiếp theo không.

Phase 1: Project Scaffolding & Setup

Tạo toàn bộ cấu trúc thư mục như Phần 3.

Viết requirements.txt (nhớ thêm h5py, pandas).

Khởi tạo các file cấu hình YAML.

Phase 2: Core Neural Components

Code chi tiết sharp_pruner.py, st_graph.py, mamba_layer.py, text_encoder.py, qwen_reasoner.py.

Phase 3: Model Assembly

Ghép nối các components vào models/next_st_mamba.py với luồng tensor rõ ràng.

Phase 4: Data Pipeline (HDF5 Loading) - Rất Quan Trọng

Agent BẮT BUỘC dùng thư viện h5py để xử lý file đặc trưng.

Trong data/nextqa_dataset.py, load train.csv. Khi __getitem__ được gọi, hãy sử dụng video_id từ row của CSV (và map qua map_vid_vidorID.json nếu cần) làm KEY để query tensor đặc trưng từ file app_mot_train.h5 bằng h5py.

Lưu ý: Tránh mở/đóng file .h5 liên tục trong __getitem__. Mở file .h5 1 lần trong __init__ hoặc cache tensor hợp lý.

Implement ActionChunk và Frame Shuffling trong transforms.py.

Phase 5: GRPO Training & Alignment

Viết thuật toán GRPO (tiết kiệm VRAM) tại training/grpo_trainer.py.

Cài đặt Causal & Temporal Rewards tại training/reward_funcs.py.

Đóng gói vòng lặp huấn luyện vào training/lightning_module.py.

Phase 6: Entry points & Utils

Hoàn thiện train.py, evaluate.py, và metrics.py.

