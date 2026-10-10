
# 项目格式
enterprise-rag-system/
├── app/
│   ├── main.py
│   ├── ingest/        # 文档导入：加载/解析/分块/入库
│   ├── retrieve/      # 检索：向量/关键词/重排
│   ├── generate/      # 生成：拼prompt调模型
│   ├── schemas/  core/
├── data/docs/         # 放你的30份测试文档
└── tests/questions.md