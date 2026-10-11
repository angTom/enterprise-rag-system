
import chromadb


chroma_client = chromadb.Client()

# 创建集合
collection = chroma_client.create_collection("my_collection")

# 要添加文档
documents = [
    "退货政策是7天无理由", "退款流程需要3个工作日", "发票开具请联系财务",
    "账号密码重置在设置页面", "会员等级分为金银铜三级", "优惠券不可叠加使用",
    "生鲜商品不支持无理由退货", "配送范围仅限中国大陆", "客服工作时间为9-18点",
    "今天的天气真不错啊"
]
# 向集合中添加文档
collection.add(
    documents=documents,
    ids=[f"doc_{id}" for id in range(len(documents))]
)

# 查询文档
query = "退货政策"
results = collection.query(
    query_texts=[query],
    n_results=10,
)
# 5. 体验返回结构
print("=== Query 返回结构解析 ===")
print(f"返回的 Keys: {list(results.keys())}")
# 遍历第一个查询文本所返回的所有结果
for i in range(len(results["ids"][0])):
    print(f"\n Top-{i+1}:")
    print(f"   ID:       {results['ids'][0][i]}")
    print(f"   Document: {results['documents'][0][i]}")
    print(f"   Distance: {results['distances'][0][i]:.4f}")