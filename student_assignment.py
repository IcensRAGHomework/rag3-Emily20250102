import datetime
import chromadb
import traceback
import pandas as pd

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration


gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"
csv_file_name = "COA_OpenData.csv"

def get_db_collection():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    # 建立collection
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    if collection.count() == 0:
        # 讀取CSV檔案
        df = pd.read_csv(csv_file_name)
        print("columns: "+df.columns)

        required_columns = {"Name", "Type", "Address", "Tel", "City", "Town", "CreateDate", "HostWords"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV 缺少欄位: {required_columns - set(df.columns)}")
        for idx, row in df.iterrows():
            metadata = {
                "file_name": csv_file_name,
                "name": row["Name"],
                "type": row["Type"],
                "address": row["Address"],
                "tel": row["Tel"],
                "city": row["City"],
                "town": row["Town"],
                "date": int(datetime.datetime.strptime(row['CreateDate'], '%Y-%m-%d').timestamp())  # 轉換為時間戳
            }
            print(str(idx)+str(metadata))
            print("\n")
            # 將資料寫入 ChromaDB
            collection.add(
                ids=[str(idx)],
                metadatas=[metadata],
                documents=[row["HostWords"]]
            )
    return collection

def query_data_from_collection_hw02(collection, question, city, store_type, start_date, end_date):
    results = collection.query(
        query_texts=[question],
        n_results=10,
        include=["metadatas", "distances"],
        where={
            "$and": [
                {"date": {"$gte": int(start_date.timestamp())}},
                {"date": {"$lte": int(end_date.timestamp())}},
                {"type": {"$in": store_type}},
                {"city": {"$in": city}}
            ]
        }
    )
    print(results)
    return results

def filter_data_hw02(data, similarity):
    filtered_results = []
    for index, distance in enumerate(data["distances"][0]):
            print(distance)
            if 1-distance > similarity:
                name = data["metadatas"][0][index]["name"]
                print("name = "+ name)
                filtered_results.append(name)
    print(filtered_results)
    return filtered_results

def query_data_from_collection_hw03(collection, question, city, store_type):
    results = collection.query(
        query_texts=[question],
        n_results=10,
        include=["metadatas", "distances"],
        where={
            "$and": [
                {"type": {"$in": store_type}},
                {"city": {"$in": city}}
            ]
        }
    )

    print(results)
    print("\n")
    return results

def filter_data_hw03(data, similarity):
    store_similarity = []
    store_name = []

    for index, distance in enumerate(data["distances"][0]):
        if 1-distance > similarity:
            new_store_name = data['metadatas'][0][index].get('new_store_name', "")
            name = data['metadatas'][0][index]['name']
            store_name.append(new_store_name if new_store_name else name)
            store_similarity.append(float(1 - distance))
    
    print(store_similarity)
    print(store_name)
    # 將 store_similarity 和 store_name 配對
    paired_list = list(zip(store_similarity, store_name))
    # 根據 similarity 進行遞減排序
    sorted_paired_list = sorted(paired_list, key=lambda x: x[0], reverse= True)
    # 拆分排序後的配對列表
    sorted_store_similarity, sorted_store_name = zip(*sorted_paired_list)
    sorted_store_similarity = list(sorted_store_similarity)
    sorted_store_name = list(sorted_store_name)
    print(sorted_store_similarity)
    print(sorted_store_name)
    return sorted_store_name


def generate_hw01():
    collection = get_db_collection()
    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    print(
    "question = " + str(question) + ",\n"
    "city = " + str(city) + ",\n"
    "store_type = " + str(store_type) + ",\n"
    "start_date = " + str(start_date) + ",\n"
    "end_date = " + str(end_date)
    )
    collection = get_db_collection()
    # add filter(where)
    # query data from db collection
    data_from_db = query_data_from_collection_hw02(collection, question, city, store_type, start_date, end_date)
    # filter data based on similarity
    return filter_data_hw02(data_from_db, 0.8)

    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    print(
    "question = " + str(question) + ",\n"
    "store_name = " + str(store_name) + ",\n"
    "new_store_name = " + str(new_store_name) + ",\n"
    "city = " + str(city) + ",\n"
    "store_type = " + str(store_type)
    )
    # 找到指定店家，並在Metadata新增新的參數，名稱為 new_store_name
    collection = get_db_collection()
    get_selected_store = collection.get(where={"name": store_name})
    metadatas = [{**meta, "new_store_name": new_store_name} for meta in get_selected_store.get("metadatas", [])]
    collection.upsert(ids=get_selected_store.get("ids", []), metadatas=metadatas, documents=get_selected_store.get("documents", []))
    
    # for doc_id, meta in enumerate(collection.metadata):
    #      if isinstance(meta, dict) and meta.get('name') == store_name:
    #         meta['new_store_name'] = new_store_name
    #         collection.update_document_metadata((doc_id, meta))

    # 透過問題取得的店家名稱，如果該店家的 Metadata 有 new_store_name 參數，請用該參數來顯示新的店家名稱
    data_from_db = query_data_from_collection_hw03(collection, question, city, store_type)
    return filter_data_hw03(data_from_db, 0.8)

    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection


if __name__ == "__main__":
    print("****************hw03_1******************")
    generate_hw01()
    print("\n")
    print("****************hw03_2******************")
    generate_hw02("我要找有關茶餐點的店家", ["宜蘭縣", "新北市"], ["美食"], datetime.datetime(2024, 4, 1), datetime.datetime(2024, 5, 1))
    print("\n")
    print("****************hw03_3******************")
    generate_hw03("我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵", "耄饕客棧", "田媽媽（耄饕客棧）", ["南投縣"], ["美食"])
    print("\n")
