import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    print("=" * 60)
    print("初始化知识库")
    print("=" * 60)
    
    mock_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "mock_data", "mock_knowledge.json")
    
    if os.path.exists(mock_data_path):
        print(f"✓ Mock数据文件已存在: {mock_data_path}")
    else:
        print(f"✗ Mock数据文件不存在，正在创建...")
        from backend.db.mock_client import MockDB
        db = MockDB()
        print("✓ Mock数据库初始化完成")
    
    print("\n" + "=" * 60)
    print("知识库初始化完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()