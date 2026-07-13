import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.orchestrator.harness import AgentHarness


def main():
    print("=" * 60)
    print("AI Agent Trainer Demo")
    print("=" * 60)
    
    harness = AgentHarness()
    
    sample_input = """我是一名计算机专业大二学生，有PyTorch基础但没有分布式系统经验。
我想学习"大模型LoRA微调与LLM部署"，但不知道从哪里开始、该学多少。"""
    
    print("\n用户输入:")
    print(sample_input)
    print("\n" + "=" * 60)
    print("开始运行Agent流程...")
    print("=" * 60)
    
    result = harness.run(session_id="demo_session_001", user_input=sample_input)
    
    print("\n运行结果:")
    print(f"状态: {result.get('status')}")
    print(f"迭代轮次: {result.get('iteration_round')}")
    print(f"正确率: {result.get('accuracy')}")
    print(f"覆盖率: {result.get('coverage')}")
    
    if result.get("reason"):
        print(f"\n终止原因: {result.get('reason')}")
    
    if result.get("content"):
        print("\n生成的学习内容:")
        print("-" * 40)
        content = result.get("content")
        print(content[:2000] + "..." if len(content) > 2000 else content)
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("Demo运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()